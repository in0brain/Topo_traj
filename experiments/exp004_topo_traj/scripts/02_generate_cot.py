import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ANSWER_MARKER = "####"
STEP_RE = re.compile(r"Step\s+\d+\s*[:：]", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed Step-format CoT outputs.")
    parser.add_argument("--config", required=True, help="Path to topo_traj config YAML.")
    parser.add_argument("--model-name", default=None, help="Override model.model_name from config.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N input samples.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip IDs already present in the output JSONL. Use --no-resume to append duplicates.",
    )
    parser.add_argument("--max-retries", type=int, default=2, help="Retry invalid generations up to N times.")
    parser.add_argument("--overwrite", action="store_true", help="Clear existing output before generation.")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL does not exist: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc
    return records


def existing_output_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = record.get("id")
            if record_id is not None:
                done_ids.add(str(record_id))
    return done_ids


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = text.replace(",", "").replace("$", "").replace("%", "")
    text = re.sub(r"\s+", "", text)
    if not text:
        return ""

    match = NUMBER_RE.search(text)
    if match:
        text = match.group(0).replace(",", "")

    try:
        decimal = Decimal(text)
    except InvalidOperation:
        return text

    normalized = decimal.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.to_integral())
    return format(normalized, "f").rstrip("0").rstrip(".")


def answers_equal(left: Any, right: Any) -> bool:
    left_norm = normalize_answer(left)
    right_norm = normalize_answer(right)
    if not left_norm or not right_norm:
        return left_norm == right_norm
    try:
        return Decimal(left_norm) == Decimal(right_norm)
    except InvalidOperation:
        return left_norm == right_norm


def extract_last_number(text: str) -> str:
    matches = NUMBER_RE.findall(text or "")
    if not matches:
        return ""
    return normalize_answer(matches[-1])


def final_segment_after_last_marker(text: str) -> str:
    if not text or ANSWER_MARKER not in text:
        return ""
    return text.rsplit(ANSWER_MARKER, 1)[1]


def extract_answer_from_raw(raw_output: str) -> str:
    if not raw_output:
        return ""
    if ANSWER_MARKER in raw_output:
        return extract_last_number(final_segment_after_last_marker(raw_output))
    return extract_last_number(raw_output[-800:])


def count_steps(text: str) -> int:
    if not text:
        return 0
    return len(STEP_RE.findall(text))


def canonicalize_step_headings(text: str) -> tuple[str, bool]:
    if not text:
        return "", False

    pattern = re.compile(r"(?m)^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*Step\s+(\d+)\s*[:：]\s*(?:\*\*)?")

    def repl(match: re.Match[str]) -> str:
        return f"Step {match.group(1)}: "

    canonical = pattern.sub(repl, text)
    return canonical, canonical != text


def canonicalize_output(raw_output: str, pred_answer: str) -> tuple[str, bool, bool]:
    canonical, canonicalized_steps = canonicalize_step_headings(raw_output)
    canonicalized_marker = False
    if ANSWER_MARKER not in canonical and pred_answer:
        canonical = canonical.rstrip() + f"\n{ANSWER_MARKER} {pred_answer}"
        canonicalized_marker = True
    return canonical, canonicalized_marker, canonicalized_steps


def strengthen_prompt(base_prompt: str) -> str:
    extra_rules = """Additional mandatory constraints:
- You must write at least 3 reasoning steps.
- Step 1 must identify known quantities.
- Step 2 must compute an intermediate relation.
- Step 3 must compute the final answer.
- Even if the problem is simple, split it into at least three steps.
- The final line must be exactly: #### <number>
- Do not write anything after the final answer.
"""
    if "You must write at least 3 reasoning steps." in base_prompt:
        return base_prompt
    marker = "\nProblem:"
    if marker in base_prompt:
        return base_prompt.replace(marker, "\n" + extra_rules + marker, 1)
    return base_prompt.rstrip() + "\n\n" + extra_rules


def prompt_for_retry(prompt: str, attempt: int) -> str:
    if attempt <= 0:
        return prompt
    retry_note = f"""Retry attempt {attempt}: The previous output was invalid.
You must include at least 3 lines starting exactly with Step N: and a final line exactly like #### <number>.
Do not use Markdown headings. Do not write anything after the final answer.
"""
    return prompt.rstrip() + "\n\n" + retry_note


def resolve_torch_dtype(value: Any) -> Any:
    if value is None or str(value).lower() == "auto":
        return "auto"
    lowered = str(value).lower()
    if lowered in {"float16", "fp16", "torch.float16"}:
        return torch.float16
    if lowered in {"bfloat16", "bf16", "torch.bfloat16"}:
        return torch.bfloat16
    if lowered in {"float32", "fp32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {value}")


def select_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def load_model_and_tokenizer(model_cfg: dict[str, Any], model_name_override: str | None) -> tuple[Any, Any, torch.device, str]:
    model_name = model_name_override or model_cfg["model_name"]
    local_files_only = bool(model_cfg.get("local_files_only", False))
    torch_dtype = resolve_torch_dtype(model_cfg.get("torch_dtype", "auto"))
    device = select_device(str(model_cfg.get("device", "auto")))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"selected device: {device}")
    print(f"model name/path: {model_name}")
    return tokenizer, model, device, model_name


def render_generation_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def generate_once(
    tokenizer: Any,
    model: Any,
    device: torch.device,
    prompt: str,
    generation_kwargs: dict[str, Any],
) -> str:
    text = render_generation_text(tokenizer, prompt)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_len = int(inputs["input_ids"].shape[1])

    with torch.no_grad():
        generated = model.generate(**inputs, **generation_kwargs)

    output_ids = generated[0, input_len:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def build_generation_kwargs(model_cfg: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    max_new_tokens = int(model_cfg.get("max_new_tokens", 512))
    temperature = float(model_cfg.get("temperature", 0.0))
    do_sample = bool(model_cfg.get("do_sample", False))
    if temperature <= 0.0:
        do_sample = False

    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample and temperature > 0.0:
        kwargs["temperature"] = temperature
    return kwargs


def make_record(
    sample: dict[str, Any],
    prompt: str,
    raw_output: str,
    model_name: str,
    generation_config: dict[str, Any],
    retry_count: int,
    error: str | None = None,
) -> dict[str, Any]:
    pred_answer = extract_answer_from_raw(raw_output)
    canonical_output, canonicalized_marker, canonicalized_steps = canonicalize_output(raw_output, pred_answer)
    raw_step_count = count_steps(raw_output)
    canonical_step_count = count_steps(canonical_output)
    has_raw_marker = ANSWER_MARKER in raw_output
    has_canonical_marker = ANSWER_MARKER in canonical_output
    final_answer_raw_segment = final_segment_after_last_marker(raw_output)
    is_correct = answers_equal(pred_answer, sample.get("gold_answer"))

    record = {
        "id": sample.get("id"),
        "question": sample.get("question"),
        "gold_answer": sample.get("gold_answer"),
        "prompt": prompt,
        "model_output_raw": raw_output,
        "model_output": canonical_output,
        "pred_answer": pred_answer,
        "is_correct": is_correct,
        "num_steps": canonical_step_count,
        "raw_step_count": raw_step_count,
        "has_raw_final_marker": has_raw_marker,
        "has_canonical_final_marker": has_canonical_marker,
        "retry_count": retry_count,
        "final_answer_raw_segment": final_answer_raw_segment,
        "has_marker_raw": has_raw_marker,
        "has_marker": has_canonical_marker,
        "canonicalized_marker": canonicalized_marker,
        "canonicalized_steps": canonicalized_steps,
        "model_name": model_name,
        "generation_config": generation_config,
    }
    if error is not None:
        record["error"] = error
    return record


def is_generation_valid(record: dict[str, Any]) -> bool:
    return (
        bool(record.get("has_raw_final_marker"))
        and int(record.get("raw_step_count", 0)) >= 3
        and bool(record.get("pred_answer"))
    )


def append_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def summarize(records: list[dict[str, Any]], total_input: int, already_done: int, output_path: Path) -> None:
    generated_this_run = len(records)
    success_count = sum(1 for record in records if "error" not in record)
    raw_marker_count = sum(1 for record in records if record.get("has_raw_final_marker", False))
    canonical_marker_count = sum(1 for record in records if record.get("has_canonical_final_marker", False))
    raw_step_positive = sum(1 for record in records if int(record.get("raw_step_count", 0)) > 0)
    raw_step_ge_3 = sum(1 for record in records if int(record.get("raw_step_count", 0)) >= 3)
    correct_count = sum(1 for record in records if record.get("is_correct", False))

    denom = max(generated_this_run, 1)
    print(f"total_input: {total_input}")
    print(f"already_done: {already_done}")
    print(f"generated_this_run: {generated_this_run}")
    print(f"success_count: {success_count}")
    print(f"raw_marker_rate: {raw_marker_count / denom:.4f}")
    print(f"canonical_marker_rate: {canonical_marker_count / denom:.4f}")
    print(f"raw_step_parse_rate: {raw_step_positive / denom:.4f}")
    print(f"num_steps_ge_3_rate: {raw_step_ge_3 / denom:.4f}")
    print(f"accuracy: {correct_count / denom:.4f}")
    print(f"output_path: {output_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths_cfg = cfg["paths"]
    model_cfg = cfg["model"]

    input_path = Path(paths_cfg["sample_file"])
    output_path = Path(paths_cfg["generations_file"])
    debug_preview_path = Path(paths_cfg.get("debug_preview_file", ""))

    samples = read_jsonl(input_path)
    if args.limit is not None:
        samples = samples[: args.limit]

    done_ids: set[str] = set()
    if args.overwrite:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
    elif args.resume:
        done_ids = existing_output_ids(output_path)

    already_done = sum(1 for sample in samples if str(sample.get("id")) in done_ids)
    samples_to_generate = [sample for sample in samples if str(sample.get("id")) not in done_ids]

    tokenizer, model, device, model_name = load_model_and_tokenizer(model_cfg, args.model_name)
    generation_kwargs = build_generation_kwargs(model_cfg, tokenizer)
    generation_config = {
        "max_new_tokens": int(model_cfg.get("max_new_tokens", 512)),
        "temperature": float(model_cfg.get("temperature", 0.0)),
        "do_sample": bool(generation_kwargs.get("do_sample", False)),
    }

    prompt_template = cfg["prompt"]["template"]
    generated_records: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as out_f:
        for sample in tqdm(samples_to_generate, desc="Generating CoT"):
            base_prompt = strengthen_prompt(prompt_template.format(question=sample.get("question", "")))
            best_record: dict[str, Any] | None = None
            max_attempt = max(int(args.max_retries), 0)

            for attempt in range(max_attempt + 1):
                current_prompt = prompt_for_retry(base_prompt, attempt)
                try:
                    raw_output = generate_once(
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        prompt=current_prompt,
                        generation_kwargs=generation_kwargs,
                    )
                    record = make_record(
                        sample=sample,
                        prompt=current_prompt,
                        raw_output=raw_output,
                        model_name=model_name,
                        generation_config=generation_config,
                        retry_count=attempt,
                    )
                    best_record = record
                    if is_generation_valid(record):
                        break
                except Exception as exc:
                    best_record = make_record(
                        sample=sample,
                        prompt=current_prompt,
                        raw_output="",
                        model_name=model_name,
                        generation_config=generation_config,
                        retry_count=attempt,
                        error=repr(exc),
                    )
                    break

            assert best_record is not None
            append_record(out_f, best_record)
            generated_records.append(best_record)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if generated_records and str(debug_preview_path):
        debug_preview_path.parent.mkdir(parents=True, exist_ok=True)
        with open(debug_preview_path, "w", encoding="utf-8") as f:
            for record in generated_records[:5]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summarize(
        records=generated_records,
        total_input=len(samples),
        already_done=already_done,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
