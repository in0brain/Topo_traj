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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed Step-format CoT for exp004_topo_traj.")
    parser.add_argument("--config", required=True, help="Path to topo_traj_v0.yaml.")
    parser.add_argument("--model-name", default=None, help="Optional HuggingFace model override.")
    parser.add_argument("--limit", type=int, default=None, help="Only generate the first N samples.")
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_answer(answer: str) -> str:
    return answer.replace(",", "").replace("$", "").strip()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def extract_pred_answer(text: str, marker: str = "####") -> str:
    if marker in text:
        return normalize_answer(text.rsplit(marker, 1)[-1])
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return normalize_answer(numbers[-1]) if numbers else ""


def numeric_equal(left: str, right: str) -> bool:
    try:
        return Decimal(normalize_answer(left)) == Decimal(normalize_answer(right))
    except (InvalidOperation, ValueError):
        return normalize_answer(left) == normalize_answer(right)


def count_steps(text: str) -> int:
    return len(re.findall(r"Step\s+\d+\s*:", text))


def canonicalize_output(raw_output: str, pred_answer: str) -> tuple[str, bool, bool]:
    step_pattern = re.compile(
        r"(?m)^(?P<indent>\s*)(?:#{1,6}\s*)?(?:\*\*)?\s*Step\s+(?P<num>\d+)\s*[:：](?:\*\*)?"
    )

    def replace_step_heading(match: re.Match[str]) -> str:
        return f"{match.group('indent')}Step {match.group('num')}:"

    step_normalized = step_pattern.sub(replace_step_heading, raw_output)
    canonicalized_steps = step_normalized != raw_output

    if "####" in step_normalized or not pred_answer:
        return step_normalized, False, canonicalized_steps

    canonical_output = f"{step_normalized.rstrip()}\n#### {pred_answer}"
    return canonical_output, True, canonicalized_steps


def select_device(config_device: str) -> str:
    if config_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return config_device


def select_dtype(dtype_name: str, device: str) -> torch.dtype | str:
    if dtype_name != "auto":
        return getattr(torch, dtype_name)
    if device == "cuda":
        return torch.float16
    return torch.float32


def build_generation_kwargs(model_config: dict[str, Any]) -> dict[str, Any]:
    temperature = float(model_config.get("temperature", 0.0))
    do_sample = as_bool(model_config.get("do_sample", False))
    if temperature == 0.0:
        do_sample = False

    kwargs = {
        "max_new_tokens": int(model_config.get("max_new_tokens", 512)),
        "do_sample": do_sample,
    }
    if do_sample:
        kwargs["temperature"] = temperature
    return kwargs


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    sample_file = Path(config["paths"]["sample_file"])
    generations_file = Path(config["paths"]["generations_file"])
    debug_preview_file = Path(config["paths"]["debug_preview_file"])
    prompt_template = config["prompt"]["template"]
    answer_marker = config.get("evaluation", {}).get("answer_marker", "####")

    model_config = config["model"]
    model_name = args.model_name or model_config["model_name"]
    local_files_only = as_bool(model_config.get("local_files_only", False))
    device = select_device(model_config.get("device", "auto"))
    dtype = select_dtype(model_config.get("torch_dtype", "auto"), device)
    generation_kwargs = build_generation_kwargs(model_config)
    generation_config_record = {
        "max_new_tokens": generation_kwargs["max_new_tokens"],
        "temperature": float(model_config.get("temperature", 0.0)),
        "do_sample": generation_kwargs["do_sample"],
    }

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Local files only: {local_files_only}")

    records = read_jsonl(sample_file)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be a positive integer.")
        records = records[: args.limit]
        print(f"Limit: {args.limit}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()

    outputs: list[dict[str, Any]] = []
    for record in tqdm(records, desc="Generating CoT"):
        prompt = prompt_template.format(question=record["question"])
        output_record = {
            "id": record["id"],
            "question": record["question"],
            "gold_answer": record["gold_answer"],
            "prompt": prompt,
            "model_output_raw": "",
            "model_output": "",
            "pred_answer": "",
            "is_correct": False,
            "num_steps": 0,
            "has_marker_raw": False,
            "has_marker": False,
            "canonicalized_marker": False,
            "canonicalized_steps": False,
            "model_name": model_name,
            "generation_config": generation_config_record,
        }

        try:
            messages = [{"role": "user", "content": prompt}]
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                input_text = prompt

            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    pad_token_id=tokenizer.eos_token_id,
                    **generation_kwargs,
                )
            new_tokens = generated_ids[0, inputs["input_ids"].shape[-1] :]
            raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            pred_answer = extract_pred_answer(raw_output, marker=answer_marker)
            canonical_output, canonicalized_marker, canonicalized_steps = canonicalize_output(
                raw_output,
                pred_answer,
            )
            output_record["model_output_raw"] = raw_output
            output_record["model_output"] = canonical_output
            output_record["pred_answer"] = pred_answer
            output_record["is_correct"] = numeric_equal(pred_answer, record["gold_answer"])
            output_record["num_steps"] = count_steps(canonical_output)
            output_record["has_marker_raw"] = answer_marker in raw_output
            output_record["has_marker"] = answer_marker in canonical_output
            output_record["canonicalized_marker"] = canonicalized_marker
            output_record["canonicalized_steps"] = canonicalized_steps
        except Exception as exc:
            output_record["error"] = repr(exc)

        outputs.append(output_record)

    write_jsonl(generations_file, outputs)
    write_jsonl(debug_preview_file, outputs[:5])

    total = len(outputs)
    successes = sum(1 for item in outputs if "error" not in item)
    step_rate = sum(1 for item in outputs if item.get("num_steps", 0) > 0) / total if total else 0.0
    raw_marker_rate = sum(1 for item in outputs if item.get("has_marker_raw")) / total if total else 0.0
    canonical_marker_rate = sum(1 for item in outputs if item.get("has_marker")) / total if total else 0.0
    accuracy = sum(1 for item in outputs if item.get("is_correct")) / total if total else 0.0

    print(f"Total samples: {total}")
    print(f"Successful generations: {successes}")
    print(f"Raw marker rate: {raw_marker_rate:.4f}")
    print(f"Canonical marker rate: {canonical_marker_rate:.4f}")
    print(f"Step parse rate: {step_rate:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Generations path: {generations_file}")
    print(f"Debug preview path: {debug_preview_file}")


if __name__ == "__main__":
    main()
