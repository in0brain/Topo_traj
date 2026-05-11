import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


STEP_PATTERN = re.compile(r"Step\s+\d+\s*:")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache selected hidden states and attentions with teacher-forcing forward."
    )
    parser.add_argument("--config", required=True, help="Path to topo_traj_v0.yaml.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N valid samples.")
    parser.add_argument("--model-name", default=None, help="Optional model path/name override.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute cache files that already exist.")
    parser.add_argument("--no-attentions", action="store_true", help="Only cache hidden states.")
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


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def count_steps(text: str) -> int:
    return len(STEP_PATTERN.findall(text))


def select_device(config_device: str) -> str:
    if config_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return config_device


def select_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name != "auto":
        return getattr(torch, dtype_name)
    if device == "cuda":
        return torch.float16
    return torch.float32


def select_cache_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported cache_dtype: {dtype_name!r}")


def load_model(
    model_name: str,
    local_files_only: bool,
    torch_dtype: torch.dtype,
    attn_implementation: str | None,
) -> AutoModelForCausalLM:
    model_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except TypeError as exc:
        if "attn_implementation" not in model_kwargs:
            raise
        print(
            "Warning: this transformers/model combination did not accept "
            f"attn_implementation={attn_implementation!r}; retrying without it. Error: {exc}"
        )
        model_kwargs.pop("attn_implementation", None)
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)


def get_record_marker_and_steps(record: dict[str, Any]) -> tuple[bool, int]:
    model_output = record.get("model_output", "")
    has_marker = record.get("has_marker")
    if has_marker is None:
        has_marker = "####" in model_output
    else:
        has_marker = as_bool(has_marker)

    num_steps = record.get("num_steps")
    if num_steps is None:
        num_steps = count_steps(model_output)
    else:
        num_steps = int(num_steps)

    return has_marker, num_steps


def build_teacher_forcing_text(
    tokenizer: AutoTokenizer,
    prompt: str,
    model_output: str,
) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": model_output},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return prompt + "\n\n" + model_output


def make_manifest_record(
    record_id: str,
    status: str,
    selected_layers: list[int],
    reason: str | None = None,
    cache_path: str | None = None,
    seq_len: int | None = None,
    has_hidden_states: bool = False,
    has_attentions: bool = False,
    label: int | None = None,
    is_correct: bool | None = None,
    num_steps: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    manifest_record = {
        "id": record_id,
        "status": status,
        "cache_path": cache_path,
        "seq_len": seq_len,
        "selected_layers": selected_layers,
        "has_hidden_states": has_hidden_states,
        "has_attentions": has_attentions,
        "label": label,
        "is_correct": is_correct,
        "num_steps": num_steps,
        "reason": reason,
    }
    if error:
        manifest_record["error"] = error
    return manifest_record


def manifest_from_existing_cache(
    cache_path: Path,
    selected_layers: list[int],
) -> dict[str, Any]:
    cached = torch.load(cache_path, map_location="cpu")
    try:
        return make_manifest_record(
            record_id=cached["id"],
            status="cached",
            selected_layers=selected_layers,
            reason=None,
            cache_path=str(cache_path),
            seq_len=int(cached.get("seq_len", 0)),
            has_hidden_states=bool(cached.get("selected_hidden_states")),
            has_attentions=bool(cached.get("selected_attentions")),
            label=cached.get("label"),
            is_correct=cached.get("is_correct"),
            num_steps=cached.get("num_steps"),
        )
    finally:
        del cached


def validate_selected_layers(
    selected_layers: list[int],
    hidden_states: tuple[torch.Tensor, ...],
    attentions: tuple[torch.Tensor, ...] | None,
    save_attentions: bool,
) -> None:
    for layer_idx in selected_layers:
        hidden_state_idx = layer_idx + 1
        if hidden_state_idx >= len(hidden_states):
            raise IndexError(
                f"Selected layer {layer_idx} maps to hidden_states[{hidden_state_idx}], "
                f"but only {len(hidden_states)} hidden-state tensors were returned."
            )
        if save_attentions:
            if attentions is None:
                raise RuntimeError(
                    "Model did not return attentions. Check cache.attn_implementation=eager."
                )
            if layer_idx >= len(attentions):
                raise IndexError(
                    f"Selected layer {layer_idx} maps to attentions[{layer_idx}], "
                    f"but only {len(attentions)} attention tensors were returned."
                )
            if attentions[layer_idx] is None:
                raise RuntimeError(
                    f"attentions[{layer_idx}] is None. Check cache.attn_implementation=eager."
                )


def cache_one_record(
    record: dict[str, Any],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    cache_path: Path,
    selected_layers: list[int],
    layer_indexing: str,
    save_hidden_states: bool,
    save_attentions: bool,
    cache_dtype_name: str,
    cache_dtype: torch.dtype,
    max_seq_len: int,
    model_name: str,
) -> dict[str, Any]:
    record_id = record["id"]
    prompt = record["prompt"]
    model_output = record.get("model_output", "")
    has_marker, num_steps = get_record_marker_and_steps(record)
    is_correct = as_bool(record.get("is_correct", False))
    label = 0 if is_correct else 1
    outputs = None

    text = build_teacher_forcing_text(tokenizer, prompt, model_output)
    inputs = tokenizer(text, return_tensors="pt")
    seq_len = int(inputs["input_ids"].shape[-1])
    print(f"[{record_id}] seq_len={seq_len}")

    if seq_len > max_seq_len:
        return make_manifest_record(
            record_id=record_id,
            status="skipped",
            selected_layers=selected_layers,
            reason="seq_len_exceeds_max",
            seq_len=seq_len,
            label=label,
            is_correct=is_correct,
            num_steps=num_steps,
        )

    input_ids_cpu = inputs["input_ids"][0].detach().cpu()
    tokens = tokenizer.convert_ids_to_tokens(input_ids_cpu.tolist())
    inputs = {key: value.to(device) for key, value in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                output_hidden_states=True,
                output_attentions=save_attentions,
                use_cache=False,
            )

        if outputs.hidden_states is None:
            raise RuntimeError("Model did not return hidden_states.")
        validate_selected_layers(
            selected_layers=selected_layers,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            save_attentions=save_attentions,
        )

        selected_hidden_states = {}
        if save_hidden_states:
            for layer_idx in selected_layers:
                selected_hidden_states[str(layer_idx)] = (
                    outputs.hidden_states[layer_idx + 1][0].detach().to(cache_dtype).cpu()
                )

        selected_attentions = {}
        if save_attentions:
            for layer_idx in selected_layers:
                selected_attentions[str(layer_idx)] = (
                    outputs.attentions[layer_idx][0].detach().to(cache_dtype).cpu()
                )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "id": record_id,
                "input_ids": input_ids_cpu,
                "tokens": tokens,
                "seq_len": seq_len,
                "selected_layers": selected_layers,
                "layer_indexing": layer_indexing,
                "hidden_state_index_rule": "hidden_states[layer_idx + 1]",
                "attention_index_rule": "attentions[layer_idx]",
                "selected_hidden_states": selected_hidden_states,
                "selected_attentions": selected_attentions,
                "label": label,
                "is_correct": is_correct,
                "num_steps": num_steps,
                "has_marker": has_marker,
                "model_name": model_name,
                "prompt": prompt,
                "model_output": model_output,
                "model_output_raw": record.get("model_output_raw", ""),
                "cache_dtype": cache_dtype_name,
            },
            cache_path,
        )

        return make_manifest_record(
            record_id=record_id,
            status="cached",
            selected_layers=selected_layers,
            reason=None,
            cache_path=str(cache_path),
            seq_len=seq_len,
            has_hidden_states=bool(selected_hidden_states),
            has_attentions=bool(selected_attentions),
            label=label,
            is_correct=is_correct,
            num_steps=num_steps,
        )
    finally:
        if outputs is not None:
            del outputs
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    paths_config = config["paths"]
    model_config = config["model"]
    cache_config = config["cache"]

    generations_file = Path(paths_config["generations_file"])
    forward_dir = Path(cache_config["forward_dir"])
    manifest_file = Path(cache_config["manifest_file"])
    selected_layers = [int(layer_idx) for layer_idx in cache_config["selected_layers"]]
    layer_indexing = cache_config["layer_indexing"]
    save_hidden_states = as_bool(cache_config.get("save_hidden_states", True))
    save_attentions = as_bool(cache_config.get("save_attentions", True)) and not args.no_attentions
    cache_dtype_name = cache_config.get("cache_dtype", "float16")
    cache_dtype = select_cache_dtype(cache_dtype_name)
    max_seq_len = int(cache_config.get("max_seq_len", 2048))
    valid_only = as_bool(cache_config.get("valid_only", True))
    min_steps = int(cache_config.get("min_steps", 2))
    attn_implementation = cache_config.get("attn_implementation")

    model_name = args.model_name or model_config["model_name"]
    local_files_only = as_bool(model_config.get("local_files_only", False))
    device = select_device(model_config.get("device", "auto"))
    torch_dtype = select_dtype(model_config.get("torch_dtype", "auto"), device)

    print(f"Selected device: {device}")
    print(f"Torch dtype: {torch_dtype}")
    print(f"Model: {model_name}")
    print(f"Selected layers: {selected_layers}")
    print(f"Save attentions: {save_attentions}")

    records = read_jsonl(generations_file)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model = load_model(
        model_name=model_name,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    model.to(device)
    model.eval()

    manifest_records = []
    valid_records = 0
    processed_valid_records = 0
    seq_lens = []

    for record in tqdm(records, desc="Caching forward pass"):
        record_id = record["id"]
        model_output = record.get("model_output", "")
        has_marker, num_steps = get_record_marker_and_steps(record)
        is_valid = bool(model_output) and has_marker and num_steps >= min_steps

        if not is_valid and valid_only:
            manifest_records.append(
                make_manifest_record(
                    record_id=record_id,
                    status="skipped",
                    selected_layers=selected_layers,
                    reason="invalid_marker_or_steps",
                    is_correct=as_bool(record.get("is_correct", False)),
                    num_steps=num_steps,
                )
            )
            continue

        if args.limit is not None and processed_valid_records >= args.limit:
            break

        valid_records += 1
        cache_path = forward_dir / f"{record_id}.pt"
        try:
            if cache_path.exists() and not args.overwrite:
                manifest_record = manifest_from_existing_cache(cache_path, selected_layers)
                manifest_records.append(manifest_record)
                if manifest_record.get("seq_len"):
                    seq_lens.append(int(manifest_record["seq_len"]))
                processed_valid_records += 1
                continue

            manifest_record = cache_one_record(
                record=record,
                tokenizer=tokenizer,
                model=model,
                device=device,
                cache_path=cache_path,
                selected_layers=selected_layers,
                layer_indexing=layer_indexing,
                save_hidden_states=save_hidden_states,
                save_attentions=save_attentions,
                cache_dtype_name=cache_dtype_name,
                cache_dtype=cache_dtype,
                max_seq_len=max_seq_len,
                model_name=model_name,
            )
            manifest_records.append(manifest_record)
            if manifest_record.get("seq_len"):
                seq_lens.append(int(manifest_record["seq_len"]))
            processed_valid_records += 1
        except Exception as exc:
            manifest_records.append(
                make_manifest_record(
                    record_id=record_id,
                    status="error",
                    selected_layers=selected_layers,
                    reason="exception",
                    cache_path=str(cache_path),
                    is_correct=as_bool(record.get("is_correct", False)),
                    num_steps=num_steps,
                    error=repr(exc),
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_jsonl(manifest_file, manifest_records)

    total_records = len(manifest_records)
    cached_records = sum(1 for item in manifest_records if item["status"] == "cached")
    skipped_records = sum(1 for item in manifest_records if item["status"] == "skipped")
    error_records = sum(1 for item in manifest_records if item["status"] == "error")
    average_seq_len = sum(seq_lens) / len(seq_lens) if seq_lens else 0.0
    max_observed_seq_len = max(seq_lens) if seq_lens else 0

    print(f"Total records: {total_records}")
    print(f"Valid records: {valid_records}")
    print(f"Cached records: {cached_records}")
    print(f"Skipped records: {skipped_records}")
    print(f"Error records: {error_records}")
    print(f"Average seq_len: {average_seq_len:.2f}")
    print(f"Max seq_len: {max_observed_seq_len}")
    print(f"Forward cache dir: {forward_dir}")
    print(f"Manifest path: {manifest_file}")


if __name__ == "__main__":
    main()
