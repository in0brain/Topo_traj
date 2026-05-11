import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer


STEP_PATTERN = re.compile(r"Step\s+\d+\s*:")
TERMINATION_MARKER = "####"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Step and termination marker token positions.")
    parser.add_argument("--config", required=True, help="Path to topo_traj_v0.yaml.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N cached samples.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute marker files that already exist.")
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


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def build_teacher_forcing_text(
    tokenizer: AutoTokenizer,
    prompt: str,
    model_output: str,
) -> tuple[str, str]:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": model_output},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            ),
            "chat_template",
        )
    return prompt + "\n\n" + model_output, "fallback_concat"


def tokenize_with_offsets(tokenizer: AutoTokenizer, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
    try:
        encoded = tokenizer(text, return_offsets_mapping=True, return_tensors=None)
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        return input_ids, [(int(start), int(end)) for start, end in offsets]
    except (NotImplementedError, TypeError, ValueError):
        encoded = tokenizer(text, return_tensors=None)
        return encoded["input_ids"], None


def is_special_offset(offset: tuple[int, int]) -> bool:
    return offset[0] == 0 and offset[1] == 0


def char_to_token_idx(offsets: list[tuple[int, int]], char_start: int) -> int | None:
    for token_idx, (offset_start, offset_end) in enumerate(offsets):
        if is_special_offset((offset_start, offset_end)):
            continue
        if offset_start <= char_start < offset_end:
            return token_idx

    for token_idx, (offset_start, offset_end) in enumerate(offsets):
        if is_special_offset((offset_start, offset_end)):
            continue
        if offset_end > char_start:
            return token_idx

    for token_idx, (offset_start, offset_end) in enumerate(offsets):
        if offset_start <= char_start < offset_end or offset_end > char_start:
            return token_idx
    return None


def fallback_char_to_token_idx(
    tokenizer: AutoTokenizer,
    input_ids: list[int],
    text: str,
    char_start: int,
) -> int | None:
    for token_idx in range(len(input_ids)):
        prefix = tokenizer.decode(input_ids[: token_idx + 1], skip_special_tokens=False)
        if len(prefix) > char_start:
            return token_idx
    return None


def locate_model_output_span(text: str, model_output: str, warnings: list[str]) -> tuple[int, int]:
    output_start = text.rfind(model_output)
    if output_start < 0:
        warnings.append("model_output_span_not_found")
        return 0, len(text)
    return output_start, output_start + len(model_output)


def make_marker_manifest_record(
    record_id: str,
    status: str,
    marker_path: str | None = None,
    num_steps_detected: int | None = None,
    has_termination: bool = False,
    selected_positions: dict[str, int] | None = None,
    seq_len: int | None = None,
    warnings: list[str] | None = None,
    reason: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    manifest_record = {
        "id": record_id,
        "status": status,
        "marker_path": marker_path,
        "num_steps_detected": num_steps_detected,
        "has_termination": has_termination,
        "selected_positions": selected_positions or {},
        "seq_len": seq_len,
        "warnings": warnings or [],
        "reason": reason,
    }
    if error:
        manifest_record["error"] = error
    return manifest_record


def load_existing_marker(marker_path: Path) -> dict[str, Any]:
    with open(marker_path, "r", encoding="utf-8") as f:
        return json.load(f)


def manifest_from_existing_marker(marker_path: Path) -> dict[str, Any]:
    marker = load_existing_marker(marker_path)
    return make_marker_manifest_record(
        record_id=marker["id"],
        status="parsed",
        marker_path=str(marker_path),
        num_steps_detected=marker.get("num_steps_detected"),
        has_termination=bool(marker.get("termination_marker")),
        selected_positions=marker.get("selected_positions", {}),
        seq_len=marker.get("seq_len"),
        warnings=marker.get("warnings", []),
        reason=None,
    )


def resolve_cache_path(manifest_record: dict[str, Any], forward_dir: Path) -> Path:
    cache_path = manifest_record.get("cache_path")
    if cache_path:
        return Path(cache_path)
    return forward_dir / f"{manifest_record['id']}.pt"


def get_cache_seq_len_and_ids(cache_path: Path) -> tuple[int | None, list[int]]:
    cached = torch.load(cache_path, map_location="cpu")
    try:
        input_ids = cached.get("input_ids")
        if input_ids is None:
            return cached.get("seq_len"), []
        if isinstance(input_ids, torch.Tensor):
            input_ids_list = [int(token_id) for token_id in input_ids.reshape(-1).tolist()]
        else:
            input_ids_list = [int(token_id) for token_id in input_ids]
        return int(cached.get("seq_len", len(input_ids_list))), input_ids_list
    finally:
        del cached


def find_token_idx(
    tokenizer: AutoTokenizer,
    input_ids: list[int],
    offsets: list[tuple[int, int]] | None,
    text: str,
    char_start: int,
    warnings: list[str],
) -> int | None:
    if offsets is not None:
        return char_to_token_idx(offsets, char_start)
    if "offset_mapping_unavailable" not in warnings:
        warnings.append("offset_mapping_unavailable")
    return fallback_char_to_token_idx(tokenizer, input_ids, text, char_start)


def extract_marker_positions(
    record: dict[str, Any],
    tokenizer: AutoTokenizer,
    cache_path: Path,
    marker_path: Path,
    require_min_steps: int,
    selected_step_keys: list[str],
) -> dict[str, Any]:
    warnings: list[str] = []
    prompt = record["prompt"]
    model_output = record.get("model_output", "")
    text, tokenization_source = build_teacher_forcing_text(tokenizer, prompt, model_output)
    input_ids, offsets = tokenize_with_offsets(tokenizer, text)
    retokenized_seq_len = len(input_ids)
    cache_seq_len, cache_input_ids = get_cache_seq_len_and_ids(cache_path)
    seq_len = cache_seq_len if cache_seq_len is not None else retokenized_seq_len

    if cache_seq_len != retokenized_seq_len:
        warnings.append("token_length_mismatch")
    if cache_seq_len is not None and retokenized_seq_len != cache_seq_len:
        warnings.append(f"cache_seq_len={cache_seq_len}")
        warnings.append(f"retokenized_seq_len={retokenized_seq_len}")
    if cache_input_ids and len(cache_input_ids) == len(input_ids) and cache_input_ids != input_ids:
        warnings.append("token_id_mismatch")

    output_start, output_end = locate_model_output_span(text, model_output, warnings)
    output_text = text[output_start:output_end]

    step_markers = []
    for match in STEP_PATTERN.finditer(output_text):
        char_start = output_start + match.start()
        char_end = output_start + match.end()
        marker_token_idx = find_token_idx(
            tokenizer=tokenizer,
            input_ids=input_ids,
            offsets=offsets,
            text=text,
            char_start=char_start,
            warnings=warnings,
        )
        if marker_token_idx is None:
            warnings.append(f"step_marker_token_not_found:{match.group(0)}")
            continue
        preceding_token_idx = marker_token_idx - 1
        step_markers.append(
            {
                "step_index": len(step_markers) + 1,
                "step_text": match.group(0),
                "char_start": char_start,
                "char_end": char_end,
                "marker_token_idx": marker_token_idx,
                "preceding_token_idx": preceding_token_idx,
            }
        )

    termination_rel_start = output_text.rfind(TERMINATION_MARKER)
    termination_marker = None
    if termination_rel_start >= 0:
        char_start = output_start + termination_rel_start
        char_end = char_start + len(TERMINATION_MARKER)
        marker_token_idx = find_token_idx(
            tokenizer=tokenizer,
            input_ids=input_ids,
            offsets=offsets,
            text=text,
            char_start=char_start,
            warnings=warnings,
        )
        if marker_token_idx is not None:
            termination_marker = {
                "text": TERMINATION_MARKER,
                "char_start": char_start,
                "char_end": char_end,
                "marker_token_idx": marker_token_idx,
                "preceding_token_idx": marker_token_idx - 1,
            }
        else:
            warnings.append("termination_marker_token_not_found")

    if len(step_markers) < require_min_steps:
        return make_marker_manifest_record(
            record_id=record["id"],
            status="skipped",
            num_steps_detected=len(step_markers),
            has_termination=termination_marker is not None,
            seq_len=seq_len,
            warnings=warnings,
            reason="insufficient_steps",
        )

    if termination_marker is None:
        return make_marker_manifest_record(
            record_id=record["id"],
            status="skipped",
            num_steps_detected=len(step_markers),
            has_termination=False,
            seq_len=seq_len,
            warnings=warnings,
            reason="missing_termination",
        )

    selected_positions_all = {
        "K-2": step_markers[-2]["preceding_token_idx"],
        "K-1": step_markers[-1]["preceding_token_idx"],
        "TERM": termination_marker["preceding_token_idx"],
    }
    selected_marker_positions_all = {
        "K-2_marker": step_markers[-2]["marker_token_idx"],
        "K-1_marker": step_markers[-1]["marker_token_idx"],
        "TERM_marker": termination_marker["marker_token_idx"],
    }
    selected_positions = {
        key: selected_positions_all[key]
        for key in selected_step_keys
        if key in selected_positions_all
    }
    selected_marker_positions = {
        f"{key}_marker": selected_marker_positions_all[f"{key}_marker"]
        for key in selected_step_keys
        if f"{key}_marker" in selected_marker_positions_all
    }

    max_valid_position = seq_len - 1 if seq_len is not None else retokenized_seq_len - 1
    for key, value in {**selected_positions, **selected_marker_positions}.items():
        if value < 0 or value > max_valid_position:
            warnings.append(f"position_out_of_bounds:{key}={value}")

    marker_record = {
        "id": record["id"],
        "seq_len": seq_len,
        "retokenized_seq_len": retokenized_seq_len,
        "num_steps_detected": len(step_markers),
        "step_markers": step_markers,
        "termination_marker": termination_marker,
        "selected_positions": selected_positions,
        "selected_marker_positions": selected_marker_positions,
        "position_rule": "use preceding token before marker",
        "tokenization_source": tokenization_source,
        "warnings": warnings,
    }
    write_json(marker_path, marker_record)

    return make_marker_manifest_record(
        record_id=record["id"],
        status="parsed",
        marker_path=str(marker_path),
        num_steps_detected=len(step_markers),
        has_termination=True,
        selected_positions=selected_positions,
        seq_len=seq_len,
        warnings=warnings,
        reason=None,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_config = config["model"]
    cache_config = config["cache"]
    markers_config = config["markers"]

    generations_file = Path(config["paths"]["generations_file"])
    forward_dir = Path(cache_config["forward_dir"])
    forward_manifest_file = Path(cache_config["manifest_file"])
    marker_dir = Path(markers_config["marker_dir"])
    marker_manifest_file = Path(markers_config["marker_manifest_file"])
    require_min_steps = int(markers_config.get("require_min_steps", 2))
    selected_step_keys = list(markers_config.get("selected_step_keys", ["K-2", "K-1", "TERM"]))
    model_name = model_config["model_name"]
    local_files_only = as_bool(model_config.get("local_files_only", False))

    generations = {record["id"]: record for record in read_jsonl(generations_file)}
    forward_manifest = read_jsonl(forward_manifest_file)
    cached_manifest_records = [record for record in forward_manifest if record.get("status") == "cached"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )

    marker_records = []
    processed_cached_records = 0
    total_cached_records = len(cached_manifest_records)

    for manifest_record in tqdm(cached_manifest_records, desc="Extracting markers"):
        if args.limit is not None and processed_cached_records >= args.limit:
            break

        record_id = manifest_record["id"]
        marker_path = marker_dir / f"{record_id}.json"
        processed_cached_records += 1

        try:
            if marker_path.exists() and not args.overwrite:
                marker_records.append(manifest_from_existing_marker(marker_path))
                continue

            generation_record = generations.get(record_id)
            if generation_record is None:
                marker_records.append(
                    make_marker_manifest_record(
                        record_id=record_id,
                        status="error",
                        reason="generation_record_missing",
                    )
                )
                continue

            cache_path = resolve_cache_path(manifest_record, forward_dir)
            if not cache_path.exists():
                marker_records.append(
                    make_marker_manifest_record(
                        record_id=record_id,
                        status="error",
                        reason="forward_cache_missing",
                        error=str(cache_path),
                    )
                )
                continue

            marker_records.append(
                extract_marker_positions(
                    record=generation_record,
                    tokenizer=tokenizer,
                    cache_path=cache_path,
                    marker_path=marker_path,
                    require_min_steps=require_min_steps,
                    selected_step_keys=selected_step_keys,
                )
            )
        except Exception as exc:
            marker_records.append(
                make_marker_manifest_record(
                    record_id=record_id,
                    status="error",
                    reason="exception",
                    error=repr(exc),
                )
            )

    write_jsonl(marker_manifest_file, marker_records)

    parsed_records = sum(1 for record in marker_records if record["status"] == "parsed")
    skipped_records = sum(1 for record in marker_records if record["status"] == "skipped")
    error_records = sum(1 for record in marker_records if record["status"] == "error")
    detected_steps = [
        int(record["num_steps_detected"])
        for record in marker_records
        if record.get("num_steps_detected") is not None
    ]
    average_steps = sum(detected_steps) / len(detected_steps) if detected_steps else 0.0
    min_steps = min(detected_steps) if detected_steps else 0
    max_steps = max(detected_steps) if detected_steps else 0
    missing_termination_count = sum(1 for record in marker_records if not record.get("has_termination"))

    print(f"Total cached records: {total_cached_records}")
    print(f"Parsed records: {parsed_records}")
    print(f"Skipped records: {skipped_records}")
    print(f"Error records: {error_records}")
    print(f"Average detected steps: {average_steps:.2f}")
    print(f"Min detected steps: {min_steps}")
    print(f"Max detected steps: {max_steps}")
    print(f"Missing termination count: {missing_termination_count}")
    print(f"Marker dir: {marker_dir}")
    print(f"Marker manifest path: {marker_manifest_file}")


if __name__ == "__main__":
    main()
