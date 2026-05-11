import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Step-Layer geometry tensors from cached hidden states."
    )
    parser.add_argument("--config", required=True, help="Path to topo_traj_v0.yaml.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N candidate records.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing geometry outputs.")
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


def write_json(path: Path, record: Any) -> None:
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


def ensure_outputs_can_be_written(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Geometry output files already exist. Use --overwrite to replace them:\n"
            f"{joined}"
        )


def index_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in records if "id" in record}


def ordered_candidate_ids(
    forward_records: list[dict[str, Any]],
    marker_records: list[dict[str, Any]],
) -> list[str]:
    seen = set()
    ordered_ids = []
    for record in forward_records + marker_records:
        record_id = record.get("id")
        if record_id and record_id not in seen:
            seen.add(record_id)
            ordered_ids.append(record_id)
    return ordered_ids


def make_manifest_record(
    record_id: str,
    status: str,
    cache_path: str | None = None,
    marker_path: str | None = None,
    label: int | None = None,
    is_correct: bool | None = None,
    seq_len: int | None = None,
    selected_positions: dict[str, int] | None = None,
    selected_layers: list[int] | None = None,
    sample_shape: list[int] | None = None,
    reason: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    record = {
        "id": record_id,
        "status": status,
        "cache_path": cache_path,
        "marker_path": marker_path,
        "label": label,
        "is_correct": is_correct,
        "seq_len": seq_len,
        "selected_positions": selected_positions or {},
        "selected_layers": selected_layers or [],
        "sample_shape": sample_shape,
        "reason": reason,
    }
    if error:
        record["error"] = error
    return record


def resolve_path(path_value: str | None, fallback: Path) -> Path:
    if path_value:
        return Path(path_value)
    return fallback


def validate_position(value: Any, key: str, seq_len: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Position {key!r} must be an int, got {type(value).__name__}.")
    if value < 0 or value >= seq_len:
        raise IndexError(f"Position {key!r}={value} is outside seq_len={seq_len}.")
    return value


def get_label_and_correct(cache: dict[str, Any]) -> tuple[int, bool]:
    is_correct = as_bool(cache.get("is_correct", False))
    label = cache.get("label")
    if label is None:
        label = 0 if is_correct else 1
    label = int(label)
    if label not in {0, 1}:
        raise ValueError(f"Label must be 0 or 1, got {label}.")
    return label, is_correct


def extract_sample_geometry(
    cache: dict[str, Any],
    marker: dict[str, Any],
    position_keys: list[str],
    layers: list[int],
    output_dtype: np.dtype,
) -> tuple[np.ndarray, dict[str, int], int, int, bool]:
    selected_hidden_states = cache.get("selected_hidden_states")
    if not isinstance(selected_hidden_states, dict):
        raise KeyError("Cache missing selected_hidden_states dict.")

    selected_positions = marker.get("selected_positions")
    if not isinstance(selected_positions, dict):
        raise KeyError("Marker file missing selected_positions dict.")

    seq_len = int(cache.get("seq_len", marker.get("seq_len", 0)))
    if seq_len <= 0:
        raise ValueError(f"Invalid seq_len: {seq_len}.")

    positions = {
        key: validate_position(selected_positions[key], key, seq_len)
        for key in position_keys
    }

    layer_tensors = {}
    hidden_dim = None
    for layer in layers:
        layer_key = str(layer)
        if layer_key not in selected_hidden_states:
            raise KeyError(f"Missing selected_hidden_states[{layer_key!r}].")
        tensor = selected_hidden_states[layer_key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"selected_hidden_states[{layer_key!r}] is not a torch.Tensor.")
        if tensor.ndim != 2:
            raise ValueError(
                f"selected_hidden_states[{layer_key!r}] must have shape [seq_len, hidden_dim], "
                f"got {tuple(tensor.shape)}."
            )
        if int(tensor.shape[0]) != seq_len:
            raise ValueError(
                f"selected_hidden_states[{layer_key!r}] seq_len mismatch: "
                f"{int(tensor.shape[0])} != {seq_len}."
            )
        tensor_hidden_dim = int(tensor.shape[1])
        if hidden_dim is None:
            hidden_dim = tensor_hidden_dim
        elif tensor_hidden_dim != hidden_dim:
            raise ValueError(
                f"Hidden dim mismatch at layer {layer}: {tensor_hidden_dim} != {hidden_dim}."
            )
        layer_tensors[layer] = tensor

    if hidden_dim is None:
        raise ValueError("No hidden states found for selected layers.")

    sample = np.empty((len(position_keys), len(layers), hidden_dim), dtype=output_dtype)
    for pos_idx, position_key in enumerate(position_keys):
        pos = positions[position_key]
        for layer_idx, layer in enumerate(layers):
            hidden_vector = layer_tensors[layer][pos]
            if hidden_vector.ndim != 1 or int(hidden_vector.shape[0]) != hidden_dim:
                raise ValueError(
                    f"Hidden vector for {position_key}/layer {layer} must have shape [{hidden_dim}], "
                    f"got {tuple(hidden_vector.shape)}."
                )
            sample[pos_idx, layer_idx, :] = hidden_vector.detach().cpu().numpy().astype(output_dtype, copy=False)

    label, is_correct = get_label_and_correct(cache)
    return sample, positions, seq_len, label, is_correct


def save_empty_outputs(
    geometry_file: Path,
    labels_file: Path,
    ids_file: Path,
    metadata_file: Path,
    position_keys: list[str],
    layers: list[int],
    output_dtype: np.dtype,
) -> None:
    geometry = np.empty((0, len(position_keys), len(layers), 0), dtype=output_dtype)
    labels = np.empty((0,), dtype=np.int64)
    np.save(geometry_file, geometry)
    np.save(labels_file, labels)
    write_json(ids_file, [])
    write_json(
        metadata_file,
        {
            "num_samples": 0,
            "geometry_shape": list(geometry.shape),
            "position_keys": position_keys,
            "layers": layers,
            "hidden_dim": 0,
            "label_rule": "0=correct, 1=wrong",
            "source": "forward cache selected_hidden_states + marker selected_positions",
            "position_rule": "preceding token before marker",
            "text_source": "canonicalized_model_output",
            "canonicalization_used": True,
            "marker_rule": "preceding token before marker",
            "termination_marker_choice": "last ####",
            "layer_indexing": "block_0_based",
            "hidden_state_index_rule": "hidden_states[layer_idx + 1]",
        },
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    cache_config = config["cache"]
    marker_config = config["markers"]
    geometry_config = config["geometry"]

    forward_manifest_file = Path(cache_config["manifest_file"])
    marker_manifest_file = Path(marker_config["marker_manifest_file"])
    forward_dir = Path(cache_config["forward_dir"])
    marker_dir = Path(marker_config["marker_dir"])

    output_dir = Path(geometry_config["output_dir"])
    geometry_file = Path(geometry_config["geometry_file"])
    labels_file = Path(geometry_config["labels_file"])
    ids_file = Path(geometry_config["ids_file"])
    metadata_file = Path(geometry_config["metadata_file"])
    manifest_file = Path(geometry_config["manifest_file"])
    position_keys = list(geometry_config["selected_position_keys"])
    layers = [int(layer) for layer in geometry_config["selected_layers"]]
    output_dtype = np.dtype(geometry_config.get("output_dtype", "float32"))

    ensure_outputs_can_be_written(
        [geometry_file, labels_file, ids_file, metadata_file, manifest_file],
        overwrite=args.overwrite,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    forward_records = read_jsonl(forward_manifest_file)
    marker_records = read_jsonl(marker_manifest_file)
    forward_by_id = index_by_id(forward_records)
    marker_by_id = index_by_id(marker_records)
    candidate_ids = ordered_candidate_ids(forward_records, marker_records)
    total_candidate_records = len(candidate_ids)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be a positive integer.")
        candidate_ids = candidate_ids[: args.limit]

    geometry_samples = []
    labels = []
    extracted_ids = []
    manifest_records = []
    hidden_dim = None

    for record_id in tqdm(candidate_ids, desc="Extracting geometry"):
        forward_record = forward_by_id.get(record_id)
        marker_record = marker_by_id.get(record_id)
        default_cache_path = forward_dir / f"{record_id}.pt"
        default_marker_path = marker_dir / f"{record_id}.json"
        cache_path = resolve_path(forward_record.get("cache_path") if forward_record else None, default_cache_path)
        marker_path = resolve_path(marker_record.get("marker_path") if marker_record else None, default_marker_path)

        if forward_record is None:
            manifest_records.append(
                make_manifest_record(
                    record_id=record_id,
                    status="skipped",
                    cache_path=str(cache_path),
                    marker_path=str(marker_path),
                    selected_layers=layers,
                    reason="missing_forward_manifest_record",
                )
            )
            continue
        if forward_record.get("status") != "cached":
            manifest_records.append(
                make_manifest_record(
                    record_id=record_id,
                    status="skipped",
                    cache_path=str(cache_path),
                    marker_path=str(marker_path),
                    selected_layers=layers,
                    reason="forward_not_cached",
                )
            )
            continue
        if marker_record is None:
            manifest_records.append(
                make_manifest_record(
                    record_id=record_id,
                    status="skipped",
                    cache_path=str(cache_path),
                    marker_path=str(marker_path),
                    selected_layers=layers,
                    reason="missing_marker_manifest_record",
                )
            )
            continue
        if marker_record.get("status") != "parsed":
            manifest_records.append(
                make_manifest_record(
                    record_id=record_id,
                    status="skipped",
                    cache_path=str(cache_path),
                    marker_path=str(marker_path),
                    selected_layers=layers,
                    reason="marker_not_parsed",
                )
            )
            continue
        if not cache_path.exists():
            manifest_records.append(
                make_manifest_record(
                    record_id=record_id,
                    status="skipped",
                    cache_path=str(cache_path),
                    marker_path=str(marker_path),
                    selected_layers=layers,
                    reason="cache_file_missing",
                )
            )
            continue
        if not marker_path.exists():
            manifest_records.append(
                make_manifest_record(
                    record_id=record_id,
                    status="skipped",
                    cache_path=str(cache_path),
                    marker_path=str(marker_path),
                    selected_layers=layers,
                    reason="marker_file_missing",
                )
            )
            continue

        try:
            cache = torch.load(cache_path, map_location="cpu")
            with open(marker_path, "r", encoding="utf-8") as f:
                marker = json.load(f)

            sample, selected_positions, seq_len, label, is_correct = extract_sample_geometry(
                cache=cache,
                marker=marker,
                position_keys=position_keys,
                layers=layers,
                output_dtype=output_dtype,
            )
            sample_hidden_dim = int(sample.shape[-1])
            if hidden_dim is None:
                hidden_dim = sample_hidden_dim
            elif sample_hidden_dim != hidden_dim:
                raise ValueError(f"Sample hidden_dim mismatch: {sample_hidden_dim} != {hidden_dim}.")

            geometry_samples.append(sample)
            labels.append(label)
            extracted_ids.append(record_id)
            manifest_records.append(
                make_manifest_record(
                    record_id=record_id,
                    status="extracted",
                    cache_path=str(cache_path),
                    marker_path=str(marker_path),
                    label=label,
                    is_correct=is_correct,
                    seq_len=seq_len,
                    selected_positions=selected_positions,
                    selected_layers=layers,
                    sample_shape=list(sample.shape),
                    reason=None,
                )
            )
        except Exception as exc:
            manifest_records.append(
                make_manifest_record(
                    record_id=record_id,
                    status="error",
                    cache_path=str(cache_path),
                    marker_path=str(marker_path),
                    selected_layers=layers,
                    reason="exception",
                    error=repr(exc),
                )
            )
        finally:
            if "cache" in locals():
                del cache

    if geometry_samples:
        geometry = np.stack(geometry_samples, axis=0).astype(output_dtype, copy=False)
        labels_array = np.asarray(labels, dtype=np.int64)
        hidden_dim = int(geometry.shape[-1])
        np.save(geometry_file, geometry)
        np.save(labels_file, labels_array)
        write_json(ids_file, extracted_ids)
        metadata = {
            "num_samples": int(geometry.shape[0]),
            "geometry_shape": list(geometry.shape),
            "position_keys": position_keys,
            "layers": layers,
            "hidden_dim": hidden_dim,
            "label_rule": "0=correct, 1=wrong",
            "source": "forward cache selected_hidden_states + marker selected_positions",
            "position_rule": "preceding token before marker",
            "text_source": "canonicalized_model_output",
            "canonicalization_used": True,
            "marker_rule": "preceding token before marker",
            "termination_marker_choice": "last ####",
            "layer_indexing": "block_0_based",
            "hidden_state_index_rule": "hidden_states[layer_idx + 1]",
            "output_dtype": str(output_dtype),
        }
        write_json(metadata_file, metadata)
    else:
        save_empty_outputs(
            geometry_file=geometry_file,
            labels_file=labels_file,
            ids_file=ids_file,
            metadata_file=metadata_file,
            position_keys=position_keys,
            layers=layers,
            output_dtype=output_dtype,
        )
        geometry = np.load(geometry_file)
        labels_array = np.load(labels_file)

    write_jsonl(manifest_file, manifest_records)

    extracted_records = sum(1 for record in manifest_records if record["status"] == "extracted")
    skipped_records = sum(1 for record in manifest_records if record["status"] == "skipped")
    error_records = sum(1 for record in manifest_records if record["status"] == "error")
    correct_count = int(np.sum(labels_array == 0)) if labels_array.size else 0
    wrong_count = int(np.sum(labels_array == 1)) if labels_array.size else 0

    print(f"Total candidate records: {total_candidate_records}")
    print(f"Extracted records: {extracted_records}")
    print(f"Skipped records: {skipped_records}")
    print(f"Error records: {error_records}")
    print(f"Geometry shape: {list(geometry.shape)}")
    print(f"Labels shape: {list(labels_array.shape)}")
    print(f"Correct count: {correct_count}")
    print(f"Wrong count: {wrong_count}")
    print(f"Output geometry path: {geometry_file}")
    print(f"Output labels path: {labels_file}")
    print(f"Metadata path: {metadata_file}")
    print(f"Manifest path: {manifest_file}")


if __name__ == "__main__":
    main()
