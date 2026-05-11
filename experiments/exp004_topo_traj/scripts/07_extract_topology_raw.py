import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm


EXPECTED_FEATURE_NAMES = [
    "edge_mean",
    "edge_std",
    "edge_max",
    "topk_mass_ratio",
    "attention_entropy",
    "node_in_entropy",
    "effective_edge_count",
    "local_window_size",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract raw local attention topology features from cached selected_attentions."
    )
    parser.add_argument("--config", required=True, help="Path to topo_traj_v0.yaml.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N candidate records.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing topology outputs.")
    parser.add_argument("--window-size", type=int, default=None, help="Override topology_raw.window_size.")
    parser.add_argument("--topk-edges", type=int, default=None, help="Override topology_raw.topk_edges.")
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


def ensure_outputs_can_be_written(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Topology output files already exist. Use --overwrite to replace them:\n"
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
        if record_id is not None and record_id not in seen:
            seen.add(record_id)
            ordered_ids.append(record_id)
    return ordered_ids


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def fallback_cache_path(forward_manifest_file: Path, record_id: str) -> Path:
    return forward_manifest_file.parent / "forward" / f"{record_id}.pt"


def fallback_marker_path(marker_manifest_file: Path, record_id: str) -> Path:
    return marker_manifest_file.parent / "markers" / f"{record_id}.json"


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
    window_size: int | None = None,
    topk_edges: int | None = None,
    warnings: list[str] | None = None,
    reason: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": record_id,
        "status": status,
        "cache_path": cache_path,
        "marker_path": marker_path,
        "label": label,
        "is_correct": is_correct,
        "seq_len": seq_len,
        "selected_positions": selected_positions,
        "selected_layers": selected_layers,
        "sample_shape": sample_shape,
        "window_size": window_size,
        "topk_edges": topk_edges,
        "warnings": warnings or [],
        "reason": reason,
    }
    if error is not None:
        record["error"] = error
    return record


def derive_label(cache: dict[str, Any]) -> tuple[int, bool]:
    if "is_correct" in cache:
        is_correct = as_bool(cache["is_correct"])
    elif "label" in cache:
        is_correct = int(cache["label"]) == 0
    else:
        raise ValueError("Cache is missing both is_correct and label.")

    label = int(cache["label"]) if "label" in cache else (0 if is_correct else 1)
    if label not in {0, 1}:
        raise ValueError(f"Invalid label: {label}.")
    return label, is_correct


def sanitize_features(features: np.ndarray, warnings_list: list[str], context: str) -> np.ndarray:
    if not np.all(np.isfinite(features)):
        warnings_list.append(f"{context}: nonfinite_feature_replaced")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def compute_attention_features(
    attention: torch.Tensor,
    pos: int,
    window_size: int,
    topk_edges: int,
    eps: float,
    context: str,
) -> tuple[np.ndarray, int, list[str]]:
    warnings_list: list[str] = []
    start = max(0, pos - window_size + 1)
    end = pos + 1
    local_len = end - start
    if local_len <= 0:
        raise ValueError(f"{context}: local window is empty.")

    local_attention = attention[:, start:end, start:end].to(torch.float32)
    attention_mean = local_attention.mean(dim=0).detach().cpu().numpy().astype(np.float64, copy=False)
    if not np.all(np.isfinite(attention_mean)):
        warnings_list.append(f"{context}: nonfinite_attention_values_replaced")
        attention_mean = np.nan_to_num(attention_mean, nan=0.0, posinf=0.0, neginf=0.0)

    edge_values = attention_mean.reshape(-1)
    edge_values = edge_values[edge_values > eps]
    if edge_values.size == 0:
        warnings_list.append(f"{context}: no_valid_edges")
        features = np.zeros(len(EXPECTED_FEATURE_NAMES), dtype=np.float64)
        features[-1] = float(local_len)
        return sanitize_features(features, warnings_list, context), local_len, warnings_list

    edge_mean = float(np.mean(edge_values))
    edge_std = float(np.std(edge_values))
    edge_max = float(np.max(edge_values))

    k = int(min(max(topk_edges, 1), edge_values.size))
    topk_values = np.partition(edge_values, -k)[-k:]
    total_mass = float(np.sum(edge_values))
    topk_mass_ratio = float(np.sum(topk_values) / (total_mass + eps))

    p = edge_values / (total_mass + eps)
    entropy_raw = float(-np.sum(p * np.log(p + eps)))
    if edge_values.size > 1:
        attention_entropy = float(entropy_raw / np.log(edge_values.size + eps))
    else:
        attention_entropy = 0.0

    node_in = attention_mean.sum(axis=0)
    node_in = np.clip(node_in, 0.0, None)
    node_in_total = float(np.sum(node_in))
    if node_in_total <= eps or local_len <= 1:
        node_in_entropy = 0.0
    else:
        q = node_in / (node_in_total + eps)
        node_in_entropy_raw = float(-np.sum(q * np.log(q + eps)))
        node_in_entropy = float(node_in_entropy_raw / np.log(local_len + eps))

    effective_edge_count = float(np.exp(entropy_raw))

    features = np.array(
        [
            edge_mean,
            edge_std,
            edge_max,
            topk_mass_ratio,
            attention_entropy,
            node_in_entropy,
            effective_edge_count,
            float(local_len),
        ],
        dtype=np.float64,
    )
    return sanitize_features(features, warnings_list, context), local_len, warnings_list


def extract_sample_topology(
    cache: dict[str, Any],
    marker: dict[str, Any],
    position_keys: list[str],
    layers: list[int],
    feature_names: list[str],
    window_size: int,
    topk_edges: int,
    eps: float,
    output_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], int, int, bool, list[str]]:
    if feature_names != EXPECTED_FEATURE_NAMES:
        raise ValueError(f"feature_names must be {EXPECTED_FEATURE_NAMES}, got {feature_names}.")

    selected_attentions = cache.get("selected_attentions")
    if not isinstance(selected_attentions, dict):
        raise ValueError("Cache is missing selected_attentions.")

    selected_positions = marker.get("selected_positions")
    if not isinstance(selected_positions, dict):
        raise ValueError("Marker JSON is missing selected_positions.")

    seq_len = int(cache.get("seq_len", 0))
    if seq_len <= 0:
        first_layer_key = str(layers[0])
        if first_layer_key not in selected_attentions:
            raise ValueError("Cache is missing seq_len and first selected attention layer.")
        seq_len = int(selected_attentions[first_layer_key].shape[1])

    missing_layers = [str(layer) for layer in layers if str(layer) not in selected_attentions]
    if missing_layers:
        raise ValueError(f"Missing selected_attentions layers: {missing_layers}.")

    missing_positions = [key for key in position_keys if key not in selected_positions]
    if missing_positions:
        raise ValueError(f"Missing selected_positions keys: {missing_positions}.")

    label, is_correct = derive_label(cache)
    sample = np.zeros((len(position_keys), len(layers), len(feature_names)), dtype=output_dtype)
    local_window_sizes = np.zeros((len(position_keys), len(layers)), dtype=np.float32)
    sample_warnings: list[str] = []

    for pos_idx, position_key in enumerate(position_keys):
        pos = selected_positions[position_key]
        if not isinstance(pos, int):
            raise ValueError(f"Position {position_key} must be int, got {type(pos).__name__}.")
        if pos < 0 or pos >= seq_len:
            raise ValueError(f"Position {position_key}={pos} is outside seq_len={seq_len}.")

        for layer_idx, layer in enumerate(layers):
            attention = selected_attentions[str(layer)]
            if not isinstance(attention, torch.Tensor):
                attention = torch.as_tensor(attention)
            if attention.ndim != 3:
                raise ValueError(f"Attention layer {layer} must have shape [num_heads, seq_len, seq_len].")
            if int(attention.shape[1]) != seq_len or int(attention.shape[2]) != seq_len:
                raise ValueError(
                    f"Attention layer {layer} shape {tuple(attention.shape)} does not match seq_len={seq_len}."
                )

            context = f"{position_key}/layer_{layer}"
            features, local_len, warnings_list = compute_attention_features(
                attention=attention,
                pos=pos,
                window_size=window_size,
                topk_edges=topk_edges,
                eps=eps,
                context=context,
            )
            sample[pos_idx, layer_idx, :] = features.astype(output_dtype, copy=False)
            local_window_sizes[pos_idx, layer_idx] = float(local_len)
            sample_warnings.extend(warnings_list)

    if not np.all(np.isfinite(sample)):
        sample_warnings.append("sample_nonfinite_features_replaced")
        sample = np.nan_to_num(sample, nan=0.0, posinf=0.0, neginf=0.0).astype(output_dtype, copy=False)

    selected_positions_out = {key: int(selected_positions[key]) for key in position_keys}
    return sample, local_window_sizes, selected_positions_out, seq_len, label, is_correct, sample_warnings


def save_empty_outputs(
    topology_file: Path,
    labels_file: Path,
    ids_file: Path,
    metadata_file: Path,
    position_keys: list[str],
    layers: list[int],
    feature_names: list[str],
    window_size: int,
    topk_edges: int,
    output_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    topology = np.zeros((0, len(position_keys), len(layers), len(feature_names)), dtype=output_dtype)
    labels = np.zeros((0,), dtype=np.int64)
    topology_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(topology_file, topology)
    np.save(labels_file, labels)
    write_json(ids_file, [])
    write_json(
        metadata_file,
        build_metadata(
            num_samples=0,
            topology_shape=list(topology.shape),
            position_keys=position_keys,
            layers=layers,
            feature_names=feature_names,
            window_size=window_size,
            topk_edges=topk_edges,
        ),
    )
    return topology, labels


def build_metadata(
    num_samples: int,
    topology_shape: list[int],
    position_keys: list[str],
    layers: list[int],
    feature_names: list[str],
    window_size: int,
    topk_edges: int,
) -> dict[str, Any]:
    return {
        "num_samples": num_samples,
        "topology_shape": topology_shape,
        "position_keys": position_keys,
        "layers": layers,
        "feature_names": feature_names,
        "window_size": window_size,
        "topk_edges": topk_edges,
        "attention_source": "selected_attentions from teacher-forced canonicalized model_output",
        "attention_type": "raw_attention",
        "head_pooling": "mean_over_heads",
        "local_graph": "window [pos-window_size+1, pos]",
        "text_source": "canonicalized_model_output",
        "canonicalization_used": True,
        "marker_rule": "preceding token before marker",
        "termination_marker_choice": "last ####",
        "layer_indexing": "block_0_based",
        "attention_index_rule": "attentions[layer_idx]",
        "feature_notes": {
            "attention_entropy": "normalized edge entropy",
            "node_in_entropy": "normalized entropy over incoming attention mass",
            "effective_edge_count": "exp(raw edge entropy)",
        },
        "label_rule": "0=correct, 1=wrong",
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    topo_cfg = cfg["topology_raw"]

    topology_file = Path(topo_cfg["topology_file"])
    labels_file = Path(topo_cfg["labels_file"])
    ids_file = Path(topo_cfg["ids_file"])
    metadata_file = Path(topo_cfg["metadata_file"])
    manifest_file = Path(topo_cfg["manifest_file"])
    forward_manifest_file = Path(topo_cfg["forward_manifest_file"])
    marker_manifest_file = Path(topo_cfg["marker_manifest_file"])

    position_keys = list(topo_cfg["selected_position_keys"])
    layers = [int(layer) for layer in topo_cfg["selected_layers"]]
    feature_names = list(topo_cfg["feature_names"])
    window_size = int(args.window_size if args.window_size is not None else topo_cfg["window_size"])
    topk_edges = int(args.topk_edges if args.topk_edges is not None else topo_cfg["topk_edges"])
    eps = float(topo_cfg.get("eps", 1.0e-12))
    output_dtype = np.dtype(topo_cfg.get("output_dtype", "float32"))

    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}.")
    if topk_edges <= 0:
        raise ValueError(f"topk_edges must be positive, got {topk_edges}.")
    if feature_names != EXPECTED_FEATURE_NAMES:
        raise ValueError(f"Configured feature_names must match {EXPECTED_FEATURE_NAMES}.")

    ensure_outputs_can_be_written(
        [topology_file, labels_file, ids_file, metadata_file, manifest_file],
        overwrite=args.overwrite,
    )

    forward_records = read_jsonl(forward_manifest_file)
    marker_records = read_jsonl(marker_manifest_file)
    forward_by_id = index_by_id(forward_records)
    marker_by_id = index_by_id(marker_records)
    candidate_ids = ordered_candidate_ids(forward_records, marker_records)
    total_candidate_records = len(candidate_ids)
    if args.limit is not None:
        candidate_ids = candidate_ids[: args.limit]

    topology_samples: list[np.ndarray] = []
    local_window_samples: list[np.ndarray] = []
    labels: list[int] = []
    extracted_ids: list[str] = []
    manifest_records: list[dict[str, Any]] = []

    for record_id in tqdm(candidate_ids, desc="Extracting raw topology"):
        forward_record = forward_by_id.get(record_id)
        marker_record = marker_by_id.get(record_id)
        cache_path = Path(forward_record.get("cache_path")) if forward_record and forward_record.get("cache_path") else fallback_cache_path(forward_manifest_file, record_id)
        marker_path = Path(marker_record.get("marker_path")) if marker_record and marker_record.get("marker_path") else fallback_marker_path(marker_manifest_file, record_id)

        if forward_record is None:
            manifest_records.append(
                make_manifest_record(
                    record_id=record_id,
                    status="skipped",
                    cache_path=str(cache_path),
                    marker_path=str(marker_path),
                    selected_layers=layers,
                    window_size=window_size,
                    topk_edges=topk_edges,
                    reason="missing_forward_manifest_record",
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
                    window_size=window_size,
                    topk_edges=topk_edges,
                    reason="missing_marker_manifest_record",
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
                    window_size=window_size,
                    topk_edges=topk_edges,
                    reason="forward_not_cached",
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
                    window_size=window_size,
                    topk_edges=topk_edges,
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
                    window_size=window_size,
                    topk_edges=topk_edges,
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
                    window_size=window_size,
                    topk_edges=topk_edges,
                    reason="marker_file_missing",
                )
            )
            continue

        try:
            cache = torch.load(cache_path, map_location="cpu")
            with open(marker_path, "r", encoding="utf-8") as f:
                marker = json.load(f)

            sample, local_windows, selected_positions, seq_len, label, is_correct, warnings_list = extract_sample_topology(
                cache=cache,
                marker=marker,
                position_keys=position_keys,
                layers=layers,
                feature_names=feature_names,
                window_size=window_size,
                topk_edges=topk_edges,
                eps=eps,
                output_dtype=output_dtype,
            )

            topology_samples.append(sample)
            local_window_samples.append(local_windows)
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
                    window_size=window_size,
                    topk_edges=topk_edges,
                    warnings=warnings_list,
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
                    window_size=window_size,
                    topk_edges=topk_edges,
                    reason="exception",
                    error=repr(exc),
                )
            )
        finally:
            if "cache" in locals():
                del cache

    if topology_samples:
        topology = np.stack(topology_samples, axis=0).astype(output_dtype, copy=False)
        labels_array = np.asarray(labels, dtype=np.int64)
        topology_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(topology_file, topology)
        np.save(labels_file, labels_array)
        write_json(ids_file, extracted_ids)
        write_json(
            metadata_file,
            build_metadata(
                num_samples=int(topology.shape[0]),
                topology_shape=list(topology.shape),
                position_keys=position_keys,
                layers=layers,
                feature_names=feature_names,
                window_size=window_size,
                topk_edges=topk_edges,
            ),
        )
    else:
        topology, labels_array = save_empty_outputs(
            topology_file=topology_file,
            labels_file=labels_file,
            ids_file=ids_file,
            metadata_file=metadata_file,
            position_keys=position_keys,
            layers=layers,
            feature_names=feature_names,
            window_size=window_size,
            topk_edges=topk_edges,
            output_dtype=output_dtype,
        )

    write_jsonl(manifest_file, manifest_records)

    extracted_records = sum(1 for record in manifest_records if record["status"] == "extracted")
    skipped_records = sum(1 for record in manifest_records if record["status"] == "skipped")
    error_records = sum(1 for record in manifest_records if record["status"] == "error")
    correct_count = int(np.sum(labels_array == 0)) if labels_array.size else 0
    wrong_count = int(np.sum(labels_array == 1)) if labels_array.size else 0
    if local_window_samples:
        average_local_window_size = float(np.mean(np.stack(local_window_samples, axis=0)))
    else:
        average_local_window_size = 0.0

    print(f"Total candidate records: {total_candidate_records}")
    print(f"Extracted records: {extracted_records}")
    print(f"Skipped records: {skipped_records}")
    print(f"Error records: {error_records}")
    print(f"Topology shape: {list(topology.shape)}")
    print(f"Labels shape: {list(labels_array.shape)}")
    print(f"Correct count: {correct_count}")
    print(f"Wrong count: {wrong_count}")
    print(f"Average local window size: {average_local_window_size:.4f}")
    print(f"Feature names: {feature_names}")
    print(f"Output topology path: {topology_file}")
    print(f"Output labels path: {labels_file}")
    print(f"Metadata path: {metadata_file}")
    print(f"Manifest path: {manifest_file}")


if __name__ == "__main__":
    main()
