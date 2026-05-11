import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


VALID_FEATURE_MODES = {"full", "term_only", "last_step_only", "late_only"}
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PCA + logistic regression geometry-only probe."
    )
    parser.add_argument("--config", required=True, help="Path to topo_traj_v0.yaml.")
    parser.add_argument("--feature-mode", choices=sorted(VALID_FEATURE_MODES), default=None)
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing probe outputs.")
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def ensure_outputs_can_be_written(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Probe output files already exist. Use --overwrite to replace them:\n"
            f"{joined}"
        )


def get_probe_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    probe = dict(config.get("probe", {}))
    if args.feature_mode is not None:
        probe["feature_mode"] = args.feature_mode
    if args.pca_components is not None:
        probe["pca_components"] = args.pca_components
    if args.cv_folds is not None:
        probe["cv_folds"] = args.cv_folds
    if args.test_size is not None:
        probe["test_size"] = args.test_size
    if args.seed is not None:
        probe["random_seed"] = args.seed
    return probe


def validate_inputs(
    geometry: np.ndarray,
    labels: np.ndarray,
    ids: list[str],
    warnings_list: list[str],
) -> None:
    if geometry.ndim != 4:
        raise ValueError(f"geometry_raw.npy must have shape [N, 3, 3, hidden_dim], got {geometry.shape}.")
    if labels.ndim != 1:
        raise ValueError(f"labels.npy must have shape [N], got {labels.shape}.")
    if geometry.shape[0] != labels.shape[0]:
        raise ValueError(f"Geometry/labels N mismatch: {geometry.shape[0]} != {labels.shape[0]}.")
    if len(ids) != geometry.shape[0]:
        raise ValueError(f"ids.json length mismatch: {len(ids)} != {geometry.shape[0]}.")
    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"labels.npy may only contain 0 and 1, got {sorted(unique_labels)}.")
    counts = class_counts(labels)
    if min(counts.values()) < 5:
        warnings_list.append(
            f"Small minority class count detected: {counts}. Metrics are for pipeline sanity checks only."
        )
    if min(counts.values()) < 2:
        warnings_list.append(
            f"Very small class count detected: {counts}. Train/test split or CV may be unstable."
        )


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "0": int(np.sum(labels == 0)),
        "1": int(np.sum(labels == 1)),
    }


def build_features(
    geometry: np.ndarray,
    metadata: dict[str, Any],
    feature_mode: str,
) -> tuple[np.ndarray, list[str], list[int]]:
    if feature_mode not in VALID_FEATURE_MODES:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}.")

    position_keys = list(metadata.get("position_keys", ["K-2", "K-1", "TERM"]))
    layers = [int(layer) for layer in metadata.get("layers", [12, 20, 27])]
    position_to_idx = {key: idx for idx, key in enumerate(position_keys)}

    def require_position(key: str) -> int:
        if key not in position_to_idx:
            raise ValueError(f"Position key {key!r} not found in metadata position_keys={position_keys}.")
        return position_to_idx[key]

    if feature_mode == "full":
        selected = geometry
    elif feature_mode == "term_only":
        selected = geometry[:, require_position("TERM"), :, :]
    elif feature_mode == "last_step_only":
        selected = geometry[:, require_position("K-1"), :, :]
    else:
        selected = geometry[:, [require_position("K-1"), require_position("TERM")], :, :]

    return selected.reshape(geometry.shape[0], -1), position_keys, layers


def adjusted_pca_components(requested: int, n_samples: int, x_dim: int) -> int:
    return int(max(1, min(requested, n_samples - 1, x_dim)))


def positive_scores(clf: LogisticRegression, X: np.ndarray) -> np.ndarray | None:
    if not hasattr(clf, "predict_proba"):
        return None
    classes = list(clf.classes_)
    if 1 not in classes:
        return None
    positive_idx = classes.index(1)
    return clf.predict_proba(X)[:, positive_idx]


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    warnings_list: list[str],
    context: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    if y_score is None or len(np.unique(y_true)) < 2:
        metrics["auroc"] = None
        metrics["auprc"] = None
        warnings_list.append(f"{context}: AUROC/AUPRC unavailable because scores are missing or labels have one class.")
    else:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["auprc"] = float(average_precision_score(y_true, y_score))

    return metrics


def make_classifier(seed: int, class_weight: str | None) -> LogisticRegression:
    return LogisticRegression(
        class_weight=class_weight,
        max_iter=2000,
        random_state=seed,
    )


def train_test_probe(
    X_raw: np.ndarray,
    labels: np.ndarray,
    ids: list[str],
    requested_pca_components: int,
    test_size: float,
    seed: int,
    class_weight: str | None,
    warnings_list: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    indices = np.arange(len(labels))
    stratify = labels if len(np.unique(labels)) == 2 and min(np.bincount(labels.astype(int), minlength=2)) >= 2 else None

    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError as exc:
        warnings_list.append(f"Stratified train/test split failed; using non-stratified split. Error: {exc}")
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )

    X_train = X_raw[train_idx]
    X_test = X_raw[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    if len(np.unique(y_train)) < 2:
        raise ValueError("Train split contains fewer than two classes; cannot fit logistic regression.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_components = adjusted_pca_components(requested_pca_components, X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    clf = make_classifier(seed=seed, class_weight=class_weight)
    clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_test_pca)
    y_score = positive_scores(clf, X_test_pca)
    metrics = binary_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        warnings_list=warnings_list,
        context="train_test",
    )
    metrics["train_size"] = int(len(train_idx))
    metrics["test_size"] = int(len(test_idx))
    metrics["pca_components_used"] = int(n_components)

    predictions = []
    for idx, pred, label, score in zip(test_idx, y_pred, y_test, y_score if y_score is not None else [None] * len(test_idx)):
        predictions.append(
            {
                "id": ids[int(idx)],
                "label": int(label),
                "pred": int(pred),
                "score": None if score is None else float(score),
                "split": "test",
            }
        )

    return metrics, predictions, n_components


def summarize_fold_metrics(fold_metrics: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    mean: dict[str, Any] = {}
    std: dict[str, Any] = {}
    for key in METRIC_KEYS:
        values = [fold[key] for fold in fold_metrics if fold.get(key) is not None]
        if values:
            mean[key] = float(np.mean(values))
            std[key] = float(np.std(values, ddof=0))
        else:
            mean[key] = None
            std[key] = None
    return mean, std


def cross_validate_probe(
    X_raw: np.ndarray,
    labels: np.ndarray,
    requested_folds: int,
    requested_pca_components: int,
    seed: int,
    class_weight: str | None,
    warnings_list: list[str],
) -> dict[str, Any]:
    counts = class_counts(labels)
    min_class_count = min(counts.values())
    n_folds = int(min(requested_folds, min_class_count))

    if n_folds < requested_folds:
        warnings_list.append(
            f"CV folds adjusted from {requested_folds} to {n_folds} because min_class_count={min_class_count}."
        )

    if n_folds < 2:
        warnings_list.append(f"Skipping CV because min_class_count={min_class_count} is less than 2.")
        return {
            "n_folds": 0,
            "fold_metrics": [],
            "mean": {key: None for key in METRIC_KEYS},
            "std": {key: None for key in METRIC_KEYS},
        }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_raw, labels), start=1):
        X_train = X_raw[train_idx]
        X_val = X_raw[val_idx]
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        n_components = adjusted_pca_components(
            requested_pca_components,
            X_train_scaled.shape[0],
            X_train_scaled.shape[1],
        )
        pca = PCA(n_components=n_components, random_state=seed)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)

        clf = make_classifier(seed=seed, class_weight=class_weight)
        clf.fit(X_train_pca, y_train)

        y_pred = clf.predict(X_val_pca)
        y_score = positive_scores(clf, X_val_pca)
        fold_result = binary_metrics(
            y_true=y_val,
            y_pred=y_pred,
            y_score=y_score,
            warnings_list=warnings_list,
            context=f"cv_fold_{fold_idx}",
        )
        fold_result["fold"] = fold_idx
        fold_result["train_size"] = int(len(train_idx))
        fold_result["val_size"] = int(len(val_idx))
        fold_result["pca_components_used"] = int(n_components)
        fold_metrics.append(fold_result)

    mean, std = summarize_fold_metrics(fold_metrics)
    return {
        "n_folds": n_folds,
        "fold_metrics": fold_metrics,
        "mean": mean,
        "std": std,
    }


def fit_final_artifacts(
    X_raw: np.ndarray,
    labels: np.ndarray,
    requested_pca_components: int,
    seed: int,
    class_weight: str | None,
    pca_file: Path,
    scaler_file: Path,
    pca_model_file: Path,
    logreg_model_file: Path,
) -> int:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    n_components = adjusted_pca_components(requested_pca_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)

    clf = make_classifier(seed=seed, class_weight=class_weight)
    clf.fit(X_pca, labels)

    pca_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(pca_file, X_pca.astype(np.float32, copy=False))

    scaler_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_file)
    joblib.dump(pca, pca_model_file)
    joblib.dump(clf, logreg_model_file)

    return int(n_components)


def write_report(
    path: Path,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    warnings_list: list[str],
) -> None:
    train_test = metrics["train_test"]
    cv = metrics["cv"]
    lines = [
        "# Geometry-only Probe Report",
        "",
        "## Goal",
        "",
        "Train a Microsoft-style Step-Layer hidden-state geometry baseline.",
        "",
        "## Input",
        "",
        f"- geometry_raw.npy shape: `{metrics['geometry_shape']}`",
        f"- labels shape: `{metrics['labels_shape']}`",
        f"- position keys: `{metadata.get('position_keys')}`",
        f"- layers: `{metadata.get('layers')}`",
        "",
        "## Feature mode",
        "",
        f"`{metrics['feature_mode']}`",
        "",
        "## Class distribution",
        "",
        f"- correct (`0`): {metrics['class_counts']['0']}",
        f"- wrong (`1`): {metrics['class_counts']['1']}",
        "",
        "## Metrics",
        "",
        "### Train/Test",
        "",
        f"- accuracy: {train_test['accuracy']}",
        f"- precision: {train_test['precision']}",
        f"- recall: {train_test['recall']}",
        f"- f1: {train_test['f1']}",
        f"- auroc: {train_test['auroc']}",
        f"- auprc: {train_test['auprc']}",
        f"- confusion_matrix: `{train_test['confusion_matrix']}`",
        "",
        "### CV",
        "",
        f"- folds: {cv['n_folds']}",
        f"- mean: `{cv['mean']}`",
        f"- std: `{cv['std']}`",
        "",
        "## Notes",
        "",
        "Current v0 has only 29 samples, so these numbers are a pipeline sanity check, not a formal result.",
    ]

    if warnings_list:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings_list)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    probe = get_probe_config(cfg, args)

    feature_mode = str(probe.get("feature_mode", "full"))
    if feature_mode not in VALID_FEATURE_MODES:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}.")

    classifier = str(probe.get("classifier", "logistic_regression"))
    if classifier != "logistic_regression":
        raise ValueError(f"Only logistic_regression is supported in v0, got {classifier}.")

    requested_pca_components = int(probe.get("pca_components", 16))
    requested_cv_folds = int(probe.get("cv_folds", 5))
    test_size = float(probe.get("test_size", 0.3))
    seed = int(probe.get("random_seed", 42))
    class_weight = probe.get("class_weight", "balanced")
    if class_weight in {"none", "None", ""}:
        class_weight = None

    geometry_file = Path(probe["geometry_file"])
    labels_file = Path(probe["labels_file"])
    ids_file = Path(probe["ids_file"])
    metadata_file = Path(probe["metadata_file"])
    pca_file = Path(probe["pca_file"])
    output_dir = Path(probe["output_dir"])
    scaler_file = Path(probe["scaler_file"])
    pca_model_file = Path(probe["pca_model_file"])
    logreg_model_file = output_dir / "logreg_model.joblib"
    metrics_file = Path(probe["metrics_file"])
    predictions_file = Path(probe["predictions_file"])
    report_file = Path(probe["report_file"])

    ensure_outputs_can_be_written(
        [
            pca_file,
            scaler_file,
            pca_model_file,
            logreg_model_file,
            metrics_file,
            predictions_file,
            report_file,
        ],
        overwrite=args.overwrite,
    )

    warnings_list: list[str] = []
    geometry = np.load(geometry_file)
    labels = np.load(labels_file)
    ids = load_json(ids_file)
    metadata = load_json(metadata_file)

    validate_inputs(geometry, labels, ids, warnings_list)
    X_raw, position_keys, layers = build_features(geometry, metadata, feature_mode)
    counts = class_counts(labels)

    train_test_metrics, predictions, split_pca_components = train_test_probe(
        X_raw=X_raw,
        labels=labels,
        ids=ids,
        requested_pca_components=requested_pca_components,
        test_size=test_size,
        seed=seed,
        class_weight=class_weight,
        warnings_list=warnings_list,
    )

    cv_metrics = cross_validate_probe(
        X_raw=X_raw,
        labels=labels,
        requested_folds=requested_cv_folds,
        requested_pca_components=requested_pca_components,
        seed=seed,
        class_weight=class_weight,
        warnings_list=warnings_list,
    )

    final_pca_components = fit_final_artifacts(
        X_raw=X_raw,
        labels=labels,
        requested_pca_components=requested_pca_components,
        seed=seed,
        class_weight=class_weight,
        pca_file=pca_file,
        scaler_file=scaler_file,
        pca_model_file=pca_model_file,
        logreg_model_file=logreg_model_file,
    )

    metrics = {
        "feature_mode": feature_mode,
        "geometry_shape": list(geometry.shape),
        "labels_shape": list(labels.shape),
        "X_raw_shape": list(X_raw.shape),
        "class_counts": counts,
        "position_keys": position_keys,
        "layers": layers,
        "classifier": classifier,
        "class_weight": class_weight,
        "random_seed": seed,
        "test_size": test_size,
        "requested_pca_components": requested_pca_components,
        "pca_components_used": split_pca_components,
        "final_pca_components_used": final_pca_components,
        "train_test": train_test_metrics,
        "cv": cv_metrics,
        "warnings": warnings_list,
        "artifacts": {
            "geometry_pca": str(pca_file),
            "scaler": str(scaler_file),
            "pca_model": str(pca_model_file),
            "logreg_model": str(logreg_model_file),
        },
    }

    write_json(metrics_file, metrics)
    write_jsonl(predictions_file, predictions)
    write_report(report_file, metrics, metadata, warnings_list)

    print(f"Geometry shape: {list(geometry.shape)}")
    print(f"Labels shape: {list(labels.shape)}")
    print(f"Class counts: {counts}")
    print(f"Feature mode: {feature_mode}")
    print(f"Raw feature shape: {list(X_raw.shape)}")
    print(f"PCA components used: {split_pca_components}")
    print(f"Train/test metrics: {train_test_metrics}")
    print(f"CV mean metrics: {cv_metrics['mean']}")
    print(f"Output metrics path: {metrics_file}")
    print(f"Output report path: {report_file}")


if __name__ == "__main__":
    main()
