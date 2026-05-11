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


VALID_FEATURE_SETS = {"geometry_only", "topology_only", "joint_concat"}
METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "auroc", "auprc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train geometry-only, topology-only, and joint concat probes."
    )
    parser.add_argument("--config", required=True, help="Path to topo_traj_v0.yaml.")
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=sorted(VALID_FEATURE_SETS),
        default=None,
        help="Feature sets to train. Defaults to joint_probe.feature_sets.",
    )
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
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
            "Joint probe output files already exist. Use --overwrite to replace them:\n"
            f"{joined}"
        )


def get_joint_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    joint_cfg = dict(config["joint_probe"])
    if args.feature_sets is not None:
        joint_cfg["feature_sets"] = args.feature_sets
    if args.test_size is not None:
        joint_cfg["test_size"] = args.test_size
    if args.cv_folds is not None:
        joint_cfg["cv_folds"] = args.cv_folds
    if args.seed is not None:
        joint_cfg["random_seed"] = args.seed
    return joint_cfg


def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "0": int(np.sum(labels == 0)),
        "1": int(np.sum(labels == 1)),
    }


def validate_inputs(
    geometry: np.ndarray,
    geometry_labels: np.ndarray,
    geometry_ids: list[str],
    geometry_metadata: dict[str, Any],
    topology: np.ndarray,
    topology_labels: np.ndarray,
    topology_ids: list[str],
    topology_metadata: dict[str, Any],
    global_warnings: list[str],
) -> np.ndarray:
    if geometry.ndim != 4:
        raise ValueError(f"geometry_raw.npy must have shape [N, 3, 3, hidden_dim], got {geometry.shape}.")
    if topology.ndim != 4:
        raise ValueError(f"topology_raw.npy must have shape [N, 3, 3, F], got {topology.shape}.")
    if geometry.shape[0] != topology.shape[0]:
        raise ValueError(f"Geometry/topology N mismatch: {geometry.shape[0]} != {topology.shape[0]}.")
    if geometry_labels.ndim != 1 or topology_labels.ndim != 1:
        raise ValueError("Label arrays must be one-dimensional.")
    if geometry_labels.shape[0] != geometry.shape[0]:
        raise ValueError("Geometry labels length does not match geometry first dimension.")
    if topology_labels.shape[0] != topology.shape[0]:
        raise ValueError("Topology labels length does not match topology first dimension.")
    if not np.array_equal(geometry_labels, topology_labels):
        raise ValueError("Geometry labels and topology labels are not identical.")
    if len(geometry_ids) != geometry.shape[0]:
        raise ValueError("Geometry ids length does not match geometry first dimension.")
    if len(topology_ids) != topology.shape[0]:
        raise ValueError("Topology ids length does not match topology first dimension.")
    if geometry_ids != topology_ids:
        raise ValueError("Geometry ids and topology ids are not aligned in the same order.")

    labels = geometry_labels.astype(np.int64, copy=False)
    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Labels may only contain 0 and 1, got {sorted(unique_labels)}.")

    counts = class_counts(labels)
    if min(counts.values()) < 5:
        global_warnings.append(
            f"Small minority class count detected: {counts}. Metrics are pipeline sanity checks only."
        )
    if min(counts.values()) < 2:
        global_warnings.append(
            f"Very small class count detected: {counts}. Stratified split or CV may be skipped."
        )

    if geometry_metadata.get("position_keys") and topology_metadata.get("position_keys"):
        if geometry_metadata["position_keys"] != topology_metadata["position_keys"]:
            global_warnings.append("Geometry and topology position_keys differ in metadata.")
    if geometry_metadata.get("layers") and topology_metadata.get("layers"):
        if geometry_metadata["layers"] != topology_metadata["layers"]:
            global_warnings.append("Geometry and topology layers differ in metadata.")

    return labels


def build_feature_matrices(geometry: np.ndarray, topology: np.ndarray) -> dict[str, np.ndarray]:
    n_samples = int(geometry.shape[0])
    X_geo = geometry.reshape(n_samples, -1)
    X_topo = topology.reshape(n_samples, -1)
    X_joint = np.concatenate([X_geo, X_topo], axis=1)
    return {
        "geometry_only": X_geo,
        "topology_only": X_topo,
        "joint_concat": X_joint,
    }


def pca_components_for_feature_set(feature_set: str, joint_cfg: dict[str, Any]) -> int | None:
    if feature_set == "geometry_only":
        value = joint_cfg.get("geometry_pca_components")
    elif feature_set == "topology_only":
        value = joint_cfg.get("topology_pca_components")
    elif feature_set == "joint_concat":
        value = joint_cfg.get("joint_pca_components")
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}.")
    return None if value is None else int(value)


def build_shared_split(
    labels: np.ndarray,
    test_size: float,
    seed: int,
    global_warnings: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    counts = class_counts(labels)
    stratify = labels if len(np.unique(labels)) == 2 and min(counts.values()) >= 2 else None
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError as exc:
        global_warnings.append(f"Stratified train/test split failed; using non-stratified split. Error: {exc}")
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
    return train_idx, test_idx


def build_shared_cv_folds(
    labels: np.ndarray,
    requested_folds: int,
    seed: int,
    global_warnings: list[str],
) -> list[tuple[np.ndarray, np.ndarray]]:
    counts = class_counts(labels)
    min_class_count = min(counts.values())
    n_folds = int(min(requested_folds, min_class_count))

    if n_folds < requested_folds:
        global_warnings.append(
            f"CV folds adjusted from {requested_folds} to {n_folds} because min_class_count={min_class_count}."
        )
    if n_folds < 2:
        global_warnings.append(f"Skipping CV because min_class_count={min_class_count} is less than 2.")
        return []

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(train_idx, val_idx) for train_idx, val_idx in splitter.split(np.zeros(len(labels)), labels)]


def make_classifier(seed: int, class_weight: str | None, max_iter: int) -> LogisticRegression:
    return LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=seed,
    )


def positive_scores(clf: LogisticRegression, X: np.ndarray) -> np.ndarray | None:
    if not hasattr(clf, "predict_proba"):
        return None
    classes = list(clf.classes_)
    if 1 not in classes:
        return None
    return clf.predict_proba(X)[:, classes.index(1)]


def transform_with_scaler_pca(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    requested_pca_components: int | None,
    seed: int,
    warnings_list: list[str],
    context: str,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, PCA | None, int | None, int]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    if requested_pca_components is None:
        return X_train_scaled, X_eval_scaled, scaler, None, None, int(X_train_scaled.shape[1])

    max_components = min(int(requested_pca_components), X_train_scaled.shape[0] - 1, X_train_scaled.shape[1])
    if max_components < 1:
        warnings_list.append(f"{context}: PCA skipped because adjusted n_components={max_components}.")
        return X_train_scaled, X_eval_scaled, scaler, None, None, int(X_train_scaled.shape[1])

    if max_components < int(requested_pca_components):
        warnings_list.append(
            f"{context}: PCA components adjusted from {requested_pca_components} to {max_components}."
        )

    pca = PCA(n_components=int(max_components), random_state=seed)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_eval_pca = pca.transform(X_eval_scaled)
    return X_train_pca, X_eval_pca, scaler, pca, int(max_components), int(max_components)


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
        warnings_list.append(f"{context}: AUROC/AUPRC unavailable because labels have one class or scores are missing.")
    else:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
    return metrics


def empty_metrics(reason: str) -> dict[str, Any]:
    return {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "auroc": None,
        "auprc": None,
        "confusion_matrix": [[0, 0], [0, 0]],
        "reason": reason,
    }


def train_test_for_feature_set(
    feature_set: str,
    X: np.ndarray,
    labels: np.ndarray,
    ids: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    requested_pca_components: int | None,
    seed: int,
    class_weight: str | None,
    max_iter: int,
    warnings_list: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    if len(np.unique(y_train)) < 2:
        warnings_list.append(f"{feature_set}: train split has fewer than two classes; skipping train/test fit.")
        metrics = empty_metrics("train_split_single_class")
        metrics.update(
            {
                "class_counts": class_counts(labels),
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "pca_components_used": None,
                "feature_dim_raw": int(X.shape[1]),
                "feature_dim_after_pca": None,
            }
        )
        return metrics, []

    X_train_model, X_test_model, _, _, pca_used, feature_dim_after_pca = transform_with_scaler_pca(
        X_train=X_train,
        X_eval=X_test,
        requested_pca_components=requested_pca_components,
        seed=seed,
        warnings_list=warnings_list,
        context=f"{feature_set}/train_test",
    )

    clf = make_classifier(seed=seed, class_weight=class_weight, max_iter=max_iter)
    clf.fit(X_train_model, y_train)

    y_pred = clf.predict(X_test_model)
    y_score = positive_scores(clf, X_test_model)
    metrics = binary_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        warnings_list=warnings_list,
        context=f"{feature_set}/train_test",
    )
    metrics.update(
        {
            "class_counts": class_counts(labels),
            "train_size": int(len(train_idx)),
            "test_size": int(len(test_idx)),
            "pca_components_used": pca_used,
            "feature_dim_raw": int(X.shape[1]),
            "feature_dim_after_pca": int(feature_dim_after_pca),
        }
    )

    predictions = []
    fallback_scores = [None] * len(test_idx)
    for idx, label, pred, score in zip(test_idx, y_test, y_pred, y_score if y_score is not None else fallback_scores):
        predictions.append(
            {
                "feature_set": feature_set,
                "id": ids[int(idx)],
                "label": int(label),
                "pred": int(pred),
                "score": None if score is None else float(score),
                "split": "test",
            }
        )

    return metrics, predictions


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


def cv_for_feature_set(
    feature_set: str,
    X: np.ndarray,
    labels: np.ndarray,
    cv_folds: list[tuple[np.ndarray, np.ndarray]],
    requested_pca_components: int | None,
    seed: int,
    class_weight: str | None,
    max_iter: int,
    warnings_list: list[str],
) -> dict[str, Any]:
    if not cv_folds:
        return {
            "n_folds": 0,
            "fold_metrics": [],
            "mean": {key: None for key in METRIC_KEYS},
            "std": {key: None for key in METRIC_KEYS},
        }

    fold_metrics: list[dict[str, Any]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv_folds, start=1):
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = labels[train_idx]
        y_val = labels[val_idx]

        if len(np.unique(y_train)) < 2:
            warnings_list.append(f"{feature_set}/cv_fold_{fold_idx}: train fold has fewer than two classes; skipped.")
            fold_result = empty_metrics("train_fold_single_class")
            fold_result.update(
                {
                    "fold": fold_idx,
                    "train_size": int(len(train_idx)),
                    "val_size": int(len(val_idx)),
                    "pca_components_used": None,
                    "feature_dim_raw": int(X.shape[1]),
                    "feature_dim_after_pca": None,
                }
            )
            fold_metrics.append(fold_result)
            continue

        X_train_model, X_val_model, _, _, pca_used, feature_dim_after_pca = transform_with_scaler_pca(
            X_train=X_train,
            X_eval=X_val,
            requested_pca_components=requested_pca_components,
            seed=seed,
            warnings_list=warnings_list,
            context=f"{feature_set}/cv_fold_{fold_idx}",
        )

        clf = make_classifier(seed=seed, class_weight=class_weight, max_iter=max_iter)
        clf.fit(X_train_model, y_train)

        y_pred = clf.predict(X_val_model)
        y_score = positive_scores(clf, X_val_model)
        fold_result = binary_metrics(
            y_true=y_val,
            y_pred=y_pred,
            y_score=y_score,
            warnings_list=warnings_list,
            context=f"{feature_set}/cv_fold_{fold_idx}",
        )
        fold_result.update(
            {
                "fold": fold_idx,
                "train_size": int(len(train_idx)),
                "val_size": int(len(val_idx)),
                "pca_components_used": pca_used,
                "feature_dim_raw": int(X.shape[1]),
                "feature_dim_after_pca": int(feature_dim_after_pca),
            }
        )
        fold_metrics.append(fold_result)

    mean, std = summarize_fold_metrics(fold_metrics)
    return {
        "n_folds": len(cv_folds),
        "fold_metrics": fold_metrics,
        "mean": mean,
        "std": std,
    }


def fit_final_model(
    feature_set: str,
    X: np.ndarray,
    labels: np.ndarray,
    requested_pca_components: int | None,
    seed: int,
    class_weight: str | None,
    max_iter: int,
    models_dir: Path,
    warnings_list: list[str],
) -> dict[str, str | None]:
    artifacts: dict[str, str | None] = {
        "scaler": None,
        "pca": None,
        "logreg": None,
    }
    if len(np.unique(labels)) < 2:
        warnings_list.append(f"{feature_set}: final model not saved because labels have fewer than two classes.")
        return artifacts

    models_dir.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = None
    X_model = X_scaled
    pca_used = None
    if requested_pca_components is not None:
        max_components = min(int(requested_pca_components), X_scaled.shape[0] - 1, X_scaled.shape[1])
        if max_components < 1:
            warnings_list.append(f"{feature_set}/final: PCA skipped because adjusted n_components={max_components}.")
        else:
            if max_components < int(requested_pca_components):
                warnings_list.append(
                    f"{feature_set}/final: PCA components adjusted from {requested_pca_components} to {max_components}."
                )
            pca = PCA(n_components=int(max_components), random_state=seed)
            X_model = pca.fit_transform(X_scaled)
            pca_used = int(max_components)

    clf = make_classifier(seed=seed, class_weight=class_weight, max_iter=max_iter)
    clf.fit(X_model, labels)

    scaler_path = models_dir / f"{feature_set}_scaler.joblib"
    logreg_path = models_dir / f"{feature_set}_logreg.joblib"
    joblib.dump(scaler, scaler_path)
    joblib.dump(clf, logreg_path)
    artifacts["scaler"] = str(scaler_path)
    artifacts["logreg"] = str(logreg_path)

    if pca is not None:
        pca_path = models_dir / f"{feature_set}_pca.joblib"
        joblib.dump(pca, pca_path)
        artifacts["pca"] = str(pca_path)
    artifacts["pca_components_used"] = pca_used
    return artifacts


def compare_auroc(feature_results: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    def auroc(name: str) -> float | None:
        value = feature_results.get(name, {}).get("train_test", {}).get("auroc")
        return None if value is None else float(value)

    geometry = auroc("geometry_only")
    topology = auroc("topology_only")
    joint = auroc("joint_concat")

    return {
        "joint_minus_geometry_auroc": None if joint is None or geometry is None else joint - geometry,
        "joint_minus_topology_auroc": None if joint is None or topology is None else joint - topology,
        "topology_minus_geometry_auroc": None if topology is None or geometry is None else topology - geometry,
    }


def format_metric(value: Any) -> str:
    return "null" if value is None else str(value)


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    dataset = metrics["dataset"]
    feature_results = metrics["feature_sets"]
    lines = [
        "# Topology + Geometry Probe Report",
        "",
        "## Goal",
        "",
        "Compare geometry-only, topology-only, and geometry+topology concat probes.",
        "",
        "## Input",
        "",
        f"- geometry_raw.npy shape: `{dataset['geometry_shape']}`",
        f"- topology_raw.npy shape: `{dataset['topology_shape']}`",
        f"- labels shape: `[{dataset['num_samples']}]`",
        f"- class distribution: `{dataset['class_counts']}`",
        f"- ids aligned: `{dataset['ids_aligned']}`",
        "",
        "## Methods",
        "",
        "- StandardScaler fit on train split or train fold only",
        "- Optional PCA fit on train split or train fold only",
        "- LogisticRegression with configured class weight",
        "",
        "## Metrics",
        "",
    ]

    for feature_set, result in feature_results.items():
        train_test = result["train_test"]
        cv = result["cv"]
        lines.extend(
            [
                f"### {feature_set}",
                "",
                f"- train/test accuracy: {format_metric(train_test.get('accuracy'))}",
                f"- train/test precision: {format_metric(train_test.get('precision'))}",
                f"- train/test recall: {format_metric(train_test.get('recall'))}",
                f"- train/test f1: {format_metric(train_test.get('f1'))}",
                f"- train/test auroc: {format_metric(train_test.get('auroc'))}",
                f"- train/test auprc: {format_metric(train_test.get('auprc'))}",
                f"- train/test confusion_matrix: `{train_test.get('confusion_matrix')}`",
                f"- CV folds: {cv.get('n_folds')}",
                f"- CV mean: `{cv.get('mean')}`",
                f"- CV std: `{cv.get('std')}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation warning",
            "",
            "Current v0 has only 29 samples and strong class imbalance, so metrics are pipeline sanity checks only, not formal evidence.",
        ]
    )

    global_warnings = metrics.get("global_warnings", [])
    feature_warnings = [
        f"{feature_set}: {warning}"
        for feature_set, result in feature_results.items()
        for warning in result.get("warnings", [])
    ]
    if global_warnings or feature_warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in global_warnings + feature_warnings)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    joint_cfg = get_joint_config(config, args)

    classifier = str(joint_cfg.get("classifier", "logistic_regression"))
    if classifier != "logistic_regression":
        raise ValueError(f"Only logistic_regression is supported in v0, got {classifier}.")

    feature_sets = list(joint_cfg.get("feature_sets", ["geometry_only", "topology_only", "joint_concat"]))
    unknown = [name for name in feature_sets if name not in VALID_FEATURE_SETS]
    if unknown:
        raise ValueError(f"Unsupported feature sets: {unknown}.")

    output_dir = Path(joint_cfg["output_dir"])
    metrics_file = Path(joint_cfg["metrics_file"])
    predictions_file = Path(joint_cfg["predictions_file"])
    report_file = Path(joint_cfg["report_file"])
    models_dir = output_dir / "models"
    model_files = []
    for feature_set in feature_sets:
        model_files.append(models_dir / f"{feature_set}_scaler.joblib")
        model_files.append(models_dir / f"{feature_set}_logreg.joblib")
        if pca_components_for_feature_set(feature_set, joint_cfg) is not None:
            model_files.append(models_dir / f"{feature_set}_pca.joblib")

    ensure_outputs_can_be_written(
        [metrics_file, predictions_file, report_file] + model_files,
        overwrite=args.overwrite,
    )

    geometry = np.load(joint_cfg["geometry_file"])
    geometry_labels = np.load(joint_cfg["geometry_labels_file"])
    geometry_ids = load_json(Path(joint_cfg["geometry_ids_file"]))
    geometry_metadata = load_json(Path(joint_cfg["geometry_metadata_file"]))

    topology = np.load(joint_cfg["topology_file"])
    topology_labels = np.load(joint_cfg["topology_labels_file"])
    topology_ids = load_json(Path(joint_cfg["topology_ids_file"]))
    topology_metadata = load_json(Path(joint_cfg["topology_metadata_file"]))

    global_warnings: list[str] = []
    labels = validate_inputs(
        geometry=geometry,
        geometry_labels=geometry_labels,
        geometry_ids=geometry_ids,
        geometry_metadata=geometry_metadata,
        topology=topology,
        topology_labels=topology_labels,
        topology_ids=topology_ids,
        topology_metadata=topology_metadata,
        global_warnings=global_warnings,
    )

    test_size = float(joint_cfg.get("test_size", 0.3))
    cv_folds_requested = int(joint_cfg.get("cv_folds", 5))
    seed = int(joint_cfg.get("random_seed", 42))
    max_iter = int(joint_cfg.get("max_iter", 2000))
    class_weight = joint_cfg.get("class_weight", "balanced")
    if class_weight in {"none", "None", ""}:
        class_weight = None

    feature_matrices = build_feature_matrices(geometry=geometry, topology=topology)
    train_idx, test_idx = build_shared_split(
        labels=labels,
        test_size=test_size,
        seed=seed,
        global_warnings=global_warnings,
    )
    cv_folds = build_shared_cv_folds(
        labels=labels,
        requested_folds=cv_folds_requested,
        seed=seed,
        global_warnings=global_warnings,
    )

    all_predictions: list[dict[str, Any]] = []
    feature_results: dict[str, dict[str, Any]] = {}

    for feature_set in feature_sets:
        X = feature_matrices[feature_set]
        requested_pca = pca_components_for_feature_set(feature_set, joint_cfg)
        feature_warnings: list[str] = []
        train_test_metrics, predictions = train_test_for_feature_set(
            feature_set=feature_set,
            X=X,
            labels=labels,
            ids=geometry_ids,
            train_idx=train_idx,
            test_idx=test_idx,
            requested_pca_components=requested_pca,
            seed=seed,
            class_weight=class_weight,
            max_iter=max_iter,
            warnings_list=feature_warnings,
        )
        cv_metrics = cv_for_feature_set(
            feature_set=feature_set,
            X=X,
            labels=labels,
            cv_folds=cv_folds,
            requested_pca_components=requested_pca,
            seed=seed,
            class_weight=class_weight,
            max_iter=max_iter,
            warnings_list=feature_warnings,
        )
        artifacts = fit_final_model(
            feature_set=feature_set,
            X=X,
            labels=labels,
            requested_pca_components=requested_pca,
            seed=seed,
            class_weight=class_weight,
            max_iter=max_iter,
            models_dir=models_dir,
            warnings_list=feature_warnings,
        )
        feature_results[feature_set] = {
            "train_test": train_test_metrics,
            "cv": cv_metrics,
            "warnings": feature_warnings,
            "artifacts": artifacts,
        }
        all_predictions.extend(predictions)

    metrics = {
        "dataset": {
            "num_samples": int(geometry.shape[0]),
            "class_counts": class_counts(labels),
            "geometry_shape": list(geometry.shape),
            "topology_shape": list(topology.shape),
            "geometry_position_keys": geometry_metadata.get("position_keys"),
            "geometry_layers": geometry_metadata.get("layers"),
            "topology_position_keys": topology_metadata.get("position_keys"),
            "topology_layers": topology_metadata.get("layers"),
            "topology_feature_names": topology_metadata.get("feature_names"),
            "ids_aligned": True,
        },
        "settings": {
            "random_seed": seed,
            "test_size": test_size,
            "cv_folds_requested": cv_folds_requested,
            "cv_folds_used": len(cv_folds),
            "class_weight": class_weight,
            "classifier": classifier,
            "max_iter": max_iter,
            "feature_sets": feature_sets,
            "geometry_pca_components": joint_cfg.get("geometry_pca_components"),
            "topology_pca_components": joint_cfg.get("topology_pca_components"),
            "joint_pca_components": joint_cfg.get("joint_pca_components"),
        },
        "feature_sets": feature_results,
        "comparison": compare_auroc(feature_results),
        "global_warnings": global_warnings,
    }

    write_json(metrics_file, metrics)
    write_jsonl(predictions_file, all_predictions)
    write_report(report_file, metrics)

    print("Dataset summary:")
    print(f"- num_samples: {metrics['dataset']['num_samples']}")
    print(f"- class_counts: {metrics['dataset']['class_counts']}")
    print(f"- geometry_shape: {metrics['dataset']['geometry_shape']}")
    print(f"- topology_shape: {metrics['dataset']['topology_shape']}")
    print(f"- ids_aligned: {metrics['dataset']['ids_aligned']}")
    for feature_set in feature_sets:
        result = feature_results[feature_set]
        train_test = result["train_test"]
        cv_mean = result["cv"]["mean"]
        print(f"Feature set: {feature_set}")
        print(f"- raw feature shape: {[int(geometry.shape[0]), int(feature_matrices[feature_set].shape[1])]}")
        print(f"- pca components used: {train_test.get('pca_components_used')}")
        print(
            "- train/test AUROC, AUPRC, F1: "
            f"{train_test.get('auroc')}, {train_test.get('auprc')}, {train_test.get('f1')}"
        )
        print(
            "- CV mean AUROC, AUPRC, F1: "
            f"{cv_mean.get('auroc')}, {cv_mean.get('auprc')}, {cv_mean.get('f1')}"
        )
    print("Output:")
    print(f"- metrics path: {metrics_file}")
    print(f"- report path: {report_file}")
    print(f"- predictions path: {predictions_file}")


if __name__ == "__main__":
    main()
