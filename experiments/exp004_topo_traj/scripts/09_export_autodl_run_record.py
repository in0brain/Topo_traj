import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = "experiments/exp004_topo_traj/configs/topo_traj_server.yaml"
DEFAULT_ARCHIVE_ROOT = "/root/autodl-fs/topo_traj_records/exp004_topo_traj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive key AutoDL run records for exp004_topo_traj."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to topo_traj_server.yaml.")
    parser.add_argument("--archive-root", default=DEFAULT_ARCHIVE_ROOT, help="Root directory for archived run records.")
    parser.add_argument("--run-name", default=None, help="Optional run name appended to archive directory.")
    parser.add_argument("--include-large", action="store_true", help="Also copy .npy files and the full cache directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned copies without writing files.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sanitize_run_name(name: str | None) -> str:
    if not name:
        return "default"
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return sanitized.strip("._-") or "default"


def make_archive_dir(archive_root: Path, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = archive_root / f"{stamp}_{run_name}"
    if not base.exists():
        return base

    suffix = 1
    while True:
        candidate = archive_root / f"{stamp}_{run_name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def run_command(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def get_git_info() -> tuple[str | None, str | None]:
    commit = run_command(["git", "rev-parse", "HEAD"])
    branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    return commit, branch


def get_torch_info() -> tuple[str | None, bool | None, str | None]:
    try:
        import torch
    except Exception:
        return None, None, None

    torch_version = getattr(torch, "__version__", None)
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = None
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = None
    return torch_version, cuda_available, gpu_name


def copy_file(
    source: Path,
    destination: Path,
    copied_files: list[str],
    missing_files: list[str],
    dry_run: bool,
) -> None:
    if not source.exists():
        missing_files.append(str(source))
        return
    copied_files.append(str(destination))
    if dry_run:
        print(f"[dry-run] copy file: {source} -> {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(
    source: Path,
    destination: Path,
    copied_files: list[str],
    missing_files: list[str],
    dry_run: bool,
) -> None:
    if not source.exists():
        missing_files.append(str(source))
        return

    if dry_run:
        for path in source.rglob("*"):
            if path.is_file():
                rel = path.relative_to(source)
                copied_files.append(str(destination / rel))
                print(f"[dry-run] copy file: {path} -> {destination / rel}")
        return

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    for path in destination.rglob("*"):
        if path.is_file():
            copied_files.append(str(path))


def copy_log_files(log_dir: Path, archive_dir: Path, copied_files: list[str], missing_files: list[str], dry_run: bool) -> None:
    if not log_dir.exists():
        missing_files.append(str(log_dir))
        return
    log_files = sorted(log_dir.glob("*.log"))
    if not log_files:
        missing_files.append(str(log_dir / "*.log"))
        return
    for log_file in log_files:
        copy_file(log_file, archive_dir / "logs" / log_file.name, copied_files, missing_files, dry_run)


def collect_and_copy_files(
    config_path: Path,
    cfg: dict[str, Any],
    archive_dir: Path,
    include_large: bool,
    dry_run: bool,
) -> tuple[list[str], list[str]]:
    copied_files: list[str] = []
    missing_files: list[str] = []

    paths = cfg.get("paths", {})
    cache = cfg.get("cache", {})
    markers = cfg.get("markers", {})
    geometry = cfg.get("geometry", {})
    probe = cfg.get("probe", {})
    topology_raw = cfg.get("topology_raw", {})
    joint_probe = cfg.get("joint_probe", {})

    copy_file(config_path, archive_dir / "config" / config_path.name, copied_files, missing_files, dry_run)
    copy_log_files(Path(paths.get("log_dir", "")), archive_dir, copied_files, missing_files, dry_run)

    output_files = [
        Path(paths.get("debug_preview_file", "")),
        Path(probe.get("metrics_file", "")),
        Path(probe.get("predictions_file", "")),
        Path(probe.get("report_file", "")),
        Path(joint_probe.get("metrics_file", "")),
        Path(joint_probe.get("predictions_file", "")),
        Path(joint_probe.get("report_file", "")),
    ]
    for source in output_files:
        if str(source):
            copy_file(source, archive_dir / "outputs" / source.name, copied_files, missing_files, dry_run)

    feature_files = [
        Path(geometry.get("metadata_file", "")),
        Path(geometry.get("manifest_file", "")),
        Path(topology_raw.get("metadata_file", "")),
        Path(topology_raw.get("manifest_file", "")),
        Path(geometry.get("ids_file", "")),
        Path(topology_raw.get("ids_file", "")),
    ]
    if include_large:
        feature_files.extend(
            [
                Path(geometry.get("geometry_file", "")),
                Path(geometry.get("labels_file", "")),
                Path(probe.get("pca_file", "")),
                Path(topology_raw.get("topology_file", "")),
                Path(topology_raw.get("labels_file", "")),
            ]
        )
    for source in feature_files:
        if str(source):
            copy_file(source, archive_dir / "features" / source.name, copied_files, missing_files, dry_run)

    cache_manifest_files = [
        Path(cache.get("manifest_file", "")),
        Path(markers.get("marker_manifest_file", "")),
    ]
    for source in cache_manifest_files:
        if str(source):
            copy_file(source, archive_dir / "cache" / source.name, copied_files, missing_files, dry_run)

    if include_large:
        cache_dir = Path(paths.get("cache_dir", ""))
        copy_tree(cache_dir, archive_dir / "cache_full", copied_files, missing_files, dry_run)

    data_files = [
        Path(paths.get("sample_file", "")),
        Path(paths.get("generations_file", "")),
    ]
    for source in data_files:
        if str(source):
            copy_file(source, archive_dir / "data" / source.name, copied_files, missing_files, dry_run)

    return copied_files, missing_files


def build_summary(
    cfg: dict[str, Any],
    config_path: Path,
    archive_dir: Path,
    run_name: str,
    copied_files: list[str],
    missing_files: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    git_commit, git_branch = get_git_info()
    torch_version, cuda_available, gpu_name = get_torch_info()
    experiment = cfg.get("experiment", {})
    model = cfg.get("model", {})
    cache = cfg.get("cache", {})
    data = cfg.get("data", {})

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "archive_dir": str(archive_dir),
        "config_path": str(config_path),
        "experiment.name": experiment.get("name"),
        "experiment.stage": experiment.get("stage"),
        "experiment.max_samples": experiment.get("max_samples"),
        "model.model_name": model.get("model_name"),
        "model.max_new_tokens": model.get("max_new_tokens"),
        "cache.selected_layers": cache.get("selected_layers"),
        "cache.save_hidden_states": cache.get("save_hidden_states"),
        "cache.save_attentions": cache.get("save_attentions"),
        "cache.attn_implementation": cache.get("attn_implementation"),
        "data.dataset_name": data.get("dataset_name"),
        "data.split": data.get("split"),
        "git_commit": git_commit,
        "git_branch": git_branch,
        "python_version": sys.version,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "copied_files": copied_files,
        "missing_files": missing_files,
        "warnings": warnings,
    }


def write_json(path: Path, payload: dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] write json: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_report(path: Path, summary: dict[str, Any], dry_run: bool) -> None:
    lines = [
        "# 实验记录",
        "",
        "## 基本信息",
        "",
        f"- 创建时间：{summary.get('created_at')}",
        f"- run_name：{summary.get('run_name')}",
        f"- 归档目录：{summary.get('archive_dir')}",
        f"- Git branch：{summary.get('git_branch')}",
        f"- Git commit：{summary.get('git_commit')}",
        f"- Python：{summary.get('python_version')}",
        f"- Torch：{summary.get('torch_version')}",
        f"- CUDA 可用：{summary.get('cuda_available')}",
        f"- GPU：{summary.get('gpu_name')}",
        "",
        "## 实验配置",
        "",
        f"- 实验名称：{summary.get('experiment.name')}",
        f"- 阶段：{summary.get('experiment.stage')}",
        f"- 最大样本数：{summary.get('experiment.max_samples')}",
        f"- 模型路径：{summary.get('model.model_name')}",
        f"- max_new_tokens：{summary.get('model.max_new_tokens')}",
        f"- selected_layers：{summary.get('cache.selected_layers')}",
        f"- save_hidden_states：{summary.get('cache.save_hidden_states')}",
        f"- save_attentions：{summary.get('cache.save_attentions')}",
        f"- attn_implementation：{summary.get('cache.attn_implementation')}",
        f"- 数据集：{summary.get('data.dataset_name')}",
        f"- split：{summary.get('data.split')}",
        "",
        "## 结果文件",
        "",
    ]
    copied_files = summary.get("copied_files", [])
    lines.extend([f"- {path}" for path in copied_files] if copied_files else ["- 无"])
    lines.extend(["", "## 缺失文件", ""])
    missing_files = summary.get("missing_files", [])
    lines.extend([f"- {path}" for path in missing_files] if missing_files else ["- 无"])
    lines.extend(["", "## 注意事项", ""])
    warnings = summary.get("warnings", [])
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- 无"])
    lines.append("")

    if dry_run:
        print(f"[dry-run] write report: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    archive_root = Path(args.archive_root)
    run_name = sanitize_run_name(args.run_name)
    archive_dir = make_archive_dir(archive_root, run_name)
    warnings: list[str] = []

    if args.include_large:
        warnings.append("include_large=true：已请求复制 .npy 文件和完整 cache 目录，归档可能很大。")
    else:
        warnings.append("默认未复制大型 .npy 文件、cache/forward、cache/markers 和 attention/hidden-state 样本文件。")
    if args.dry_run:
        warnings.append("dry_run=true：本次只打印计划，不实际复制或写入归档文件。")

    cfg = load_yaml(config_path)
    copied_files, missing_files = collect_and_copy_files(
        config_path=config_path,
        cfg=cfg,
        archive_dir=archive_dir,
        include_large=args.include_large,
        dry_run=args.dry_run,
    )
    summary = build_summary(
        cfg=cfg,
        config_path=config_path,
        archive_dir=archive_dir,
        run_name=run_name,
        copied_files=copied_files,
        missing_files=missing_files,
        warnings=warnings,
    )

    summary_path = archive_dir / "run_summary.json"
    report_path = archive_dir / "run_report.md"
    write_json(summary_path, summary, args.dry_run)
    write_report(report_path, summary, args.dry_run)

    print(f"archive_dir: {archive_dir}")
    print(f"copied file count: {len(copied_files)}")
    print(f"missing file count: {len(missing_files)}")
    print(f"run_summary.json: {summary_path}")
    print(f"run_report.md: {report_path}")


if __name__ == "__main__":
    main()
