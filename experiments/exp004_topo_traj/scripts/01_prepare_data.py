import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small GSM8K sample for exp004_topo_traj.")
    parser.add_argument("--config", required=True, help="Path to topo_traj_v0.yaml.")
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_answer(answer: str) -> str:
    return answer.replace(",", "").replace("$", "").strip()


def extract_gold_answer(raw_answer: str, marker: str = "####") -> str:
    if marker in raw_answer:
        return normalize_answer(raw_answer.rsplit(marker, 1)[-1])
    numbers = re.findall(r"-?\d+(?:\.\d+)?", raw_answer.replace(",", ""))
    return normalize_answer(numbers[-1]) if numbers else ""


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config["experiment"].get("seed", 42))
    max_samples = int(config["experiment"].get("max_samples", 30))
    dataset_name = config["data"].get("dataset_name", "gsm8k")
    dataset_config = config["data"].get("dataset_config", "main")
    split = config["data"].get("split", "test")
    sample_file = Path(config["paths"]["sample_file"])
    answer_marker = config.get("evaluation", {}).get("answer_marker", "####")

    random.seed(seed)
    sample_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset(dataset_name, dataset_config)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download/load dataset {dataset_name!r} with config {dataset_config!r}. "
            "Check network access and the datasets installation."
        ) from exc

    if split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Split {split!r} not found in dataset. Available splits: {available}")

    split_data = dataset[split]
    indices = list(range(len(split_data)))
    random.shuffle(indices)
    selected_indices = indices[:max_samples]

    records = []
    for output_idx, dataset_idx in enumerate(selected_indices):
        item = split_data[dataset_idx]
        raw_answer = item["answer"]
        records.append(
            {
                "id": f"gsm8k_{split}_{output_idx:06d}",
                "question": item["question"],
                "raw_answer": raw_answer,
                "gold_answer": extract_gold_answer(raw_answer, marker=answer_marker),
            }
        )

    with open(sample_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved to: {sample_file}")
    print(f"Sample count: {len(records)}")
    print("Preview:")
    for record in records[:2]:
        print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
