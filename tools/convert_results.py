from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JSON result files under outputs/ into CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing JSON files. Defaults to outputs/.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern used to find JSON files recursively. Defaults to *.json.",
    )
    return parser.parse_args()


def load_json_payload(file_path: Path) -> Any:
    raw_text = file_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError("File is empty.")

    payload = json.loads(raw_text)

    # Some result files may store a JSON document as a JSON string.
    while isinstance(payload, str):
        payload = json.loads(payload)

    return payload


def normalize_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def flatten_record(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in record.items():
        column_name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_record(value, prefix=column_name))
        else:
            flattened[column_name] = normalize_value(value)
    return flattened


def to_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return [flatten_record(payload)]

    if isinstance(payload, list):
        if not payload:
            return []

        if all(isinstance(item, dict) for item in payload):
            return [flatten_record(item) for item in payload]

        return [{"value": normalize_value(item)} for item in payload]

    return [{"value": normalize_value(payload)}]


def collect_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def build_output_path(source_path: Path) -> Path:
    candidate = source_path.with_name(f"{source_path.stem}_json2csv.csv")
    suffix_index = 1
    while candidate.exists():
        candidate = source_path.with_name(f"{source_path.stem}_json2csv_{suffix_index}.csv")
        suffix_index += 1
    return candidate


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = collect_fieldnames(rows)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def convert_file(file_path: Path) -> Path:
    payload = load_json_payload(file_path)
    rows = to_rows(payload)
    output_path = build_output_path(file_path)
    write_csv(rows, output_path)
    return output_path


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    json_files = sorted(path for path in input_dir.rglob(args.pattern) if path.is_file())
    if not json_files:
        print(f"No files matched {args.pattern!r} under {input_dir}.")
        return 0

    converted_count = 0
    skipped_count = 0

    for json_file in json_files:
        try:
            output_path = convert_file(json_file)
            converted_count += 1
            print(f"Converted: {json_file} -> {output_path}")
        except Exception as exc:
            skipped_count += 1
            print(f"Skipped: {json_file} ({exc})")

    print(
        f"Finished. Converted {converted_count} file(s), skipped {skipped_count} file(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
