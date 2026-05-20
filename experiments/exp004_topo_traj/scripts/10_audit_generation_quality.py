import argparse
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "experiments/exp004_topo_traj/data/generations.jsonl"
DEFAULT_OUTPUT_DIR = "experiments/exp004_topo_traj/outputs/audit_generation"
ANSWER_MARKER = "####"
STEP_RE = re.compile(r"Step\s+\d+\s*[:：]", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit generation quality and answer extraction for exp004_topo_traj."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input generations JSONL file.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for audit outputs.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL does not exist: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {exc}") from exc
    return records


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = text.replace(",", "").replace("$", "")
    text = re.sub(r"\s+", "", text)
    if not text:
        return ""

    match = NUMBER_RE.search(text)
    if match:
        text = match.group(0).replace(",", "")

    try:
        decimal = Decimal(text)
    except InvalidOperation:
        return text

    normalized = decimal.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.to_integral())
    return format(normalized, "f").rstrip("0").rstrip(".")


def answers_equal(left: Any, right: Any) -> bool:
    left_norm = normalize_answer(left)
    right_norm = normalize_answer(right)
    if not left_norm or not right_norm:
        return left_norm == right_norm

    try:
        return Decimal(left_norm) == Decimal(right_norm)
    except InvalidOperation:
        return left_norm == right_norm


def extract_last_number(text: str) -> str:
    matches = NUMBER_RE.findall(text or "")
    if not matches:
        return ""
    return normalize_answer(matches[-1])


def final_segment_after_last_marker(text: str) -> str:
    if not text or ANSWER_MARKER not in text:
        return ""
    return text.rsplit(ANSWER_MARKER, 1)[1]


def extract_answer(text: str) -> str:
    if not text:
        return ""
    if ANSWER_MARKER in text:
        segment = final_segment_after_last_marker(text)
        extracted = extract_last_number(segment)
        if extracted:
            return extracted
    return extract_last_number(text)


def step_count(text: str) -> int:
    if not text:
        return 0
    return len(STEP_RE.findall(text))


def tail(text: Any, n_chars: int = 800) -> str:
    if text is None:
        return ""
    return str(text)[-n_chars:]


def bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def make_bad_sample(
    record: dict[str, Any],
    extracted_answer_from_raw: str,
    extracted_answer_from_canonical: str,
    is_correct_recomputed: bool,
    raw_step_count: int,
    canonical_step_count: int,
    final_answer_raw_segment: str,
) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "question": record.get("question"),
        "gold_answer": record.get("gold_answer"),
        "pred_answer": record.get("pred_answer"),
        "extracted_answer_from_raw": extracted_answer_from_raw,
        "extracted_answer_from_canonical": extracted_answer_from_canonical,
        "is_correct": record.get("is_correct"),
        "is_correct_recomputed": is_correct_recomputed,
        "raw_step_count": raw_step_count,
        "canonical_step_count": canonical_step_count,
        "model_output_raw_tail": tail(record.get("model_output_raw")),
        "model_output_tail": tail(record.get("model_output")),
        "final_answer_raw_segment": final_answer_raw_segment,
    }


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    records = read_jsonl(input_path)

    total = len(records)
    raw_marker_count = 0
    canonical_marker_count = 0
    raw_step_positive_count = 0
    canonical_step_positive_count = 0
    raw_step_ge_3_count = 0
    canonical_step_ge_3_count = 0
    existing_correct_count = 0
    recomputed_raw_correct_count = 0
    recomputed_canonical_correct_count = 0
    num_pred_answer_mismatch = 0
    num_raw_correct_but_existing_wrong = 0
    num_existing_correct_but_raw_wrong = 0
    num_missing_raw_output = 0
    num_missing_model_output = 0
    retry_counts = Counter()
    retry_field_seen = False

    bad_no_raw_marker: list[dict[str, Any]] = []
    bad_step_less_than_3: list[dict[str, Any]] = []
    suspicious_extraction_mismatch: list[dict[str, Any]] = []

    for record in records:
        raw_output = record.get("model_output_raw") or ""
        canonical_output = record.get("model_output") or ""
        gold_answer = record.get("gold_answer")
        pred_answer = record.get("pred_answer")

        if not raw_output:
            num_missing_raw_output += 1
        if not canonical_output:
            num_missing_model_output += 1

        raw_has_marker = ANSWER_MARKER in raw_output
        canonical_has_marker = ANSWER_MARKER in canonical_output
        raw_step_count = step_count(raw_output)
        canonical_step_count = step_count(canonical_output)
        extracted_answer_from_raw = extract_answer(raw_output)
        extracted_answer_from_canonical = extract_answer(canonical_output)
        final_answer_raw_segment = record.get("final_answer_raw_segment")
        if final_answer_raw_segment is None:
            final_answer_raw_segment = final_segment_after_last_marker(raw_output)

        existing_is_correct = bool_or_none(record.get("is_correct"))
        if existing_is_correct is True:
            existing_correct_count += 1

        raw_correct = answers_equal(extracted_answer_from_raw, gold_answer)
        canonical_correct = answers_equal(extracted_answer_from_canonical, gold_answer)
        if raw_correct:
            recomputed_raw_correct_count += 1
        if canonical_correct:
            recomputed_canonical_correct_count += 1

        extraction_match_existing = answers_equal(extracted_answer_from_canonical, pred_answer)
        if not extraction_match_existing:
            num_pred_answer_mismatch += 1

        if raw_correct and existing_is_correct is False:
            num_raw_correct_but_existing_wrong += 1
        if existing_is_correct is True and not raw_correct:
            num_existing_correct_but_raw_wrong += 1

        if raw_has_marker:
            raw_marker_count += 1
        if canonical_has_marker:
            canonical_marker_count += 1
        if raw_step_count > 0:
            raw_step_positive_count += 1
        if canonical_step_count > 0:
            canonical_step_positive_count += 1
        if raw_step_count >= 3:
            raw_step_ge_3_count += 1
        if canonical_step_count >= 3:
            canonical_step_ge_3_count += 1

        if "retry_count" in record:
            retry_field_seen = True
            retry_counts[str(record.get("retry_count"))] += 1

        bad_sample = make_bad_sample(
            record=record,
            extracted_answer_from_raw=extracted_answer_from_raw,
            extracted_answer_from_canonical=extracted_answer_from_canonical,
            is_correct_recomputed=canonical_correct,
            raw_step_count=raw_step_count,
            canonical_step_count=canonical_step_count,
            final_answer_raw_segment=final_answer_raw_segment,
        )

        if not raw_has_marker:
            bad_no_raw_marker.append(bad_sample)
        if raw_step_count < 3 or canonical_step_count < 3:
            bad_step_less_than_3.append(bad_sample)
        if not extraction_match_existing:
            suspicious_extraction_mismatch.append(
                {
                    **bad_sample,
                    "extraction_match_existing": extraction_match_existing,
                    "gold_answer_normalized": normalize_answer(gold_answer),
                    "pred_answer_normalized": normalize_answer(pred_answer),
                }
            )

    summary = {
        "total": total,
        "raw_marker_count": raw_marker_count,
        "canonical_marker_count": canonical_marker_count,
        "raw_marker_rate": safe_rate(raw_marker_count, total),
        "canonical_marker_rate": safe_rate(canonical_marker_count, total),
        "raw_step_parse_rate": safe_rate(raw_step_positive_count, total),
        "canonical_step_parse_rate": safe_rate(canonical_step_positive_count, total),
        "raw_num_steps_ge_3_rate": safe_rate(raw_step_ge_3_count, total),
        "canonical_num_steps_ge_3_rate": safe_rate(canonical_step_ge_3_count, total),
        "existing_accuracy": safe_rate(existing_correct_count, total),
        "recomputed_accuracy_from_raw": safe_rate(recomputed_raw_correct_count, total),
        "recomputed_accuracy_from_canonical": safe_rate(recomputed_canonical_correct_count, total),
        "num_pred_answer_mismatch": num_pred_answer_mismatch,
        "num_raw_correct_but_existing_wrong": num_raw_correct_but_existing_wrong,
        "num_existing_correct_but_raw_wrong": num_existing_correct_but_raw_wrong,
        "num_missing_raw_output": num_missing_raw_output,
        "num_missing_model_output": num_missing_model_output,
        "retry_count_distribution": dict(sorted(retry_counts.items())) if retry_field_seen else {},
    }

    write_json(output_dir / "generation_quality_summary.json", summary)
    write_jsonl(output_dir / "bad_no_raw_marker.jsonl", bad_no_raw_marker)
    write_jsonl(output_dir / "bad_step_less_than_3.jsonl", bad_step_less_than_3)
    write_jsonl(output_dir / "suspicious_extraction_mismatch.jsonl", suspicious_extraction_mismatch)

    print("Generation quality audit summary")
    print(f"input: {input_path}")
    print(f"total: {total}")
    print(f"raw marker rate: {summary['raw_marker_rate']:.4f}")
    print(f"canonical marker rate: {summary['canonical_marker_rate']:.4f}")
    print(f"raw step parse rate: {summary['raw_step_parse_rate']:.4f}")
    print(f"canonical step parse rate: {summary['canonical_step_parse_rate']:.4f}")
    print(f"existing accuracy: {summary['existing_accuracy']:.4f}")
    print(f"recomputed raw accuracy: {summary['recomputed_accuracy_from_raw']:.4f}")
    print(f"recomputed canonical accuracy: {summary['recomputed_accuracy_from_canonical']:.4f}")
    print(f"pred answer mismatches: {num_pred_answer_mismatch}")
    print(f"output dir: {output_dir}")


if __name__ == "__main__":
    main()
