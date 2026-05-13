from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import pandas as pd
import requests


DEFAULT_ENDPOINT = "https://noelalex-samuel2023--bank-parser.modal.run/parse/bank-statement"
DEFAULT_PDF_DIR = Path("test_pdfs")
DEFAULT_GROUND_TRUTH_DIR = Path("parsed test")
DEFAULT_OUTPUT_DIR = Path("quality-compare-results")

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
DATE_SEPARATORS_RE = re.compile(r"[-./\s]+")


@dataclass(frozen=True)
class ComparisonResult:
    stem: str
    ok: bool
    status_code: int | None
    latency_seconds: float | None
    ground_truth_rows: int
    parsed_rows: int
    matched_rows: int
    recall: float
    precision: float
    f1: float
    missing_rows: int
    extra_rows: int
    missing_columns: list[str]
    mapped_columns: dict[str, str]
    response_path: str | None
    report_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare bank-parser JSON responses against CSV ground truth for test PDFs."
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Parser /parse/bank-statement endpoint.")
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR), help="Directory containing test PDFs.")
    parser.add_argument("--ground-truth-dir", default=str(DEFAULT_GROUND_TRUTH_DIR), help="Directory containing expected CSVs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for JSON responses and reports.")
    parser.add_argument("--only", nargs="*", default=[], help="Optional PDF stems to compare, e.g. axis-1 sbi-1.")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-PDF HTTP timeout in seconds.")
    parser.add_argument("--reuse-responses", action="store_true", help="Reuse existing response JSON files if present.")
    parser.add_argument("--fail-under-f1", type=float, default=0.98, help="Exit non-zero if any comparable PDF is below this F1.")
    return parser.parse_args()


def normalize_header(value: object) -> str:
    return NON_ALNUM_RE.sub("", str(value or "").lower())


def header_kind(column: str) -> str:
    normalized = normalize_header(column)
    if "date" in normalized:
        return "date"
    if any(token in normalized for token in ("debit", "credit", "balance", "withdraw", "deposit")):
        return "money"
    if any(token in normalized for token in ("description", "particular", "narration")):
        return "text"
    return "generic"


def normalize_money(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    sign = "-"
    if "(dr)" in text.lower() or text.startswith("-"):
        prefix = sign
    else:
        prefix = ""
    text = re.sub(r"\((cr|dr)\)", "", text, flags=re.IGNORECASE)
    text = text.replace(",", "").replace("₹", "").strip()
    text = re.sub(r"[^0-9.]", "", text)
    if not text:
        return ""
    try:
        amount = Decimal(text).quantize(Decimal("0.01")).normalize()
    except InvalidOperation:
        return prefix + text
    normalized = format(amount, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return prefix + (normalized or "0")


def normalize_date(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parenthesized = re.search(r"\(([^)]+)\)", text)
    candidate = parenthesized.group(1) if parenthesized else text
    candidate = candidate.replace("Sept", "Sep")
    parsed = pd.to_datetime(candidate, dayfirst=True, errors="coerce")
    if not pd.isna(parsed):
        return parsed.strftime("%Y%m%d")
    return DATE_SEPARATORS_RE.sub("", candidate.lower())


def normalize_text(value: object) -> str:
    return NON_ALNUM_RE.sub("", str(value or "").lower())


def normalize_cell(value: object, column: str) -> str:
    kind = header_kind(column)
    if kind == "date":
        return normalize_date(value)
    if kind == "money":
        return normalize_money(value)
    return normalize_text(value)


def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype=str).fillna("")


def load_response_rows(response_path: Path) -> pd.DataFrame:
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    columns = payload.get("columns", [])
    if not isinstance(rows, list):
        return pd.DataFrame()
    if isinstance(columns, list) and columns:
        return pd.DataFrame(rows, columns=columns).fillna("")
    return pd.DataFrame(rows).fillna("")


def call_parser(endpoint: str, pdf_path: Path, response_path: Path, timeout: int) -> tuple[int, float]:
    started = time.perf_counter()
    with pdf_path.open("rb") as handle:
        response = requests.post(
            endpoint,
            files={"file": (pdf_path.name, handle, "application/pdf")},
            timeout=timeout,
        )
    latency = time.perf_counter() - started
    response_path.write_text(json.dumps(response.json(), indent=2, ensure_ascii=True), encoding="utf-8")
    return response.status_code, latency


def best_column_mapping(expected_columns: list[str], parsed_columns: list[str]) -> tuple[dict[str, str], list[str]]:
    parsed_by_norm = {normalize_header(column): column for column in parsed_columns}
    parsed_norms = list(parsed_by_norm)
    mapping: dict[str, str] = {}
    missing: list[str] = []

    for expected in expected_columns:
        expected_norm = normalize_header(expected)
        if expected_norm in parsed_by_norm:
            mapping[expected] = parsed_by_norm[expected_norm]
            continue
        close = get_close_matches(expected_norm, parsed_norms, n=1, cutoff=0.78)
        if close:
            mapping[expected] = parsed_by_norm[close[0]]
        else:
            missing.append(expected)

    return mapping, missing


def normalized_row_counter(df: pd.DataFrame, column_mapping: dict[str, str]) -> Counter[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for _, row in df.iterrows():
        normalized = []
        for expected_column, actual_column in column_mapping.items():
            normalized.append(normalize_cell(row.get(actual_column, ""), expected_column))
        rows.append(tuple(normalized))
    return Counter(rows)


def sample_counter(counter: Counter[tuple[str, ...]], columns: list[str], limit: int = 10) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for row_tuple, count in counter.most_common(limit):
        sample = {column: value for column, value in zip(columns, row_tuple)}
        sample["_count"] = str(count)
        samples.append(sample)
    return samples


def compare_one(
    *,
    stem: str,
    pdf_path: Path,
    csv_path: Path,
    endpoint: str,
    output_dir: Path,
    timeout: int,
    reuse_responses: bool,
) -> ComparisonResult:
    response_path = output_dir / "responses" / f"{stem}.json"
    report_path = output_dir / "reports" / f"{stem}.json"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    status_code: int | None = None
    latency_seconds: float | None = None
    if not (reuse_responses and response_path.exists()):
        status_code, latency_seconds = call_parser(endpoint, pdf_path, response_path, timeout)

    expected = load_ground_truth(csv_path)
    parsed = load_response_rows(response_path)
    mapping, missing_columns = best_column_mapping(list(expected.columns), list(parsed.columns))

    expected_for_compare = expected.rename(columns={column: column for column in mapping})
    parsed_for_compare = parsed.rename(columns={actual: expected_col for expected_col, actual in mapping.items()})
    expected_counter = normalized_row_counter(expected_for_compare, {column: column for column in mapping})
    parsed_counter = normalized_row_counter(parsed_for_compare, {column: column for column in mapping})

    matched = sum((expected_counter & parsed_counter).values())
    expected_rows = len(expected)
    parsed_rows = len(parsed)
    missing_counter = expected_counter - parsed_counter
    extra_counter = parsed_counter - expected_counter
    recall = matched / expected_rows if expected_rows else 1.0
    precision = matched / parsed_rows if parsed_rows else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    ok = not missing_columns and f1 >= 0.98

    report: dict[str, Any] = {
        "stem": stem,
        "status_code": status_code,
        "latency_seconds": round(latency_seconds, 6) if latency_seconds is not None else None,
        "ground_truth_csv": str(csv_path),
        "pdf": str(pdf_path),
        "response_json": str(response_path),
        "ground_truth_rows": expected_rows,
        "parsed_rows": parsed_rows,
        "matched_rows": matched,
        "recall": round(recall, 6),
        "precision": round(precision, 6),
        "f1": round(f1, 6),
        "missing_rows": sum(missing_counter.values()),
        "extra_rows": sum(extra_counter.values()),
        "missing_columns": missing_columns,
        "mapped_columns": mapping,
        "missing_samples": sample_counter(missing_counter, list(mapping), limit=12),
        "extra_samples": sample_counter(extra_counter, list(mapping), limit=12),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    return ComparisonResult(
        stem=stem,
        ok=ok,
        status_code=status_code,
        latency_seconds=latency_seconds,
        ground_truth_rows=expected_rows,
        parsed_rows=parsed_rows,
        matched_rows=matched,
        recall=recall,
        precision=precision,
        f1=f1,
        missing_rows=sum(missing_counter.values()),
        extra_rows=sum(extra_counter.values()),
        missing_columns=missing_columns,
        mapped_columns=mapping,
        response_path=str(response_path),
        report_path=str(report_path),
    )


def main() -> int:
    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    ground_truth_dir = Path(args.ground_truth_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    only = set(args.only)
    csv_paths = sorted(ground_truth_dir.glob("*.csv"))
    results: list[ComparisonResult] = []
    for csv_path in csv_paths:
        stem = csv_path.stem
        if only and stem not in only:
            continue
        pdf_path = pdf_dir / f"{stem}.pdf"
        if not pdf_path.exists():
            print(f"SKIP {stem}: missing {pdf_path}", file=sys.stderr)
            continue
        print(f"Comparing {stem}...", flush=True)
        result = compare_one(
            stem=stem,
            pdf_path=pdf_path,
            csv_path=csv_path,
            endpoint=args.endpoint,
            output_dir=output_dir,
            timeout=args.timeout,
            reuse_responses=args.reuse_responses,
        )
        results.append(result)
        print(
            f"{stem}: f1={result.f1:.3f} recall={result.recall:.3f} precision={result.precision:.3f} "
            f"rows expected={result.ground_truth_rows} parsed={result.parsed_rows} "
            f"missing={result.missing_rows} extra={result.extra_rows} report={result.report_path}",
            flush=True,
        )

    summary = {
        "endpoint": args.endpoint,
        "count": len(results),
        "fail_under_f1": args.fail_under_f1,
        "results": [result.__dict__ for result in results],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    failed = [
        result
        for result in results
        if result.f1 < args.fail_under_f1 or result.missing_columns
    ]
    if failed:
        print("FAILED:", ", ".join(result.stem for result in failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
