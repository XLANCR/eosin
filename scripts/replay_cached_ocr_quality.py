from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_parser_to_ground_truth import normalize_date, normalize_header, normalize_money


ReconstructPageOCR = Callable[[list[dict[str, object]]], tuple[pd.DataFrame, dict[str, object]]]


def _pick_column(columns: list[object], kind: str) -> object | None:
    normalized = [(column, normalize_header(column)) for column in columns]
    if kind == "date":
        transaction_dates = [
            column
            for column, header in normalized
            if "date" in header and "value" not in header
        ]
        if transaction_dates:
            return transaction_dates[0]
        return next((column for column, header in normalized if "date" in header), None)
    if kind == "balance":
        return next((column for column, header in normalized if "balance" in header), None)
    raise ValueError(f"unknown column kind: {kind}")


def _amount_columns(columns: list[object]) -> list[object]:
    result: list[object] = []
    for column in columns:
        header = normalize_header(column)
        if "balance" in header:
            continue
        if any(token in header for token in ("debit", "credit", "withdraw", "deposit", "amount")):
            result.append(column)
    return result


def _financial_key_counter(frame: pd.DataFrame) -> Counter[tuple[str, str, str]]:
    date_column = _pick_column(list(frame.columns), "date")
    balance_column = _pick_column(list(frame.columns), "balance")
    amount_columns = _amount_columns(list(frame.columns))
    if date_column is None or balance_column is None or not amount_columns:
        return Counter()

    keys: list[tuple[str, str, str]] = []
    for _, row in frame.fillna("").iterrows():
        amounts = [normalize_money(row.get(column, "")).lstrip("-") for column in amount_columns]
        amounts = [value for value in amounts if value and value != "0"]
        keys.append(
            (
                normalize_date(row.get(date_column, "")),
                amounts[0] if amounts else "",
                normalize_money(row.get(balance_column, "")).lstrip("-"),
            )
        )
    return Counter(keys)


def _default_reconstructor() -> ReconstructPageOCR:
    from eosin.backend.eosin_pipeline import BankStatementParser

    parser = BankStatementParser.__new__(BankStatementParser)

    def reconstruct(page_ocr: list[dict[str, object]]) -> tuple[pd.DataFrame, dict[str, object]]:
        with contextlib.redirect_stdout(io.StringIO()):
            expected_headers: list[str] | None = None
            evaluations: list[dict[str, object]] = []
            for item in page_ocr:
                evaluation = parser._evaluate_page_ocr_result(
                    page_idx=int(item["page_index"]),
                    html_content=str(item.get("raw_html", "")),
                    expected_headers=expected_headers,
                    pass_label="cached_replay",
                )
                if expected_headers is None and evaluation.get("raw_headers"):
                    expected_headers = list(evaluation["raw_headers"])
                evaluations.append(evaluation)
            frames, headers, _ = parser._materialize_selected_tables(evaluations)
        if not frames:
            return pd.DataFrame(), {
                "selected_source_rows": 0,
                "emitted_transaction_sources": 0,
                "absorbed_continuation_sources": 0,
                "excluded_non_transaction_sources": 0,
                "rejected_unclassified_sources": 0,
                "conservation_ok": True,
            }
        with contextlib.redirect_stdout(io.StringIO()):
            dataframe = parser._finalize_extracted_tables(frames, headers)
        return dataframe, dict(parser._last_transaction_reconstruction)

    return reconstruct


def evaluate_cached_responses(
    *,
    responses_dir: Path,
    corpus_dir: Path,
    reconstruct_page_ocr: ReconstructPageOCR | None = None,
    output_path: Path | None = None,
    baseline: dict[str, object] | None = None,
) -> dict[str, object]:
    reconstruct = reconstruct_page_ocr or _default_reconstructor()
    documents: list[dict[str, object]] = []
    total_ground_truth = 0
    total_parsed = 0
    total_matched = 0
    conservation_failures: list[str] = []

    for response_path in sorted(responses_dir.glob("*.json")):
        response = json.loads(response_path.read_text(encoding="utf-8"))
        request = response.get("request", {})
        payload = response.get("payload", {})
        relative_pdf = Path(str(request.get("pdf_path", "")))
        if not relative_pdf.name:
            continue
        ground_truth_path = corpus_dir / relative_pdf.with_suffix(".csv")
        if not ground_truth_path.exists():
            continue
        page_ocr = payload.get("debug", {}).get("page_ocr", [])
        if not isinstance(page_ocr, list):
            page_ocr = []
        parsed, diagnostics = reconstruct(page_ocr)
        expected = pd.read_csv(ground_truth_path, dtype=str).fillna("")
        parsed_counter = _financial_key_counter(parsed)
        expected_counter = _financial_key_counter(expected)
        matched = sum((parsed_counter & expected_counter).values())
        expected_rows = len(expected)
        parsed_rows = len(parsed)
        recall = matched / expected_rows if expected_rows else 1.0
        precision = matched / parsed_rows if parsed_rows else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if not diagnostics.get("conservation_ok", False):
            conservation_failures.append(str(relative_pdf))
        documents.append(
            {
                "pdf_path": str(relative_pdf),
                "ground_truth_rows": expected_rows,
                "parsed_rows": parsed_rows,
                "matched_financial_rows": matched,
                "financial_recall": round(recall, 6),
                "financial_precision": round(precision, 6),
                "financial_f1": round(f1, 6),
                "reconstruction": diagnostics,
            }
        )
        total_ground_truth += expected_rows
        total_parsed += parsed_rows
        total_matched += matched

    micro_recall = total_matched / total_ground_truth if total_ground_truth else 1.0
    micro_precision = total_matched / total_parsed if total_parsed else 1.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    regressions: list[str] = []
    if baseline and "micro_financial_f1" in baseline:
        baseline_f1 = float(baseline["micro_financial_f1"])
        if micro_f1 < baseline_f1:
            regressions.append(f"micro_financial_f1: {micro_f1:.6f} < baseline {baseline_f1:.6f}")

    summary: dict[str, object] = {
        "document_count": len(documents),
        "ground_truth_rows_total": total_ground_truth,
        "parsed_rows_total": total_parsed,
        "matched_financial_rows_total": total_matched,
        "micro_financial_recall": round(micro_recall, 6),
        "micro_financial_precision": round(micro_precision, 6),
        "micro_financial_f1": round(micro_f1, 6),
        "conservation_failures": conservation_failures,
        "regressions": regressions,
        "documents": documents,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay captured Modal OCR through the current transaction finalizer.")
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    summary = evaluate_cached_responses(
        responses_dir=args.responses_dir,
        corpus_dir=args.corpus_dir,
        output_path=args.output,
        baseline=baseline,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "documents"}, indent=2))
    if args.fail_on_regression and (summary["regressions"] or summary["conservation_failures"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
