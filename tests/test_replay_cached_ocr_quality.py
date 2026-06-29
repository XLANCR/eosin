from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from scripts.replay_cached_ocr_quality import evaluate_cached_responses


ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_cached_responses_reports_financial_metrics(tmp_path: Path) -> None:
    responses = tmp_path / "responses"
    corpus = tmp_path / "corpus"
    output = tmp_path / "summary.json"
    responses.mkdir()
    (corpus / "Bank").mkdir(parents=True)
    (corpus / "Bank" / "sample.csv").write_text(
        "Date,Description,Debit,Credit,Balance\n"
        "01/01/2024,Payment,10.00,,90.00\n",
        encoding="utf-8",
    )
    response = {
        "request": {"pdf_path": "Bank/sample.pdf"},
        "payload": {
            "debug": {
                "page_ocr": [
                    {
                        "page_index": 0,
                        "raw_html": "<table><tr><td>cached</td></tr></table>",
                    }
                ]
            }
        },
    }
    (responses / "0001__sample.json").write_text(json.dumps(response), encoding="utf-8")

    def reconstruct(_page_ocr: list[dict[str, object]]) -> tuple[pd.DataFrame, dict[str, object]]:
        return (
            pd.DataFrame(
                [
                    {
                        "Date": "01/01/2024",
                        "Description": "Payment",
                        "Debit": "10.00",
                        "Credit": "",
                        "Balance": "90.00",
                    }
                ]
            ),
            {
                "selected_source_rows": 1,
                "emitted_transaction_sources": 1,
                "absorbed_continuation_sources": 0,
                "excluded_non_transaction_sources": 0,
                "rejected_unclassified_sources": 0,
                "conservation_ok": True,
            },
        )

    summary = evaluate_cached_responses(
        responses_dir=responses,
        corpus_dir=corpus,
        reconstruct_page_ocr=reconstruct,
        output_path=output,
    )

    assert summary["document_count"] == 1
    assert summary["micro_financial_f1"] == 1.0
    assert summary["conservation_failures"] == []
    assert summary["documents"][0]["parsed_rows"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["micro_financial_f1"] == 1.0


def test_evaluate_cached_responses_detects_baseline_regression(tmp_path: Path) -> None:
    responses = tmp_path / "responses"
    corpus = tmp_path / "corpus"
    responses.mkdir()
    (corpus / "Bank").mkdir(parents=True)
    (corpus / "Bank" / "sample.csv").write_text(
        "Date,Description,Debit,Credit,Balance\n01/01/2024,Payment,10.00,,90.00\n",
        encoding="utf-8",
    )
    (responses / "sample.json").write_text(
        json.dumps(
            {
                "request": {"pdf_path": "Bank/sample.pdf"},
                "payload": {"debug": {"page_ocr": [{"page_index": 0, "raw_html": "<table></table>"}]}},
            }
        ),
        encoding="utf-8",
    )

    summary = evaluate_cached_responses(
        responses_dir=responses,
        corpus_dir=corpus,
        reconstruct_page_ocr=lambda _: (
            pd.DataFrame(columns=["Date", "Description", "Debit", "Credit", "Balance"]),
            {"selected_source_rows": 0, "conservation_ok": True},
        ),
        baseline={"micro_financial_f1": 0.5},
    )

    assert summary["regressions"] == ["micro_financial_f1: 0.000000 < baseline 0.500000"]


def test_replay_cli_can_be_invoked_by_file_path() -> None:
    completed = subprocess.run(
        ["bash", "-lc", "python scripts/replay_cached_ocr_quality.py --help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--responses-dir" in completed.stdout
