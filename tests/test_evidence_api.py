from __future__ import annotations

from eosin.backend.bank_parser_api import evidence_payload_from_result
from eosin.backend.bank_parser_service import BankParserResult


def test_evidence_payload_from_result_returns_page_raw_html() -> None:
    payload = evidence_payload_from_result(
        BankParserResult(
            source_pdf="statement.pdf",
            page_count=1,
            pages_with_tables=[1],
            columns=["Date"],
            rows=[{"Date": "01/01/2024"}],
            timings={"ocr_pages": 1.5},
            debug={
                "page_ocr": [
                    {
                        "page_index": 0,
                        "raw_html": "<table><tr><td>Date</td></tr></table>",
                        "raw_headers": ["Date"],
                        "quality_score": 100,
                        "suspicious": False,
                        "reasons": [],
                        "row_count": 1,
                    }
                ]
            },
        )
    )

    assert payload["source_pdf"] == "statement.pdf"
    assert payload["page_count"] == 1
    assert payload["pages_with_tables"] == [1]
    assert payload["pages"][0]["page_number"] == 1
    assert payload["pages"][0]["raw_html"] == "<table><tr><td>Date</td></tr></table>"
    assert payload["pages"][0]["headers"] == ["Date"]
