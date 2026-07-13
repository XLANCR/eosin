from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.materialize_live_ocr_fixture import materialize_fixture


def test_materializes_warmed_live_artifacts_with_provenance(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    artifacts = tmp_path / "artifacts"
    output = tmp_path / "fixture"
    (corpus / "Bank").mkdir(parents=True)
    artifacts.mkdir()
    pdf = corpus / "Bank" / "sample.pdf"
    pdf.write_bytes(b"%PDF-sample")
    response = {
        "request": {"pdf_path": "Bank/sample.pdf"},
        "payload": {
            "direct_ocr_response": {
                "page_count": 2,
                "pages": [
                    {"page_number": 1, "raw_html": "one", "suspicious": False},
                    {"page_number": 2, "raw_html": "two", "suspicious": True},
                ],
                "timings": {"service_total": 3.0},
                "ocr_metrics": {"retry_count": 1},
            }
        },
    }
    (artifacts / "0001.json").write_text(json.dumps(response), encoding="utf-8")

    manifest = materialize_fixture(
        artifacts, output, corpus_root=corpus, expected_documents=1
    )

    assert manifest["document_count"] == 1
    assert manifest["total_pages"] == 2
    assert json.loads((output / "sample" / "glm_html_cache.json").read_text()) == {
        "1": "one",
        "2": "two",
    }
    metadata = json.loads(
        (output / "sample" / "live_ocr_metadata.json").read_text()
    )
    assert metadata["content_sha256"]
    assert metadata["artifact_sha256"]
    assert metadata["suspicious_page_count"] == 1
    assert metadata["timings"] == {"service_total": 3.0}


def test_rejects_incomplete_or_overwriting_fixture(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    with pytest.raises(ValueError, match="expected 55"):
        materialize_fixture(artifacts, tmp_path / "fixture", expected_documents=55)
