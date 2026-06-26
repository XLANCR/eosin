from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

pipeline_stub = types.ModuleType("eosin.backend.eosin_pipeline")
pipeline_stub.SAVE_DEBUG_IMAGES = False
pipeline_stub.PARSE_TESTING = False
pipeline_stub.ENABLE_OCR_BATCHING = False
pipeline_stub.OCR_BATCH_SIZE = 1
pipeline_stub.ENABLE_PAGE_OCR_RETRY = False
pipeline_stub.CAPTURE_RAW_OCR_DEBUG = False


class BankStatementParser:
    pass


pipeline_stub.BankStatementParser = BankStatementParser
sys.modules.setdefault("eosin.backend.eosin_pipeline", pipeline_stub)


class FakeService:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready

    def is_ready(self) -> bool:
        return self.ready


def test_validate_pdf_upload_rejects_upload_larger_than_limit(monkeypatch) -> None:
    from eosin.backend.bank_parser_api import validate_pdf_upload

    monkeypatch.setenv("BANK_PARSER_MAX_UPLOAD_BYTES", "12")

    with pytest.raises(HTTPException) as exc_info:
        validate_pdf_upload("statement.pdf", b"%PDF-" + (b"x" * 20))

    assert exc_info.value.status_code == 413


def test_validate_pdf_upload_rejects_non_pdf_signature() -> None:
    from eosin.backend.bank_parser_api import validate_pdf_upload

    with pytest.raises(HTTPException) as exc_info:
        validate_pdf_upload("statement.pdf", b"not a pdf")

    assert exc_info.value.status_code == 400
    assert "valid PDF" in exc_info.value.detail


def test_validate_pdf_upload_rejects_pdf_over_page_limit(monkeypatch) -> None:
    from eosin.backend.bank_parser_api import validate_pdf_upload

    monkeypatch.setenv("BANK_PARSER_MAX_PAGES", "1")

    with pytest.raises(HTTPException) as exc_info:
        validate_pdf_upload("statement.pdf", _minimal_pdf(page_count=2))

    assert exc_info.value.status_code == 413


def test_read_limited_pdf_upload_rejects_before_unbounded_read(monkeypatch) -> None:
    import asyncio
    from eosin.backend.bank_parser_api import read_limited_pdf_upload

    class ChunkedUpload:
        filename = "statement.pdf"

        def __init__(self) -> None:
            self.read_sizes: list[int] = []
            self._chunks = [b"%PDF-12345", b"67890"]

        async def read(self, size: int = -1) -> bytes:
            self.read_sizes.append(size)
            if not self._chunks:
                return b""
            return self._chunks.pop(0)

    monkeypatch.setenv("BANK_PARSER_MAX_UPLOAD_BYTES", "12")
    upload = ChunkedUpload()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(read_limited_pdf_upload(upload))

    assert exc_info.value.status_code == 413
    assert upload.read_sizes == [13, 13]


def test_timeout_error_maps_to_503() -> None:
    from eosin.backend.bank_parser_api import http_exception_for_timeout

    exc = http_exception_for_timeout(TimeoutError("parser pool saturated"))

    assert exc.status_code == 503
    assert "parser pool saturated" in exc.detail


def test_evidence_endpoint_tracks_metrics_lifecycle() -> None:
    tree = ast.parse(Path("eosin/backend/bank_parser_api.py").read_text())
    create_app_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_app"
    )
    method = next(
        node
        for node in create_app_node.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "extract_bank_statement_evidence"
    )
    called_attributes = {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert {
        "track_request_started",
        "track_request_success",
        "track_request_failure",
        "track_request_finished",
        "schedule_push",
    }.issubset(called_attributes)


def test_evidence_top_level_ocr_metrics_are_observed(monkeypatch) -> None:
    from eosin.backend.metrics import MetricsManager

    manager = MetricsManager()
    observed: list[dict] = []
    monkeypatch.setattr(manager, "_observe_ocr_metrics", observed.append)

    manager.track_request_success({"page_count": 1, "pages": [], "ocr_metrics": {"task_count": 2.0}})

    assert observed == [{"task_count": 2.0}]


def test_ready_payload_reports_service_unavailable() -> None:
    from eosin.backend.bank_parser_api import readiness_payload

    with pytest.raises(HTTPException) as exc_info:
        readiness_payload(FakeService(ready=False))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["status"] == "unavailable"


def _minimal_pdf(*, page_count: int = 1) -> bytes:
    try:
        import fitz
    except Exception:
        return b"%PDF-1.4\n%%EOF\n"

    doc = fitz.open()
    for _ in range(page_count):
        doc.new_page(width=72, height=72)
    payload = doc.tobytes()
    doc.close()
    return payload
