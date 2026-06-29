from __future__ import annotations

import ast
import asyncio
import importlib
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
_original_pipeline = sys.modules.get("eosin.backend.eosin_pipeline")
sys.modules["eosin.backend.eosin_pipeline"] = pipeline_stub
try:
    importlib.import_module("eosin.backend.bank_parser_api")
finally:
    if _original_pipeline is None:
        sys.modules.pop("eosin.backend.eosin_pipeline", None)
    else:
        sys.modules["eosin.backend.eosin_pipeline"] = _original_pipeline


class FakeService:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready

    def is_ready(self) -> bool:
        return self.ready


class FakeEvidenceService(FakeService):
    def extract_glm_page_html_bytes(self, filename: str, pdf_bytes: bytes) -> dict:
        from eosin.backend.bank_parser_api import validate_pdf_upload

        page_count = validate_pdf_upload(filename, pdf_bytes)
        return {
            "source_pdf": filename,
            "page_count": page_count,
            "pages": [],
            "timings": {},
            "ocr_metrics": {},
        }


def test_validate_pdf_upload_ignores_retired_byte_limit(monkeypatch) -> None:
    from eosin.backend.bank_parser_api import validate_pdf_upload

    monkeypatch.setenv("BANK_PARSER_MAX_UPLOAD_BYTES", "12")
    payload = _minimal_pdf(page_count=2)

    assert len(payload) > 12
    assert validate_pdf_upload("statement.pdf", payload) == 2


def test_validate_pdf_upload_rejects_non_pdf_signature() -> None:
    from eosin.backend.bank_parser_api import validate_pdf_upload

    with pytest.raises(HTTPException) as exc_info:
        validate_pdf_upload("statement.pdf", b"not a pdf")

    assert exc_info.value.status_code == 400
    assert "valid PDF" in exc_info.value.detail


def test_validate_pdf_upload_ignores_retired_page_limit(monkeypatch) -> None:
    from eosin.backend.bank_parser_api import validate_pdf_upload

    monkeypatch.setenv("BANK_PARSER_MAX_PAGES", "1")

    assert validate_pdf_upload("statement.pdf", _minimal_pdf(page_count=2)) == 2


def test_read_pdf_upload_uses_fixed_chunks_without_application_limit(monkeypatch) -> None:
    import asyncio
    from eosin.backend import bank_parser_api

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
    read_pdf_upload = getattr(bank_parser_api, "read_pdf_upload")

    filename, payload = asyncio.run(read_pdf_upload(upload))

    assert filename == "statement.pdf"
    assert payload == b"%PDF-1234567890"
    assert upload.read_sizes == [1024 * 1024, 1024 * 1024, 1024 * 1024]


def test_evidence_endpoint_accepts_pdf_above_retired_limits(monkeypatch) -> None:
    from eosin.backend import bank_parser_api

    async def run_inline(function):
        return function()

    monkeypatch.setenv("BANK_PARSER_MAX_UPLOAD_BYTES", "12")
    monkeypatch.setenv("BANK_PARSER_MAX_PAGES", "1")
    monkeypatch.setattr(bank_parser_api, "run_in_threadpool", run_inline)
    payload = _minimal_pdf(page_count=2)
    app = bank_parser_api.create_app(service=FakeEvidenceService())
    endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/v2/extract/bank-statement-evidence"
    )

    class AsyncUpload:
        filename = "statement.pdf"

        def __init__(self, content: bytes) -> None:
            self._content = content

        async def read(self, size: int = -1) -> bytes:
            if not self._content:
                return b""
            chunk = self._content[:size]
            self._content = self._content[size:]
            return chunk

    upload = AsyncUpload(payload)

    response = asyncio.run(endpoint(upload))

    assert response["page_count"] == 2


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
