from __future__ import annotations

import ast
import threading
import time
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from eosin.backend.bank_parser_service import BankParserService
from eosin.backend.bank_parser_api import evidence_payload_from_result


class FakeParser:
    def __init__(self, parser_id: int, barrier: threading.Barrier) -> None:
        self.parser_id = parser_id
        self.barrier = barrier
        self.layout_guard = None
        self.last_run_stats: dict[str, object] = {}
        self.closed = False

    def parse_pdf(self, pdf_path: str) -> pd.DataFrame:
        self.barrier.wait(timeout=2)
        time.sleep(0.01)
        self.last_run_stats = {
            "pages_with_tables": [self.parser_id],
            "timings": {"ocr_tables": float(self.parser_id)},
            "page_ocr": [
                {
                    "page_index": 0,
                    "raw_html": f"<table><tr><td>{Path(pdf_path).name}</td></tr></table>",
                }
            ],
        }
        return pd.DataFrame([{"parser_id": self.parser_id, "source": Path(pdf_path).name}])

    def close(self) -> None:
        self.closed = True


class FakeDirectOCRParser(FakeParser):
    pdf_dpi = 200
    last_task_type = None

    def _render_specific_pages(self, _pdf_path: str, page_indices, _dpi: int):
        return {page_idx: Image.new("RGB", (8, 8), "white") for page_idx in page_indices}

    def _normalize_ocr_image(self, image: Image.Image) -> Image.Image:
        return image

    def _ocr_tables_parallel(self, ocr_images, *, task_type="table"):
        self.last_task_type = task_type
        return [
            (page_idx, f"<table><tr><td>page-{page_idx + 1}</td></tr></table>")
            for page_idx, _ in ocr_images
        ], {
            "task_count": float(len(ocr_images)),
            "success_count": float(len(ocr_images)),
        }

    def _evaluate_page_ocr_result(
        self,
        *,
        page_idx: int,
        html_content: str,
        expected_headers,
        pass_label: str,
    ):
        assert html_content
        assert expected_headers is None
        assert pass_label == "evidence"
        return {
            "quality_score": 91 - page_idx,
            "suspicious": page_idx > 0,
            "reasons": ["low_quality"] if page_idx > 0 else [],
        }


class CriticalRetryParser(FakeDirectOCRParser):
    def _evaluate_page_ocr_result(self, *, page_idx, html_content, expected_headers, pass_label):
        result = super()._evaluate_page_ocr_result(
            page_idx=page_idx,
            html_content=html_content,
            expected_headers=expected_headers,
            pass_label=pass_label,
        )
        if page_idx == 1:
            result["reasons"] = ["money_in_non_amount_columns"]
        return result

    def _retry_suspicious_pages_without_layout(self, *, pdf_path, page_evaluations):
        assert pdf_path.endswith(".pdf")
        assert [item["page_index"] for item in page_evaluations] == [1]
        return [{
            **page_evaluations[0],
            "raw_html": "<table><tr><td>repaired-page-2</td></tr></table>",
            "quality_score": 100,
            "suspicious": False,
            "reasons": [],
        }], {"task_count": 1.0, "success_count": 1.0}

    @staticmethod
    def _merge_ocr_metric_summaries(primary, retry):
        return {
            "task_count": primary.get("task_count", 0) + retry.get("task_count", 0),
            "success_count": primary.get("success_count", 0) + retry.get("success_count", 0),
        }


class CountingParser:
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def __init__(self, parser_id: int) -> None:
        self.parser_id = parser_id
        self.layout_guard = None
        self.last_run_stats: dict[str, object] = {}

    def parse_pdf(self, pdf_path: str) -> pd.DataFrame:
        with self.lock:
            type(self).active += 1
            type(self).maximum_active = max(type(self).maximum_active, type(self).active)
        try:
            time.sleep(0.02)
            self.last_run_stats = {"pages_with_tables": [], "timings": {"ocr_tables": 0.01}}
            return pd.DataFrame([{"source": Path(pdf_path).name}])
        finally:
            with self.lock:
                type(self).active -= 1

    def close(self) -> None:
        return None


def test_bank_parser_service_uses_parser_pool_for_concurrent_requests(monkeypatch, tmp_path) -> None:
    barrier = threading.Barrier(2)
    created: list[FakeParser] = []

    def fake_build_parser(self):
        parser = FakeParser(len(created) + 1, barrier)
        created.append(parser)
        return parser

    monkeypatch.setattr(BankParserService, "_build_parser", fake_build_parser)
    service = BankParserService(parser_pool_size=2)
    first_pdf = tmp_path / "first.pdf"
    second_pdf = tmp_path / "second.pdf"
    first_pdf.write_bytes(b"not a real pdf")
    second_pdf.write_bytes(b"not a real pdf")

    results = []
    threads = [
        threading.Thread(target=lambda path=path: results.append(service.parse_pdf(path)))
        for path in (first_pdf, second_pdf)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    try:
        assert len(results) == 2
        assert sorted(result.pages_with_tables[0] for result in results) == [1, 2]
        assert sorted(result.rows[0]["parser_id"] for result in results) == [1, 2]
    finally:
        service.close()

    assert all(parser.closed for parser in created)


def test_borrow_parser_times_out_when_pool_is_saturated(monkeypatch) -> None:
    created: list[FakeParser] = []

    def fake_build_parser(self):
        parser = FakeParser(len(created) + 1, threading.Barrier(1))
        created.append(parser)
        return parser

    monkeypatch.setattr(BankParserService, "_build_parser", fake_build_parser)
    service = BankParserService(parser_pool_size=1, parser_pool_wait_timeout=0.01)
    borrowed = service._parser_pool.get_nowait()
    try:
        with pytest.raises(TimeoutError, match="parser pool"):
            with service._borrow_parser():
                raise AssertionError("should not acquire parser")
    finally:
        service._parser_pool.put(borrowed)
        service.close()


def test_service_admits_35_requests_behind_two_parser_slots(monkeypatch, tmp_path) -> None:
    CountingParser.active = 0
    CountingParser.maximum_active = 0
    created: list[CountingParser] = []

    def fake_build_parser(self):
        parser = CountingParser(len(created) + 1)
        created.append(parser)
        return parser

    monkeypatch.setattr(BankParserService, "_build_parser", fake_build_parser)
    service = BankParserService(parser_pool_size=2)
    start = threading.Barrier(35)
    results = []
    errors = []

    def parse(index: int) -> None:
        path = tmp_path / f"statement-{index}.pdf"
        path.write_bytes(b"not a real pdf")
        start.wait(timeout=5)
        try:
            results.append(service.parse_pdf(path))
        except Exception as exc:  # pragma: no cover - assertion captures details
            errors.append(exc)

    threads = [threading.Thread(target=parse, args=(index,)) for index in range(35)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    try:
        assert errors == []
        assert len(results) == 35
        assert CountingParser.maximum_active == 2
        assert service._parser_pool_wait_timeout == 1800.0
        assert max(result.timings["parser_queue_wait"] for result in results) > 0
        assert max(result.debug["parser_queue_depth_at_submit"] for result in results) >= 1
    finally:
        service.close()


def test_service_readiness_reflects_lifecycle_not_temporary_saturation(monkeypatch) -> None:
    created: list[FakeParser] = []

    def fake_build_parser(self):
        parser = FakeParser(len(created) + 1, threading.Barrier(1))
        created.append(parser)
        return parser

    monkeypatch.setattr(BankParserService, "_build_parser", fake_build_parser)
    service = BankParserService(parser_pool_size=1)
    assert service.is_ready() is True

    borrowed = service._parser_pool.get_nowait()
    try:
        assert service.is_ready() is True
    finally:
        service._parser_pool.put(borrowed)

    service.close()
    assert service.is_ready() is False


def test_extract_glm_page_html_bytes_returns_cache_compatible_pages(monkeypatch, tmp_path) -> None:
    created: list[FakeDirectOCRParser] = []

    def fake_build_parser(self):
        parser = FakeDirectOCRParser(len(created) + 1, threading.Barrier(1))
        created.append(parser)
        return parser

    class FakeDocument:
        page_count = 3

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(BankParserService, "_build_parser", fake_build_parser)
    monkeypatch.setattr("eosin.backend.bank_parser_service.fitz.open", lambda _path: FakeDocument())
    clock = iter([100.0, 101.0, 101.25, 102.0, 102.75, 103.0])
    monkeypatch.setattr(
        "eosin.backend.bank_parser_service.time.time",
        lambda: next(clock),
    )

    service = BankParserService(parser_pool_size=1)
    try:
        payload = service.extract_glm_page_html_bytes(
            "statement.pdf",
            b"fake pdf",
            page_numbers=[1, 3],
            dpi=180,
        )
    finally:
        service.close()

    assert payload["source_pdf"] == "statement.pdf"
    assert payload["page_count"] == 3
    assert payload["pages"] == [
        {
            "page_number": 1,
            "raw_html": "<table><tr><td>page-1</td></tr></table>",
            "quality_score": 91,
            "suspicious": False,
            "reasons": [],
            "provider": "eosin_glm",
            "model": "default",
            "timing_ms": 0.0,
        },
        {
            "page_number": 3,
            "raw_html": "<table><tr><td>page-3</td></tr></table>",
            "quality_score": 89,
            "suspicious": True,
            "reasons": ["low_quality"],
            "provider": "eosin_glm",
            "model": "default",
            "timing_ms": 0.0,
        },
    ]
    assert payload["timings"]["ocr_pages"] == 0.75
    assert payload["ocr_metrics"]["task_count"] == 2.0


def test_document_evidence_uses_text_ocr_for_invoice(monkeypatch) -> None:
    created: list[FakeDirectOCRParser] = []

    def fake_build_parser(self):
        parser = FakeDirectOCRParser(len(created) + 1, threading.Barrier(1))
        created.append(parser)
        return parser

    class FakeDocument:
        page_count = 1

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(BankParserService, "_build_parser", fake_build_parser)
    monkeypatch.setattr("eosin.backend.bank_parser_service.fitz.open", lambda _path: FakeDocument())

    service = BankParserService(parser_pool_size=1)
    try:
        payload = service.extract_glm_page_html_bytes(
            "invoice.pdf", b"fake pdf", task_type="text"
        )
    finally:
        service.close()

    assert created[0].last_task_type == "text"
    assert payload["ocr_task_type"] == "text"
    assert payload["pages"][0]["quality_score"] == 100
    assert payload["pages"][0]["suspicious"] is False


def test_document_evidence_retries_only_critical_table_pages(monkeypatch) -> None:
    def fake_build_parser(self):
        return CriticalRetryParser(1, threading.Barrier(1))

    class FakeDocument:
        page_count = 2

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(BankParserService, "_build_parser", fake_build_parser)
    monkeypatch.setattr("eosin.backend.bank_parser_service.fitz.open", lambda _path: FakeDocument())

    service = BankParserService(parser_pool_size=1)
    try:
        payload = service.extract_glm_page_html_bytes("statement.pdf", b"fake pdf")
    finally:
        service.close()

    assert payload["pages"][0]["raw_html"].endswith("page-1</td></tr></table>")
    assert payload["pages"][1]["raw_html"].endswith("repaired-page-2</td></tr></table>")
    assert payload["pages"][1]["suspicious"] is False
    assert payload["ocr_metrics"]["task_count"] == 3.0


def test_evidence_payload_preserves_direct_extraction_page_contract() -> None:
    direct_payload = {
        "source_pdf": "statement.pdf",
        "page_count": 1,
        "pages": [
            {
                "page_number": 1,
                "raw_html": "<table><tr><td>page-1</td></tr></table>",
                "quality_score": 91,
                "suspicious": False,
                "reasons": [],
                "provider": "eosin_glm",
                "model": "test-model",
                "timing_ms": 12.5,
            }
        ],
        "timings": {
            "render_pages": 0.1,
            "ocr_pages": 0.2,
            "service_total": 0.3,
        },
        "ocr_metrics": {"task_count": 1.0},
    }

    payload = evidence_payload_from_result(direct_payload)

    assert payload["pages"] == direct_payload["pages"]
    assert payload["timings"] == direct_payload["timings"]
    assert payload["ocr_metrics"] == direct_payload["ocr_metrics"]


def test_http_and_modal_rpc_return_identical_shaped_evidence_payloads() -> None:
    direct_payload = {
        "source_pdf": "statement.pdf",
        "page_count": 1,
        "pages": [
            {
                "page_number": 1,
                "raw_html": "<table><tr><td>page-1</td></tr></table>",
                "quality_score": 91,
                "suspicious": True,
                "reasons": ["low_quality"],
                "provider": "eosin_glm",
                "model": "test-model",
                "timing_ms": 12.5,
            }
        ],
        "timings": {"service_total": 0.3},
        "ocr_metrics": {"task_count": 1.0},
    }

    expected_payload = evidence_payload_from_result(direct_payload)

    api_tree = ast.parse(Path("eosin/backend/bank_parser_api.py").read_text())
    create_app_node = next(
        node
        for node in api_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_app"
    )
    http_method = next(
        node
        for node in create_app_node.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "extract_bank_statement_evidence"
    )
    http_payload_call = next(
        node
        for node in ast.walk(http_method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "evidence_payload_from_result"
    )

    modal_tree = ast.parse(Path("modal_app.py").read_text())
    modal_class = next(
        node
        for node in modal_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BankParserModalApp"
    )
    modal_method = next(
        node
        for node in modal_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "extract_evidence"
    )
    modal_return = next(
        node
        for node in modal_method.body
        if isinstance(node, ast.Return)
    )

    assert expected_payload["pages"] == direct_payload["pages"]
    assert isinstance(http_payload_call, ast.Call)
    assert isinstance(modal_return.value, ast.Call)
    assert isinstance(modal_return.value.func, ast.Name)
    assert modal_return.value.func.id == "evidence_payload_from_result"
    direct_call = modal_return.value.args[0]
    assert isinstance(direct_call, ast.Call)
    assert isinstance(direct_call.func, ast.Attribute)
    assert direct_call.func.attr == "extract_glm_page_html_bytes"
