from __future__ import annotations

from concurrent.futures import Future
from concurrent.futures import wait
from types import SimpleNamespace

from PIL import Image

import eosin.backend.eosin_pipeline as pipeline_module
from eosin.backend.eosin_pipeline import BankStatementParser
from eosin.backend.ocr_pipeline import (
    BatchDrainOCRBackend,
    DocumentOCRTaskResult,
    HTTPDocumentOCRBackend,
    build_document_request,
)


class FakePageLoader:
    def build_request_from_image(self, image: Image.Image, task_type: str = "text") -> dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"image://{image.size[0]}x{image.size[1]}"},
                        },
                        {"type": "text", "text": f"task={task_type}"},
                    ],
                }
            ],
            "max_tokens": 10,
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "repetition_penalty": 1,
        }


class FakeOCRClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def process(self, request: dict) -> tuple[dict, int]:
        self.requests.append(request)
        image_count = sum(
            1
            for item in request["messages"][0]["content"]
            if item["type"] == "image_url"
        )
        tables = "".join(
            f"<table><tr><td>page-{index}</td></tr></table>"
            for index in range(image_count)
        )
        return {"choices": [{"message": {"content": tables}}]}, 200


def make_image() -> Image.Image:
    return Image.new("RGB", (8, 8), color="white")


def test_build_document_request_merges_multiple_images_into_one_payload() -> None:
    request = build_document_request(
        FakePageLoader(),
        [make_image(), make_image()],
        task_type="table",
    )

    content = request["messages"][0]["content"]
    image_items = [item for item in content if item["type"] == "image_url"]
    text_items = [item for item in content if item["type"] == "text"]

    assert len(image_items) == 2
    assert len(text_items) == 1
    assert "task=table" in text_items[0]["text"]
    assert "same bank statement transaction table" in text_items[0]["text"]
    assert "Do not repeat header rows" in text_items[0]["text"]
    assert request["max_tokens"] == 20


def test_http_document_backend_returns_one_result_per_document_page() -> None:
    backend = HTTPDocumentOCRBackend(
        page_loader=FakePageLoader(),
        ocr_client=FakeOCRClient(),
        max_workers=1,
        queue_size=8,
    )
    try:
        future = backend.submit([make_image(), make_image()], page_indices=[0, 1], task_type="table")
        result = future.result(timeout=2)
    finally:
        backend.close()

    assert isinstance(result, DocumentOCRTaskResult)
    assert result.status_code == 200
    assert result.contents == ("<table><tr><td>page-0</td></tr></table>", "<table><tr><td>page-1</td></tr></table>")
    assert result.batch_size == 1


def test_batch_drain_backend_groups_ready_documents_into_one_flush() -> None:
    backend = BatchDrainOCRBackend(
        page_loader=FakePageLoader(),
        ocr_client=FakeOCRClient(),
        max_workers=2,
        queue_size=8,
        max_batch_size=2,
        max_wait_seconds=1.0,
    )
    try:
        first = backend.submit([make_image()], page_indices=[0], task_type="table")
        second = backend.submit([make_image()], page_indices=[1], task_type="table")
        wait([first, second], timeout=3)
        first_result = first.result(timeout=1)
        second_result = second.result(timeout=1)
    finally:
        backend.close()

    assert first_result.batch_size == 2
    assert second_result.batch_size == 2
    assert first_result.flush_reason == "batch_full"
    assert second_result.flush_reason == "batch_full"


def test_ocr_text_submits_single_image_with_page_index_zero() -> None:
    parser = BankStatementParser.__new__(BankStatementParser)

    class FakeDispatcher:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def submit(
            self,
            image: Image.Image,
            *,
            page_index: int = 0,
            task_type: str,
        ) -> Future[DocumentOCRTaskResult]:
            self.calls.append(
                {
                    "image_size": image.size,
                    "task_type": task_type,
                }
            )
            future: Future[DocumentOCRTaskResult] = Future()
            future.set_result(
                DocumentOCRTaskResult(
                    contents=(" header text ",),
                    status_code=200,
                    queue_wait_seconds=0.0,
                    build_request_seconds=0.0,
                    request_seconds=0.0,
                    total_seconds=0.0,
                    queue_size_at_submit=0,
                    batch_size=1,
                    flush_reason="immediate",
                    backend_name="document_http",
                )
            )
            return future

    parser.ocr_dispatcher = FakeDispatcher()

    result = parser._ocr_text(make_image())

    assert result == "header text"
    assert parser.ocr_dispatcher.calls == [
        {"image_size": (8, 8), "task_type": "text"}
    ]


def test_ocr_tables_parallel_submits_entire_pdf_as_one_document_task() -> None:
    parser = BankStatementParser.__new__(BankStatementParser)
    parser.ocr_max_workers = 32
    parser.ocr_connection_pool_size = 32

    class FakeDispatcher:
        def __init__(self) -> None:
            self.page_calls: list[dict] = []

        def submit(
            self,
            image: Image.Image,
            *,
            page_index: int = 0,
            task_type: str,
        ) -> Future[DocumentOCRTaskResult]:
            self.page_calls.append(
                {
                    "image_size": image.size,
                    "page_index": page_index,
                    "task_type": task_type,
                }
            )
            future: Future[DocumentOCRTaskResult] = Future()
            future.set_result(
                DocumentOCRTaskResult(
                    contents=(f"page {page_index}",),
                    status_code=200,
                    queue_wait_seconds=0.0,
                    build_request_seconds=0.0,
                    request_seconds=1.0,
                    total_seconds=1.0,
                    queue_size_at_submit=0,
                    batch_size=1,
                    flush_reason="immediate",
                    backend_name="page_http",
                )
            )
            return future

    parser.ocr_dispatcher = FakeDispatcher()

    results, metrics = parser._ocr_tables_parallel(
        [(0, make_image()), (1, make_image()), (2, make_image())]
    )

    assert parser.ocr_dispatcher.page_calls == [
        {"image_size": (8, 8), "page_index": 0, "task_type": "table"},
        {"image_size": (8, 8), "page_index": 1, "task_type": "table"},
        {"image_size": (8, 8), "page_index": 2, "task_type": "table"},
    ]
    assert results == [(0, "page 0"), (1, "page 1"), (2, "page 2")]
    assert metrics["task_count"] == 3.0
    assert metrics["document_page_count_mean"] == 1.0
    assert metrics["document_page_count_max"] == 1.0
    assert metrics["backend_batch_size_mean"] == 1.0
    assert metrics["backend_batch_size_max"] == 1.0


def test_ocr_tables_parallel_chunks_large_documents_when_configured(monkeypatch) -> None:
    parser = BankStatementParser.__new__(BankStatementParser)
    parser.ocr_max_workers = 32
    parser.ocr_connection_pool_size = 32

    class FakeDispatcher:
        def __init__(self) -> None:
            self.page_calls: list[dict] = []

        def submit(
            self,
            image: Image.Image,
            *,
            page_index: int = 0,
            task_type: str,
        ) -> Future[DocumentOCRTaskResult]:
            self.page_calls.append(
                {
                    "image_size": image.size,
                    "page_index": page_index,
                    "task_type": task_type,
                }
            )
            future: Future[DocumentOCRTaskResult] = Future()
            future.set_result(
                DocumentOCRTaskResult(
                    contents=(f"<table><tr><td>{page_index}</td></tr></table>",),
                    status_code=200,
                    queue_wait_seconds=0.0,
                    build_request_seconds=0.0,
                    request_seconds=1.0,
                    total_seconds=1.0,
                    queue_size_at_submit=0,
                    batch_size=1,
                    flush_reason="immediate",
                    backend_name="page_http",
                )
            )
            return future

    parser.ocr_dispatcher = FakeDispatcher()
    monkeypatch.setattr(pipeline_module, "OCR_DOCUMENT_MAX_IMAGES_PER_REQUEST", 2)

    results, metrics = parser._ocr_tables_parallel(
        [(0, make_image()), (1, make_image()), (2, make_image()), (3, make_image()), (4, make_image())]
    )

    assert parser.ocr_dispatcher.page_calls == [
        {"image_size": (8, 8), "page_index": 0, "task_type": "table"},
        {"image_size": (8, 8), "page_index": 1, "task_type": "table"},
        {"image_size": (8, 8), "page_index": 2, "task_type": "table"},
        {"image_size": (8, 8), "page_index": 3, "task_type": "table"},
        {"image_size": (8, 8), "page_index": 4, "task_type": "table"},
    ]
    assert results == [
        (0, "<table><tr><td>0</td></tr></table>"),
        (1, "<table><tr><td>1</td></tr></table>"),
        (2, "<table><tr><td>2</td></tr></table>"),
        (3, "<table><tr><td>3</td></tr></table>"),
        (4, "<table><tr><td>4</td></tr></table>"),
    ]
    assert metrics["task_count"] == 5.0


def test_parser_init_passes_dispatcher_backend_configuration(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeLayoutDetector:
        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class FakeOCRClientForInit:
        def __init__(self, _config) -> None:
            self._pool_maxsize = 11

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class FakeDispatcher:
        def __init__(
            self,
            page_loader,
            ocr_client,
            *,
            max_workers: int,
            queue_size: int,
            backend_mode: str,
            batch_drain_max_batch_size: int,
            batch_drain_max_wait_seconds: float,
        ) -> None:
            captured["page_loader"] = page_loader
            captured["ocr_client"] = ocr_client
            captured["max_workers"] = max_workers
            captured["queue_size"] = queue_size
            captured["backend_mode"] = backend_mode
            captured["batch_drain_max_batch_size"] = batch_drain_max_batch_size
            captured["batch_drain_max_wait_seconds"] = batch_drain_max_wait_seconds

        def close(self) -> None:
            return None

    fake_sdk_config = SimpleNamespace(
        pipeline=SimpleNamespace(
            layout=SimpleNamespace(),
            page_loader=SimpleNamespace(pdf_dpi=200),
            ocr_api=SimpleNamespace(),
            max_workers=32,
        )
    )

    monkeypatch.setattr(pipeline_module, "sdk_load_config", lambda _path: fake_sdk_config)
    monkeypatch.setattr(pipeline_module, "PPDocLayoutDetector", lambda _cfg: FakeLayoutDetector())
    monkeypatch.setattr(pipeline_module, "PageLoader", lambda cfg: SimpleNamespace(pdf_dpi=cfg.pdf_dpi))
    monkeypatch.setattr(pipeline_module, "OCRClient", FakeOCRClientForInit)
    monkeypatch.setattr(pipeline_module, "OCRPipelineDispatcher", FakeDispatcher)
    monkeypatch.setattr(pipeline_module, "OCR_PIPELINE_WORKERS", 13)
    monkeypatch.setattr(pipeline_module, "OCR_PIPELINE_QUEUE_SIZE", 111)
    monkeypatch.setattr(pipeline_module, "OCR_BACKEND_MODE", "page_http")
    monkeypatch.setattr(pipeline_module, "OCR_BATCH_DRAIN_MAX_BATCH_SIZE", 96)
    monkeypatch.setattr(pipeline_module, "OCR_BATCH_DRAIN_MAX_WAIT_SECONDS", 1.0)

    parser = BankStatementParser(config_path="dummy.yaml")
    parser.close()

    assert captured["max_workers"] == 11
    assert captured["queue_size"] == 111
    assert captured["backend_mode"] == "page_http"
    assert captured["batch_drain_max_batch_size"] == 96
    assert captured["batch_drain_max_wait_seconds"] == 1.0
