from __future__ import annotations

from concurrent.futures import Future
from concurrent.futures import wait

from PIL import Image

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
    assert text_items == [{"type": "text", "text": "task=table"}]


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

        def submit(self, image: Image.Image, *, page_index: int, task_type: str) -> Future[DocumentOCRTaskResult]:
            self.calls.append(
                {
                    "image_size": image.size,
                    "page_index": page_index,
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
        {"image_size": (8, 8), "page_index": 0, "task_type": "text"}
    ]
