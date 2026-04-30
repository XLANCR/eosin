from __future__ import annotations

import queue
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from bs4 import BeautifulSoup
from PIL import Image


@dataclass(frozen=True)
class OCRPipelineTask:
    images: tuple[Image.Image, ...]
    page_indices: tuple[int, ...]
    task_type: str
    future: Future["DocumentOCRTaskResult"]
    enqueued_at: float
    queue_size_at_submit: int


@dataclass(frozen=True)
class DocumentOCRTaskResult:
    contents: tuple[Optional[str], ...]
    status_code: int
    queue_wait_seconds: float
    build_request_seconds: float
    request_seconds: float
    total_seconds: float
    queue_size_at_submit: int
    batch_size: int
    flush_reason: str
    backend_name: str


def _extract_tables(response_text: str) -> list[str]:
    matched_tables = re.findall(
        r"<table[^>]*>.*?</table>",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    if matched_tables:
        return matched_tables

    soup = BeautifulSoup(response_text, "html.parser")
    return [str(table) for table in soup.find_all("table")]


def split_document_response(response_text: str, expected_count: int) -> tuple[Optional[str], ...]:
    if expected_count <= 0:
        return ()

    tables = _extract_tables(response_text)
    if not tables:
        first = response_text.strip() or None
        return tuple([first] + [None] * max(0, expected_count - 1))

    if len(tables) >= expected_count:
        return tuple(tables[:expected_count])

    padded = list(tables)
    padded.extend([None] * (expected_count - len(padded)))
    return tuple(padded)


def build_document_request(page_loader, images: Sequence[Image.Image], *, task_type: str) -> dict:
    if not images:
        raise ValueError("images must not be empty")

    base_request = page_loader.build_request_from_image(images[0], task_type=task_type)
    base_content = list(base_request["messages"][0]["content"])
    prompt_items = [item for item in base_content if item.get("type") == "text"]
    image_items = [item for item in base_content if item.get("type") == "image_url"]

    for image in images[1:]:
        image_request = page_loader.build_request_from_image(image, task_type=task_type)
        image_content = image_request["messages"][0]["content"]
        image_items.extend(
            item for item in image_content if item.get("type") == "image_url"
        )

    request = dict(base_request)
    request["messages"] = [
        {
            "role": "user",
            "content": [*image_items, *prompt_items],
        }
    ]
    return request


def _response_to_content(response: dict, status_code: int) -> Optional[str]:
    if status_code != 200:
        return None

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None

    text = str(content).strip()
    return text or None


def _build_task_result(
    *,
    response: dict,
    status_code: int,
    task: OCRPipelineTask,
    queue_wait_seconds: float,
    build_request_seconds: float,
    request_seconds: float,
    total_seconds: float,
    batch_size: int,
    flush_reason: str,
    backend_name: str,
) -> DocumentOCRTaskResult:
    content = _response_to_content(response, status_code)
    contents = split_document_response(content or "", len(task.page_indices))
    if content is None:
        contents = tuple([None] * len(task.page_indices))

    return DocumentOCRTaskResult(
        contents=contents,
        status_code=int(status_code),
        queue_wait_seconds=queue_wait_seconds,
        build_request_seconds=build_request_seconds,
        request_seconds=request_seconds,
        total_seconds=total_seconds,
        queue_size_at_submit=task.queue_size_at_submit,
        batch_size=batch_size,
        flush_reason=flush_reason,
        backend_name=backend_name,
    )


class HTTPDocumentOCRBackend:
    """Process one document task per worker from a bounded queue."""

    def __init__(
        self,
        *,
        page_loader,
        ocr_client,
        max_workers: int,
        queue_size: int,
    ) -> None:
        self._page_loader = page_loader
        self._ocr_client = ocr_client
        self._queue: queue.Queue[OCRPipelineTask | None] = queue.Queue(maxsize=max(1, queue_size))
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

        for index in range(max(1, max_workers)):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"ocr-http-{index + 1}",
                daemon=True,
            )
            worker.start()
            self._threads.append(worker)

    def submit(
        self,
        images: Sequence[Image.Image],
        *,
        page_indices: Sequence[int],
        task_type: str,
    ) -> Future[DocumentOCRTaskResult]:
        future: Future[DocumentOCRTaskResult] = Future()
        task = OCRPipelineTask(
            images=tuple(images),
            page_indices=tuple(page_indices),
            task_type=task_type,
            future=future,
            enqueued_at=time.perf_counter(),
            queue_size_at_submit=self._queue.qsize(),
        )
        self._queue.put(task)
        return future

    def close(self) -> None:
        self._stop_event.set()
        for _ in self._threads:
            self._queue.put(None)
        for worker in self._threads:
            worker.join(timeout=5)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            task = self._queue.get()
            if task is None:
                self._queue.task_done()
                break

            try:
                request_started_at = time.perf_counter()
                queue_wait_seconds = request_started_at - task.enqueued_at
                build_started_at = request_started_at
                request = build_document_request(
                    self._page_loader,
                    task.images,
                    task_type=task.task_type,
                )
                build_request_seconds = time.perf_counter() - build_started_at
                ocr_started_at = time.perf_counter()
                response, status_code = self._ocr_client.process(request)
                request_seconds = time.perf_counter() - ocr_started_at
                total_seconds = time.perf_counter() - task.enqueued_at
                task.future.set_result(
                    _build_task_result(
                        response=response,
                        status_code=status_code,
                        task=task,
                        queue_wait_seconds=queue_wait_seconds,
                        build_request_seconds=build_request_seconds,
                        request_seconds=request_seconds,
                        total_seconds=total_seconds,
                        batch_size=1,
                        flush_reason="immediate",
                        backend_name="http_document",
                    )
                )
            except Exception as exc:
                task.future.set_exception(exc)
            finally:
                self._queue.task_done()


class BatchDrainOCRBackend:
    """Batch document OCR jobs briefly, then flush them together to vLLM."""

    def __init__(
        self,
        *,
        page_loader,
        ocr_client,
        max_workers: int,
        queue_size: int,
        max_batch_size: int,
        max_wait_seconds: float,
    ) -> None:
        self._page_loader = page_loader
        self._ocr_client = ocr_client
        self._queue: queue.Queue[OCRPipelineTask | None] = queue.Queue(maxsize=max(1, queue_size))
        self._stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="ocr-batch")
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_wait_seconds = max(0.001, float(max_wait_seconds))
        self._thread = threading.Thread(
            target=self._drain_loop,
            name="ocr-batch-drain",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        images: Sequence[Image.Image],
        *,
        page_indices: Sequence[int],
        task_type: str,
    ) -> Future[DocumentOCRTaskResult]:
        future: Future[DocumentOCRTaskResult] = Future()
        task = OCRPipelineTask(
            images=tuple(images),
            page_indices=tuple(page_indices),
            task_type=task_type,
            future=future,
            enqueued_at=time.perf_counter(),
            queue_size_at_submit=self._queue.qsize(),
        )
        self._queue.put(task)
        return future

    def close(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        self._thread.join(timeout=5)
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _drain_loop(self) -> None:
        while not self._stop_event.is_set():
            first = self._queue.get()
            if first is None:
                self._queue.task_done()
                break

            tasks = [first]
            flush_reason = "timeout"
            deadline = time.perf_counter() + self._max_wait_seconds

            while len(tasks) < self._max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break

                if item is None:
                    self._queue.task_done()
                    self._stop_event.set()
                    break

                tasks.append(item)
                if len(tasks) >= self._max_batch_size:
                    flush_reason = "batch_full"
                    break

            batch_size = len(tasks)
            for task in tasks:
                self._executor.submit(
                    self._execute_task,
                    task,
                    batch_size,
                    flush_reason,
                )

    def _execute_task(self, task: OCRPipelineTask, batch_size: int, flush_reason: str) -> None:
        try:
            request_started_at = time.perf_counter()
            queue_wait_seconds = request_started_at - task.enqueued_at
            build_started_at = request_started_at
            request = build_document_request(
                self._page_loader,
                task.images,
                task_type=task.task_type,
            )
            build_request_seconds = time.perf_counter() - build_started_at
            ocr_started_at = time.perf_counter()
            response, status_code = self._ocr_client.process(request)
            request_seconds = time.perf_counter() - ocr_started_at
            total_seconds = time.perf_counter() - task.enqueued_at
            task.future.set_result(
                _build_task_result(
                    response=response,
                    status_code=status_code,
                    task=task,
                    queue_wait_seconds=queue_wait_seconds,
                    build_request_seconds=build_request_seconds,
                    request_seconds=request_seconds,
                    total_seconds=total_seconds,
                    batch_size=batch_size,
                    flush_reason=flush_reason,
                    backend_name="batch_document",
                )
            )
        except Exception as exc:
            task.future.set_exception(exc)
        finally:
            self._queue.task_done()


class OCRPipelineDispatcher:
    """Adapter around selectable OCR execution backends."""

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
        mode = backend_mode.strip().lower()
        self.backend_mode = mode

        if mode == "batch_document":
            self._backend = BatchDrainOCRBackend(
                page_loader=page_loader,
                ocr_client=ocr_client,
                max_workers=max_workers,
                queue_size=queue_size,
                max_batch_size=batch_drain_max_batch_size,
                max_wait_seconds=batch_drain_max_wait_seconds,
            )
        else:
            self._backend = HTTPDocumentOCRBackend(
                page_loader=page_loader,
                ocr_client=ocr_client,
                max_workers=max_workers,
                queue_size=queue_size,
            )

    def submit_document(
        self,
        images: Sequence[Image.Image],
        *,
        page_indices: Sequence[int],
        task_type: str,
    ) -> Future[DocumentOCRTaskResult]:
        return self._backend.submit(images, page_indices=page_indices, task_type=task_type)

    def submit(self, image: Image.Image, *, page_index: int, task_type: str) -> Future[DocumentOCRTaskResult]:
        return self.submit_document([image], page_indices=[page_index], task_type=task_type)

    def close(self) -> None:
        self._backend.close()
