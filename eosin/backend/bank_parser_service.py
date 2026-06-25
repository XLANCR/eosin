from __future__ import annotations

import os
import tempfile
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd
import fitz

import eosin.backend.eosin_pipeline as impl


@dataclass(frozen=True)
class BankParserResult:
    source_pdf: str
    page_count: int
    pages_with_tables: list[int]
    columns: list[str]
    rows: list[dict[str, Any]]
    timings: dict[str, float]
    debug: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_pdf": self.source_pdf,
            "page_count": self.page_count,
            "pages_with_tables": self.pages_with_tables,
            "columns": self.columns,
            "rows": self.rows,
            "timings": self.timings,
            "debug": self.debug,
        }


class BankParserService:
    """Server-side wrapper around the bank statement parser script."""

    def __init__(
        self,
        config_path: str | None = None,
        *,
        layout_mode: Optional[str] = None,
        save_debug_images: Optional[bool] = None,
        parse_testing: Optional[bool] = None,
        enable_ocr_batching: Optional[bool] = None,
        ocr_batch_size: Optional[int] = None,
        pdf_render_dpi_override: Optional[int] = None,
        layout_max_concurrency: Optional[int] = None,
        ocr_pipeline_workers: Optional[int] = None,
        ocr_pipeline_queue_size: Optional[int] = None,
        ocr_backend_mode: Optional[str] = None,
        ocr_batch_drain_max_batch_size: Optional[int] = None,
        ocr_batch_drain_max_wait_seconds: Optional[float] = None,
        enable_page_ocr_retry: Optional[bool] = None,
        capture_raw_ocr_debug: Optional[bool] = None,
        page_ocr_retry_dpi: Optional[int] = None,
        parser_pool_size: int = 1,
        backend_startup_timeout: float = 180.0,
        backend_retry_interval: float = 5.0,
    ):
        default_config_path = Path(__file__).resolve().parent / "config.yaml"
        self.config_path = str(config_path or default_config_path)
        self.layout_mode = layout_mode
        self._settings = {
            "ENABLE_OCR_HEADER_FALLBACK": False,
        }
        if save_debug_images is not None:
            self._settings["SAVE_DEBUG_IMAGES"] = save_debug_images
        if parse_testing is not None:
            self._settings["PARSE_TESTING"] = parse_testing
        if enable_ocr_batching is not None:
            self._settings["ENABLE_OCR_BATCHING"] = enable_ocr_batching
        if ocr_batch_size is not None:
            self._settings["OCR_BATCH_SIZE"] = ocr_batch_size
        if pdf_render_dpi_override is not None:
            self._settings["PDF_RENDER_DPI_OVERRIDE"] = pdf_render_dpi_override
        if ocr_pipeline_workers is not None:
            self._settings["OCR_PIPELINE_WORKERS"] = ocr_pipeline_workers
        if ocr_pipeline_queue_size is not None:
            self._settings["OCR_PIPELINE_QUEUE_SIZE"] = ocr_pipeline_queue_size
        if ocr_backend_mode is not None:
            self._settings["OCR_BACKEND_MODE"] = ocr_backend_mode
        if ocr_batch_drain_max_batch_size is not None:
            self._settings["OCR_BATCH_DRAIN_MAX_BATCH_SIZE"] = ocr_batch_drain_max_batch_size
        if ocr_batch_drain_max_wait_seconds is not None:
            self._settings["OCR_BATCH_DRAIN_MAX_WAIT_SECONDS"] = ocr_batch_drain_max_wait_seconds
        if enable_page_ocr_retry is not None:
            self._settings["ENABLE_PAGE_OCR_RETRY"] = enable_page_ocr_retry
        if capture_raw_ocr_debug is not None:
            self._settings["CAPTURE_RAW_OCR_DEBUG"] = capture_raw_ocr_debug
        if page_ocr_retry_dpi is not None:
            self._settings["PAGE_OCR_RETRY_DPI"] = page_ocr_retry_dpi

        self._backend_startup_timeout = backend_startup_timeout
        self._backend_retry_interval = backend_retry_interval
        self._layout_max_concurrency = max(1, int(layout_max_concurrency or 1))
        self._layout_guard = threading.BoundedSemaphore(self._layout_max_concurrency)
        self._parser_pool_size = max(1, int(parser_pool_size))
        self._parser_pool: queue.Queue[impl.BankStatementParser] = queue.Queue(maxsize=self._parser_pool_size)
        self._parsers: list[impl.BankStatementParser] = []
        self._apply_impl_settings()
        self._build_parser_pool()

    def _apply_impl_settings(self) -> None:
        for name, value in self._settings.items():
            setattr(impl, name, value)

    def _build_parser(self) -> impl.BankStatementParser:
        deadline = time.monotonic() + self._backend_startup_timeout
        while True:
            try:
                parser = impl.BankStatementParser(
                    self.config_path,
                    layout_mode=self.layout_mode,
                )
                parser.layout_guard = self._layout_guard
                return parser
            except (ConnectionError, TimeoutError) as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for GLM OCR backend after {self._backend_startup_timeout:.0f}s"
                    ) from exc
                time.sleep(self._backend_retry_interval)

    def _build_parser_pool(self) -> None:
        for _ in range(self._parser_pool_size):
            parser = self._build_parser()
            self._parsers.append(parser)
            self._parser_pool.put(parser)

    @contextmanager
    def _borrow_parser(self) -> Iterator[impl.BankStatementParser]:
        parser = self._parser_pool.get()
        try:
            yield parser
        finally:
            self._parser_pool.put(parser)

    @staticmethod
    def _json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []

        safe_df = df.replace([np.inf, -np.inf], np.nan)
        safe_df = safe_df.astype(object).where(pd.notna(safe_df), None)
        return safe_df.to_dict(orient="records")

    def parse_pdf(self, pdf_path: str | Path) -> BankParserResult:
        pdf_path = Path(pdf_path)
        started_at = time.time()

        with self._borrow_parser() as parser:
            df = parser.parse_pdf(str(pdf_path))
            parser_stats = dict(getattr(parser, "last_run_stats", {}) or {})

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()

        try:
            with fitz.open(str(pdf_path)) as document:
                page_count = int(document.page_count)
        except Exception:
            page_count = 0

        return BankParserResult(
            source_pdf=pdf_path.name,
            page_count=page_count,
            pages_with_tables=[
                int(page_number)
                for page_number in parser_stats.get("pages_with_tables", [])
            ],
            columns=[str(column) for column in df.columns],
            rows=self._json_safe_records(df),
            timings={
                **{
                    str(name): float(value)
                    for name, value in parser_stats.get("timings", {}).items()
                },
                "service_total": round(time.time() - started_at, 3),
            },
            debug={
                key: value
                for key, value in parser_stats.items()
                if key != "timings" and key != "pages_with_tables"
            },
        )

    def parse_pdf_bytes(self, filename: str, pdf_bytes: bytes) -> BankParserResult:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(pdf_bytes)
            temp_path = Path(handle.name)

        try:
            result = self.parse_pdf(temp_path)
            return BankParserResult(
                source_pdf=filename,
                page_count=result.page_count,
                pages_with_tables=result.pages_with_tables,
                columns=result.columns,
                rows=result.rows,
                timings=result.timings,
                debug=result.debug,
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def extract_glm_page_html_bytes(
        self,
        filename: str,
        pdf_bytes: bytes,
        page_numbers: list[int] | None = None,
        *,
        dpi: int | None = None,
    ) -> dict[str, Any]:
        """Return raw GLM table HTML per page with quality metadata.

        Per plan.md: eosin is the evidence-only service.  This method runs
        PDF rendering → GLM OCR → quality evaluation and returns raw HTML
        + quality scores + retry/timing metadata.  It does NOT run the full
        parser pipeline (no DataFrame assembly, no column alignment, no
        row repairs).
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(pdf_bytes)
            temp_path = Path(handle.name)

        started_at = time.time()
        try:
            with fitz.open(str(temp_path)) as document:
                page_count = int(document.page_count)

            requested_pages = page_numbers or list(range(1, page_count + 1))
            page_indices = [
                page_number - 1
                for page_number in requested_pages
                if 1 <= int(page_number) <= page_count
            ]

            with self._borrow_parser() as parser:
                render_dpi = int(dpi or parser.pdf_dpi)
                render_started_at = time.time()
                rendered_pages = parser._render_specific_pages(str(temp_path), page_indices, render_dpi)
                render_seconds = time.time() - render_started_at
                ocr_images = [
                    (page_idx, parser._normalize_ocr_image(rendered_pages[page_idx]))
                    for page_idx in sorted(rendered_pages)
                ]
                ocr_started_at = time.time()
                ocr_results, ocr_metrics = parser._ocr_tables_parallel(ocr_images)
                ocr_seconds = time.time() - ocr_started_at

                # Quality evaluation per page (lightweight: no DataFrame assembly)
                page_quality = {}
                for page_idx, html in ocr_results:
                    if not html:
                        page_quality[page_idx] = {"quality_score": 0, "suspicious": True, "reasons": ["empty_ocr_output"]}
                        continue
                    try:
                        evaluation = parser._evaluate_page_ocr_result(
                            page_idx=page_idx,
                            html_content=html,
                            expected_headers=None,
                            pass_label="evidence",
                        )
                        page_quality[page_idx] = {
                            "quality_score": int(evaluation.get("quality_score", 50)),
                            "suspicious": bool(evaluation.get("suspicious", False)),
                            "reasons": [str(r) for r in evaluation.get("reasons", [])],
                        }
                    except Exception:
                        page_quality[page_idx] = {
                            "quality_score": 50, "suspicious": True,
                            "reasons": ["quality_evaluation_failed"],
                        }

            pages = [
                {
                    "page_number": page_idx + 1,
                    "raw_html": html or "",
                    "quality_score": page_quality.get(page_idx, {}).get("quality_score", 50),
                    "suspicious": page_quality.get(page_idx, {}).get("suspicious", True),
                    "reasons": page_quality.get(page_idx, {}).get("reasons", []),
                    "provider": "eosin_glm",
                    "model": os.getenv("GLMOCR_OCR_MODEL", "default"),
                    "timing_ms": 0.0,
                }
                for page_idx, html in sorted(ocr_results, key=lambda item: item[0])
            ]
            return {
                "source_pdf": filename,
                "page_count": page_count,
                "pages": pages,
                "timings": {
                    "render_pages": round(render_seconds, 3),
                    "ocr_pages": round(ocr_seconds, 3),
                    "service_total": round(time.time() - started_at, 3),
                },
                "ocr_metrics": ocr_metrics,
            }
        finally:
            temp_path.unlink(missing_ok=True)

    def close(self) -> None:
        for parser in self._parsers:
            parser.close()
