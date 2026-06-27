from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

import eosin.backend.eosin_pipeline as impl
from eosin.backend.bank_parser_service import BankParserService
from eosin.backend.bank_parser_service import BankParserResult
from eosin.backend.metrics import get_metrics_manager


def evidence_payload_from_result(result: dict | BankParserResult) -> dict:
    """Transform evidence-only extraction result to the v2 evidence contract.

    Per plan.md: eosin returns raw per-page GLM HTML + quality metadata +
    retry/timing diagnostics.  No DataFrame assembly, no column alignment,
    no row repairs.  Almond owns all parser intelligence.

    Accepts both dict payloads and BankParserResult objects (from Modal-returned
    results that may not have been serialized to dict).
    """
    if not isinstance(result, dict):
        result = result.to_payload()
    source_pages = result.get("pages")
    if source_pages is None:
        source_pages = result.get("debug", {}).get("page_ocr", [])
    pages = []
    for item in source_pages:
        page_number = item.get("page_number")
        if page_number is None:
            page_number = int(item.get("page_index", -1)) + 1
        page = {
            "page_number": int(page_number),
            "raw_html": str(item.get("raw_html", "")),
            "quality_score": int(item.get("quality_score", 50)),
            "suspicious": bool(item.get("suspicious", True)),
            "reasons": [str(r) for r in item.get("reasons", [])],
            "provider": str(item.get("provider", "eosin_glm")),
            "model": str(item.get("model", os.getenv("GLMOCR_OCR_MODEL", "default"))),
            "timing_ms": float(item.get("timing_ms", 0.0)),
        }
        if "headers" in item or "raw_headers" in item:
            page["headers"] = [
                str(header)
                for header in item.get("headers", item.get("raw_headers", []))
            ]
        if "row_count" in item:
            page["row_count"] = int(item["row_count"])
        pages.append(page)

    return {
        "source_pdf": str(result.get("source_pdf", "")),
        "page_count": int(result.get("page_count", 0)),
        "pages_with_tables": [
            int(page_number)
            for page_number in result.get(
                "pages_with_tables",
                [page["page_number"] for page in pages if page["raw_html"]],
            )
        ],
        "pages": pages,
        "timings": result.get("timings", {}),
        "ocr_metrics": result.get("ocr_metrics", {}),
    }


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _optional_env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def validate_pdf_upload(filename: str, pdf_bytes: bytes) -> int:
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="uploaded file must be a PDF")

    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="uploaded file must be a valid PDF")

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            page_count = int(document.page_count)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="uploaded file must be a valid PDF") from exc

    return page_count


async def read_pdf_upload(file: UploadFile) -> tuple[str, bytes]:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="uploaded file must be a PDF")

    chunks: list[bytes] = []
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return filename, b"".join(chunks)


def http_exception_for_timeout(exc: TimeoutError) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


def readiness_payload(service: object) -> dict:
    is_ready = True
    if hasattr(service, "is_ready"):
        is_ready = bool(service.is_ready())
    if not is_ready:
        raise HTTPException(status_code=503, detail={"status": "unavailable"})
    return {"status": "ok"}


def create_app(service: Optional[BankParserService] = None) -> FastAPI:
    app = FastAPI()
    app.state.bank_parser_service = service
    app.state.bank_parser_service_lock = threading.Lock()
    app.state.metrics_manager = get_metrics_manager()
    app.state.metrics_manager.start_background_samplers()
    default_config_path = Path(__file__).resolve().parent / "config.yaml"

    def get_service() -> BankParserService:
        existing = app.state.bank_parser_service
        if existing is not None:
            return existing

        with app.state.bank_parser_service_lock:
            existing = app.state.bank_parser_service
            if existing is not None:
                return existing

            created = BankParserService(
                config_path=os.getenv("BANK_PARSER_CONFIG", str(default_config_path)),
                layout_mode=os.getenv("BANK_PARSER_LAYOUT_MODE"),
                save_debug_images=_env_flag("BANK_PARSER_SAVE_DEBUG_IMAGES", impl.SAVE_DEBUG_IMAGES),
                parse_testing=_env_flag("BANK_PARSER_PARSE_TESTING", impl.PARSE_TESTING),
                enable_ocr_batching=_env_flag("BANK_PARSER_ENABLE_OCR_BATCHING", impl.ENABLE_OCR_BATCHING),
                ocr_batch_size=int(os.getenv("BANK_PARSER_OCR_BATCH_SIZE", str(impl.OCR_BATCH_SIZE))),
                pdf_render_dpi_override=_optional_env_int("BANK_PARSER_PDF_DPI"),
                layout_max_concurrency=_optional_env_int("BANK_PARSER_LAYOUT_MAX_CONCURRENCY"),
                ocr_pipeline_workers=_optional_env_int("BANK_PARSER_OCR_PIPELINE_WORKERS"),
                ocr_pipeline_queue_size=_optional_env_int("BANK_PARSER_OCR_PIPELINE_QUEUE_SIZE"),
                ocr_backend_mode=os.getenv("BANK_PARSER_OCR_BACKEND_MODE"),
                ocr_batch_drain_max_batch_size=_optional_env_int("BANK_PARSER_OCR_BATCH_DRAIN_MAX_BATCH_SIZE"),
                ocr_batch_drain_max_wait_seconds=_env_float(
                    "BANK_PARSER_OCR_BATCH_DRAIN_MAX_WAIT_SECONDS",
                    1.0,
                ),
                enable_page_ocr_retry=_env_flag("BANK_PARSER_ENABLE_PAGE_OCR_RETRY", impl.ENABLE_PAGE_OCR_RETRY),
                capture_raw_ocr_debug=_env_flag("BANK_PARSER_CAPTURE_RAW_OCR_DEBUG", impl.CAPTURE_RAW_OCR_DEBUG),
                page_ocr_retry_dpi=_optional_env_int("BANK_PARSER_PAGE_OCR_RETRY_DPI"),
                parser_pool_wait_timeout=_env_float("BANK_PARSER_POOL_WAIT_TIMEOUT", 30.0),
                backend_startup_timeout=_env_float("BANK_PARSER_BACKEND_STARTUP_TIMEOUT", 180.0),
                backend_retry_interval=_env_float("BANK_PARSER_BACKEND_RETRY_INTERVAL", 5.0),
            )
            app.state.bank_parser_service = created
            return created

    @app.post("/parse/bank-statement")
    async def parse_bank_statement(file: UploadFile = File(...)):
        filename, pdf_bytes = await read_pdf_upload(file)
        validate_pdf_upload(filename, pdf_bytes)
        metrics_manager = app.state.metrics_manager
        metrics_manager.track_request_started(len(pdf_bytes))
        started_at = time.perf_counter()

        try:
            result = await run_in_threadpool(
                lambda: get_service().parse_pdf_bytes(filename, pdf_bytes).to_payload()
            )
            metrics_manager.track_request_success(result)
            return result
        except TimeoutError as exc:
            metrics_manager.track_request_failure("timeout")
            raise http_exception_for_timeout(exc) from exc
        except Exception:
            metrics_manager.track_request_failure("exception")
            raise
        finally:
            metrics_manager.track_request_finished(time.perf_counter() - started_at)
            metrics_manager.schedule_push()

    @app.post("/v2/extract/bank-statement-evidence")
    async def extract_bank_statement_evidence(file: UploadFile = File(...)):
        """Evidence-only endpoint — returns raw GLM HTML + quality metadata.

        Per plan.md: eosin produces evidence only.  This endpoint runs PDF
        rendering → GLM OCR → quality evaluation and returns per-page raw
        HTML, quality scores, suspicious flags, and timing metadata.

        It does NOT run DataFrame assembly, column alignment, or any row
        repairs.  Almond owns all parser intelligence.
        """
        filename, pdf_bytes = await read_pdf_upload(file)
        validate_pdf_upload(filename, pdf_bytes)
        metrics_manager = app.state.metrics_manager
        metrics_manager.track_request_started(len(pdf_bytes))
        started_at = time.perf_counter()
        try:
            result = await run_in_threadpool(
                lambda: get_service().extract_glm_page_html_bytes(filename, pdf_bytes)
            )
            payload = evidence_payload_from_result(result)
            metrics_manager.track_request_success(payload)
            return payload
        except TimeoutError as exc:
            metrics_manager.track_request_failure("timeout")
            raise http_exception_for_timeout(exc) from exc
        except Exception:
            metrics_manager.track_request_failure("exception")
            raise
        finally:
            metrics_manager.track_request_finished(time.perf_counter() - started_at)
            metrics_manager.schedule_push()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        try:
            return readiness_payload(get_service())
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"status": "unavailable"}) from exc

    @app.get("/metrics")
    async def metrics():
        rendered = app.state.metrics_manager.render_metrics()
        return Response(content=rendered.payload, media_type=rendered.media_type)

    return app


app = create_app(service=None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("BANK_PARSER_PORT", "8090")),
    )
