from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_ENDPOINT = os.getenv("EOSIN_PARSER_BASE_URL", "").rstrip("/")
DEFAULT_CORPUS_DIR = Path("bank statements")
DEFAULT_OUTPUT_ROOT = Path("load-test-results")


@dataclass(frozen=True)
class PlannedRequest:
    request_index: int
    corpus_cycle: int
    pdf_path: str
    bank_name: str
    file_size_bytes: int


@dataclass(frozen=True)
class RequestResult:
    request_index: int
    corpus_cycle: int
    pdf_path: str
    bank_name: str
    file_size_bytes: int
    started_at_utc: str
    completed_at_utc: str
    latency_seconds: float
    status_code: int | None
    ok: bool
    error_type: str | None
    error_message: str | None
    response_bytes: int
    rows_returned: int | None
    columns_returned: int | None
    page_count: int | None
    pages_with_tables_count: int | None
    server_total_seconds: float | None
    server_render_seconds: float | None
    server_layout_seconds: float | None
    server_ocr_seconds: float | None
    server_parse_seconds: float | None
    server_ocr_task_count: float | None
    server_ocr_queue_wait_mean: float | None
    server_ocr_queue_wait_max: float | None
    server_ocr_request_mean: float | None
    server_ocr_request_max: float | None
    response_json_path: str | None
    dataframe_markdown_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hammer the bank parser endpoint with many PDFs.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Base URL for the parser service.")
    parser.add_argument("--corpus-dir", default=str(DEFAULT_CORPUS_DIR), help="Directory containing PDF corpus.")
    parser.add_argument("--target-requests", type=int, default=100, help="Total requests to send.")
    parser.add_argument("--concurrency", type=int, default=32, help="Client-side concurrent requests.")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-request timeout in seconds.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory for saved reports.")
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(UTC)


def isoformat(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def discover_pdfs(corpus_dir: Path) -> list[Path]:
    return sorted(path for path in corpus_dir.rglob("*.pdf") if path.is_file())


def build_request_plan(corpus_dir: Path, pdf_paths: list[Path], target_requests: int) -> list[PlannedRequest]:
    plan: list[PlannedRequest] = []
    for request_index, pdf_path in zip(range(1, target_requests + 1), itertools.cycle(pdf_paths)):
        relative = pdf_path.relative_to(corpus_dir)
        bank_name = relative.parts[0] if relative.parts else pdf_path.parent.name
        plan.append(
            PlannedRequest(
                request_index=request_index,
                corpus_cycle=(request_index - 1) // len(pdf_paths) + 1,
                pdf_path=str(relative),
                bank_name=bank_name,
                file_size_bytes=pdf_path.stat().st_size,
            )
        )
    return plan


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]

    values = sorted(values)
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[int(position)]
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def safe_git_value(args: list[str]) -> str | None:
    try:
        output = subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
        return output or None
    except Exception:
        return None


def make_results_dir(output_root: Path) -> Path:
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    for attempt in range(1000):
        suffix = "" if attempt == 0 else f"-{attempt:03d}"
        results_dir = output_root / f"{timestamp}{suffix}"
        try:
            results_dir.mkdir(parents=True, exist_ok=False)
            return results_dir
        except FileExistsError:
            continue
    raise RuntimeError(f"Unable to allocate unique results directory under {output_root}")


def safe_artifact_stem(planned: PlannedRequest) -> str:
    raw = "__".join(
        [
            f"{planned.request_index:04d}",
            planned.bank_name,
            Path(planned.pdf_path).stem,
        ]
    )
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in raw)[:180]


def escape_markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    if len(dataframe.columns) == 0:
        return "_No columns returned._"

    headers = [escape_markdown_cell(column) for column in dataframe.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in dataframe.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(escape_markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def iter_rows(results: Iterable[RequestResult]) -> Iterable[dict[str, Any]]:
    for result in results:
        yield asdict(result)


def send_request(
    endpoint: str,
    timeout: int,
    corpus_dir: Path,
    server_responses_dir: Path,
    dataframes_dir: Path,
    planned: PlannedRequest,
) -> RequestResult:
    pdf_path = corpus_dir / planned.pdf_path
    started_at = utc_now()
    start_monotonic = time.perf_counter()

    try:
        with pdf_path.open("rb") as handle:
            response = requests.post(
                f"{endpoint}/parse/bank-statement",
                files={"file": (pdf_path.name, handle, "application/pdf")},
                timeout=timeout,
            )

        completed_at = utc_now()
        latency = time.perf_counter() - start_monotonic
        response_bytes = len(response.content or b"")

        payload: dict[str, Any] | None = None
        rows_returned = None
        columns_returned = None
        page_count = None
        pages_with_tables_count = None
        server_total_seconds = None
        server_render_seconds = None
        server_layout_seconds = None
        server_ocr_seconds = None
        server_parse_seconds = None
        server_ocr_task_count = None
        server_ocr_queue_wait_mean = None
        server_ocr_queue_wait_max = None
        server_ocr_request_mean = None
        server_ocr_request_max = None
        error_type = None
        error_message = None
        response_json_path = None
        dataframe_markdown_path = None

        try:
            payload = response.json()
        except Exception:
            payload = None

        artifact_stem = safe_artifact_stem(planned)
        response_artifact = server_responses_dir / f"{artifact_stem}.json"
        response_record: dict[str, Any] = {
            "request": asdict(planned),
            "status_code": response.status_code,
            "ok": response.ok,
            "latency_seconds": round(latency, 6),
            "completed_at_utc": isoformat(completed_at),
            "payload": payload,
        }
        if payload is None:
            response_record["response_text"] = response.text
        with response_artifact.open("w", encoding="utf-8") as handle:
            json.dump(response_record, handle, indent=2, ensure_ascii=True)
        response_json_path = str(response_artifact)

        if isinstance(payload, dict):
            rows = payload.get("rows")
            columns = payload.get("columns")
            page_count = payload.get("page_count")
            pages_with_tables = payload.get("pages_with_tables")
            timings = payload.get("timings")
            debug = payload.get("debug")
            rows_returned = len(rows) if isinstance(rows, list) else None
            columns_returned = len(columns) if isinstance(columns, list) else None
            pages_with_tables_count = len(pages_with_tables) if isinstance(pages_with_tables, list) else None
            if response.ok and isinstance(rows, list):
                dataframe = pd.DataFrame(
                    rows,
                    columns=columns if isinstance(columns, list) and columns else None,
                )
                markdown_artifact = dataframes_dir / f"{artifact_stem}.md"
                with markdown_artifact.open("w", encoding="utf-8") as handle:
                    handle.write(f"# Request {planned.request_index}: {planned.pdf_path}\n\n")
                    handle.write(f"- Bank: {planned.bank_name}\n")
                    handle.write(f"- Status: {response.status_code}\n")
                    handle.write(f"- Latency seconds: {round(latency, 6)}\n")
                    handle.write(f"- Rows: {len(dataframe)}\n")
                    handle.write(f"- Columns: {len(dataframe.columns)}\n\n")
                    handle.write(dataframe_to_markdown(dataframe))
                    handle.write("\n")
                dataframe_markdown_path = str(markdown_artifact)
            if isinstance(timings, dict):
                server_total_seconds = timings.get("total")
                server_render_seconds = timings.get("render_pages")
                server_layout_seconds = timings.get("layout_detection")
                server_ocr_seconds = timings.get("ocr_tables")
                server_parse_seconds = timings.get("parse_html_tables")
            if isinstance(debug, dict):
                ocr_metrics = debug.get("ocr_metrics")
                if isinstance(ocr_metrics, dict):
                    server_ocr_task_count = ocr_metrics.get("task_count")
                    server_ocr_queue_wait_mean = ocr_metrics.get("queue_wait_mean")
                    server_ocr_queue_wait_max = ocr_metrics.get("queue_wait_max")
                    server_ocr_request_mean = ocr_metrics.get("request_mean")
                    server_ocr_request_max = ocr_metrics.get("request_max")

        ok = response.ok
        if not ok:
            error_type = "http_error"
            error_message = (response.text or "")[:2000]

        return RequestResult(
            request_index=planned.request_index,
            corpus_cycle=planned.corpus_cycle,
            pdf_path=planned.pdf_path,
            bank_name=planned.bank_name,
            file_size_bytes=planned.file_size_bytes,
            started_at_utc=isoformat(started_at),
            completed_at_utc=isoformat(completed_at),
            latency_seconds=round(latency, 6),
            status_code=response.status_code,
            ok=ok,
            error_type=error_type,
            error_message=error_message,
            response_bytes=response_bytes,
            rows_returned=rows_returned,
            columns_returned=columns_returned,
            page_count=page_count if isinstance(page_count, int) else None,
            pages_with_tables_count=pages_with_tables_count,
            server_total_seconds=server_total_seconds if isinstance(server_total_seconds, (int, float)) else None,
            server_render_seconds=server_render_seconds if isinstance(server_render_seconds, (int, float)) else None,
            server_layout_seconds=server_layout_seconds if isinstance(server_layout_seconds, (int, float)) else None,
            server_ocr_seconds=server_ocr_seconds if isinstance(server_ocr_seconds, (int, float)) else None,
            server_parse_seconds=server_parse_seconds if isinstance(server_parse_seconds, (int, float)) else None,
            server_ocr_task_count=server_ocr_task_count if isinstance(server_ocr_task_count, (int, float)) else None,
            server_ocr_queue_wait_mean=server_ocr_queue_wait_mean if isinstance(server_ocr_queue_wait_mean, (int, float)) else None,
            server_ocr_queue_wait_max=server_ocr_queue_wait_max if isinstance(server_ocr_queue_wait_max, (int, float)) else None,
            server_ocr_request_mean=server_ocr_request_mean if isinstance(server_ocr_request_mean, (int, float)) else None,
            server_ocr_request_max=server_ocr_request_max if isinstance(server_ocr_request_max, (int, float)) else None,
            response_json_path=response_json_path,
            dataframe_markdown_path=dataframe_markdown_path,
        )
    except Exception as exc:
        completed_at = utc_now()
        latency = time.perf_counter() - start_monotonic
        return RequestResult(
            request_index=planned.request_index,
            corpus_cycle=planned.corpus_cycle,
            pdf_path=planned.pdf_path,
            bank_name=planned.bank_name,
            file_size_bytes=planned.file_size_bytes,
            started_at_utc=isoformat(started_at),
            completed_at_utc=isoformat(completed_at),
            latency_seconds=round(latency, 6),
            status_code=None,
            ok=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
            response_bytes=0,
            rows_returned=None,
            columns_returned=None,
            page_count=None,
            pages_with_tables_count=None,
            server_total_seconds=None,
            server_render_seconds=None,
            server_layout_seconds=None,
            server_ocr_seconds=None,
            server_parse_seconds=None,
            server_ocr_task_count=None,
            server_ocr_queue_wait_mean=None,
            server_ocr_queue_wait_max=None,
            server_ocr_request_mean=None,
            server_ocr_request_max=None,
            response_json_path=None,
            dataframe_markdown_path=None,
        )


def summarize_results(
    *,
    endpoint: str,
    corpus_dir: Path,
    plan: list[PlannedRequest],
    results: list[RequestResult],
    started_at: datetime,
    completed_at: datetime,
    concurrency: int,
    timeout: int,
) -> dict[str, Any]:
    latencies = [result.latency_seconds for result in results]
    ok_results = [result for result in results if result.ok]
    failed_results = [result for result in results if not result.ok]
    duration_seconds = (completed_at - started_at).total_seconds()

    status_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    bank_counts: dict[str, dict[str, int]] = {}

    for result in results:
        status_key = str(result.status_code) if result.status_code is not None else "exception"
        status_counts[status_key] = status_counts.get(status_key, 0) + 1
        if result.error_type:
            error_counts[result.error_type] = error_counts.get(result.error_type, 0) + 1

        bank_summary = bank_counts.setdefault(result.bank_name, {"total": 0, "ok": 0, "failed": 0})
        bank_summary["total"] += 1
        if result.ok:
            bank_summary["ok"] += 1
        else:
            bank_summary["failed"] += 1

    slowest = sorted(results, key=lambda item: item.latency_seconds, reverse=True)[:10]
    fastest = sorted(results, key=lambda item: item.latency_seconds)[:10]

    rows_total = sum(result.rows_returned or 0 for result in ok_results)
    pages_total = sum(result.page_count or 0 for result in ok_results)
    bytes_uploaded_total = sum(item.file_size_bytes for item in plan)
    bytes_downloaded_total = sum(result.response_bytes for result in results)
    successful_with_server_total = [item.server_total_seconds for item in ok_results if item.server_total_seconds is not None]
    successful_with_render = [item.server_render_seconds for item in ok_results if item.server_render_seconds is not None]
    successful_with_layout = [item.server_layout_seconds for item in ok_results if item.server_layout_seconds is not None]
    successful_with_ocr = [item.server_ocr_seconds for item in ok_results if item.server_ocr_seconds is not None]
    successful_with_parse = [item.server_parse_seconds for item in ok_results if item.server_parse_seconds is not None]
    successful_with_ocr_queue_mean = [item.server_ocr_queue_wait_mean for item in ok_results if item.server_ocr_queue_wait_mean is not None]
    successful_with_ocr_queue_max = [item.server_ocr_queue_wait_max for item in ok_results if item.server_ocr_queue_wait_max is not None]
    successful_with_ocr_request_mean = [item.server_ocr_request_mean for item in ok_results if item.server_ocr_request_mean is not None]
    successful_with_ocr_request_max = [item.server_ocr_request_max for item in ok_results if item.server_ocr_request_max is not None]
    successful_with_ocr_task_count = [item.server_ocr_task_count for item in ok_results if item.server_ocr_task_count is not None]
    end_to_end_queue_gap = [
        item.latency_seconds - item.server_total_seconds
        for item in ok_results
        if item.server_total_seconds is not None
    ]

    return {
        "test_started_at_utc": isoformat(started_at),
        "test_completed_at_utc": isoformat(completed_at),
        "test_duration_seconds": round(duration_seconds, 6),
        "endpoint": endpoint,
        "corpus_dir": str(corpus_dir),
        "unique_pdfs_discovered": len({item.pdf_path for item in plan}),
        "target_requests": len(plan),
        "actual_results": len(results),
        "configured_concurrency": concurrency,
        "request_timeout_seconds": timeout,
        "successful_requests": len(ok_results),
        "failed_requests": len(failed_results),
        "success_rate": round((len(ok_results) / len(results)) * 100, 3) if results else 0.0,
        "requests_per_second": round(len(results) / duration_seconds, 6) if duration_seconds > 0 else None,
        "rows_returned_total": rows_total,
        "pages_processed_total": pages_total,
        "pages_per_second": round(pages_total / duration_seconds, 6) if duration_seconds > 0 else None,
        "bytes_uploaded_total": bytes_uploaded_total,
        "bytes_downloaded_total": bytes_downloaded_total,
        "server_responses_dir": None,
        "dataframes_dir": None,
        "latency_seconds": {
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "server_total_seconds": {
            "mean": statistics.fmean(successful_with_server_total) if successful_with_server_total else None,
            "p50": percentile(successful_with_server_total, 0.50),
            "p90": percentile(successful_with_server_total, 0.90),
            "p95": percentile(successful_with_server_total, 0.95),
            "max": max(successful_with_server_total) if successful_with_server_total else None,
        },
        "stage_seconds": {
            "render_mean": statistics.fmean(successful_with_render) if successful_with_render else None,
            "layout_mean": statistics.fmean(successful_with_layout) if successful_with_layout else None,
            "ocr_mean": statistics.fmean(successful_with_ocr) if successful_with_ocr else None,
            "parse_mean": statistics.fmean(successful_with_parse) if successful_with_parse else None,
        },
        "ocr_pipeline": {
            "task_count_mean": statistics.fmean(successful_with_ocr_task_count) if successful_with_ocr_task_count else None,
            "queue_wait_mean_mean": statistics.fmean(successful_with_ocr_queue_mean) if successful_with_ocr_queue_mean else None,
            "queue_wait_max_max": max(successful_with_ocr_queue_max) if successful_with_ocr_queue_max else None,
            "request_mean_mean": statistics.fmean(successful_with_ocr_request_mean) if successful_with_ocr_request_mean else None,
            "request_max_max": max(successful_with_ocr_request_max) if successful_with_ocr_request_max else None,
        },
        "client_minus_server_seconds": {
            "mean": statistics.fmean(end_to_end_queue_gap) if end_to_end_queue_gap else None,
            "p50": percentile(end_to_end_queue_gap, 0.50),
            "p90": percentile(end_to_end_queue_gap, 0.90),
            "max": max(end_to_end_queue_gap) if end_to_end_queue_gap else None,
        },
        "status_counts": status_counts,
        "error_counts": error_counts,
        "bank_counts": bank_counts,
        "slowest_requests": [asdict(item) for item in slowest],
        "fastest_requests": [asdict(item) for item in fastest],
        "git": {
            "branch": safe_git_value(["git", "branch", "--show-current"]),
            "commit": safe_git_value(["git", "rev-parse", "HEAD"]),
        },
    }


def write_results(results_dir: Path, plan: list[PlannedRequest], results: list[RequestResult], summary: dict[str, Any]) -> None:
    requests_csv = results_dir / "requests.csv"
    plan_json = results_dir / "plan.json"
    raw_jsonl = results_dir / "requests.jsonl"
    summary_json = results_dir / "summary.json"
    summary_txt = results_dir / "summary.txt"

    with plan_json.open("w", encoding="utf-8") as handle:
        json.dump([asdict(item) for item in plan], handle, indent=2)

    with raw_jsonl.open("w", encoding="utf-8") as handle:
        for row in iter_rows(results):
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    with requests_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()) if results else [])
        if results:
            writer.writeheader()
            writer.writerows(iter_rows(results))

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    summary_lines = [
        f"Endpoint: {summary['endpoint']}",
        f"Duration seconds: {summary['test_duration_seconds']}",
        f"Requests sent: {summary['actual_results']}",
        f"Success / failure: {summary['successful_requests']} / {summary['failed_requests']}",
        f"Requests/sec: {summary['requests_per_second']}",
        f"Pages processed total: {summary['pages_processed_total']}",
        f"Pages/sec: {summary['pages_per_second']}",
        f"Latency p50/p95/p99: {summary['latency_seconds']['p50']} / {summary['latency_seconds']['p95']} / {summary['latency_seconds']['p99']}",
        f"Server total mean/p95/max: {summary['server_total_seconds']['mean']} / {summary['server_total_seconds']['p95']} / {summary['server_total_seconds']['max']}",
        f"Stage means render/layout/ocr/parse: {summary['stage_seconds']['render_mean']} / {summary['stage_seconds']['layout_mean']} / {summary['stage_seconds']['ocr_mean']} / {summary['stage_seconds']['parse_mean']}",
        f"OCR queue mean(max-mean)/request mean(max-mean): {summary['ocr_pipeline']['queue_wait_mean_mean']} ({summary['ocr_pipeline']['queue_wait_max_max']}) / {summary['ocr_pipeline']['request_mean_mean']} ({summary['ocr_pipeline']['request_max_max']})",
        f"Client minus server mean/p90/max: {summary['client_minus_server_seconds']['mean']} / {summary['client_minus_server_seconds']['p90']} / {summary['client_minus_server_seconds']['max']}",
        f"Status counts: {summary['status_counts']}",
        f"Error counts: {summary['error_counts']}",
        f"Rows returned total: {summary['rows_returned_total']}",
        f"Bytes uploaded total: {summary['bytes_uploaded_total']}",
        f"Bytes downloaded total: {summary['bytes_downloaded_total']}",
        f"Server responses dir: {summary['server_responses_dir']}",
        f"DataFrames dir: {summary['dataframes_dir']}",
        f"Git branch: {summary['git']['branch']}",
        f"Git commit: {summary['git']['commit']}",
    ]
    with summary_txt.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")


def main() -> None:
    args = parse_args()
    if not args.endpoint:
        raise SystemExit("Missing --endpoint and EOSIN_PARSER_BASE_URL is not set.")

    endpoint = args.endpoint.rstrip("/")
    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.exists():
        raise SystemExit(f"Corpus directory does not exist: {corpus_dir}")

    pdf_paths = discover_pdfs(corpus_dir)
    if not pdf_paths:
        raise SystemExit(f"No PDFs found under: {corpus_dir}")

    plan = build_request_plan(corpus_dir, pdf_paths, args.target_requests)
    output_root = Path(args.output_root)
    results_dir = make_results_dir(output_root)
    server_responses_dir = results_dir / "server-responses"
    dataframes_dir = results_dir / "dataframes"
    server_responses_dir.mkdir(parents=True, exist_ok=True)
    dataframes_dir.mkdir(parents=True, exist_ok=True)

    print(f"Endpoint: {endpoint}")
    print(f"Corpus dir: {corpus_dir}")
    print(f"Unique PDFs: {len(pdf_paths)}")
    print(f"Target requests: {len(plan)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Results dir: {results_dir}")

    started_at = utc_now()
    completed_counter = 0
    completed_lock = threading.Lock()
    results: list[RequestResult] = []

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        future_map = {
            executor.submit(
                send_request,
                endpoint,
                args.timeout,
                corpus_dir,
                server_responses_dir,
                dataframes_dir,
                planned,
            ): planned
            for planned in plan
        }
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            with completed_lock:
                completed_counter += 1
                print(
                    f"[{completed_counter}/{len(plan)}] "
                    f"#{result.request_index} {result.bank_name} {result.pdf_path} "
                    f"status={result.status_code} ok={result.ok} latency={result.latency_seconds:.3f}s"
                )

    completed_at = utc_now()
    results.sort(key=lambda item: item.request_index)
    summary = summarize_results(
        endpoint=endpoint,
        corpus_dir=corpus_dir,
        plan=plan,
        results=results,
        started_at=started_at,
        completed_at=completed_at,
        concurrency=args.concurrency,
        timeout=args.timeout,
    )
    summary["server_responses_dir"] = str(server_responses_dir)
    summary["dataframes_dir"] = str(dataframes_dir)
    write_results(results_dir, plan, results, summary)

    print("Summary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved results to: {results_dir}")


if __name__ == "__main__":
    main()
