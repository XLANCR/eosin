from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


MAX_CONCURRENCY = 35


@dataclass(frozen=True)
class PlannedRequest:
    request_index: int
    repetition: int
    pdf_path: Path
    artifact_pdf_path: str


@dataclass(frozen=True)
class RequestRecord:
    pdf_path: str
    page_count: int
    latency_seconds: float
    success: bool
    retry_count: int
    quality_artifact: str | None
    error: str | None
    service_timings: Mapping[str, object] | None = None
    ocr_metrics: Mapping[str, object] | None = None


@dataclass(frozen=True)
class GpuSample:
    utilization_percent: float
    memory_used_bytes: int | None


def clamp_concurrency(value: int) -> int:
    return max(1, min(MAX_CONCURRENCY, value))


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "sample_count": len(values),
        "mean": _rounded(statistics.fmean(values)) if values else None,
        "median": _rounded(statistics.median(values)) if values else None,
        "p95": _rounded(percentile(values, 0.95)),
        "peak": _rounded(max(values)) if values else None,
    }


def aggregate_records(
    records: Sequence[RequestRecord],
    *,
    elapsed_seconds: float,
    cold_request_seconds: float,
    gpu_samples: Sequence[GpuSample] = (),
) -> dict[str, object]:
    latencies = [record.latency_seconds for record in records]
    utilization = [sample.utilization_percent for sample in gpu_samples]
    memory = [
        sample.memory_used_bytes
        for sample in gpu_samples
        if sample.memory_used_bytes is not None
    ]
    total_pages = sum(record.page_count for record in records)
    successful = sum(record.success for record in records)
    artifacts = [record.quality_artifact for record in records if record.quality_artifact]
    request_records = [
        {
            "pdf_path": record.pdf_path,
            "page_count": record.page_count,
            "latency_seconds": record.latency_seconds,
            "success": record.success,
            "retry_count": record.retry_count,
            "quality_artifact": record.quality_artifact,
            "error": record.error,
            "service_timings": dict(record.service_timings or {}),
            "ocr_metrics": dict(record.ocr_metrics or {}),
        }
        for record in records
    ]
    return {
        "request_count": len(records),
        "successful_requests": successful,
        "total_pages": total_pages,
        "pages_per_second": _rounded(total_pages / elapsed_seconds) if elapsed_seconds > 0 else None,
        "success_rate": _rounded(successful / len(records)) if records else 0.0,
        "retry_count": sum(record.retry_count for record in records),
        "cold_request_seconds": _rounded(cold_request_seconds),
        "document_latency_seconds": {
            "median": _rounded(statistics.median(latencies)) if latencies else None,
            "p95": _rounded(percentile(latencies, 0.95)),
        },
        "gpu_utilization_percent": _distribution(utilization),
        "peak_gpu_memory_bytes": max(memory) if memory else None,
        "quality_artifact_references": artifacts,
        "request_records": request_records,
        "errors": [record.error for record in records if record.error],
    }


def resolve_modal_instance(app_name: str, class_name: str):
    import modal

    remote_class = modal.Cls.from_name(app_name, class_name)
    return remote_class()


def _retry_count(payload: Mapping[str, object]) -> int:
    metrics = payload.get("ocr_metrics")
    metric_values = metrics if isinstance(metrics, Mapping) else {}
    explicit = payload.get(
        "retry_count",
        metric_values.get("retry_count", metric_values.get("retries")),
    )
    if explicit is not None:
        return int(explicit)
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return 0
    return sum(bool(page.get("retried")) for page in pages if isinstance(page, Mapping))


def _artifact_payload(
    item: PlannedRequest,
    response: Mapping[str, object],
) -> dict[str, object]:
    pages = response.get("pages")
    page_rows = pages if isinstance(pages, list) else []
    page_ocr = [
        {
            "page_index": int(page.get("page_number", index + 1)) - 1,
            "raw_html": str(page.get("raw_html", "")),
        }
        for index, page in enumerate(page_rows)
        if isinstance(page, Mapping)
    ]
    return {
        "request": {"pdf_path": item.artifact_pdf_path},
        "payload": {
            "debug": {"page_ocr": page_ocr},
            "direct_ocr_response": dict(response),
        },
    }


def _write_quality_artifact(
    artifact_dir: Path,
    item: PlannedRequest,
    response: Mapping[str, object],
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(
        character if character.isalnum() else "_"
        for character in item.pdf_path.stem
    )
    path = artifact_dir / f"{item.request_index:04d}_r{item.repetition:02d}_{safe_stem}.json"
    path.write_text(json.dumps(_artifact_payload(item, response), indent=2), encoding="utf-8")
    return path


def _failed_record(
    item: PlannedRequest,
    started_at: float,
    clock: Callable[[], float],
    error: BaseException,
) -> RequestRecord:
    return RequestRecord(
        str(item.pdf_path),
        0,
        _rounded(clock() - started_at) or 0.0,
        False,
        0,
        None,
        f"{type(error).__name__}: {error}",
    )


def invoke_request(
    instance,
    item: PlannedRequest,
    artifact_dir: Path,
    *,
    request_timeout_seconds: float = 600.0,
    document_type: str | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RequestRecord:
    started_at = clock()
    function_call = None
    try:
        arguments = (item.pdf_path.name, item.pdf_path.read_bytes())
        if document_type:
            function_call = instance.extract_document_evidence.spawn(
                *arguments, document_type
            )
        else:
            function_call = instance.extract_glm_page_html.spawn(*arguments)
        response = function_call.get(timeout=request_timeout_seconds)
        if not isinstance(response, Mapping):
            raise TypeError("extract_glm_page_html must return a mapping")
        artifact = _write_quality_artifact(artifact_dir, item, response)
        pages = response.get("pages")
        page_count = int(
            response.get("page_count", len(pages) if isinstance(pages, list) else 0)
        )
        return RequestRecord(
            str(item.pdf_path),
            page_count,
            _rounded(clock() - started_at) or 0.0,
            True,
            _retry_count(response),
            str(artifact),
            None,
            response.get("timings") if isinstance(response.get("timings"), Mapping) else None,
            response.get("ocr_metrics") if isinstance(response.get("ocr_metrics"), Mapping) else None,
        )
    except TimeoutError as error:
        if function_call is not None:
            function_call.cancel(terminate_containers=False)
        return _failed_record(item, started_at, clock, error)
    except Exception as error:
        return _failed_record(item, started_at, clock, error)


def _gpu_samples(payload: object) -> tuple[GpuSample, ...]:
    if isinstance(payload, Mapping):
        raw_samples = payload.get("samples", [payload])
    else:
        raw_samples = payload
    if not isinstance(raw_samples, list):
        return ()
    samples: list[GpuSample] = []
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, Mapping):
            continue
        utilization = raw_sample.get("gpu_utilization_percent", raw_sample.get("gpu_utilization"))
        memory = raw_sample.get("gpu_memory_used_bytes", raw_sample.get("gpu_memory_bytes"))
        if utilization is not None:
            samples.append(GpuSample(float(utilization), int(memory) if memory is not None else None))
    return tuple(samples)


def _metrics_method(instance, metrics_method_name: str | None):
    if not metrics_method_name:
        return None
    if not metrics_method_name.isidentifier() or metrics_method_name.startswith("_"):
        raise ValueError("metrics method must be a public Python identifier")
    return getattr(instance, metrics_method_name)


def reset_gpu_metrics(instance, metrics_method_name: str | None) -> None:
    method = _metrics_method(instance, metrics_method_name)
    if method is not None:
        method.remote(reset=True)


def fetch_gpu_samples(instance, metrics_method_name: str | None) -> tuple[GpuSample, ...]:
    method = _metrics_method(instance, metrics_method_name)
    if method is None:
        return ()
    payload = method.remote()
    return _gpu_samples(payload)


def run_benchmark(
    instance,
    plan: Sequence[PlannedRequest],
    *,
    concurrency: int,
    artifact_dir: Path,
    request_timeout_seconds: float = 600.0,
    metrics_method_name: str | None = None,
    document_type: str | None = None,
    invoke: Callable[..., RequestRecord] = invoke_request,
    clock: Callable[[], float] = time.monotonic,
    executor_factory: Callable[..., object] = ThreadPoolExecutor,
) -> dict[str, object]:
    fixed_plan = tuple(plan)
    if not fixed_plan:
        raise ValueError("benchmark plan must contain at least one PDF")
    cold_started_at = clock()
    cold_kwargs = {"request_timeout_seconds": request_timeout_seconds}
    if document_type:
        cold_kwargs["document_type"] = document_type
    cold_record = invoke(instance, fixed_plan[0], artifact_dir / "cold", **cold_kwargs)
    cold_seconds = clock() - cold_started_at
    bounded_concurrency = clamp_concurrency(concurrency)
    if not cold_record.success:
        summary = aggregate_records(
            (),
            elapsed_seconds=0.0,
            cold_request_seconds=cold_seconds,
        )
        return {
            **summary,
            "configured_concurrency": bounded_concurrency,
            "warmed_elapsed_seconds": 0.0,
            "cold_request_success": False,
            "cold_quality_artifact_reference": None,
            "cold_request_record": aggregate_records(
                (cold_record,), elapsed_seconds=cold_seconds, cold_request_seconds=cold_seconds
            )["request_records"][0],
        }
    reset_gpu_metrics(instance, metrics_method_name)
    warmed_started_at = clock()
    executor = executor_factory(max_workers=bounded_concurrency)
    try:
        warmed_records = tuple(
            executor.map(
                lambda item: invoke(instance, item, artifact_dir, **cold_kwargs),
                fixed_plan,
            )
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    warmed_seconds = clock() - warmed_started_at
    summary = aggregate_records(
        warmed_records,
        elapsed_seconds=warmed_seconds,
        cold_request_seconds=cold_seconds,
        gpu_samples=fetch_gpu_samples(instance, metrics_method_name),
    )
    return {
        **summary,
        "configured_concurrency": bounded_concurrency,
        "warmed_elapsed_seconds": _rounded(warmed_seconds),
        "cold_request_success": cold_record.success,
        "cold_quality_artifact_reference": cold_record.quality_artifact,
        "cold_request_record": aggregate_records(
            (cold_record,), elapsed_seconds=cold_seconds, cold_request_seconds=cold_seconds
        )["request_records"][0],
    }


def build_request_plan(
    *,
    pdf_paths: Sequence[Path] | None,
    corpus_root: Path | None,
    repetitions: int,
) -> tuple[PlannedRequest, ...]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if pdf_paths:
        if corpus_root is None:
            raise ValueError("explicit PDF requires corpus root")
        paths = [Path(path) for path in pdf_paths]
        resolved_root = corpus_root.resolve()
        try:
            artifact_paths = [
                str(path.resolve().relative_to(resolved_root)) for path in paths
            ]
        except ValueError as error:
            raise ValueError("explicit PDF is outside corpus root") from error
    elif corpus_root is not None:
        paths = sorted(
            path
            for path in corpus_root.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
        artifact_paths = [path.relative_to(corpus_root).as_posix() for path in paths]
    else:
        paths = []
        artifact_paths = []
    if not paths:
        raise ValueError("no PDF files found")
    return tuple(
        PlannedRequest(index, repetition, path, artifact_path)
        for index, (repetition, path, artifact_path) in enumerate(
            (
                (repetition, path, artifact_path)
                for repetition in range(1, repetitions + 1)
                for path, artifact_path in zip(paths, artifact_paths)
            ),
            start=1,
        )
    )


def _positive_float(raw_value: str) -> float:
    value = float(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark warmed GLM-OCR through direct Modal RPC."
    )
    parser.add_argument("--app-name", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--pdf", dest="pdf_paths", type=Path, nargs="+")
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--request-timeout-seconds",
        type=_positive_float,
        default=600.0,
    )
    parser.add_argument("--metrics-method")
    parser.add_argument(
        "--document-type",
        choices=("bank_statement", "invoice", "receipt"),
        help="Use the typed document-evidence RPC with the selected OCR task.",
    )
    args = parser.parse_args(argv)
    if args.pdf_paths and args.corpus_root is None:
        parser.error("--pdf requires --corpus-root")
    if not args.pdf_paths and args.corpus_root is None:
        parser.error("one of --pdf or --corpus-root is required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_request_plan(
        pdf_paths=args.pdf_paths,
        corpus_root=args.corpus_root,
        repetitions=args.repetitions,
    )
    instance = resolve_modal_instance(args.app_name, args.class_name)
    artifact_dir = args.output_json.parent / f"{args.output_json.stem}_quality_artifacts"
    summary = run_benchmark(
        instance,
        plan,
        concurrency=args.concurrency,
        artifact_dir=artifact_dir,
        request_timeout_seconds=args.request_timeout_seconds,
        metrics_method_name=args.metrics_method,
        document_type=args.document_type,
    )
    output = {
        **summary,
        "app_name": args.app_name,
        "class_name": args.class_name,
        "repetitions": args.repetitions,
        "document_type": args.document_type or "bank_statement",
        "request_plan": [item.artifact_pdf_path for item in plan],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if summary["success_rate"] == 1.0 and summary["cold_request_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
