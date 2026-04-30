from __future__ import annotations

import os
import queue
import socket
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

import requests

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
except Exception:  # pragma: no cover - local env fallback
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    class _MetricHandle:
        def inc(self, amount: float = 1.0) -> None:
            return None

        def dec(self, amount: float = 1.0) -> None:
            return None

        def set(self, value: float) -> None:
            return None

        def observe(self, value: float) -> None:
            return None

    class _Metric:
        def labels(self, **_: str) -> _MetricHandle:
            return _MetricHandle()

        def inc(self, amount: float = 1.0) -> None:
            return None

        def dec(self, amount: float = 1.0) -> None:
            return None

        def set(self, value: float) -> None:
            return None

        def observe(self, value: float) -> None:
            return None

    def Counter(*args, **kwargs):  # type: ignore[misc]
        return _Metric()

    def Gauge(*args, **kwargs):  # type: ignore[misc]
        return _Metric()

    def Histogram(*args, **kwargs):  # type: ignore[misc]
        return _Metric()

    def generate_latest() -> bytes:
        return b""

try:
    import pynvml
except Exception:  # pragma: no cover - optional at runtime
    pynvml = None


REQUESTS_TOTAL = Counter("eosin_parse_requests_total", "Total number of PDF parse requests handled by the API.", labelnames=("status",))
REQUEST_FAILURES_TOTAL = Counter("eosin_parse_failures_total", "Total number of failed PDF parse requests.", labelnames=("reason",))
REQUEST_INFLIGHT = Gauge("eosin_parse_inflight_requests", "Number of PDF parse requests currently being processed.")
REQUEST_DURATION_SECONDS = Histogram("eosin_parse_duration_seconds", "End-to-end parser API request duration.", buckets=(1, 2.5, 5, 10, 20, 40, 80, 160, 320, 640, 1280))
PDF_BYTES = Histogram("eosin_pdf_bytes", "Uploaded PDF size in bytes.", buckets=(2e5, 5e5, 1e6, 3e6, 5e6, 1e7, 2e7, 5e7, 1e8))
PDF_PAGE_COUNT = Histogram("eosin_pdf_page_count", "Rendered page count per PDF.", buckets=(1, 2, 4, 8, 16, 32, 64, 128))
TABLE_ROW_COUNT = Histogram("eosin_table_row_count", "Extracted bank statement rows per PDF.", buckets=(0, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000))
PIPELINE_STAGE_SECONDS = Histogram("eosin_pipeline_stage_seconds", "Per-stage parser timings reported by the pipeline.", labelnames=("stage",), buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160, 320, 640))
OCR_TASK_COUNT = Histogram("eosin_ocr_task_count", "Number of OCR tasks emitted by one parser request.", buckets=(1, 2, 4, 8, 16, 32, 64, 128))
OCR_QUEUE_WAIT_SECONDS = Histogram("eosin_ocr_queue_wait_seconds", "Mean or max OCR queue wait per parser request.", labelnames=("aggregation",), buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160))
OCR_BUILD_REQUEST_SECONDS = Histogram("eosin_ocr_build_request_seconds", "Mean or max OCR request-build time per parser request.", labelnames=("aggregation",), buckets=(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
OCR_REQUEST_SECONDS = Histogram("eosin_ocr_request_seconds", "Mean or max OCR network/inference request time per parser request.", labelnames=("aggregation",), buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160, 320))
OCR_TOTAL_SECONDS = Histogram("eosin_ocr_total_seconds", "Mean or max total OCR task time per parser request.", labelnames=("aggregation",), buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 40, 80, 160, 320))
OCR_QUEUE_SIZE = Histogram("eosin_ocr_queue_size_at_submit", "Largest OCR dispatcher queue depth seen while submitting tasks for one parser request.", buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096))
VLLM_METRICS_PROXY_UP = Gauge("eosin_vllm_metrics_proxy_up", "Whether the API successfully fetched the local vLLM /metrics endpoint on the last scrape.")
VLLM_METRICS_SCRAPE_DURATION_SECONDS = Gauge("eosin_vllm_metrics_scrape_duration_seconds", "Wall-clock duration of the last local vLLM metrics scrape.")
METRICS_PUSH_SUCCESS_TOTAL = Counter("eosin_metrics_push_total", "Total metric snapshot push attempts.", labelnames=("status",))
GPU_METRICS_AVAILABLE = Gauge("eosin_gpu_metrics_available", "Whether NVML-based GPU metrics sampling is available in this container.")
GPU_UTILIZATION_RATIO = Gauge("eosin_gpu_utilization_ratio", "Current GPU utilization as a ratio in the range [0, 1].", labelnames=("gpu_index",))
GPU_MEMORY_USED_BYTES = Gauge("eosin_gpu_memory_used_bytes", "Current GPU memory used in bytes.", labelnames=("gpu_index",))
GPU_MEMORY_TOTAL_BYTES = Gauge("eosin_gpu_memory_total_bytes", "Total GPU memory in bytes.", labelnames=("gpu_index",))
GPU_TEMPERATURE_CELSIUS = Gauge("eosin_gpu_temperature_celsius", "Current GPU temperature in Celsius.", labelnames=("gpu_index",))
GPU_POWER_USAGE_WATTS = Gauge("eosin_gpu_power_usage_watts", "Current GPU power draw in watts.", labelnames=("gpu_index",))
GPU_SM_CLOCK_HZ = Gauge("eosin_gpu_sm_clock_hz", "Current GPU SM clock in hertz.", labelnames=("gpu_index",))
GPU_MEM_CLOCK_HZ = Gauge("eosin_gpu_mem_clock_hz", "Current GPU memory clock in hertz.", labelnames=("gpu_index",))
CONTAINER_INFO = Gauge("eosin_container_info", "Static metadata about the container currently exporting metrics.", labelnames=("hostname", "pid"))


@dataclass(frozen=True)
class RenderedMetrics:
    payload: bytes
    media_type: str


class MetricsPushClient:
    def __init__(
        self,
        *,
        enabled: bool,
        push_url: str,
        timeout_seconds: float,
        auth_header_name: str,
        auth_header_value: str,
        transport: str,
        socks5_url: str,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.enabled = enabled and bool(push_url.strip())
        self.push_url = push_url.strip()
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.auth_header_name = auth_header_name.strip()
        self.auth_header_value = auth_header_value.strip()
        self.transport = transport.strip().lower()
        self.socks5_url = socks5_url.strip()
        self.session = session or requests.Session()

    def push(self, payload: bytes) -> None:
        if not self.enabled or not payload:
            return

        headers = {"Content-Type": "text/plain; version=0.0.4"}
        if self.auth_header_name and self.auth_header_value:
            headers[self.auth_header_name] = self.auth_header_value

        proxies = None
        if self.transport == "tailscale-socks5" and self.socks5_url:
            proxies = {
                "http": self.socks5_url,
                "https": self.socks5_url,
            }

        self.session.post(
            self.push_url,
            data=payload,
            headers=headers,
            timeout=self.timeout_seconds,
            proxies=proxies,
        )


class MetricsManager:
    def __init__(self) -> None:
        self._vllm_metrics_enabled = os.getenv("BANK_PARSER_METRICS_ENABLE_VLLM_PROXY", "true").strip().lower() in {"1", "true", "yes", "on"}
        self._vllm_metrics_url = os.getenv("BANK_PARSER_VLLM_METRICS_URL", "http://127.0.0.1:8000/metrics")
        self._vllm_metrics_timeout = max(0.25, float(os.getenv("BANK_PARSER_VLLM_METRICS_TIMEOUT", "1.5")))
        self._vllm_metrics_cache_ttl = max(1.0, float(os.getenv("BANK_PARSER_VLLM_METRICS_CACHE_TTL", "2.0")))
        self._gpu_metrics_enabled = os.getenv("BANK_PARSER_GPU_METRICS_ENABLE", "true").strip().lower() in {"1", "true", "yes", "on"}
        self._gpu_metrics_interval = max(0.5, float(os.getenv("BANK_PARSER_GPU_METRICS_INTERVAL", "1.0")))
        self._push_queue: queue.Queue[object] = queue.Queue(maxsize=1)
        self._push_thread_started = False
        self._push_thread_lock = threading.Lock()
        self._vllm_cache_lock = threading.Lock()
        self._gpu_thread_started = False
        self._gpu_thread_lock = threading.Lock()
        self._cached_vllm_metrics = b""
        self._cached_vllm_metrics_at = 0.0
        self._cached_vllm_metrics_ok = False
        self._push_client = MetricsPushClient(
            enabled=os.getenv("BANK_PARSER_METRICS_PUSH_ENABLE", "false").strip().lower() in {"1", "true", "yes", "on"},
            push_url=os.getenv("BANK_PARSER_METRICS_PUSH_URL", ""),
            timeout_seconds=float(os.getenv("BANK_PARSER_METRICS_PUSH_TIMEOUT", "1.0")),
            auth_header_name=os.getenv("BANK_PARSER_METRICS_PUSH_AUTH_HEADER", ""),
            auth_header_value=os.getenv("BANK_PARSER_METRICS_PUSH_AUTH_VALUE", ""),
            transport=os.getenv("BANK_PARSER_METRICS_PUSH_TRANSPORT", "direct"),
            socks5_url=os.getenv("BANK_PARSER_METRICS_PUSH_SOCKS5_URL", "socks5h://127.0.0.1:1055"),
        )

        CONTAINER_INFO.labels(hostname=socket.gethostname(), pid=str(os.getpid())).set(1)

    def start_background_samplers(self) -> None:
        self._start_push_worker()
        if not self._gpu_metrics_enabled:
            GPU_METRICS_AVAILABLE.set(0)
            return

        with self._gpu_thread_lock:
            if self._gpu_thread_started:
                return
            worker = threading.Thread(target=self._gpu_sampler_loop, name="eosin-gpu-metrics", daemon=True)
            worker.start()
            self._gpu_thread_started = True

    def _start_push_worker(self) -> None:
        with self._push_thread_lock:
            if self._push_thread_started or not self._push_client.enabled:
                return
            worker = threading.Thread(target=self._push_loop, name="eosin-metrics-push", daemon=True)
            worker.start()
            self._push_thread_started = True

    def schedule_push(self) -> None:
        if not self._push_client.enabled:
            return
        try:
            self._push_queue.put_nowait(object())
        except queue.Full:
            return

    def track_request_started(self, pdf_bytes: int) -> None:
        REQUEST_INFLIGHT.inc()
        PDF_BYTES.observe(pdf_bytes)

    def track_request_finished(self, elapsed_seconds: float) -> None:
        REQUEST_DURATION_SECONDS.observe(elapsed_seconds)
        REQUEST_INFLIGHT.dec()

    def track_request_success(self, result_payload: dict[str, object]) -> None:
        REQUESTS_TOTAL.labels(status="success").inc()
        page_count = int(result_payload.get("page_count", 0) or 0)
        PDF_PAGE_COUNT.observe(page_count)

        rows = result_payload.get("rows", [])
        if isinstance(rows, list):
            TABLE_ROW_COUNT.observe(len(rows))

        timings = result_payload.get("timings", {})
        if isinstance(timings, dict):
            for stage_name, stage_value in timings.items():
                try:
                    PIPELINE_STAGE_SECONDS.labels(stage=str(stage_name)).observe(float(stage_value))
                except Exception:
                    continue

        debug = result_payload.get("debug", {})
        if isinstance(debug, dict):
            ocr_metrics = debug.get("ocr_metrics")
            if isinstance(ocr_metrics, dict):
                self._observe_ocr_metrics(ocr_metrics)

    def track_request_failure(self, reason: str) -> None:
        REQUESTS_TOTAL.labels(status="failure").inc()
        REQUEST_FAILURES_TOTAL.labels(reason=reason).inc()

    def render_metrics(self) -> RenderedMetrics:
        app_metrics = generate_latest()
        vllm_metrics = self._get_vllm_metrics()
        payload = app_metrics if not vllm_metrics else app_metrics + b"\n" + vllm_metrics
        return RenderedMetrics(payload=payload, media_type=CONTENT_TYPE_LATEST)

    def _observe_ocr_metrics(self, ocr_metrics: dict[str, object]) -> None:
        try:
            OCR_TASK_COUNT.observe(float(ocr_metrics.get("task_count", 0.0) or 0.0))
            OCR_QUEUE_SIZE.observe(float(ocr_metrics.get("max_queue_size_at_submit", 0.0) or 0.0))
            for metric_name, histogram in (("queue_wait", OCR_QUEUE_WAIT_SECONDS), ("build_request", OCR_BUILD_REQUEST_SECONDS), ("request", OCR_REQUEST_SECONDS), ("total", OCR_TOTAL_SECONDS)):
                mean_value = float(ocr_metrics.get(f"{metric_name}_mean", 0.0) or 0.0)
                max_value = float(ocr_metrics.get(f"{metric_name}_max", 0.0) or 0.0)
                histogram.labels(aggregation="mean").observe(mean_value)
                histogram.labels(aggregation="max").observe(max_value)
        except Exception:
            return

    def _get_vllm_metrics(self) -> bytes:
        if not self._vllm_metrics_enabled:
            VLLM_METRICS_PROXY_UP.set(0)
            return b""

        now = time.monotonic()
        with self._vllm_cache_lock:
            if now - self._cached_vllm_metrics_at <= self._vllm_metrics_cache_ttl:
                VLLM_METRICS_PROXY_UP.set(1 if self._cached_vllm_metrics_ok else 0)
                return self._cached_vllm_metrics

        started_at = time.perf_counter()
        try:
            with urllib.request.urlopen(self._vllm_metrics_url, timeout=self._vllm_metrics_timeout) as response:
                payload = response.read()
            ok = True
        except Exception:
            payload = b""
            ok = False
        finally:
            VLLM_METRICS_SCRAPE_DURATION_SECONDS.set(time.perf_counter() - started_at)

        with self._vllm_cache_lock:
            self._cached_vllm_metrics = payload
            self._cached_vllm_metrics_at = time.monotonic()
            self._cached_vllm_metrics_ok = ok

        VLLM_METRICS_PROXY_UP.set(1 if ok else 0)
        return payload

    def _push_loop(self) -> None:
        while True:
            self._push_queue.get()
            try:
                rendered = self.render_metrics()
                self._push_client.push(rendered.payload)
                METRICS_PUSH_SUCCESS_TOTAL.labels(status="success").inc()
            except Exception:
                METRICS_PUSH_SUCCESS_TOTAL.labels(status="failure").inc()
            finally:
                self._push_queue.task_done()

    def _gpu_sampler_loop(self) -> None:
        if pynvml is None:
            GPU_METRICS_AVAILABLE.set(0)
            return

        try:
            pynvml.nvmlInit()
            GPU_METRICS_AVAILABLE.set(1)
        except Exception:
            GPU_METRICS_AVAILABLE.set(0)
            return

        while True:
            try:
                count = pynvml.nvmlDeviceGetCount()
                for gpu_index in range(count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power_usage_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    sm_clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                    mem_clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    label = str(gpu_index)
                    GPU_UTILIZATION_RATIO.labels(gpu_index=label).set(utilization.gpu / 100.0)
                    GPU_MEMORY_USED_BYTES.labels(gpu_index=label).set(float(memory.used))
                    GPU_MEMORY_TOTAL_BYTES.labels(gpu_index=label).set(float(memory.total))
                    GPU_TEMPERATURE_CELSIUS.labels(gpu_index=label).set(float(temperature))
                    GPU_POWER_USAGE_WATTS.labels(gpu_index=label).set(power_usage_mw / 1000.0)
                    GPU_SM_CLOCK_HZ.labels(gpu_index=label).set(sm_clock_mhz * 1_000_000.0)
                    GPU_MEM_CLOCK_HZ.labels(gpu_index=label).set(mem_clock_mhz * 1_000_000.0)
            except Exception:
                GPU_METRICS_AVAILABLE.set(0)
            time.sleep(self._gpu_metrics_interval)


_metrics_manager: Optional[MetricsManager] = None
_metrics_manager_lock = threading.Lock()


def get_metrics_manager() -> MetricsManager:
    global _metrics_manager
    if _metrics_manager is not None:
        return _metrics_manager
    with _metrics_manager_lock:
        if _metrics_manager is None:
            _metrics_manager = MetricsManager()
        return _metrics_manager
