from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from pathlib import Path

import modal
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


def _bounded_modal_inputs() -> tuple[int, int]:
    safe_max_inputs = 35
    max_inputs = max(1, min(_env_int("EOSIN_MODAL_MAX_INPUTS", safe_max_inputs), safe_max_inputs))
    target_inputs = max(1, min(_env_int("EOSIN_MODAL_TARGET_INPUTS", max_inputs), max_inputs))
    return max_inputs, target_inputs


APP_NAME = os.getenv("EOSIN_MODAL_APP_NAME", "eosin-glm-ocr")
WEB_LABEL = os.getenv("EOSIN_MODAL_WEB_LABEL", "bank-parser")
REQUIRES_PROXY_AUTH = _env_bool("EOSIN_MODAL_REQUIRES_PROXY_AUTH", True)
GPU_TYPE = os.getenv("EOSIN_MODAL_GPU", "L40S")
CPU_COUNT = float(os.getenv("EOSIN_MODAL_CPU", "8.0"))
MAX_CONTAINERS = int(os.getenv("EOSIN_MODAL_MAX_CONTAINERS", "1"))
MIN_CONTAINERS = int(os.getenv("EOSIN_MODAL_MIN_CONTAINERS", "0"))
ALLOW_ALWAYS_ON = _env_bool("EOSIN_MODAL_ALLOW_ALWAYS_ON", False)
if MIN_CONTAINERS > 0 and not ALLOW_ALWAYS_ON:
    raise ValueError(
        "EOSIN_MODAL_MIN_CONTAINERS keeps paid GPU containers running. "
        "Set EOSIN_MODAL_ALLOW_ALWAYS_ON=true only if you intentionally want an always-on deployment."
    )
MAX_INPUTS, TARGET_INPUTS = _bounded_modal_inputs()
SCALEDOWN_WINDOW_SECONDS = int(os.getenv("EOSIN_MODAL_SCALEDOWN_WINDOW", "120"))
FUNCTION_TIMEOUT_SECONDS = int(os.getenv("EOSIN_MODAL_TIMEOUT", "1800"))
STARTUP_TIMEOUT_SECONDS = int(os.getenv("EOSIN_MODAL_STARTUP_TIMEOUT", "900"))
MEMORY_MIB = int(os.getenv("EOSIN_MODAL_MEMORY_MIB", "32768"))
OCR_PIPELINE_WORKERS = int(os.getenv("EOSIN_MODAL_OCR_PIPELINE_WORKERS", "16"))
OCR_PIPELINE_QUEUE_SIZE = int(os.getenv("EOSIN_MODAL_OCR_PIPELINE_QUEUE_SIZE", "512"))
PARSER_POOL_SIZE = int(os.getenv("EOSIN_MODAL_PARSER_POOL_SIZE", "2"))
INFERENCE_BACKEND = os.getenv("EOSIN_MODAL_INFERENCE_BACKEND", "sglang").strip().lower()
if INFERENCE_BACKEND not in {"vllm", "sglang"}:
    raise ValueError("EOSIN_MODAL_INFERENCE_BACKEND must be vllm or sglang")
VLLM_IMAGE = os.getenv("EOSIN_MODAL_VLLM_IMAGE", "vllm/vllm-openai:v0.19.0-ubuntu2404")
SGLANG_IMAGE = os.getenv("EOSIN_MODAL_SGLANG_IMAGE", "lmsysorg/sglang:v0.5.10")
INFERENCE_IMAGE = SGLANG_IMAGE if INFERENCE_BACKEND == "sglang" else VLLM_IMAGE
VLLM_PORT = int(os.getenv("EOSIN_MODAL_VLLM_PORT", "8000"))
VLLM_MODEL = os.getenv("EOSIN_MODAL_VLLM_MODEL", "zai-org/GLM-OCR")
VLLM_MODEL_REVISION = os.getenv("EOSIN_MODAL_VLLM_MODEL_REVISION", "").strip()
VLLM_SERVED_MODEL_NAME = os.getenv("EOSIN_MODAL_SERVED_MODEL_NAME", "default")
VLLM_GPU_MEMORY_UTILIZATION = os.getenv("EOSIN_MODAL_VLLM_GPU_MEMORY_UTILIZATION", "0.92")
VLLM_MAX_MODEL_LEN = os.getenv("EOSIN_MODAL_VLLM_MAX_MODEL_LEN", "12288")
VLLM_MAX_NUM_SEQS = os.getenv("EOSIN_MODAL_VLLM_MAX_NUM_SEQS", "196")
VLLM_MAX_BATCHED_TOKENS = os.getenv("EOSIN_MODAL_VLLM_MAX_BATCHED_TOKENS", "24576")
VLLM_SPECULATIVE_CONFIG = os.getenv(
    "EOSIN_MODAL_VLLM_SPECULATIVE_CONFIG",
    '{"method": "mtp", "num_speculative_tokens": 3}',
)
VLLM_ENABLE_PREFIX_CACHING = _env_bool("EOSIN_MODAL_VLLM_ENABLE_PREFIX_CACHING", True)
VLLM_KV_CACHE_METRICS = _env_bool("EOSIN_MODAL_VLLM_KV_CACHE_METRICS", True)
VLLM_ENABLE_MFU_METRICS = _env_bool("EOSIN_MODAL_VLLM_ENABLE_MFU_METRICS", True)
VLLM_ENABLE_LOGGING_ITERATION_DETAILS = _env_bool("EOSIN_MODAL_VLLM_ENABLE_LOGGING_ITERATION_DETAILS", False)
VLLM_ENABLE_LOG_REQUESTS = _env_bool("EOSIN_MODAL_VLLM_ENABLE_LOG_REQUESTS", False)
VLLM_FAST_BOOT = _env_bool("EOSIN_MODAL_VLLM_FAST_BOOT", False)
VLLM_ENABLE_SLEEP_MODE = _env_bool("EOSIN_MODAL_VLLM_ENABLE_SLEEP_MODE", False)
VLLM_ENABLE_GPU_SNAPSHOT = _env_bool("EOSIN_MODAL_VLLM_ENABLE_GPU_SNAPSHOT", False)
VLLM_SNAPSHOT_WARMUP_ENABLE = _env_bool("EOSIN_MODAL_VLLM_SNAPSHOT_WARMUP_ENABLE", False)
VLLM_SNAPSHOT_WARMUP_REQUESTS = int(os.getenv("EOSIN_MODAL_VLLM_SNAPSHOT_WARMUP_REQUESTS", "3"))
VLLM_SNAPSHOT_WARMUP_MODE = os.getenv("EOSIN_MODAL_VLLM_SNAPSHOT_WARMUP_MODE", "multimodal").strip().lower()
VLLM_ATTENTION_BACKEND = os.getenv("EOSIN_MODAL_VLLM_ATTENTION_BACKEND", "").strip()
VLLM_ENABLE_SERVER_LOAD_TRACKING = _env_bool("EOSIN_MODAL_VLLM_ENABLE_SERVER_LOAD_TRACKING", True)
if VLLM_ENABLE_GPU_SNAPSHOT and not _env_bool("EOSIN_MODAL_ALLOW_GPU_SNAPSHOT", False):
    raise ValueError(
        "GPU snapshots were removed from this production Modal path after NCCL heartbeat log storms. "
        "Keep EOSIN_MODAL_VLLM_ENABLE_GPU_SNAPSHOT=false; test snapshots in a separate deployment."
    )
VLLM_COMMIT_CACHE_AFTER_START = _env_bool("EOSIN_MODAL_VLLM_COMMIT_CACHE_AFTER_START", False)
HF_CACHE_PATH = "/root/.cache/huggingface"
VLLM_CACHE_PATH = "/root/.cache/vllm"
TRITON_CACHE_DIR = f"{VLLM_CACHE_PATH}/triton"
TORCHINDUCTOR_CACHE_DIR = f"{VLLM_CACHE_PATH}/torchinductor"
VLLM_HEALTH_URL = f"http://127.0.0.1:{VLLM_PORT}/health"
VLLM_WARMUP_IMAGE_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4Bf4Z9AAAAAElFTkSuQmCC"
)
TAILSCALE_ENABLE = os.getenv("EOSIN_MODAL_TAILSCALE_ENABLE", "false").lower() == "true"
TAILSCALE_SOCKS5_URL = os.getenv("EOSIN_MODAL_TAILSCALE_SOCKS5_URL", "socks5h://127.0.0.1:1055")
TAILSCALE_PORT = int(os.getenv("EOSIN_MODAL_TAILSCALE_SOCKS5_PORT", "1055"))
TAILSCALE_HOSTNAME = os.getenv("EOSIN_MODAL_TAILSCALE_HOSTNAME", "eosin-modal-worker")
OCR_BACKEND_MODE = os.getenv("EOSIN_MODAL_BANK_PARSER_OCR_BACKEND_MODE", "page_batch")
OCR_BATCH_DRAIN_MAX_BATCH_SIZE = int(os.getenv("BANK_PARSER_OCR_BATCH_DRAIN_MAX_BATCH_SIZE", "96"))
OCR_BATCH_DRAIN_MAX_WAIT_SECONDS = float(os.getenv("BANK_PARSER_OCR_BATCH_DRAIN_MAX_WAIT_SECONDS", "0.05"))
ENABLE_PAGE_OCR_RETRY = os.getenv("BANK_PARSER_ENABLE_PAGE_OCR_RETRY", "false").lower() == "true"
PAGE_OCR_RETRY_DPI = int(os.getenv("BANK_PARSER_PAGE_OCR_RETRY_DPI", "300"))
CAPTURE_RAW_OCR_DEBUG = os.getenv("BANK_PARSER_CAPTURE_RAW_OCR_DEBUG", "false").lower() == "true"
SAVE_DEBUG_IMAGES = os.getenv("BANK_PARSER_SAVE_DEBUG_IMAGES", "false").lower() == "true"
METRICS_PUSH_ENABLE = _env_bool("EOSIN_MODAL_METRICS_PUSH_ENABLE", False)
METRICS_PUSH_URL = os.getenv("EOSIN_MODAL_METRICS_PUSH_URL", "")
METRICS_PUSH_TIMEOUT = os.getenv("EOSIN_MODAL_METRICS_PUSH_TIMEOUT", "1.0")
METRICS_PUSH_AUTH_HEADER = os.getenv("EOSIN_MODAL_METRICS_PUSH_AUTH_HEADER", "X-Eosin-Key")
METRICS_PUSH_TRANSPORT = os.getenv("EOSIN_MODAL_METRICS_PUSH_TRANSPORT", "direct")
METRICS_PUSH_SOCKS5_URL = os.getenv("EOSIN_MODAL_METRICS_PUSH_SOCKS5_URL", TAILSCALE_SOCKS5_URL)

app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name("eosin-hf-cache", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("eosin-vllm-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        INFERENCE_IMAGE,
        setup_dockerfile_commands=["ENTRYPOINT []"],
    )
    .run_commands(
        "apt-get update && apt-get install -y curl git ghostscript python3-tk && rm -rf /var/lib/apt/lists/*",
        "curl -fsSL https://tailscale.com/install.sh | sh",
        "ln -sf $(command -v python3) /usr/local/bin/python",
        "pip install --upgrade pip",
        "pip install --ignore-installed --no-deps blinker glmocr pandas beautifulsoup4",
        "pip install --no-deps accelerate beautifulsoup4 fastapi numpy opencv-python-headless pandas pillow portalocker prometheus-client pydantic pymupdf python-dotenv python-multipart pyyaml requests PySocks sentencepiece tqdm uvicorn eliot 'camelot-py[cv]' liteparse img2table nvidia-ml-py",
    )
    .run_commands(
        "pip install --no-cache-dir --force-reinstall 'setuptools<81'",
        "pip install --no-cache-dir --force-reinstall --no-deps 'huggingface-hub==1.13.0'",
        "pip install --no-cache-dir --force-reinstall --no-deps 'transformers==5.6.2'",
    )
    .add_local_dir("eosin", remote_path="/root/eosin", copy=True)
    .entrypoint([])
    .env(
        {
            "PYTHONPATH": "/root/eosin:/root",
            "EOSIN_MODAL_INFERENCE_BACKEND": INFERENCE_BACKEND,
            "HF_HOME": HF_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": os.getenv("EOSIN_MODAL_HF_XET_HIGH_PERFORMANCE", "1"),
            "VLLM_CACHE_ROOT": VLLM_CACHE_PATH,
            "TRITON_CACHE_DIR": TRITON_CACHE_DIR,
            "TORCHINDUCTOR_CACHE_DIR": TORCHINDUCTOR_CACHE_DIR,
            "TORCHINDUCTOR_COMPILE_THREADS": os.getenv("EOSIN_MODAL_TORCHINDUCTOR_COMPILE_THREADS", "1"),
            "TORCH_NCCL_ENABLE_MONITORING": os.getenv("EOSIN_MODAL_TORCH_NCCL_ENABLE_MONITORING", "0"),
            "VLLM_SERVER_DEV_MODE": "1" if VLLM_ENABLE_SLEEP_MODE else "0",
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib",
            "GLMOCR_MODE": "selfhosted",
            "GLMOCR_OCR_API_HOST": "127.0.0.1",
            "GLMOCR_OCR_API_PORT": str(VLLM_PORT),
            "GLMOCR_OCR_MODEL": VLLM_SERVED_MODEL_NAME,
            "BANK_PARSER_SAVE_DEBUG_IMAGES": "true" if SAVE_DEBUG_IMAGES else "false",
            "BANK_PARSER_PARSE_TESTING": "false",
            "BANK_PARSER_ENABLE_OCR_BATCHING": "false",
            "BANK_PARSER_LAYOUT_MODE": os.getenv("BANK_PARSER_LAYOUT_MODE", "auto"),
            "BANK_PARSER_RECOVER_LAYOUT_MISSED_PAGES": os.getenv(
                "BANK_PARSER_RECOVER_LAYOUT_MISSED_PAGES", "true"
            ),
            "BANK_PARSER_LAYOUT_MAX_CONCURRENCY": "1",
            "BANK_PARSER_OCR_PIPELINE_WORKERS": str(OCR_PIPELINE_WORKERS),
            "BANK_PARSER_OCR_PIPELINE_QUEUE_SIZE": str(OCR_PIPELINE_QUEUE_SIZE),
            "BANK_PARSER_OCR_BACKEND_MODE": OCR_BACKEND_MODE,
            "BANK_PARSER_ENABLE_PAGE_OCR_RETRY": "true" if ENABLE_PAGE_OCR_RETRY else "false",
            "BANK_PARSER_PAGE_OCR_RETRY_ALL": os.getenv("BANK_PARSER_PAGE_OCR_RETRY_ALL", "false"),
            "BANK_PARSER_PAGE_OCR_RETRY_DPI": str(PAGE_OCR_RETRY_DPI),
            "BANK_PARSER_CAPTURE_RAW_OCR_DEBUG": "true" if CAPTURE_RAW_OCR_DEBUG else "false",
            "BANK_PARSER_OCR_MAX_IMAGE_SIDE": os.getenv("BANK_PARSER_OCR_MAX_IMAGE_SIDE", "3500"),
            "BANK_PARSER_OCR_MAX_IMAGE_PIXELS": os.getenv("BANK_PARSER_OCR_MAX_IMAGE_PIXELS", "9000000"),
            "BANK_PARSER_OCR_DOCUMENT_MAX_TOKENS_CAP": os.getenv("BANK_PARSER_OCR_DOCUMENT_MAX_TOKENS_CAP", "4096"),
            "BANK_PARSER_OCR_DOCUMENT_MAX_IMAGES_PER_REQUEST": os.getenv("BANK_PARSER_OCR_DOCUMENT_MAX_IMAGES_PER_REQUEST", "4"),
            "BANK_PARSER_OCR_BATCH_DRAIN_MAX_BATCH_SIZE": str(OCR_BATCH_DRAIN_MAX_BATCH_SIZE),
            "BANK_PARSER_OCR_BATCH_DRAIN_MAX_WAIT_SECONDS": str(OCR_BATCH_DRAIN_MAX_WAIT_SECONDS),
            "BANK_PARSER_BACKEND_STARTUP_TIMEOUT": "60",
            "BANK_PARSER_BACKEND_RETRY_INTERVAL": "1",
            "BANK_PARSER_METRICS_ENABLE_VLLM_PROXY": "true",
            "BANK_PARSER_VLLM_METRICS_URL": VLLM_HEALTH_URL.replace("/health", "/metrics"),
            "BANK_PARSER_VLLM_METRICS_TIMEOUT": "1.5",
            "BANK_PARSER_GPU_METRICS_ENABLE": "true",
            "BANK_PARSER_GPU_METRICS_INTERVAL": "1.0",
            "BANK_PARSER_METRICS_PUSH_ENABLE": "true" if METRICS_PUSH_ENABLE else "false",
            "BANK_PARSER_METRICS_PUSH_URL": METRICS_PUSH_URL,
            "BANK_PARSER_METRICS_PUSH_TIMEOUT": METRICS_PUSH_TIMEOUT,
            "BANK_PARSER_METRICS_PUSH_AUTH_HEADER": METRICS_PUSH_AUTH_HEADER,
            "BANK_PARSER_METRICS_PUSH_TRANSPORT": METRICS_PUSH_TRANSPORT,
            "BANK_PARSER_METRICS_PUSH_SOCKS5_URL": METRICS_PUSH_SOCKS5_URL,
        }
    )
)


def _wait_for_http_health(
    url: str,
    timeout_seconds: int,
    *,
    process: subprocess.Popen[str] | None = None,
) -> None:
    started_at = time.monotonic()
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM exited before becoming healthy with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    print(
                        f"vLLM health ready after {time.monotonic() - started_at:.3f}s",
                        flush=True,
                    )
                    return
        except Exception as exc:  # pragma: no cover - startup retry path
            last_error = exc
            time.sleep(1)

    raise TimeoutError(f"Timed out waiting for backend health at {url}: {last_error}")


def _post_local(url: str, *, timeout_seconds: int = 30) -> None:
    request = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if response.status >= 400:
            raise RuntimeError(f"Unexpected response {response.status} from {url}")


def _build_vllm_warmup_payload() -> dict:
    if VLLM_SNAPSHOT_WARMUP_MODE == "text":
        content: str | list[dict[str, object]] = "Return exactly the word ready."
    else:
        content = [
            {"type": "image_url", "image_url": {"url": VLLM_WARMUP_IMAGE_DATA_URL}},
            {"type": "text", "text": "Read this tiny image and return exactly the word ready."},
        ]

    return {
        "model": VLLM_SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 8,
        "temperature": 0,
    }


def _warmup_vllm(request_count: int) -> None:
    if request_count <= 0:
        return

    import json

    payload = _build_vllm_warmup_payload()
    for _ in range(request_count):
        request = urllib.request.Request(
            f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            if response.status != 200:
                raise RuntimeError(f"Warmup request failed with status {response.status}")


def _sleep_vllm() -> None:
    _post_local(f"http://127.0.0.1:{VLLM_PORT}/sleep?level=1")


def _wake_vllm() -> None:
    _post_local(f"http://127.0.0.1:{VLLM_PORT}/wake_up")


def _terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
        process.kill()
        process.wait(timeout=10)


def _prepare_cache_directories() -> None:
    Path(HF_CACHE_PATH).mkdir(parents=True, exist_ok=True)
    Path(VLLM_CACHE_PATH).mkdir(parents=True, exist_ok=True)
    Path(TRITON_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    Path(TORCHINDUCTOR_CACHE_DIR).mkdir(parents=True, exist_ok=True)


def _start_tailscale() -> tuple[subprocess.Popen[str], subprocess.CompletedProcess[str]]:
    authkey = os.getenv("TAILSCALE_AUTHKEY", "").strip() or os.getenv("TAILSCALE_AUTH_KEY", "").strip()
    if not authkey:
        raise RuntimeError(
            "TAILSCALE_AUTHKEY or TAILSCALE_AUTH_KEY is required when EOSIN_MODAL_TAILSCALE_ENABLE=true"
        )

    daemon = subprocess.Popen(
        [
            "tailscaled",
            "--tun=userspace-networking",
            f"--socks5-server=127.0.0.1:{TAILSCALE_PORT}",
            "--state=mem:",
        ],
        text=True,
    )
    time.sleep(2)
    up = subprocess.run(
        [
            "tailscale",
            "up",
            f"--auth-key={authkey}",
            f"--hostname={TAILSCALE_HOSTNAME}",
        ],
        text=True,
        check=True,
        capture_output=True,
    )
    return daemon, up


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    cpu=CPU_COUNT,
    memory=MEMORY_MIB,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    timeout=FUNCTION_TIMEOUT_SECONDS,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
    enable_memory_snapshot=False,
    experimental_options={},
    secrets=[
        modal.Secret.from_name("eosin-tailscale"),
        modal.Secret.from_name("eosin-metrics-push"),
    ],
)
@modal.concurrent(max_inputs=MAX_INPUTS, target_inputs=TARGET_INPUTS)
class BankParserModalApp:
    def _build_parser_service(self):
        from eosin.backend.bank_parser_api import create_app
        from eosin.backend.bank_parser_service import BankParserService

        self.service = BankParserService(
            config_path=str(self.config_path),
            save_debug_images=SAVE_DEBUG_IMAGES,
            parse_testing=False,
            enable_ocr_batching=False,
            layout_max_concurrency=1,
            ocr_pipeline_workers=OCR_PIPELINE_WORKERS,
            ocr_pipeline_queue_size=OCR_PIPELINE_QUEUE_SIZE,
            ocr_backend_mode=OCR_BACKEND_MODE,
            ocr_batch_drain_max_batch_size=OCR_BATCH_DRAIN_MAX_BATCH_SIZE,
            ocr_batch_drain_max_wait_seconds=OCR_BATCH_DRAIN_MAX_WAIT_SECONDS,
            enable_page_ocr_retry=ENABLE_PAGE_OCR_RETRY,
            page_ocr_retry_dpi=PAGE_OCR_RETRY_DPI,
            capture_raw_ocr_debug=CAPTURE_RAW_OCR_DEBUG,
            parser_pool_size=PARSER_POOL_SIZE,
            parser_pool_wait_timeout=float(os.getenv("BANK_PARSER_POOL_WAIT_TIMEOUT", "1800")),
            backend_startup_timeout=60.0,
            backend_retry_interval=1.0,
        )
        self.web_app = create_app(service=self.service)

    def _launch_vllm(self) -> None:
        self.vllm_started_at = time.monotonic()
        backend_env = None
        if INFERENCE_BACKEND == "sglang":
            vllm_args = [
                "sglang",
                "serve",
                "--model-path",
                VLLM_MODEL,
                "--host",
                "127.0.0.1",
                "--port",
                str(VLLM_PORT),
                "--served-model-name",
                VLLM_SERVED_MODEL_NAME,
                "--trust-remote-code",
                "--dtype",
                "bfloat16",
                "--context-length",
                VLLM_MAX_MODEL_LEN,
                "--mem-fraction-static",
                VLLM_GPU_MEMORY_UTILIZATION,
                "--max-running-requests",
                VLLM_MAX_NUM_SEQS,
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
            ]
            backend_env = {**os.environ, "SGLANG_ENABLE_SPEC_V2": "1"}
        else:
            vllm_args = [
                "vllm",
                "serve",
                VLLM_MODEL,
                "--host",
                "127.0.0.1",
                "--port",
                str(VLLM_PORT),
                "--uvicorn-log-level=info",
                "--dtype",
                "bfloat16",
                "--trust-remote-code",
                "--max-model-len",
                VLLM_MAX_MODEL_LEN,
                "--gpu-memory-utilization",
                VLLM_GPU_MEMORY_UTILIZATION,
                "--max-num-seqs",
                VLLM_MAX_NUM_SEQS,
                "--max-num-batched-tokens",
                VLLM_MAX_BATCHED_TOKENS,
                "--async-scheduling",
                "--enable-chunked-prefill",
                "--served-model-name",
                VLLM_SERVED_MODEL_NAME,
            ]
            if VLLM_MODEL_REVISION:
                vllm_args.extend(["--revision", VLLM_MODEL_REVISION])
            if VLLM_SPECULATIVE_CONFIG:
                vllm_args.extend(["--speculative-config", VLLM_SPECULATIVE_CONFIG])
            if VLLM_FAST_BOOT:
                vllm_args.append("--enforce-eager")
            else:
                vllm_args.append("--no-enforce-eager")
            if VLLM_ENABLE_SLEEP_MODE:
                vllm_args.append("--enable-sleep-mode")
            if VLLM_ATTENTION_BACKEND:
                vllm_args.extend(["--attention-backend", VLLM_ATTENTION_BACKEND])
            if VLLM_ENABLE_PREFIX_CACHING:
                vllm_args.append("--enable-prefix-caching")
            if VLLM_KV_CACHE_METRICS:
                vllm_args.append("--kv-cache-metrics")
            if VLLM_ENABLE_MFU_METRICS:
                vllm_args.append("--enable-mfu-metrics")
            if VLLM_ENABLE_SERVER_LOAD_TRACKING:
                vllm_args.append("--enable-server-load-tracking")
            if VLLM_ENABLE_LOGGING_ITERATION_DETAILS:
                vllm_args.append("--enable-logging-iteration-details")
            if VLLM_ENABLE_LOG_REQUESTS:
                vllm_args.append("--enable-log-requests")
        self.vllm_process = subprocess.Popen(vllm_args, text=True, env=backend_env)
        print(f"Started {INFERENCE_BACKEND} subprocess", flush=True)

    def _await_vllm_ready(self) -> None:
        print(f"Waiting for {INFERENCE_BACKEND} readiness", flush=True)
        _wait_for_http_health(
            VLLM_HEALTH_URL,
            timeout_seconds=STARTUP_TIMEOUT_SECONDS,
            process=self.vllm_process,
        )
        print(
            f"{INFERENCE_BACKEND} startup phase completed in "
            f"{time.monotonic() - self.vllm_started_at:.3f}s",
            flush=True,
        )
        if VLLM_COMMIT_CACHE_AFTER_START:
            cache_commit_started_at = time.monotonic()
            try:
                hf_cache_volume.commit()
                vllm_cache_volume.commit()
            except Exception as exc:  # pragma: no cover - best-effort cache persistence
                print(f"Volume cache commit failed after vLLM start: {exc}", flush=True)
            finally:
                print(
                    "vLLM cache commit phase completed in "
                    f"{time.monotonic() - cache_commit_started_at:.3f}s",
                    flush=True,
                )

    def _prepare_runtime(self) -> None:
        prepare_started_at = time.monotonic()
        print("Runtime preparation: importing parser modules", flush=True)
        import eosin.backend.bank_parser_api  # noqa: F401
        import eosin.backend.bank_parser_service as bank_parser_service_module
        import eosin.backend.eosin_pipeline  # noqa: F401

        self.config_path = Path(bank_parser_service_module.__file__).resolve().parent / "config.modal.yaml"
        os.environ["BANK_PARSER_CONFIG"] = str(self.config_path)
        print(f"Runtime preparation completed in {time.monotonic() - prepare_started_at:.3f}s", flush=True)

    @modal.enter(snap=False)
    def enter(self) -> None:
        enter_started_at = time.monotonic()
        print("Restore phase: entering live container", flush=True)
        _prepare_cache_directories()
        self._launch_vllm()
        self._prepare_runtime()
        if TAILSCALE_ENABLE:
            self.tailscale_process, _ = _start_tailscale()
        self._await_vllm_ready()
        parser_service_started_at = time.monotonic()
        self._build_parser_service()
        print(
            "Parser service construction completed in "
            f"{time.monotonic() - parser_service_started_at:.3f}s",
            flush=True,
        )
        print(f"Restore phase completed in {time.monotonic() - enter_started_at:.3f}s", flush=True)

    @modal.exit()
    def exit(self) -> None:
        service = getattr(self, "service", None)
        if service is not None:
            service.close()

        _terminate_process(getattr(self, "vllm_process", None))
        _terminate_process(getattr(self, "tailscale_process", None))

    @modal.asgi_app(label=WEB_LABEL, requires_proxy_auth=REQUIRES_PROXY_AUTH)
    def web(self):
        return self.web_app

    @modal.method()
    def extract_evidence(self, filename: str, pdf_bytes: bytes) -> dict:
        from eosin.backend.bank_parser_api import evidence_payload_from_result, validate_pdf_upload

        validate_pdf_upload(filename, pdf_bytes)
        return evidence_payload_from_result(
            self.service.extract_glm_page_html_bytes(filename, pdf_bytes)
        )

    @modal.method()
    def extract_document_evidence(
        self, filename: str, pdf_bytes: bytes, document_type: str
    ) -> dict:
        from eosin.backend.bank_parser_api import evidence_payload_from_result, validate_pdf_upload

        normalized_type = document_type.strip().lower().replace("-", "_")
        if normalized_type not in {"bank_statement", "invoice", "receipt"}:
            raise ValueError("document_type must be bank_statement, invoice, or receipt")
        validate_pdf_upload(filename, pdf_bytes)
        result = self.service.extract_glm_page_html_bytes(
            filename,
            pdf_bytes,
            task_type="table" if normalized_type == "bank_statement" else "text",
        )
        payload = evidence_payload_from_result(result)
        payload["document_type"] = normalized_type
        payload["ocr_task_type"] = result.get("ocr_task_type", "table")
        return payload

    @modal.method()
    def extract_glm_page_html(
        self,
        filename: str,
        pdf_bytes: bytes,
        page_numbers: list[int] | None = None,
        dpi: int | None = None,
    ) -> dict:
        from eosin.backend.bank_parser_api import validate_pdf_upload

        validate_pdf_upload(filename, pdf_bytes)
        return self.service.extract_glm_page_html_bytes(
            filename,
            pdf_bytes,
            page_numbers=page_numbers,
            dpi=dpi,
        )
