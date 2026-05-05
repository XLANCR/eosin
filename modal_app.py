from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from pathlib import Path

import modal
from dotenv import load_dotenv

load_dotenv()

APP_NAME = os.getenv("EOSIN_MODAL_APP_NAME", "eosin-glm-ocr")
WEB_LABEL = os.getenv("EOSIN_MODAL_WEB_LABEL", "bank-parser")
GPU_TYPE = os.getenv("EOSIN_MODAL_GPU", "A100-80GB")
MAX_CONTAINERS = int(os.getenv("EOSIN_MODAL_MAX_CONTAINERS", "1"))
MAX_INPUTS = int(os.getenv("EOSIN_MODAL_MAX_INPUTS", "128"))
TARGET_INPUTS = int(os.getenv("EOSIN_MODAL_TARGET_INPUTS", str(MAX_INPUTS)))
SCALEDOWN_WINDOW_SECONDS = int(os.getenv("EOSIN_MODAL_SCALEDOWN_WINDOW", "180"))
FUNCTION_TIMEOUT_SECONDS = int(os.getenv("EOSIN_MODAL_TIMEOUT", "1800"))
MEMORY_MIB = int(os.getenv("EOSIN_MODAL_MEMORY_MIB", "40960"))
OCR_PIPELINE_WORKERS = int(os.getenv("EOSIN_MODAL_OCR_PIPELINE_WORKERS", "128"))
OCR_PIPELINE_QUEUE_SIZE = int(os.getenv("EOSIN_MODAL_OCR_PIPELINE_QUEUE_SIZE", "2048"))
VLLM_PORT = int(os.getenv("EOSIN_MODAL_VLLM_PORT", "8000"))
VLLM_MODEL = os.getenv("EOSIN_MODAL_VLLM_MODEL", "zai-org/GLM-OCR")
VLLM_SERVED_MODEL_NAME = os.getenv("EOSIN_MODAL_SERVED_MODEL_NAME", "default")
VLLM_GPU_MEMORY_UTILIZATION = os.getenv("EOSIN_MODAL_VLLM_GPU_MEMORY_UTILIZATION", "0.90")
VLLM_MAX_MODEL_LEN = os.getenv("EOSIN_MODAL_VLLM_MAX_MODEL_LEN", "22480")
VLLM_MAX_NUM_SEQS = os.getenv("EOSIN_MODAL_VLLM_MAX_NUM_SEQS", "192")
VLLM_MAX_BATCHED_TOKENS = os.getenv("EOSIN_MODAL_VLLM_MAX_BATCHED_TOKENS", "32768")
VLLM_SPECULATIVE_CONFIG = os.getenv(
    "EOSIN_MODAL_VLLM_SPECULATIVE_CONFIG",
    '{"method": "mtp", "num_speculative_tokens": 1}',
)
VLLM_ENABLE_PREFIX_CACHING = os.getenv("EOSIN_MODAL_VLLM_ENABLE_PREFIX_CACHING", "true").lower() == "true"
VLLM_KV_CACHE_METRICS = os.getenv("EOSIN_MODAL_VLLM_KV_CACHE_METRICS", "true").lower() == "true"
VLLM_ENABLE_MFU_METRICS = os.getenv("EOSIN_MODAL_VLLM_ENABLE_MFU_METRICS", "true").lower() == "true"
VLLM_ENABLE_LOGGING_ITERATION_DETAILS = (
    os.getenv("EOSIN_MODAL_VLLM_ENABLE_LOGGING_ITERATION_DETAILS", "false").lower() == "true"
)
VLLM_ENABLE_LOG_REQUESTS = os.getenv("EOSIN_MODAL_VLLM_ENABLE_LOG_REQUESTS", "false").lower() == "true"
HF_CACHE_PATH = "/root/.cache/huggingface"
VLLM_CACHE_PATH = "/root/.cache/vllm"
VLLM_HEALTH_URL = f"http://127.0.0.1:{VLLM_PORT}/health"
TAILSCALE_ENABLE = os.getenv("EOSIN_MODAL_TAILSCALE_ENABLE", "false").lower() == "true"
TAILSCALE_SOCKS5_URL = os.getenv("EOSIN_MODAL_TAILSCALE_SOCKS5_URL", "socks5h://127.0.0.1:1055")
TAILSCALE_PORT = int(os.getenv("EOSIN_MODAL_TAILSCALE_SOCKS5_PORT", "1055"))
TAILSCALE_HOSTNAME = os.getenv("EOSIN_MODAL_TAILSCALE_HOSTNAME", "eosin-modal-worker")
OCR_BACKEND_MODE = os.getenv("BANK_PARSER_OCR_BACKEND_MODE", "document_http")
OCR_BATCH_DRAIN_MAX_BATCH_SIZE = int(os.getenv("BANK_PARSER_OCR_BATCH_DRAIN_MAX_BATCH_SIZE", "64"))
OCR_BATCH_DRAIN_MAX_WAIT_SECONDS = float(os.getenv("BANK_PARSER_OCR_BATCH_DRAIN_MAX_WAIT_SECONDS", "1.0"))

app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name("eosin-hf-cache", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("eosin-vllm-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:nightly",
        setup_dockerfile_commands=["ENTRYPOINT []"],
    )
    .run_commands(
        "apt-get update && apt-get install -y curl git ghostscript python3-tk && rm -rf /var/lib/apt/lists/*",
        "curl -fsSL https://tailscale.com/install.sh | sh",
        "ln -sf $(command -v python3) /usr/local/bin/python",
        "pip install --upgrade pip",
        "pip install --ignore-installed blinker 'glmocr[selfhosted,server]' pandas beautifulsoup4",
        "pip install accelerate beautifulsoup4 fastapi numpy opencv-python-headless pandas pillow portalocker prometheus-client pydantic pymupdf python-dotenv python-multipart pyyaml requests PySocks sentencepiece tqdm uvicorn eliot 'camelot-py[cv]' liteparse img2table nvidia-ml-py",
    )
    .add_local_dir("eosin", remote_path="/root/eosin", copy=True)
    .run_commands(
        "pip install --no-cache-dir --force-reinstall 'setuptools<81'",
        "pip install --no-cache-dir --force-reinstall --no-deps 'transformers==5.6.2'",
    )
    .entrypoint([])
    .env(
        {
            "PYTHONPATH": "/root",
            "HF_HOME": HF_CACHE_PATH,
            "VLLM_CACHE_ROOT": VLLM_CACHE_PATH,
            "TRITON_CACHE_DIR": f"{VLLM_CACHE_PATH}/triton",
            "TORCHINDUCTOR_CACHE_DIR": f"{VLLM_CACHE_PATH}/torchinductor",
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib",
            "GLMOCR_MODE": "selfhosted",
            "GLMOCR_OCR_API_HOST": "127.0.0.1",
            "GLMOCR_OCR_API_PORT": str(VLLM_PORT),
            "GLMOCR_OCR_MODEL": VLLM_SERVED_MODEL_NAME,
            "BANK_PARSER_SAVE_DEBUG_IMAGES": "false",
            "BANK_PARSER_PARSE_TESTING": "false",
            "BANK_PARSER_ENABLE_OCR_BATCHING": "false",
            "BANK_PARSER_LAYOUT_MODE": os.getenv("BANK_PARSER_LAYOUT_MODE", "auto"),
            "BANK_PARSER_LAYOUT_MAX_CONCURRENCY": "1",
            "BANK_PARSER_OCR_PIPELINE_WORKERS": str(OCR_PIPELINE_WORKERS),
            "BANK_PARSER_OCR_PIPELINE_QUEUE_SIZE": str(OCR_PIPELINE_QUEUE_SIZE),
            "BANK_PARSER_OCR_BACKEND_MODE": OCR_BACKEND_MODE,
            "BANK_PARSER_OCR_BATCH_DRAIN_MAX_BATCH_SIZE": str(OCR_BATCH_DRAIN_MAX_BATCH_SIZE),
            "BANK_PARSER_OCR_BATCH_DRAIN_MAX_WAIT_SECONDS": str(OCR_BATCH_DRAIN_MAX_WAIT_SECONDS),
            "BANK_PARSER_BACKEND_STARTUP_TIMEOUT": "60",
            "BANK_PARSER_BACKEND_RETRY_INTERVAL": "1",
            "BANK_PARSER_METRICS_ENABLE_VLLM_PROXY": "true",
            "BANK_PARSER_VLLM_METRICS_URL": VLLM_HEALTH_URL.replace("/health", "/metrics"),
            "BANK_PARSER_VLLM_METRICS_TIMEOUT": "1.5",
            "BANK_PARSER_GPU_METRICS_ENABLE": "true",
            "BANK_PARSER_GPU_METRICS_INTERVAL": "1.0",
            "BANK_PARSER_METRICS_PUSH_ENABLE": os.getenv("BANK_PARSER_METRICS_PUSH_ENABLE", "false"),
            "BANK_PARSER_METRICS_PUSH_URL": os.getenv("BANK_PARSER_METRICS_PUSH_URL", ""),
            "BANK_PARSER_METRICS_PUSH_TIMEOUT": os.getenv("BANK_PARSER_METRICS_PUSH_TIMEOUT", "1.0"),
            "BANK_PARSER_METRICS_PUSH_AUTH_HEADER": os.getenv("BANK_PARSER_METRICS_PUSH_AUTH_HEADER", "X-Eosin-Key"),
            "BANK_PARSER_METRICS_PUSH_AUTH_VALUE": os.getenv("BANK_PARSER_METRICS_PUSH_AUTH_VALUE", ""),
            "BANK_PARSER_METRICS_PUSH_TRANSPORT": os.getenv("BANK_PARSER_METRICS_PUSH_TRANSPORT", "direct"),
            "BANK_PARSER_METRICS_PUSH_SOCKS5_URL": os.getenv("BANK_PARSER_METRICS_PUSH_SOCKS5_URL", TAILSCALE_SOCKS5_URL),
        }
    )
)


def _wait_for_http_health(url: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - startup retry path
            last_error = exc
            time.sleep(1)

    raise TimeoutError(f"Timed out waiting for backend health at {url}: {last_error}")


def _terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
        process.kill()
        process.wait(timeout=10)


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
    cpu=12.0,
    memory=MEMORY_MIB,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    timeout=FUNCTION_TIMEOUT_SECONDS,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
    enable_memory_snapshot=True,
    secrets=[
        modal.Secret.from_name("eosin-tailscale"),
        modal.Secret.from_name("eosin-metrics-push"),
    ],
)
@modal.concurrent(max_inputs=MAX_INPUTS, target_inputs=TARGET_INPUTS)
class BankParserModalApp:
    @modal.enter(snap=True)
    def prepare_snapshot(self) -> None:
        import eosin.backend.bank_parser_service as bank_parser_service_module

        self.config_path = Path(bank_parser_service_module.__file__).resolve().parent / "config.modal.yaml"
        os.environ["BANK_PARSER_CONFIG"] = str(self.config_path)

    @modal.enter(snap=False)
    def enter(self) -> None:
        from eosin.backend.bank_parser_api import create_app
        from eosin.backend.bank_parser_service import BankParserService

        if TAILSCALE_ENABLE:
            self.tailscale_process, _ = _start_tailscale()

        vllm_args = [
            "vllm",
            "serve",
            VLLM_MODEL,
            "--port",
            str(VLLM_PORT),
            "--dtype",
            "bfloat16",
            "--trust-remote-code",
            "--speculative-config",
            VLLM_SPECULATIVE_CONFIG,
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
        if VLLM_ENABLE_PREFIX_CACHING:
            vllm_args.append("--enable-prefix-caching")
        if VLLM_KV_CACHE_METRICS:
            vllm_args.append("--kv-cache-metrics")
        if VLLM_ENABLE_MFU_METRICS:
            vllm_args.append("--enable-mfu-metrics")
        if VLLM_ENABLE_LOGGING_ITERATION_DETAILS:
            vllm_args.append("--enable-logging-iteration-details")
        if VLLM_ENABLE_LOG_REQUESTS:
            vllm_args.append("--enable-log-requests")

        self.vllm_process = subprocess.Popen(vllm_args, text=True)
        _wait_for_http_health(VLLM_HEALTH_URL, timeout_seconds=300)

        self.service = BankParserService(
            config_path=str(self.config_path),
            save_debug_images=False,
            parse_testing=False,
            enable_ocr_batching=False,
            layout_max_concurrency=1,
            ocr_pipeline_workers=OCR_PIPELINE_WORKERS,
            ocr_pipeline_queue_size=OCR_PIPELINE_QUEUE_SIZE,
            ocr_backend_mode=OCR_BACKEND_MODE,
            ocr_batch_drain_max_batch_size=OCR_BATCH_DRAIN_MAX_BATCH_SIZE,
            ocr_batch_drain_max_wait_seconds=OCR_BATCH_DRAIN_MAX_WAIT_SECONDS,
            backend_startup_timeout=60.0,
            backend_retry_interval=1.0,
        )
        self.web_app = create_app(service=self.service)

    @modal.exit()
    def exit(self) -> None:
        service = getattr(self, "service", None)
        if service is not None:
            service.close()

        _terminate_process(getattr(self, "vllm_process", None))
        _terminate_process(getattr(self, "tailscale_process", None))

    @modal.asgi_app(label=WEB_LABEL)
    def web(self):
        return self.web_app
