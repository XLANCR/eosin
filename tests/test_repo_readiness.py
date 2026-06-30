from __future__ import annotations

import asyncio
import ast
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

load_test_spec = importlib.util.spec_from_file_location(
    "load_test_bank_parser",
    Path("scripts/load_test_bank_parser.py"),
)
assert load_test_spec is not None
load_test_module = importlib.util.module_from_spec(load_test_spec)
assert load_test_spec.loader is not None
sys.modules[load_test_spec.name] = load_test_module
load_test_spec.loader.exec_module(load_test_module)

local_quality_spec = importlib.util.spec_from_file_location(
    "run_local_quality_bank_parser",
    Path("scripts/run_local_quality_bank_parser.py"),
)
assert local_quality_spec is not None
local_quality_module = importlib.util.module_from_spec(local_quality_spec)
assert local_quality_spec.loader is not None
sys.modules[local_quality_spec.name] = local_quality_module
local_quality_spec.loader.exec_module(local_quality_module)


def _load_eosin_provider_module():
    import types

    exceptions_module = types.ModuleType("app.thesmos.exceptions")

    class ProviderFailedToDig(Exception):
        pass

    exceptions_module.ProviderFailedToDig = ProviderFailedToDig

    providers_base_module = types.ModuleType("app.thesmos.providers.base")

    class Provider:
        pass

    providers_base_module.Provider = Provider

    app_module = types.ModuleType("app")
    thesmos_module = types.ModuleType("app.thesmos")
    providers_module = types.ModuleType("app.thesmos.providers")

    sys.modules["app"] = app_module
    sys.modules["app.thesmos"] = thesmos_module
    sys.modules["app.thesmos.exceptions"] = exceptions_module
    sys.modules["app.thesmos.providers"] = providers_module
    sys.modules["app.thesmos.providers.base"] = providers_base_module

    eliot_module = types.ModuleType("eliot")

    def log_call(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    eliot_module.log_call = log_call
    sys.modules["eliot"] = eliot_module

    eosin_provider_spec = importlib.util.spec_from_file_location("legacy_eosin_provider", Path("eosin.py"))
    assert eosin_provider_spec is not None
    eosin_provider_module = importlib.util.module_from_spec(eosin_provider_spec)
    assert eosin_provider_spec.loader is not None
    eosin_provider_spec.loader.exec_module(eosin_provider_module)
    return eosin_provider_module


def test_env_example_has_modal_defaults() -> None:
    env_example = Path(".env.example").read_text()

    assert "EOSIN_MODAL_GPU=L40S" in env_example
    assert "EOSIN_MODAL_CPU=8.0" in env_example
    assert "EOSIN_MODAL_MIN_CONTAINERS=0" in env_example
    assert "EOSIN_MODAL_ALLOW_ALWAYS_ON=false" in env_example
    assert "EOSIN_MODAL_MAX_CONTAINERS=1" in env_example
    assert "EOSIN_MODAL_REQUIRES_PROXY_AUTH=true" in env_example
    assert "EOSIN_MODAL_MAX_INPUTS=35" in env_example
    assert "EOSIN_MODAL_TARGET_INPUTS=35" in env_example
    assert "EOSIN_MODAL_SCALEDOWN_WINDOW=120" in env_example
    assert "EOSIN_MODAL_STARTUP_TIMEOUT=900" in env_example
    assert "EOSIN_MODAL_MEMORY_MIB=32768" in env_example
    assert "EOSIN_MODAL_OCR_PIPELINE_WORKERS=16" in env_example
    assert "EOSIN_MODAL_VLLM_MAX_MODEL_LEN=12288" in env_example
    assert 'EOSIN_MODAL_VLLM_SPECULATIVE_CONFIG={"method": "mtp", "num_speculative_tokens": 3}' in env_example
    assert "EOSIN_MODAL_VLLM_MAX_NUM_SEQS=196" in env_example
    assert "EOSIN_MODAL_VLLM_MAX_BATCHED_TOKENS=24576" in env_example
    assert "EOSIN_MODAL_VLLM_ENABLE_GPU_SNAPSHOT=false" in env_example
    assert "EOSIN_MODAL_ALLOW_GPU_SNAPSHOT=false" in env_example
    assert "EOSIN_MODAL_VLLM_ENABLE_SLEEP_MODE=false" in env_example
    assert "EOSIN_MODAL_VLLM_SNAPSHOT_WARMUP_ENABLE=false" in env_example
    assert "EOSIN_MODAL_VLLM_COMMIT_CACHE_AFTER_START=false" in env_example
    assert "EOSIN_MODAL_VLLM_SNAPSHOT_WARMUP_MODE=multimodal" in env_example
    assert "EOSIN_MODAL_HF_XET_HIGH_PERFORMANCE=1" in env_example
    assert "EOSIN_MODAL_TORCH_NCCL_ENABLE_MONITORING=0" in env_example
    assert "EOSIN_MODAL_PARSER_POOL_SIZE=2" in env_example
    assert "EOSIN_MODAL_BANK_PARSER_OCR_BACKEND_MODE=page_batch" in env_example
    assert "EOSIN_MODAL_VLLM_ENABLE_PREFIX_CACHING=true" in env_example
    assert "BANK_PARSER_OCR_BACKEND_MODE=page_batch" in env_example
    assert "BANK_PARSER_LAYOUT_MODE=auto" in env_example
    assert "BANK_PARSER_RECOVER_LAYOUT_MISSED_PAGES=true" in env_example
    assert "BANK_PARSER_ENABLE_PAGE_OCR_RETRY=false" in env_example
    assert "BANK_PARSER_PAGE_OCR_RETRY_ALL=false" in env_example
    assert "BANK_PARSER_PAGE_OCR_RETRY_DPI=300" in env_example
    assert "BANK_PARSER_CAPTURE_RAW_OCR_DEBUG=false" in env_example
    assert "BANK_PARSER_MAX_UPLOAD_BYTES" not in env_example
    assert "BANK_PARSER_MAX_PAGES" not in env_example
    assert "BANK_PARSER_POOL_WAIT_TIMEOUT=1800" in env_example
    assert "BANK_PARSER_OCR_MAX_IMAGE_SIDE=3500" in env_example
    assert "BANK_PARSER_OCR_MAX_IMAGE_PIXELS=9000000" in env_example
    assert "BANK_PARSER_OCR_DOCUMENT_MAX_TOKENS_CAP=4096" in env_example
    assert "BANK_PARSER_OCR_DOCUMENT_MAX_IMAGES_PER_REQUEST=4" in env_example
    assert "BANK_PARSER_OCR_BATCH_DRAIN_MAX_BATCH_SIZE=96" in env_example
    assert "BANK_PARSER_OCR_BATCH_DRAIN_MAX_WAIT_SECONDS=0.05" in env_example
    assert "EOSIN_MODAL_METRICS_PUSH_ENABLE=false" in env_example
    assert "EOSIN_MODAL_METRICS_PUSH_TRANSPORT=direct" in env_example
    assert "BANK_PARSER_METRICS_PUSH_ENABLE=false" in env_example
    assert "BANK_PARSER_METRICS_PUSH_TRANSPORT=tailscale-socks5" in env_example


def test_modal_image_uses_nested_eosin_package_path() -> None:
    modal_app = Path("modal_app.py").read_text()

    assert '"PYTHONPATH": "/root/eosin:/root"' in modal_app


def test_local_artifacts_are_ignored() -> None:
    gitignore = Path(".gitignore").read_text()

    assert "bank statements/" in gitignore
    assert "/load-test-results*/" in gitignore
    assert ".env" in gitignore
    assert "!.env.example" in gitignore


def test_readme_does_not_include_hardcoded_modal_endpoint() -> None:
    readme = Path("README.md").read_text()

    assert "noelalex" not in readme
    assert "https://<workspace>--bank-parser.modal.run" in readme


def test_readme_documents_proxy_auth_headers() -> None:
    readme = Path("README.md").read_text()

    assert "Modal-Key" in readme
    assert "Modal-Secret" in readme
    assert "requires_proxy_auth" in readme


def test_load_test_dataframe_markdown_escapes_cells() -> None:
    dataframe = pd.DataFrame([{"Description": "A | B", "Amount": 10}])

    markdown = load_test_module.dataframe_to_markdown(dataframe)

    assert "| Description | Amount |" in markdown
    assert "A \\| B" in markdown


def test_local_quality_runner_targets_local_page_http_backend() -> None:
    overrides = local_quality_module.build_quality_env(
        vllm_host="127.0.0.1",
        vllm_port=8080,
        model_name="default",
    )

    assert overrides["GLMOCR_OCR_API_HOST"] == "127.0.0.1"
    assert overrides["GLMOCR_OCR_API_PORT"] == "8080"
    assert overrides["GLMOCR_OCR_MODEL"] == "default"
    assert overrides["BANK_PARSER_OCR_BACKEND_MODE"] == "page_http"
    assert overrides["BANK_PARSER_ENABLE_OCR_BATCHING"] == "false"
    assert overrides["BANK_PARSER_ENABLE_PAGE_OCR_RETRY"] == "false"
    assert overrides["BANK_PARSER_LAYOUT_MODE"] == "auto"
    assert overrides["BANK_PARSER_PAGE_OCR_RETRY_ALL"] == "false"
    assert overrides["BANK_PARSER_CAPTURE_RAW_OCR_DEBUG"] == "true"
    assert overrides["BANK_PARSER_OCR_MAX_IMAGE_SIDE"] == "3500"
    assert overrides["BANK_PARSER_OCR_MAX_IMAGE_PIXELS"] == "9000000"


def test_legacy_eosin_provider_adds_modal_proxy_auth_headers(monkeypatch) -> None:
    monkeypatch.setenv("EOSIN_PARSER_BASE_URL", "https://example.modal.run")
    monkeypatch.setenv("MODAL_PROXY_AUTH_TOKEN_ID", "wk-test")
    monkeypatch.setenv("MODAL_PROXY_AUTH_TOKEN_SECRET", "ws-test")

    module = _load_eosin_provider_module()
    provider = module.EosinPDFProvider()

    assert provider._request_headers()["Modal-Key"] == "wk-test"
    assert provider._request_headers()["Modal-Secret"] == "ws-test"


def test_legacy_eosin_provider_accepts_personal_modal_proxy_auth_aliases(monkeypatch) -> None:
    monkeypatch.setenv("EOSIN_PARSER_BASE_URL", "https://example.modal.run")
    monkeypatch.delenv("MODAL_PROXY_AUTH_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_PROXY_AUTH_TOKEN_SECRET", raising=False)
    monkeypatch.setenv("MODAL_PERSONAL_PROXY_AUTH_TOKEN_ID", "wk-personal")
    monkeypatch.setenv("MODAL_PERSONAL_PROXY_AUTH_TOKEN_SECRET", "ws-personal")

    module = _load_eosin_provider_module()
    provider = module.EosinPDFProvider()

    assert provider._request_headers()["Modal-Key"] == "wk-personal"
    assert provider._request_headers()["Modal-Secret"] == "ws-personal"


def test_legacy_eosin_provider_rejects_missing_required_modal_proxy_auth(monkeypatch) -> None:
    monkeypatch.setenv("EOSIN_PARSER_BASE_URL", "https://example.modal.run")
    monkeypatch.setenv("EOSIN_PARSER_REQUIRE_PROXY_AUTH", "true")
    monkeypatch.delenv("MODAL_PROXY_AUTH_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_PROXY_AUTH_TOKEN_SECRET", raising=False)

    module = _load_eosin_provider_module()

    with pytest.raises(ValueError, match="Modal proxy auth is required"):
        module.EosinPDFProvider()


def test_load_test_summary_includes_pages_per_second() -> None:
    started_at = pd.Timestamp("2026-05-06T12:00:00Z").to_pydatetime()
    completed_at = pd.Timestamp("2026-05-06T12:00:20Z").to_pydatetime()
    plan = [
        load_test_module.PlannedRequest(1, 1, "bank-a/a.pdf", "bank-a", 100),
        load_test_module.PlannedRequest(2, 1, "bank-b/b.pdf", "bank-b", 200),
    ]
    results = [
        load_test_module.RequestResult(
            request_index=1,
            corpus_cycle=1,
            pdf_path="bank-a/a.pdf",
            bank_name="bank-a",
            file_size_bytes=100,
            started_at_utc="2026-05-06T12:00:00Z",
            completed_at_utc="2026-05-06T12:00:10Z",
            latency_seconds=10.0,
            status_code=200,
            ok=True,
            error_type=None,
            error_message=None,
            response_bytes=1000,
            rows_returned=10,
            columns_returned=4,
            page_count=4,
            pages_with_tables_count=2,
            server_total_seconds=9.0,
            server_render_seconds=1.0,
            server_layout_seconds=1.0,
            server_ocr_seconds=6.0,
            server_parse_seconds=1.0,
            server_ocr_task_count=1.0,
            server_ocr_queue_wait_mean=0.2,
            server_ocr_queue_wait_max=0.3,
            server_ocr_request_mean=5.0,
            server_ocr_request_max=5.0,
            response_json_path="resp-a.json",
            dataframe_markdown_path="df-a.md",
        ),
        load_test_module.RequestResult(
            request_index=2,
            corpus_cycle=1,
            pdf_path="bank-b/b.pdf",
            bank_name="bank-b",
            file_size_bytes=200,
            started_at_utc="2026-05-06T12:00:02Z",
            completed_at_utc="2026-05-06T12:00:20Z",
            latency_seconds=18.0,
            status_code=200,
            ok=True,
            error_type=None,
            error_message=None,
            response_bytes=2000,
            rows_returned=20,
            columns_returned=4,
            page_count=6,
            pages_with_tables_count=3,
            server_total_seconds=17.0,
            server_render_seconds=1.5,
            server_layout_seconds=1.5,
            server_ocr_seconds=13.0,
            server_parse_seconds=1.0,
            server_ocr_task_count=1.0,
            server_ocr_queue_wait_mean=0.4,
            server_ocr_queue_wait_max=0.5,
            server_ocr_request_mean=11.0,
            server_ocr_request_max=11.0,
            response_json_path="resp-b.json",
            dataframe_markdown_path="df-b.md",
        ),
    ]

    summary = load_test_module.summarize_results(
        endpoint="https://example.com",
        corpus_dir=Path("bank statements"),
        plan=plan,
        results=results,
        started_at=started_at,
        completed_at=completed_at,
        concurrency=2,
        timeout=1800,
    )

    assert summary["pages_processed_total"] == 10
    assert summary["pages_per_second"] == 0.5


def test_make_results_dir_adds_suffix_on_collision(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(load_test_module, "utc_now", lambda: pd.Timestamp("2026-04-24T16:17:39Z").to_pydatetime())

    first = load_test_module.make_results_dir(tmp_path)
    second = load_test_module.make_results_dir(tmp_path)

    assert first.name == "20260424T161739Z"
    assert second.name == "20260424T161739Z-001"


def test_monitoring_stack_files_exist() -> None:
    assert Path("monitoring/docker-compose.metrics.yml").exists()
    assert Path("monitoring/README.md").exists()
    assert Path("monitoring/grafana/dashboards/eosin-overview.json").exists()
    assert not Path("monitoring/vmagent/promscrape.yml").exists()
    assert not Path("monitoring/blackbox-exporter/config.yml").exists()


def test_monitoring_stack_uses_env_driven_modal_target() -> None:
    compose = Path("monitoring/docker-compose.metrics.yml").read_text()
    stack_readme = Path("monitoring/README.md").read_text()
    alert_rules = Path("monitoring/vmalert/eosin.rules.yml").read_text()
    dashboard = Path("monitoring/grafana/dashboards/eosin-overview.json").read_text()

    assert "victoriametrics" in compose
    assert "grafana" in compose
    assert "blackbox-exporter" not in compose
    assert "vmagent" not in compose
    assert "push-only" in stack_readme
    assert "Do not point Prometheus-style scrapers at the Modal worker endpoint." in stack_readme
    assert "probe_success" not in alert_rules
    assert "probe_success" not in dashboard


def test_readme_documents_push_metrics_and_secret_setup() -> None:
    readme = Path("README.md").read_text()

    assert "modal secret create eosin-tailscale" in readme
    assert "modal secret create eosin-metrics-push" in readme
    assert "pull-based monitoring is intentionally disabled for the Modal worker" in readme


def test_modal_config_uses_model_native_table_prompt() -> None:
    modal_config = Path("eosin/backend/config.modal.yaml").read_text()

    assert 'table: "Table Recognition:"' in modal_config
    assert "max_tokens: 7000" in modal_config
    assert "request_timeout: 300" in modal_config
    assert "Return exactly one valid HTML <table>" not in modal_config


def test_modal_image_reinstalls_stable_transformers_last() -> None:
    modal_app = Path("modal_app.py").read_text()
    pyproject = Path("pyproject.toml").read_text()

    assert '"vllm/vllm-openai:v0.19.0-ubuntu2404"' in modal_app
    assert "pip install --no-cache-dir --force-reinstall 'setuptools<81'" in modal_app
    assert "pip install --no-cache-dir --force-reinstall --no-deps 'huggingface-hub==1.13.0'" in modal_app
    assert "pip install --no-cache-dir --force-reinstall --no-deps 'transformers==5.6.2'" in modal_app
    assert "pip install --ignore-installed --no-deps blinker glmocr pandas beautifulsoup4" in modal_app
    assert "pip install --ignore-installed blinker 'glmocr[selfhosted,server]' pandas beautifulsoup4" not in modal_app
    assert 'transformers = ">=5.6.0"' in pyproject
    assert '"HF_XET_HIGH_PERFORMANCE": os.getenv("EOSIN_MODAL_HF_XET_HIGH_PERFORMANCE", "1")' in modal_app
    assert '"TORCHINDUCTOR_COMPILE_THREADS": os.getenv("EOSIN_MODAL_TORCHINDUCTOR_COMPILE_THREADS", "1")' in modal_app
    assert '"TORCH_NCCL_ENABLE_MONITORING": os.getenv("EOSIN_MODAL_TORCH_NCCL_ENABLE_MONITORING", "0")' in modal_app
    assert "git+https://github.com/huggingface/transformers.git" not in modal_app
    assert "--outbound-http-proxy-listen" not in modal_app


def test_modal_image_uses_page_http_default_and_metrics_auth_value() -> None:
    modal_app = Path("modal_app.py").read_text()

    assert 'OCR_BACKEND_MODE = os.getenv("EOSIN_MODAL_BANK_PARSER_OCR_BACKEND_MODE", "page_batch")' in modal_app
    assert 'ENABLE_PAGE_OCR_RETRY = os.getenv("BANK_PARSER_ENABLE_PAGE_OCR_RETRY", "false").lower() == "true"' in modal_app
    assert '"BANK_PARSER_ENABLE_PAGE_OCR_RETRY": "true" if ENABLE_PAGE_OCR_RETRY else "false"' in modal_app
    assert '"BANK_PARSER_SAVE_DEBUG_IMAGES": "true" if SAVE_DEBUG_IMAGES else "false"' in modal_app
    assert '"BANK_PARSER_PAGE_OCR_RETRY_ALL": os.getenv("BANK_PARSER_PAGE_OCR_RETRY_ALL", "false")' in modal_app
    assert '"BANK_PARSER_CAPTURE_RAW_OCR_DEBUG": "true" if CAPTURE_RAW_OCR_DEBUG else "false"' in modal_app
    assert '"BANK_PARSER_OCR_MAX_IMAGE_SIDE": os.getenv("BANK_PARSER_OCR_MAX_IMAGE_SIDE", "3500")' in modal_app
    assert '"BANK_PARSER_OCR_MAX_IMAGE_PIXELS": os.getenv("BANK_PARSER_OCR_MAX_IMAGE_PIXELS", "9000000")' in modal_app
    assert 'METRICS_PUSH_ENABLE = _env_bool("EOSIN_MODAL_METRICS_PUSH_ENABLE", False)' in modal_app
    assert '"BANK_PARSER_METRICS_PUSH_ENABLE": "true" if METRICS_PUSH_ENABLE else "false"' in modal_app
    assert '"BANK_PARSER_METRICS_PUSH_ENABLE": os.getenv("BANK_PARSER_METRICS_PUSH_ENABLE", "false")' not in modal_app
    assert "enable_page_ocr_retry=ENABLE_PAGE_OCR_RETRY" in modal_app
    assert "save_debug_images=SAVE_DEBUG_IMAGES" in modal_app
    assert "capture_raw_ocr_debug=CAPTURE_RAW_OCR_DEBUG" in modal_app
    assert '"BANK_PARSER_METRICS_PUSH_AUTH_VALUE": os.getenv("BANK_PARSER_METRICS_PUSH_AUTH_VALUE", "")' not in modal_app
    assert 'modal.Secret.from_name("eosin-metrics-push")' in modal_app
    assert "vllm_cache_volume.commit()" in modal_app
    assert "hf_cache_volume.commit()" in modal_app
    assert "min_containers=MIN_CONTAINERS" in modal_app
    assert "EOSIN_MODAL_ALLOW_ALWAYS_ON" in modal_app


def test_modal_starts_vllm_before_parser_imports_and_waits_afterward() -> None:
    modal_app = Path("modal_app.py").read_text()
    enter_body = modal_app.split("def enter(self) -> None:", maxsplit=1)[1].split(
        "@modal.exit()", maxsplit=1
    )[0]

    launch_index = enter_body.index("self._launch_vllm()")
    prepare_index = enter_body.index("self._prepare_runtime()")
    await_index = enter_body.index("self._await_vllm_ready()")

    assert launch_index < prepare_index < await_index


def test_modal_can_disable_speculative_decoding_for_benchmarks() -> None:
    modal_app = Path("modal_app.py").read_text()

    assert "if VLLM_SPECULATIVE_CONFIG:" in modal_app
    assert 'vllm_args.extend(["--speculative-config", VLLM_SPECULATIVE_CONFIG])' in modal_app


def test_debug_mode_saves_all_header_stitching_artifacts() -> None:
    pipeline = Path("eosin/backend/eosin_pipeline.py").read_text()
    runner = Path("scripts/run_local_quality_bank_parser.py").read_text()
    modal_app = Path("modal_app.py").read_text()

    assert "page_idx == 7" not in pipeline
    assert 'f"page_{page_idx + 1}_crop.png"' in pipeline
    assert 'f"page_{page_idx + 1}_stitched.png"' in pipeline
    assert 'f"page_{page_idx + 1}_ocr_input.png"' in pipeline
    assert '"extracted_header.png"' in pipeline
    assert '"debug_image_dir"' in pipeline
    assert "--debug" in runner
    assert "save_debug_images=bool(args.debug)" in runner
    assert "EOSIN_MODAL_ALLOW_GPU_SNAPSHOT" in modal_app
    assert "startup_timeout=STARTUP_TIMEOUT_SECONDS" in modal_app
    assert "Runtime preparation: importing parser modules" in modal_app
    assert "Snapshot phase: starting vLLM" not in modal_app
    assert "Restore phase: waking vLLM" not in modal_app
    assert "process=self.vllm_process" in modal_app
    assert "import eosin.backend.eosin_pipeline" in modal_app
    assert 'VLLM_SNAPSHOT_WARMUP_REQUESTS = int(os.getenv("EOSIN_MODAL_VLLM_SNAPSHOT_WARMUP_REQUESTS", "3"))' in modal_app
    assert 'Path(TRITON_CACHE_DIR).mkdir(parents=True, exist_ok=True)' in modal_app
    assert 'Path(TORCHINDUCTOR_CACHE_DIR).mkdir(parents=True, exist_ok=True)' in modal_app
    assert 'VLLM_FAST_BOOT = _env_bool("EOSIN_MODAL_VLLM_FAST_BOOT", False)' in modal_app
    assert 'VLLM_ENABLE_GPU_SNAPSHOT = _env_bool("EOSIN_MODAL_VLLM_ENABLE_GPU_SNAPSHOT", False)' in modal_app
    assert 'VLLM_SNAPSHOT_WARMUP_MODE = os.getenv("EOSIN_MODAL_VLLM_SNAPSHOT_WARMUP_MODE", "multimodal").strip().lower()' in modal_app
    assert "--disable-frontend-multiprocessing" not in modal_app
    assert "VLLM_WARMUP_IMAGE_DATA_URL" in modal_app
    assert '"type": "image_url"' in modal_app
    assert '@modal.enter(snap=True)' not in modal_app
    assert "enable_memory_snapshot=False" in modal_app
    assert "experimental_options={}" in modal_app


def test_metrics_endpoint_is_exposed(monkeypatch) -> None:
    pytest.importorskip("prometheus_client")
    monkeypatch.setenv("BANK_PARSER_METRICS_ENABLE_VLLM_PROXY", "false")
    monkeypatch.setattr(
        "eosin.backend.bank_parser_api.get_metrics_manager",
        lambda: _NoopMetricsManager(),
    )

    from eosin.backend.bank_parser_api import create_app

    app = create_app(service=object())
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/metrics")
    response = asyncio.run(endpoint())

    assert response.status_code == 200
    assert "eosin_container_info" in response.body.decode()


class _NoopMetricsManager:
    def start_background_samplers(self) -> None:
        return None

    def render_metrics(self):
        from eosin.backend.metrics import get_metrics_manager

        return get_metrics_manager().render_metrics()


def test_modal_asgi_app_requires_proxy_auth() -> None:
    modal_app = Path("modal_app.py").read_text()

    assert 'REQUIRES_PROXY_AUTH = _env_bool("EOSIN_MODAL_REQUIRES_PROXY_AUTH", True)' in modal_app
    assert "@modal.asgi_app(label=WEB_LABEL, requires_proxy_auth=REQUIRES_PROXY_AUTH)" in modal_app


def test_modal_concurrency_is_bounded_to_measured_capacity() -> None:
    modal_app = Path("modal_app.py").read_text()

    assert "def _bounded_modal_inputs() -> tuple[int, int]:" in modal_app
    assert "safe_max_inputs = 35" in modal_app
    assert 'MAX_INPUTS, TARGET_INPUTS = _bounded_modal_inputs()' in modal_app
    assert 'min(_env_int("EOSIN_MODAL_TARGET_INPUTS", max_inputs), max_inputs)' in modal_app


def test_modal_startup_logs_readiness_critical_phases() -> None:
    modal_app = Path("modal_app.py").read_text()

    assert "vLLM health ready after" in modal_app
    assert "vLLM cache commit phase completed in" in modal_app
    assert "Parser service construction completed in" in modal_app
    assert "Restore phase completed in" in modal_app


def test_modal_extract_evidence_uses_direct_evidence_only_service_path() -> None:
    tree = ast.parse(Path("modal_app.py").read_text())
    modal_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BankParserModalApp"
    )
    method = next(
        node
        for node in modal_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "extract_evidence"
    )
    called_attributes = {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "extract_glm_page_html_bytes" in called_attributes
    assert "parse_pdf_bytes" not in called_attributes


def test_modal_rpc_evidence_methods_validate_pdf_payloads() -> None:
    tree = ast.parse(Path("modal_app.py").read_text())
    modal_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BankParserModalApp"
    )

    for method_name in ("extract_evidence", "extract_glm_page_html"):
        method = next(
            node
            for node in modal_class.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "validate_pdf_upload"
        ]
        assert calls, method_name
