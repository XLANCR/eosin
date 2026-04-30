from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

load_test_spec = importlib.util.spec_from_file_location(
    "load_test_bank_parser",
    Path("scripts/load_test_bank_parser.py"),
)
assert load_test_spec is not None
load_test_module = importlib.util.module_from_spec(load_test_spec)
assert load_test_spec.loader is not None
sys.modules[load_test_spec.name] = load_test_module
load_test_spec.loader.exec_module(load_test_module)


def test_env_example_has_modal_defaults() -> None:
    env_example = Path(".env.example").read_text()

    assert "EOSIN_MODAL_GPU=A100-80GB" in env_example
    assert "EOSIN_MODAL_MAX_CONTAINERS=1" in env_example
    assert "EOSIN_MODAL_MAX_INPUTS=256" in env_example
    assert "EOSIN_MODAL_TARGET_INPUTS=256" in env_example
    assert "EOSIN_MODAL_SCALEDOWN_WINDOW=180" in env_example
    assert "EOSIN_MODAL_MEMORY_MIB=40960" in env_example
    assert "EOSIN_MODAL_OCR_PIPELINE_WORKERS=128" in env_example
    assert "EOSIN_MODAL_VLLM_MAX_BATCHED_TOKENS=32768" in env_example
    assert "EOSIN_MODAL_VLLM_ENABLE_PREFIX_CACHING=true" in env_example
    assert "BANK_PARSER_OCR_BACKEND_MODE=document_http" in env_example
    assert "BANK_PARSER_OCR_BATCH_DRAIN_MAX_BATCH_SIZE=96" in env_example
    assert "BANK_PARSER_OCR_BATCH_DRAIN_MAX_WAIT_SECONDS=1.0" in env_example
    assert "BANK_PARSER_METRICS_PUSH_ENABLE=true" in env_example
    assert "BANK_PARSER_METRICS_PUSH_TRANSPORT=tailscale-socks5" in env_example


def test_local_artifacts_are_ignored() -> None:
    gitignore = Path(".gitignore").read_text()

    assert "/bank statements" in gitignore
    assert "/load-test-results/" in gitignore
    assert ".env" in gitignore
    assert "!.env.example" in gitignore


def test_readme_does_not_include_hardcoded_modal_endpoint() -> None:
    readme = Path("README.md").read_text()

    assert "noelalex" not in readme
    assert "https://<workspace>--bank-parser.modal.run" in readme


def test_load_test_dataframe_markdown_escapes_cells() -> None:
    dataframe = pd.DataFrame([{"Description": "A | B", "Amount": 10}])

    markdown = load_test_module.dataframe_to_markdown(dataframe)

    assert "| Description | Amount |" in markdown
    assert "A \\| B" in markdown


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
    assert Path("docs/monitoring/2026-04-24-monitoring-stack-report.md").exists()


def test_monitoring_stack_uses_env_driven_modal_target() -> None:
    compose = Path("monitoring/docker-compose.metrics.yml").read_text()
    stack_readme = Path("monitoring/README.md").read_text()

    assert "victoriametrics" in compose
    assert "grafana" in compose
    assert "blackbox-exporter" not in compose
    assert "vmagent" not in compose
    assert "push-only" in stack_readme
    assert "Do not point Prometheus-style scrapers at the Modal worker endpoint." in stack_readme


def test_readme_documents_push_metrics_and_secret_setup() -> None:
    readme = Path("README.md").read_text()

    assert "modal secret create eosin-tailscale" in readme
    assert "modal secret create eosin-metrics-push" in readme
    assert "pull-based monitoring is intentionally disabled for the Modal worker" in readme


def test_modal_image_reinstalls_stable_transformers_last() -> None:
    modal_app = Path("modal_app.py").read_text()

    assert "pip install --no-cache-dir --force-reinstall 'setuptools<81'" in modal_app
    assert "pip install --no-cache-dir --force-reinstall --no-deps 'transformers==5.6.2'" in modal_app
    assert "git+https://github.com/huggingface/transformers.git" not in modal_app
    assert "--outbound-http-proxy-listen" not in modal_app


def test_modal_image_uses_document_http_default_and_metrics_auth_value() -> None:
    modal_app = Path("modal_app.py").read_text()

    assert 'OCR_BACKEND_MODE = os.getenv("BANK_PARSER_OCR_BACKEND_MODE", "document_http")' in modal_app
    assert '"BANK_PARSER_METRICS_PUSH_AUTH_VALUE": os.getenv("BANK_PARSER_METRICS_PUSH_AUTH_VALUE", "")' in modal_app


def test_metrics_endpoint_is_exposed(monkeypatch) -> None:
    pytest.importorskip("prometheus_client")
    monkeypatch.setenv("BANK_PARSER_METRICS_ENABLE_VLLM_PROXY", "false")

    from eosin.backend.bank_parser_api import create_app

    client = TestClient(create_app(service=object()))
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "eosin_container_info" in response.text
