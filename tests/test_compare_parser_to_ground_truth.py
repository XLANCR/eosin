from pathlib import Path

import pytest

from scripts import compare_parser_to_ground_truth as compare


def test_production_endpoint_requires_explicit_opt_in() -> None:
    endpoint = "https://workspace--bank-parser.modal.run/parse/bank-statement"

    with pytest.raises(ValueError, match="Refusing the production"):
        compare.validate_endpoint(endpoint, allow_production=False)

    assert compare.validate_endpoint(endpoint, allow_production=True) == endpoint


def test_proxy_auth_headers_use_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODAL_PROXY_AUTH_TOKEN_ID", "wk-test")
    monkeypatch.setenv("MODAL_PROXY_AUTH_TOKEN_SECRET", "ws-test")

    assert compare.proxy_auth_headers() == {
        "Modal-Key": "wk-test",
        "Modal-Secret": "ws-test",
    }


def test_dev_deploy_script_cannot_inherit_production_names() -> None:
    script = Path("scripts/deploy_dev.ps1").read_text()

    assert '$env:PYTHONUTF8 = "1"' in script
    assert '$env:MODAL_ENVIRONMENT = "dev"' in script
    assert '$env:EOSIN_MODAL_APP_NAME = "eosin-glm-ocr-dev"' in script
    assert '$env:EOSIN_MODAL_WEB_LABEL = "bank-parser-dev"' in script
    assert '$env:EOSIN_MODAL_TAILSCALE_ENABLE = "false"' in script
    assert '$env:EOSIN_MODAL_METRICS_PUSH_ENABLE = "false"' in script
    assert "--env dev" in script
