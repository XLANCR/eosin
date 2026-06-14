from __future__ import annotations

import time

from eosin.backend.metrics import MetricsPushClient
from eosin.backend.metrics import MetricsManager


class RecordingSession:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def post(self, url: str, data: bytes, headers: dict[str, str], timeout: float, proxies: dict[str, str] | None = None):
        self.calls.append(
            {
                "url": url,
                "data": data,
                "headers": headers,
                "timeout": timeout,
                "proxies": proxies,
            }
        )
        return None


def test_metrics_push_client_sends_auth_and_tailscale_proxy() -> None:
    session = RecordingSession()
    client = MetricsPushClient(
        enabled=True,
        push_url="http://100.64.0.1:8428/api/v1/import/prometheus",
        timeout_seconds=1.5,
        auth_header_name="X-Eosin-Key",
        auth_header_value="secret",
        transport="tailscale-socks5",
        socks5_url="socks5h://127.0.0.1:1055",
        session=session,
    )

    client.push(b"metric 1\n")

    assert len(session.calls) == 1
    assert session.calls[0]["headers"]["X-Eosin-Key"] == "secret"
    assert session.calls[0]["proxies"] == {
        "http": "socks5h://127.0.0.1:1055",
        "https": "socks5h://127.0.0.1:1055",
    }


def test_metrics_push_client_noops_when_disabled() -> None:
    session = RecordingSession()
    client = MetricsPushClient(
        enabled=False,
        push_url="http://example.invalid",
        timeout_seconds=1.0,
        auth_header_name="",
        auth_header_value="",
        transport="direct",
        socks5_url="",
        session=session,
    )

    client.push(b"metric 1\n")

    assert session.calls == []


def test_metrics_manager_pushes_periodically_while_requests_are_active(monkeypatch) -> None:
    monkeypatch.setenv("BANK_PARSER_METRICS_PUSH_ENABLE", "true")
    monkeypatch.setenv("BANK_PARSER_METRICS_PUSH_INTERVAL", "1.0")
    manager = MetricsManager()

    class RecordingPushClient:
        enabled = True

        def __init__(self) -> None:
            self.calls = 0

        def push(self, payload: bytes) -> None:
            self.calls += 1

    push_client = RecordingPushClient()
    manager._push_client = push_client
    manager._active_push_interval = 0.05
    manager.start_background_samplers()
    manager.track_request_started(1024)
    time.sleep(0.16)
    manager.track_request_finished(0.2)
    time.sleep(0.08)

    assert push_client.calls >= 2
