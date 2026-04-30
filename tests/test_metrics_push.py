from __future__ import annotations

from eosin.backend.metrics import MetricsPushClient


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
        push_url="http://100.99.119.108:8428/api/v1/import/prometheus",
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
