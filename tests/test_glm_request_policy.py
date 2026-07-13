from __future__ import annotations

from PIL import Image

from eosin.backend.glm_request_policy import (
    ADAPTIVE_FREQ_PENALTIES,
    apply_glm_generation_policy,
    glm_output_quality,
)



class FakePageLoader:
    def build_request_from_image(self, _image, *, task_type: str) -> dict:
        return {
            "messages": [{"role": "user", "content": [{"type": "text", "text": task_type}]}],
            "max_tokens": 100,
        }


def make_image() -> Image.Image:
    return Image.new("RGB", (8, 8), "white")


def test_eosin_glm_retry_policy_starts_at_zero_frequency_penalty() -> None:
    assert ADAPTIVE_FREQ_PENALTIES == [0.0, 0.10, 0.20, 0.30, 0.50]


def test_apply_glm_generation_policy_sets_attempt_parameters() -> None:
    request = FakePageLoader().build_request_from_image(make_image(), task_type="table")

    updated = apply_glm_generation_policy(request, frequency_penalty=0.15)

    assert updated is request
    assert request["frequency_penalty"] == 0.15
    assert request["repetition_penalty"] == 1.2
    assert request["stop"] == ["</html>"]


def test_eosin_glm_quality_detects_runaway_repetition() -> None:
    html = "<table><tr><td>" + ("AB" * 30) + "</td></tr></table>"

    assert glm_output_quality(html) == "bad"


def test_http_backend_retries_with_higher_frequency_penalty_on_repetition() -> None:
    from eosin.backend.ocr_pipeline import HTTPDocumentOCRBackend

    class RepeatingThenGoodOCRClient:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        def process(self, request: dict) -> tuple[dict, int]:
            self.requests.append(dict(request))
            if len(self.requests) == 1:
                content = "<table><tr><td>" + ("AB" * 30) + "</td></tr></table>"
            else:
                content = "<table><tr><td>Date</td></tr><tr><td>01/01/2024</td></tr></table>"
            return {"choices": [{"message": {"content": content}}]}, 200

    client = RepeatingThenGoodOCRClient()
    backend = HTTPDocumentOCRBackend(
        page_loader=FakePageLoader(),
        ocr_client=client,
        max_workers=1,
        queue_size=8,
    )
    try:
        result = backend.submit([make_image()], page_indices=[0], task_type="table").result(timeout=2)
    finally:
        backend.close()

    assert [request["frequency_penalty"] for request in client.requests] == [0.0, 0.10]
    assert result.content == "<table><tr><td>Date</td></tr><tr><td>01/01/2024</td></tr></table>"


def test_http_backend_keeps_least_repetitive_bad_retry() -> None:
    from eosin.backend.ocr_pipeline import HTTPDocumentOCRBackend

    class BadRetryOCRClient:
        def __init__(self) -> None:
            self.requests: list[dict] = []

        def process(self, request: dict) -> tuple[dict, int]:
            self.requests.append(dict(request))
            repeats = 500 if len(self.requests) == 1 else 30
            content = "<table><tr><td>" + ("AB" * repeats) + "</td></tr></table>"
            return {"choices": [{"message": {"content": content}}]}, 200

    client = BadRetryOCRClient()
    backend = HTTPDocumentOCRBackend(
        page_loader=FakePageLoader(),
        ocr_client=client,
        max_workers=1,
        queue_size=8,
    )
    try:
        result = backend.submit([make_image()], page_indices=[0], task_type="table").result(timeout=2)
    finally:
        backend.close()

    assert len(client.requests) == len(ADAPTIVE_FREQ_PENALTIES)
    assert result.content == "<table><tr><td>" + ("AB" * 30) + "</td></tr></table>"
