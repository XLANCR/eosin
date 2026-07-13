from PIL import Image

from eosin.backend.ocr_pipeline import build_document_request


class PageLoader:
    def build_request_from_image(
        self, image: Image.Image, task_type: str = "text"
    ) -> dict:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "image://page"}},
                        {"type": "text", "text": f"task={task_type}"},
                    ],
                }
            ],
            "max_tokens": 7000,
        }


def test_build_document_request_honors_page_token_cap(monkeypatch) -> None:
    monkeypatch.setenv("BANK_PARSER_OCR_PAGE_MAX_TOKENS", "4096")

    request = build_document_request(
        PageLoader(),
        [Image.new("RGB", (1, 1))],
        task_type="table",
    )

    assert request["max_tokens"] == 4096
