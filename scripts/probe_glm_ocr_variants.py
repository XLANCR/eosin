from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path
from typing import Iterable

import fitz
import requests
from PIL import Image, ImageChops, ImageOps


DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_MODEL = "default"
PROMPT = "Table Recognition:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe raw GLM-OCR output for PDF page image variants.")
    parser.add_argument("pdf", help="PDF path")
    parser.add_argument("--pages", nargs="+", type=int, required=True, help="1-based page numbers to probe")
    parser.add_argument("--dpi", nargs="+", type=int, default=[200, 300], help="Render DPI values")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="OpenAI-compatible chat completions endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Served model name")
    parser.add_argument("--output-dir", default="glm-ocr-probes", help="Directory for probe artifacts")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[],
        help="Optional variant names to run. Defaults to all variants.",
    )
    return parser.parse_args()


def render_page(pdf_path: Path, page_number: int, dpi: int) -> Image.Image:
    scale = dpi / 72.0
    with fitz.open(pdf_path) as document:
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        mode = "RGB" if pixmap.n < 4 else "RGBA"
        return Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples).convert("RGB")


def autocrop_whitespace(image: Image.Image, padding: int = 16) -> Image.Image:
    gray = ImageOps.grayscale(image)
    background = Image.new("L", gray.size, 255)
    diff = ImageChops.difference(gray, background)
    bbox = diff.point(lambda value: 255 if value > 12 else 0).getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)
    return image.crop((left, top, right, bottom))


def crop_bottom_table_area(image: Image.Image) -> Image.Image:
    return image.crop((0, int(image.height * 0.12), image.width, image.height))


def crop_center_table_area(image: Image.Image) -> Image.Image:
    return image.crop(
        (
            int(image.width * 0.03),
            int(image.height * 0.12),
            int(image.width * 0.98),
            int(image.height * 0.98),
        )
    )


def variants(image: Image.Image) -> Iterable[tuple[str, Image.Image]]:
    yield "full_page", image
    yield "autocrop", autocrop_whitespace(image)
    yield "bottom_table_area", autocrop_whitespace(crop_bottom_table_area(image))
    yield "center_table_area", autocrop_whitespace(crop_center_table_area(image))
    gray = ImageOps.autocontrast(ImageOps.grayscale(image))
    yield "full_page_autocontrast_gray", gray.convert("RGB")


def image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def call_glm(endpoint: str, model: str, image: Image.Image, max_tokens: int) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url(image)}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    started = time.perf_counter()
    response = requests.post(endpoint, json=payload, timeout=600)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return str(content), elapsed


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)[:180]


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf)
    output_dir = Path(args.output_dir) / safe_name(pdf_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for page_number in args.pages:
        for dpi in args.dpi:
            page_image = render_page(pdf_path, page_number, dpi)
            for variant_name, variant_image in variants(page_image):
                if args.variants and variant_name not in set(args.variants):
                    continue
                stem = f"page-{page_number:03d}-dpi-{dpi}-{variant_name}"
                image_path = output_dir / f"{stem}.png"
                html_path = output_dir / f"{stem}.html"
                meta_path = output_dir / f"{stem}.json"
                variant_image.save(image_path)
                print(f"OCR {stem} size={variant_image.size}", flush=True)
                content, elapsed = call_glm(args.endpoint, args.model, variant_image, args.max_tokens)
                html_path.write_text(content, encoding="utf-8")
                meta = {
                    "pdf": str(pdf_path),
                    "page": page_number,
                    "dpi": dpi,
                    "variant": variant_name,
                    "image_size": list(variant_image.size),
                    "elapsed_seconds": round(elapsed, 6),
                    "html_path": str(html_path),
                    "image_path": str(image_path),
                }
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                summary.append(meta)
                print(f"  -> {elapsed:.2f}s {html_path}", flush=True)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
