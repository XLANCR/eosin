from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fitz
import pandas as pd
import requests
from PIL import Image

from eosin.backend.eosin_pipeline import headers_are_valid, parse_html_table
from scripts.compare_parser_to_ground_truth import best_column_mapping, normalized_row_counter, sample_counter


DEFAULT_ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_MODEL = "default"
DEFAULT_OUTPUT_ROOT = Path("minimal-glm-results")
PROMPT = "Table Recognition:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal GLM-OCR bank table runner: no layout, no stitching, no repairs.")
    parser.add_argument("pdf", help="PDF to parse")
    parser.add_argument("--ground-truth", default="", help="Optional CSV ground truth for comparison")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pages", nargs="*", type=int, default=[], help="Optional 1-based pages. Defaults to all pages.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def render_pages(pdf_path: Path, dpi: int, requested_pages: list[int]) -> list[tuple[int, Image.Image]]:
    scale = dpi / 72.0
    rendered: list[tuple[int, Image.Image]] = []
    with fitz.open(pdf_path) as document:
        page_numbers = requested_pages or list(range(1, document.page_count + 1))
        for page_number in page_numbers:
            page = document.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            mode = "RGB" if pixmap.n < 4 else "RGBA"
            image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples).convert("RGB")
            rendered.append((page_number, image))
    return rendered


def image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def call_glm(endpoint: str, model: str, image: Image.Image, max_tokens: int, timeout: int) -> tuple[str, float]:
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
    response = requests.post(endpoint, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    payload = response.json()
    return str(payload["choices"][0]["message"]["content"]), elapsed


def safe_stem(path: Path) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in path.stem)[:180]


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.fillna("").to_json(orient="records"))


def compare_to_ground_truth(parsed: pd.DataFrame, ground_truth_path: Path) -> dict[str, Any]:
    expected = pd.read_csv(ground_truth_path, dtype=str).fillna("")
    mapping, missing_columns = best_column_mapping(list(expected.columns), list(parsed.columns))
    parsed_for_compare = parsed.rename(columns={actual: expected_col for expected_col, actual in mapping.items()})
    expected_counter = normalized_row_counter(expected, {column: column for column in mapping})
    parsed_counter = normalized_row_counter(parsed_for_compare, {column: column for column in mapping})
    matched = sum((expected_counter & parsed_counter).values())
    precision = matched / len(parsed) if len(parsed) else 0.0
    recall = matched / len(expected) if len(expected) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    missing_counter = expected_counter - parsed_counter
    extra_counter = parsed_counter - expected_counter
    return {
        "ground_truth_rows": len(expected),
        "parsed_rows": len(parsed),
        "matched_rows": matched,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "missing_rows": sum(missing_counter.values()),
        "extra_rows": sum(extra_counter.values()),
        "missing_columns": missing_columns,
        "mapped_columns": mapping,
        "missing_samples": sample_counter(missing_counter, list(mapping), limit=12),
        "extra_samples": sample_counter(extra_counter, list(mapping), limit=12),
    }


def main() -> int:
    args = parse_args()
    pdf_path = Path(args.pdf)
    output_dir = Path(args.output_root) / safe_stem(pdf_path)
    raw_dir = output_dir / "raw-html"
    image_dir = output_dir / "images"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    expected_headers: list[str] | None = None
    page_frames: list[pd.DataFrame] = []
    page_summaries: list[dict[str, Any]] = []

    for page_number, image in render_pages(pdf_path, args.dpi, args.pages):
        image_path = image_dir / f"page-{page_number:03d}.png"
        html_path = raw_dir / f"page-{page_number:03d}.html"
        image.save(image_path)
        print(f"page {page_number}: GLM full-page OCR image={image.size}", flush=True)
        html, elapsed = call_glm(args.endpoint, args.model, image, args.max_tokens, args.timeout)
        html_path.write_text(html, encoding="utf-8")

        df = parse_html_table(html, expected_headers=expected_headers)
        if expected_headers is None and headers_are_valid([str(column) for column in df.columns]):
            expected_headers = [str(column) for column in df.columns]
        if expected_headers is not None and len(df.columns) == len(expected_headers):
            df = df.set_axis(expected_headers, axis=1)
        page_frames.append(df)
        page_summaries.append(
            {
                "page": page_number,
                "elapsed_seconds": round(elapsed, 6),
                "rows": int(len(df)),
                "columns": [str(column) for column in df.columns],
                "image_path": str(image_path),
                "html_path": str(html_path),
            }
        )
        print(f"  rows={len(df)} elapsed={elapsed:.2f}s", flush=True)

    combined = pd.concat(page_frames, ignore_index=True) if page_frames else pd.DataFrame()
    csv_path = output_dir / "parsed.csv"
    json_path = output_dir / "parsed.json"
    summary_path = output_dir / "summary.json"
    combined.to_csv(csv_path, index=False)
    parsed_payload = {"columns": list(combined.columns), "rows": dataframe_to_records(combined)}
    json_path.write_text(json.dumps(parsed_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    summary: dict[str, Any] = {
        "pdf": str(pdf_path),
        "dpi": args.dpi,
        "prompt": PROMPT,
        "pages": page_summaries,
        "rows": int(len(combined)),
        "columns": [str(column) for column in combined.columns],
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }
    if args.ground_truth:
        summary["comparison"] = compare_to_ground_truth(combined, Path(args.ground_truth))
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary.get("comparison", {"rows": len(combined)}), indent=2), flush=True)
    print(f"summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
