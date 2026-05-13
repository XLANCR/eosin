from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from eosin.backend.bank_parser_service import BankParserService

DEFAULT_INPUT = Path("bank statements")
DEFAULT_OUTPUT_ROOT = Path("local-quality-results")
DEFAULT_VLLM_HOST = "127.0.0.1"
DEFAULT_VLLM_PORT = 8080
DEFAULT_MODEL_NAME = "default"


def build_quality_env(
    *,
    vllm_host: str,
    vllm_port: int,
    model_name: str,
) -> dict[str, str]:
    return {
        "GLMOCR_MODE": "selfhosted",
        "GLMOCR_OCR_API_HOST": vllm_host,
        "GLMOCR_OCR_API_PORT": str(vllm_port),
        "GLMOCR_OCR_MODEL": model_name,
        "BANK_PARSER_OCR_BACKEND_MODE": "page_http",
        "BANK_PARSER_ENABLE_OCR_BATCHING": "false",
        "BANK_PARSER_ENABLE_PAGE_OCR_RETRY": "false",
        "BANK_PARSER_PAGE_OCR_RETRY_ALL": "false",
        "BANK_PARSER_CAPTURE_RAW_OCR_DEBUG": "true",
        "BANK_PARSER_OCR_MAX_IMAGE_SIDE": "3500",
        "BANK_PARSER_OCR_MAX_IMAGE_PIXELS": "9000000",
        "BANK_PARSER_LAYOUT_MODE": "auto",
        "BANK_PARSER_LAYOUT_MAX_CONCURRENCY": "1",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bank parser locally against a local vLLM GLM-OCR server in quality-first mode."
    )
    parser.add_argument("input_path", nargs="?", default=str(DEFAULT_INPUT), help="PDF file or directory.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Where to save artifacts.")
    parser.add_argument("--config", default="", help="Optional parser config YAML path.")
    parser.add_argument("--vllm-host", default=DEFAULT_VLLM_HOST, help="Host for the local vLLM OpenAI server.")
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT, help="Port for the local vLLM OpenAI server.")
    parser.add_argument("--served-model-name", default=DEFAULT_MODEL_NAME, help="Served model name exposed by vLLM.")
    parser.add_argument("--ocr-workers", type=int, default=8, help="OCR pipeline workers for per-page requests.")
    parser.add_argument("--ocr-queue-size", type=int, default=256, help="OCR pipeline queue size.")
    parser.add_argument("--pdf-dpi", type=int, default=0, help="Optional fixed PDF render DPI. 0 keeps the existing pipeline default.")
    parser.add_argument(
        "--layout-mode",
        choices=("auto", "disabled", "required"),
        default="auto",
        help="Parser layout mode. auto keeps layout/header stitching; disabled runs full-page OCR.",
    )
    parser.add_argument("--enable-page-retry", action="store_true", help="Opt into suspicious-page OCR retries.")
    parser.add_argument("--debug", action="store_true", help="Save raw OCR plus crop/header/stitched image artifacts.")
    return parser.parse_args()


def discover_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*.pdf") if path.is_file())


def escape_markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    if len(dataframe.columns) == 0:
        return "_No columns returned._"

    headers = [escape_markdown_cell(column) for column in dataframe.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in dataframe.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(escape_markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def safe_artifact_stem(pdf_path: Path) -> str:
    stem = pdf_path.stem
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)[:180]


def apply_quality_env(overrides: dict[str, str]) -> None:
    for name, value in overrides.items():
        os.environ[name] = value


def write_page_debug_artifacts(output_root: Path, artifact_stem: str, debug_payload: dict[str, Any]) -> None:
    page_ocr = debug_payload.get("page_ocr")
    if not isinstance(page_ocr, list) or not page_ocr:
        return

    artifact_dir = output_root / f"{artifact_stem}_debug"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = []
    for item in page_ocr:
        if not isinstance(item, dict):
            continue
        page_number = int(item.get("page_index", 0)) + 1
        pass_label = str(item.get("pass_label", "unknown"))
        reasons = ", ".join(str(reason) for reason in item.get("reasons", []))
        summary_lines.append(
            f"page={page_number} pass={pass_label} score={item.get('quality_score', 0)} "
            f"suspicious={bool(item.get('suspicious'))} selected={bool(item.get('selected'))} reasons={reasons}"
        )
        raw_html = item.get("raw_html")
        if raw_html:
            raw_path = artifact_dir / f"page-{page_number:03d}-{pass_label}.html"
            raw_path.write_text(str(raw_html), encoding="utf-8")

    quality_summary = debug_payload.get("quality_summary")
    if quality_summary:
        (artifact_dir / "quality-summary.json").write_text(
            json.dumps(quality_summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
    if summary_lines:
        (artifact_dir / "page-ocr-summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pdf_paths = discover_pdfs(input_path)
    if not pdf_paths:
        print(f"No PDFs found at {input_path}")
        return 1

    apply_quality_env(
        build_quality_env(
            vllm_host=args.vllm_host,
            vllm_port=args.vllm_port,
            model_name=args.served_model_name,
        )
    )

    config_path = args.config or None
    pdf_dpi = args.pdf_dpi if args.pdf_dpi > 0 else None
    service = BankParserService(
        config_path=config_path,
        layout_mode=args.layout_mode,
        enable_ocr_batching=False,
        ocr_pipeline_workers=max(1, int(args.ocr_workers)),
        ocr_pipeline_queue_size=max(1, int(args.ocr_queue_size)),
        ocr_backend_mode="page_http",
        enable_page_ocr_retry=bool(args.enable_page_retry),
        save_debug_images=bool(args.debug),
        capture_raw_ocr_debug=True,
        pdf_render_dpi_override=pdf_dpi,
        layout_max_concurrency=1,
        backend_startup_timeout=30.0,
        backend_retry_interval=1.0,
    )

    try:
        for pdf_path in pdf_paths:
            result = service.parse_pdf(pdf_path)
            dataframe = pd.DataFrame(result.rows, columns=result.columns)
            artifact_stem = safe_artifact_stem(pdf_path)
            response_path = output_root / f"{artifact_stem}.json"
            markdown_path = output_root / f"{artifact_stem}.md"

            with response_path.open("w", encoding="utf-8") as handle:
                json.dump(result.to_payload(), handle, indent=2, ensure_ascii=True)
            markdown_path.write_text(dataframe_to_markdown(dataframe), encoding="utf-8")
            write_page_debug_artifacts(output_root, artifact_stem, result.debug)

            print(
                f"{pdf_path.name}: rows={len(dataframe)} pages={result.page_count} "
                f"tables={len(result.pages_with_tables)} json={response_path} md={markdown_path}"
            )
    finally:
        service.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
