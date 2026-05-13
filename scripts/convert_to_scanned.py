"""Convert text-based PDFs to scanned versions (rasterized) for OCR testing."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm


def convert_pdf_to_scanned(
    input_path: Path,
    output_path: Path,
    dpi: int = 150,
    jpeg_quality: int = 75,
) -> None:
    """Convert a PDF to a scanned version by rasterizing pages.

    Args:
        input_path: Source PDF path.
        output_path: Destination PDF path.
        dpi: Resolution for rendering (lower = smaller file).
        jpeg_quality: JPEG compression quality (1-100, lower = smaller).
    """
    doc = fitz.open(input_path)
    output_doc = fitz.open()

    zoom = dpi / 72  # 72 is default PDF DPI
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Render page to pixmap
        pix = page.get_pixmap(matrix=matrix)

        # Convert to PIL Image for JPEG compression
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Compress as JPEG
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        img_buffer.seek(0)

        # Create new page with same dimensions
        rect = page.rect
        new_page = output_doc.new_page(width=rect.width, height=rect.height)

        # Insert the compressed image
        new_page.insert_image(rect, stream=img_buffer.getvalue())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_doc.save(output_path, garbage=4, deflate=True)
    output_doc.close()
    doc.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert text-based PDFs to scanned versions for OCR testing."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="bank statements",
        help="Source directory containing PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="bank statements scanned",
        help="Destination directory for scanned PDFs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Render resolution (default: 150, lower = smaller files).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=75,
        help="JPEG quality 1-100 (default: 75, lower = smaller files).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    pdf_files = list(input_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {input_dir}")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF(s) in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Settings: DPI={args.dpi}, JPEG quality={args.quality}")
    print()

    for pdf_path in tqdm(pdf_files, desc="Converting"):
        relative_path = pdf_path.relative_to(input_dir)
        output_path = output_dir / relative_path

        try:
            convert_pdf_to_scanned(
                pdf_path,
                output_path,
                dpi=args.dpi,
                jpeg_quality=args.quality,
            )
        except Exception as e:
            print(f"\nError converting {pdf_path}: {e}")
            continue

    print(f"\nDone! Scanned PDFs saved to: {output_dir}")


if __name__ == "__main__":
    main()
