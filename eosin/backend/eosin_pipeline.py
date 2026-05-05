"""
Bank Statement PDF → Unified pandas DataFrame

Strategy (Header-Stitching):
  1. Render PDF pages to PIL images using SDK's pdf_to_images_pil()
  2. Run layout detection on ALL pages to find table bounding boxes
  3. Identify the main transaction table per page (using single-table page anchors)
  4. Crop the first table and OCR it to determine row count
  5. Crop the header region from page 1's table
  6. Stitch the header onto each continuation page's table crop
  7. OCR each stitched image in parallel via vLLM
  8. Parse HTML → DataFrames, drop all-NaN columns, combine
"""

import os
import re
import threading
import time
from concurrent.futures import Future, as_completed
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

import cv2
import fitz
import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from pandas.api.types import is_string_dtype
from PIL import Image

# --- Configuration ---
HEADER_EXTRACTION_METHOD = 'cv2'  # 'cv2' (Lines), 'whitespace' (Gaps), or 'ocr' (HTML rows)
ENABLE_OCR_HEADER_FALLBACK = False
SAVE_DEBUG_IMAGES = False
ENABLE_OCR_BATCHING = False
OCR_BATCH_SIZE = 25
OCR_PIPELINE_WORKERS = None
OCR_PIPELINE_QUEUE_SIZE = 128
PDF_RENDER_DPI_OVERRIDE = None
PDF_RENDER_DPI_AUTO = False
PDF_RENDER_TARGET_LONG_SIDE_PX = 2600
PDF_RENDER_DPI_SAFETY_FACTOR = 0.9
PDF_RENDER_DPI_MIN = 150
PDF_RENDER_DPI_MAX = 300
PARSE_TESTING = False
PARSE_TESTING_TABLE_PAGES_PER_SIDE = 3
HEADER_MATCH_RGB_THRESHOLD = 0.94
HEADER_MATCH_GRAY_THRESHOLD = 0.92
HEADER_MATCH_EDGE_THRESHOLD = 0.90
HEADER_MATCH_VERTICAL_TOLERANCE_PX = 8
HEADER_MATCH_MAX_NORMED_SQDIFF = 0.12
HEADER_TEXT_MIN_SHARED_TOKENS = 4
HEADER_TEXT_MIN_OVERLAP_RATIO = 0.6
LAYOUT_MODE_REQUIRED = "required"
LAYOUT_MODE_DISABLED = "disabled"
LAYOUT_MODE_AUTO = "auto"
LAYOUT_MODE_CHOICES = {
    LAYOUT_MODE_REQUIRED,
    LAYOUT_MODE_DISABLED,
    LAYOUT_MODE_AUTO,
}
# ---------------------

# SDK utilities
from glmocr.config import load_config as sdk_load_config
from glmocr.dataloader import PageLoader
from glmocr.layout import PPDocLayoutDetector
from glmocr.ocr_client import OCRClient
from glmocr.utils.image_utils import crop_image_region, pdf_to_images_pil

from eosin.backend.ocr_pipeline import OCRPipelineDispatcher, OCRTaskResult


# ---------------------------------------------------------------------------
# HTML table parsing
# ---------------------------------------------------------------------------


T = TypeVar("T")
HEADER_KEYWORDS = {
    "date",
    "tran date",
    "txn date",
    "value date",
    "description",
    "narration",
    "particulars",
    "details",
    "chq",
    "cheque",
    "ref",
    "debit",
    "credit",
    "withdrawal",
    "deposit",
    "amount",
    "balance",
    "dr/cr",
    "branch",
}
BANK_HEADER_TERMS = {
    "date",
    "transaction date",
    "tran date",
    "txn date",
    "value date",
    "post date",
    "posting date",
    "entry date",
    "narration date",
    "description",
    "transaction description",
    "particular",
    "particulars",
    "transaction particulars",
    "narration",
    "details",
    "remarks",
    "transaction details",
    "reference",
    "reference no",
    "reference number",
    "ref no",
    "ref. no",
    "utr",
    "rrn",
    "cheque",
    "cheque no",
    "cheque number",
    "chq",
    "chq no",
    "chq. no",
    "debit",
    "debit amount",
    "withdrawal",
    "withdrawals",
    "dr",
    "dr.",
    "credit",
    "credit amount",
    "deposit",
    "deposits",
    "cr",
    "cr.",
    "amount",
    "transaction amount",
    "balance",
    "closing balance",
    "available balance",
    "running balance",
    "branch",
    "branch name",
    "sol id",
    "mode",
    "type",
    "transaction type",
    "instrument id",
    "instrument number",
    "serial no",
    "s no",
    "sr no",
    "value",
    "post",
    "debit(Dr.)",
    "credit(Cr.)",
    "dr/cr",
}
BANK_HEADER_DATE_TERMS = {
    "date",
    "transaction date",
    "tran date",
    "txn date",
    "value date",
    "post date",
    "posting date",
    "entry date",
}
BANK_HEADER_TEXT_TERMS = {
    "description",
    "transaction description",
    "particular",
    "particulars",
    "transaction particulars",
    "narration",
    "details",
    "remarks",
    "transaction details",
}
BANK_HEADER_MONEY_TERMS = {
    "debit",
    "debit amount",
    "withdrawal",
    "withdrawals",
    "dr",
    "dr.",
    "credit",
    "credit amount",
    "deposit",
    "deposits",
    "cr",
    "cr.",
    "amount",
    "transaction amount",
    "balance",
    "closing balance",
    "available balance",
    "running balance",
    "dr/cr",
    "debit(Dr.)",
    "credit(Cr.)",
}
BANK_HEADER_REFERENCE_TERMS = {
    "reference",
    "reference no",
    "reference number",
    "ref no",
    "ref. no",
    "utr",
    "rrn",
    "cheque",
    "cheque no",
    "cheque number",
    "chq",
    "chq no",
    "chq. no",
}
TRANSACTION_HEADER_TEXT_TERMS = BANK_HEADER_TEXT_TERMS | BANK_HEADER_REFERENCE_TERMS
SUMMARY_TABLE_REJECTION_PHRASES = (
    "deposit accounts",
    "account holder",
    "account holder name",
    "customer name",
    "customer id",
    "account summary",
    "account details",
    "account information",
    "branch address",
    "statement summary",
)
SUMMARY_TABLE_REJECTION_HEADER_GROUPS = (
    {"account type", "account number", "current balance"},
    {"account name", "account number", "current balance"},
)
NON_TRANSACTION_ROW_MARKERS = (
    "legends used in the statement",
    "transaction total",
    "closing balance",
    "opening balance",
    "iconn",
    "auto sweep",
    "rev sweep",
    "sweep trf",
)
BANK_HEADER_TERM_PATTERNS = tuple(
    re.compile(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", re.IGNORECASE)
    for term in sorted(BANK_HEADER_TERMS, key=len, reverse=True)
)
BANK_HEADER_NUMERIC_NOISE_RE = re.compile(r"(?:upi|imps|neft|rtgs|ach|nach|atm|pos|ecom)?\d{6,}", re.IGNORECASE)
DATE_ATOM_RE = (
    r"(?:"
    r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
    r"|"
    r"\d{1,2}\s+[A-Za-z]{3,9}\s+[']?\d{2,4}"
    r"|"
    r"\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4}"
    r")"
)
DATE_VALUE_RE = re.compile(
    rf"^\s*{DATE_ATOM_RE}(?:\s*\(\s*{DATE_ATOM_RE}\s*\))?\s*$",
    re.IGNORECASE,
)
DATE_EMBEDDED_RE = re.compile(
    rf"\b{DATE_ATOM_RE}\b",
    re.IGNORECASE,
)


def is_date_like(value: object) -> bool:
    text = str(value).strip()
    return bool(text and DATE_VALUE_RE.match(text))


def collect_pdf_paths(input_path: str) -> List[str]:
    """Return sorted PDF paths from a file or recursively from a directory."""
    path = Path(input_path)
    if path.is_file():
        return [str(path)] if path.suffix.lower() == ".pdf" else []
    if not path.is_dir():
        return []

    pdfs = [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() == ".pdf"
    ]
    return sorted(str(candidate) for candidate in pdfs)


def chunk_items(items: Sequence[T], chunk_size: int) -> List[List[T]]:
    """Split *items* into fixed-size chunks while preserving order."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [list(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]


def select_parse_testing_pages(
    page_indices: Sequence[int],
    pages_per_side: int,
) -> List[int]:
    """Keep only the first and last table pages for fast parser iteration."""
    if pages_per_side <= 0 or len(page_indices) <= pages_per_side * 2:
        return list(page_indices)

    selected = list(page_indices[:pages_per_side]) + list(page_indices[-pages_per_side:])
    return sorted(dict.fromkeys(selected))

def extract_all_tables_from_response(response_text: str) -> List[str]:
    """Extract all <table ...>...</table> blocks from model response text."""
    matched_tables = re.findall(
        r"<table[^>]*>.*?</table>", response_text, re.DOTALL | re.IGNORECASE
    )
    if matched_tables:
        return matched_tables

    # Some OCR responses are truncated and never emit a closing </table>.
    # BeautifulSoup can still recover a usable table tree from that HTML.
    soup = BeautifulSoup(response_text, "html.parser")
    recovered_tables = [str(table) for table in soup.find_all("table")]
    return recovered_tables


def extract_headers_from_html(html: str) -> List[str]:
    """Extract column headers from an HTML table."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    best_header_row = None
    best_score = 0

    for tr in table.find_all("tr"):
        th_cells = tr.find_all("th")
        if not th_cells:
            continue
        named_count = sum(1 for c in th_cells if c.get_text(strip=True))
        total_cols = sum(int(c.get("colspan", 1)) for c in th_cells)

        for c in tr.find_all("td"):
            total_cols += int(c.get("colspan", 1))
            if c.get_text(strip=True):
                named_count += 1

        if named_count > best_score:
            best_score = named_count
            best_header_row = tr

    headers: List[str] = []
    if best_header_row:
        for cell in best_header_row.find_all(["th", "td"]):
            colspan = int(cell.get("colspan", 1))
            text = cell.get_text(strip=True)
            headers.extend([text] + [""] * (colspan - 1))

    # Check if data rows have more columns than header
    data_ncols = 0
    for tr in table.find_all("tr"):
        td_cells = tr.find_all("td")
        if not td_cells:
            continue
        n = sum(int(c.get("colspan", 1)) for c in tr.find_all(["td", "th"]))
        data_ncols = max(data_ncols, n)

    if data_ncols > len(headers):
        headers.extend([""] * (data_ncols - len(headers)))

    # Trim trailing empty headers
    while headers and not headers[-1]:
        headers.pop()

    if not headers_are_valid(headers):
        return []

    return headers


def headers_are_valid(headers: List[str]) -> bool:
    cleaned = [h.strip() for h in headers if h and h.strip()]
    if len(cleaned) < 2:
        return False

    header_text = " | ".join(cleaned).lower()
    keyword_hits = sum(1 for kw in HEADER_KEYWORDS if kw in header_text)
    if keyword_hits >= 2:
        return True

    # Reject rows that mostly look like transaction data.
    date_like = sum(1 for value in cleaned if DATE_EMBEDDED_RE.search(value))
    numeric_like = sum(
        1
        for value in cleaned
        if re.fullmatch(r"[\d,./\-]+", value) is not None
    )
    return (date_like + numeric_like) <= 1


def resolve_layout_mode(layout_mode: Optional[str]) -> str:
    raw_mode = (layout_mode or os.getenv("BANK_PARSER_LAYOUT_MODE", LAYOUT_MODE_REQUIRED)).strip().lower()
    if raw_mode not in LAYOUT_MODE_CHOICES:
        valid = ", ".join(sorted(LAYOUT_MODE_CHOICES))
        raise ValueError(f"invalid BANK_PARSER_LAYOUT_MODE={raw_mode!r}; expected one of: {valid}")
    return raw_mode


def _normalize_header_text(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def _header_contains_term(header: str, terms: Sequence[str]) -> bool:
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", header)
        for term in terms
    )


def _has_transaction_header_anchor(headers: Sequence[str]) -> bool:
    normalized_headers = [_normalize_header_text(header) for header in headers if _normalize_header_text(header)]
    if not normalized_headers:
        return False

    date_hits = sum(1 for header in normalized_headers if _header_contains_term(header, BANK_HEADER_DATE_TERMS))
    text_hits = sum(1 for header in normalized_headers if _header_contains_term(header, TRANSACTION_HEADER_TEXT_TERMS))
    money_hits = sum(1 for header in normalized_headers if _header_contains_term(header, BANK_HEADER_MONEY_TERMS))
    return date_hits >= 1 and text_hits >= 1 and money_hits >= 2


def _row_text(values: Sequence[object]) -> str:
    return " ".join(str(value).strip().lower() for value in values if str(value).strip())


def _is_non_transaction_text(text: str) -> bool:
    return bool(text) and any(marker in text for marker in NON_TRANSACTION_ROW_MARKERS)


def _table_looks_like_summary_or_profile(headers: Sequence[str], df: pd.DataFrame) -> bool:
    normalized_headers = [_normalize_header_text(header) for header in headers if _normalize_header_text(header)]
    header_text = " | ".join(normalized_headers)

    if any(phrase in header_text for phrase in SUMMARY_TABLE_REJECTION_PHRASES):
        return True

    if any(group.issubset(set(normalized_headers)) for group in SUMMARY_TABLE_REJECTION_HEADER_GROUPS):
        return True

    if df.empty:
        return False

    preview_rows = [
        _row_text(row.tolist())
        for _, row in df.head(3).fillna("").astype(str).iterrows()
    ]
    preview_text = " | ".join(text for text in preview_rows if text)
    if any(phrase in preview_text for phrase in SUMMARY_TABLE_REJECTION_PHRASES):
        return True

    return False


def _count_transaction_rows(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    rows_as_strings = df.fillna("").astype(str)
    transaction_rows = 0
    for _, row in rows_as_strings.iterrows():
        row_values = [value.strip() for value in row.tolist()]
        row_text = _row_text(row_values)
        if _is_non_transaction_text(row_text):
            continue
        if any(is_date_like(value) for value in row_values):
            transaction_rows += 1
    return transaction_rows


def make_columns_unique(columns: List[str]) -> List[str]:
    counts: Dict[str, int] = {}
    unique_columns: List[str] = []

    for idx, column in enumerate(columns):
        base = column.strip() if column and column.strip() else f"col_{idx}"
        count = counts.get(base, 0) + 1
        counts[base] = count
        unique_columns.append(base if count == 1 else f"{base}_{count}")

    return unique_columns


def parse_html_table(
    html: str, expected_headers: Optional[List[str]] = None
) -> pd.DataFrame:
    """Parse an HTML <table> string into a DataFrame."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return pd.DataFrame()

    rows: List[List[str]] = []
    header_row: Optional[List[str]] = None

    # Extract thead
    thead = table.find("thead")
    if thead:
        tr = thead.find("tr")
        if tr:
            cells = tr.find_all(["th", "td"])
            header_cells: List[str] = []
            for cell in cells:
                colspan = int(cell.get("colspan", 1))
                text = cell.get_text(strip=True)
                header_cells.extend([text] + [""] * (colspan - 1))
            header_row = header_cells

    # Extract tbody rows
    tbody = table.find("tbody")
    if tbody:
        body_rows = tbody.find_all("tr")
    else:
        thead_trs = set()
        if thead:
            for tr in thead.find_all("tr"):
                thead_trs.add(id(tr))
        body_rows = [tr for tr in table.find_all("tr") if id(tr) not in thead_trs]

    for tr in body_rows:
        cells = tr.find_all(["td", "th"])
        row_data: List[str] = []
        for cell in cells:
            colspan = int(cell.get("colspan", 1))
            text = cell.get_text(strip=True)
            row_data.extend([text] + [""] * (colspan - 1))
        rows.append(row_data)

    if header_row and not headers_are_valid(header_row):
        rows.insert(0, header_row)
        header_row = None

    if not rows and not header_row:
        return pd.DataFrame()

    if not header_row and rows:
        first_row_lower = " ".join(rows[0]).lower()
        valid_keywords = {'date', 'narration', 'description', 'particulars', 'debit', 'credit', 'amount', 'balance', 'value date', 'txn date'}
        # A legitimate financial header row will almost certainly contain multiples of these terms organically.
        if sum(1 for kw in valid_keywords if kw in first_row_lower) >= 2:
            header_row = rows.pop(0)

    if expected_headers:
        ncols = len(expected_headers)
    elif header_row:
        ncols = len(header_row)
    else:
        ncols = max(len(r) for r in rows) if rows else 0

    if expected_headers and header_row:
        header_text = " ".join(header_row).strip()
        date_pattern = re.compile(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")
        if date_pattern.search(header_text):
            rows.insert(0, header_row)
            header_row = None

    normalized_rows: List[List[str]] = []
    header_lower = (
        {h.lower().strip() for h in expected_headers if h.strip()}
        if expected_headers
        else set()
    )

    for row in rows:
        if header_lower:
            non_empty = [c.strip().lower() for c in row if c.strip()]
            if non_empty and all(v in header_lower for v in non_empty):
                continue

        if len(row) < ncols:
            row = row + [""] * (ncols - len(row))
        elif len(row) > ncols:
            row = row[:ncols]
        normalized_rows.append(row)

    if expected_headers:
        cols = list(expected_headers)
    elif header_row:
        cols = list(header_row)
        if len(cols) < ncols:
            cols += [f"col_{i}" for i in range(len(cols), ncols)]
        elif len(cols) > ncols:
            cols = cols[:ncols]
    else:
        cols = [f"col_{i}" for i in range(ncols)]

    return pd.DataFrame(normalized_rows, columns=make_columns_unique(cols))


def _score_table_candidate(
    df: pd.DataFrame,
    raw_headers: List[str],
    expected_headers: Optional[List[str]],
) -> int:
    if df.empty:
        return -1

    score = 0
    cleaned_headers = [header.strip() for header in raw_headers if header and header.strip()]
    header_text = " ".join(cleaned_headers).lower()
    anchor_headers = cleaned_headers or [
        str(column).strip()
        for column in df.columns
        if str(column).strip() and not str(column).startswith("col_")
    ]
    if not anchor_headers and expected_headers:
        anchor_headers = [header.strip() for header in expected_headers if header and header.strip()]

    if cleaned_headers:
        score += 20
        score += sum(1 for kw in HEADER_KEYWORDS if kw in header_text) * 10

    if _has_transaction_header_anchor(anchor_headers):
        score += 140
    elif cleaned_headers:
        score -= 60

    if _table_looks_like_summary_or_profile(anchor_headers, df):
        score -= 220

    date_hits = 0
    if expected_headers:
        expected = {header.strip().lower() for header in expected_headers if header.strip()}
        actual = {str(column).strip().lower() for column in df.columns if str(column).strip()}
        raw = {header.strip().lower() for header in cleaned_headers}
        score += len(expected & actual) * 15
        score += len(expected & raw) * 20
        if len(df.columns) == len(expected_headers):
            score += 10
        if raw and not (expected & raw):
            score -= 40

    for column in df.columns:
        values = df[column].fillna("").astype(str).str.strip()
        date_hits = max(date_hits, int(values.apply(is_date_like).sum()))
    score += min(date_hits, 10) * 8
    if expected_headers and date_hits == 0:
        score -= 20

    non_empty_cells = int(
        df.fillna("").astype(str).apply(lambda column: column.str.strip().ne("")).sum().sum()
    )
    score += min(non_empty_cells, 100)
    score += min(len(df), 50) * 4

    transaction_rows = _count_transaction_rows(df)
    score += min(transaction_rows, 20) * 30
    if transaction_rows == 0:
        score -= 200
    return score


def select_best_table_candidate(
    table_htmls: Sequence[str],
    expected_headers: Optional[List[str]] = None,
) -> Optional[Tuple[int, str, pd.DataFrame, List[str]]]:
    best_candidate: Optional[Tuple[int, str, pd.DataFrame, List[str], int]] = None

    for index, table_html in enumerate(table_htmls):
        raw_headers = extract_headers_from_html(table_html)
        parsed_df = parse_html_table(table_html, expected_headers=expected_headers)
        score = _score_table_candidate(parsed_df, raw_headers, expected_headers)
        if best_candidate is None or score > best_candidate[4]:
            best_candidate = (index, table_html, parsed_df, raw_headers, score)

    if best_candidate is None or best_candidate[4] < 0:
        return None

    return best_candidate[:4]


# ---------------------------------------------------------------------------
# BankStatementParser
# ---------------------------------------------------------------------------

class BankStatementParser:
    """Parses multi-page bank statement PDFs into a single pandas DataFrame."""

    HEADER_MIN_PX = 40
    HEADER_MAX_PX = 200
    MIN_TABLE_AREA_RATIO = 0.05
    OCR_MAX_IMAGE_SIDE = 1600
    OCR_MAX_IMAGE_PIXELS = 2_200_000

    def __init__(self, config_path: str | None = None, *, layout_mode: Optional[str] = None):
        default_config_path = Path(__file__).resolve().parent / "config.yaml"
        self.config_path = str(config_path or default_config_path)
        self.last_run_stats: Dict[str, object] = {}
        self.layout_mode = resolve_layout_mode(layout_mode)

        print("  Loading SDK config...")
        sdk_cfg = sdk_load_config(self.config_path)

        self.layout_detector = self._build_layout_detector(sdk_cfg.pipeline.layout)

        print("  Initializing PageLoader & OCR Client...")
        self.page_loader = PageLoader(sdk_cfg.pipeline.page_loader)
        self.ocr_client = OCRClient(sdk_cfg.pipeline.ocr_api)
        self.ocr_client.start()
        print("  OCR Client ready.")

        self.pdf_dpi = sdk_cfg.pipeline.page_loader.pdf_dpi
        self.layout_guard: threading.Semaphore | None = None
        self.ocr_max_workers = sdk_cfg.pipeline.max_workers
        self.ocr_connection_pool_size = max(
            1,
            getattr(self.ocr_client, "_pool_maxsize", self.ocr_max_workers),
        )
        self.ocr_pipeline_workers = self._resolve_ocr_pipeline_workers()
        self.ocr_pipeline_queue_size = self._resolve_ocr_pipeline_queue_size()
        self.ocr_dispatcher = OCRPipelineDispatcher(
            self.page_loader,
            self.ocr_client,
            max_workers=self.ocr_pipeline_workers,
            queue_size=self.ocr_pipeline_queue_size,
        )

        if self.ocr_connection_pool_size < self.ocr_max_workers:
            print(
                "  Note: limiting OCR concurrency to connection pool size "
                f"({self.ocr_connection_pool_size}) instead of max_workers "
                f"({self.ocr_max_workers})"
            )
        print(
            "  OCR Pipeline ready "
            f"({self.ocr_pipeline_workers} worker(s), queue size {self.ocr_pipeline_queue_size})."
        )

    def close(self):
        try:
            self.ocr_dispatcher.close()
        except Exception:
            pass
        try:
            if self.layout_detector is not None:
                self.layout_detector.stop()
        except Exception:
            pass
        try:
            self.ocr_client.stop()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _resolve_ocr_pipeline_workers(self) -> int:
        configured = OCR_PIPELINE_WORKERS
        if configured is None:
            configured = self.ocr_max_workers
        return min(
            max(1, int(configured)),
            max(1, int(self.ocr_connection_pool_size)),
        )

    def _resolve_ocr_pipeline_queue_size(self) -> int:
        configured = OCR_PIPELINE_QUEUE_SIZE
        return max(
            self._resolve_ocr_pipeline_workers(),
            int(configured),
        )

    def _build_layout_detector(self, layout_config: object):
        print(f"  Initializing Layout Detector (mode={self.layout_mode})...")
        print(
            "  Layout backend symbol: "
            f"value={PPDocLayoutDetector!r}, type={type(PPDocLayoutDetector).__name__}"
        )

        if self.layout_mode == LAYOUT_MODE_DISABLED:
            print("  Layout Detector disabled by configuration.")
            return None

        try:
            if PPDocLayoutDetector is None:
                raise TypeError("PPDocLayoutDetector import resolved to None")
            detector = PPDocLayoutDetector(layout_config)
            if detector is None:
                raise TypeError("PPDocLayoutDetector(...) returned None")
            detector.start()
            print("  Layout Detector ready.")
            return detector
        except Exception as exc:
            if self.layout_mode == LAYOUT_MODE_REQUIRED:
                raise RuntimeError("failed to initialize layout detector") from exc
            print(f"  Layout Detector unavailable; continuing with layout disabled: {exc}")
            return None

    def _geometry_driven_pdf_dpi(
        self,
        page_sizes: Sequence[Tuple[float, float]],
    ) -> int:
        if not page_sizes:
            return int(self.pdf_dpi)

        long_sides = [max(width, height) for width, height in page_sizes if width > 0 and height > 0]
        if not long_sides:
            return int(self.pdf_dpi)

        # Use the smallest long side in the document as the most demanding page for
        # preserving detail, then apply the requested safety factor and clamp it.
        reference_long_side = min(long_sides)
        required_dpi = (PDF_RENDER_TARGET_LONG_SIDE_PX * 72.0) / reference_long_side
        adjusted_dpi = required_dpi * PDF_RENDER_DPI_SAFETY_FACTOR

        lower_bound = max(int(self.pdf_dpi), int(PDF_RENDER_DPI_MIN))
        bounded_dpi = min(int(round(adjusted_dpi)), int(PDF_RENDER_DPI_MAX))
        return max(lower_bound, bounded_dpi)

    def _effective_pdf_dpi(self, pdf_path: Optional[str] = None) -> int:
        if PDF_RENDER_DPI_OVERRIDE:
            return int(PDF_RENDER_DPI_OVERRIDE)

        if PDF_RENDER_DPI_AUTO and pdf_path:
            try:
                with fitz.open(pdf_path) as document:
                    page_sizes = [
                        (float(page.rect.width), float(page.rect.height))
                        for page in document
                    ]
                return self._geometry_driven_pdf_dpi(page_sizes)
            except Exception as exc:
                print(f"        ! Auto DPI detection failed, falling back to config DPI: {exc}")

        return int(self.pdf_dpi)

    @staticmethod
    def _header_term_hits(text: str, terms: set[str]) -> int:
        lowered = text.lower()
        return sum(
            1
            for term in terms
            if re.search(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])", lowered)
        )

    def _header_text_looks_like_bank_statement_header(self, text: str) -> bool:
        normalized = " ".join(str(text).lower().split())
        if not normalized:
            return False

        total_hits = sum(1 for pattern in BANK_HEADER_TERM_PATTERNS if pattern.search(normalized))
        date_hits = self._header_term_hits(normalized, BANK_HEADER_DATE_TERMS)
        text_hits = self._header_term_hits(normalized, BANK_HEADER_TEXT_TERMS)
        money_hits = self._header_term_hits(normalized, BANK_HEADER_MONEY_TERMS)
        reference_hits = self._header_term_hits(normalized, BANK_HEADER_REFERENCE_TERMS)
        numeric_noise = len(BANK_HEADER_NUMERIC_NOISE_RE.findall(normalized))
        bank_name_hits = sum(
            bank in normalized
            for bank in ("icici bank", "axis bank", "yes bank", "state bank", "hdfc bank")
        )

        if total_hits >= 4 and sum(hit > 0 for hit in (date_hits, text_hits, money_hits)) >= 2:
            return True

        if total_hits >= 3 and date_hits > 0 and text_hits > 0 and (money_hits > 0 or reference_hits > 0):
            return True

        if numeric_noise >= 3 and total_hits < 3:
            return False

        if bank_name_hits >= 2 and total_hits < 3:
            return False

        if DATE_EMBEDDED_RE.search(normalized) and total_hits < 3:
            return False

        return False

    def _select_header_source_table(
        self,
        table_crops: List[Tuple[int, Image.Image]],
    ) -> Tuple[int, int, Image.Image, str]:
        fallback_index = 0
        fallback_height = self._calculate_header_height(table_crops[0][1])
        fallback_image = table_crops[0][1].crop((0, 0, table_crops[0][1].width, fallback_height))
        fallback_text = ""

        for table_index, (page_idx, crop) in enumerate(table_crops):
            header_h = self._calculate_header_height(crop)
            header_img = crop.crop((0, 0, crop.width, header_h))
            header_text = self._ocr_text(header_img)
            if table_index == 0:
                fallback_text = header_text
            if self._header_text_looks_like_bank_statement_header(header_text):
                if table_index > 0:
                    print(
                        f"        → Rejected earlier table(s); using page {page_idx + 1} "
                        "as header source"
                    )
                return table_index, header_h, header_img, header_text

            print(
                f"        → Header probe page {page_idx + 1}: rejected "
                "(missing bank-header keywords)"
            )

        print("        → No validated bank-header table found; falling back to first table")
        return fallback_index, fallback_height, fallback_image, fallback_text

    def _run_layout_detection(self, page_images: List[Image.Image]):
        if self.layout_detector is None:
            raise RuntimeError("layout detector is disabled")

        original_batch_size = max(1, int(getattr(self.layout_detector, "batch_size", 1)))
        batch_size = original_batch_size
        guard = self.layout_guard if self.layout_guard is not None else nullcontext()

        while True:
            try:
                self.layout_detector.batch_size = batch_size
                with guard:
                    return self.layout_detector.process(page_images)
            except torch.OutOfMemoryError:
                if batch_size <= 1:
                    raise
                next_batch_size = max(1, batch_size // 2)
                print(
                    "        ! Layout CUDA OOM at batch size "
                    f"{batch_size}; retrying with batch size {next_batch_size}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                batch_size = next_batch_size
            else:
                break

    def _extract_transaction_dataframes(
        self,
        all_ocr_results: List[Tuple[int, Optional[str]]],
        pdf_path: str,
    ) -> Tuple[List[pd.DataFrame], Optional[List[str]], List[int]]:
        all_dfs: List[pd.DataFrame] = []
        expected_headers: Optional[List[str]] = None
        selected_pages: List[int] = []

        for page_idx, html_content in all_ocr_results:
            if not html_content:
                print(f"        Page {page_idx + 1}: No HTML returned")
                continue

            if SAVE_DEBUG_IMAGES:
                debug_dir = os.path.join("output", "debug", os.path.splitext(os.path.basename(pdf_path))[0])
                os.makedirs(debug_dir, exist_ok=True)
                with open(os.path.join(debug_dir, f"page_{page_idx + 1}_raw.html"), "w") as handle:
                    handle.write(html_content)

            table_htmls = extract_all_tables_from_response(html_content)
            if not table_htmls:
                print(f"        Page {page_idx + 1}: No <table> found in response")
                continue

            selected_table = select_best_table_candidate(
                table_htmls,
                expected_headers=expected_headers,
            )
            if selected_table is None:
                print(f"        Page {page_idx + 1}: No usable table found in response")
                continue

            table_index, _, df, raw_headers = selected_table
            selected_pages.append(page_idx)
            if table_index > 0:
                print(
                    f"        Page {page_idx + 1}: selected table "
                    f"{table_index + 1}/{len(table_htmls)}"
                )

            if expected_headers is None and raw_headers:
                expected_headers = raw_headers
                print(f"        Headers: {expected_headers}")
                all_dfs = [
                    self._align_columns_with_headers(existing_df, expected_headers)
                    for existing_df in all_dfs
                ]

            df = self._align_columns_with_headers(df, expected_headers or [])
            if not df.empty:
                all_dfs.append(df)
                print(f"        Page {page_idx + 1}: {len(df)} rows")

        return all_dfs, expected_headers, selected_pages

    def _finalize_extracted_tables(
        self,
        all_dfs: List[pd.DataFrame],
        expected_headers: Optional[List[str]],
    ) -> pd.DataFrame:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = self._remove_header_rows(combined, expected_headers or [])
        combined = self._merge_multiline_rows(combined)
        combined = self._filter_valid_date_rows(combined)
        return combined.replace(r'^\s*$', np.nan, regex=True).dropna(axis=1, how='all')

    def _parse_pdf_without_layout(
        self,
        pdf_path: str,
        page_images: List[Image.Image],
        effective_dpi: int,
        started_at: float,
        step_timings: Dict[str, float],
    ) -> pd.DataFrame:
        print("  [2/4] Layout detection disabled; OCR-ing full pages...")
        step_started_at = time.time()
        ocr_images = [
            (page_idx, self._normalize_ocr_image(page_image))
            for page_idx, page_image in enumerate(page_images)
        ]
        all_ocr_results, ocr_metrics = self._ocr_tables_parallel(ocr_images)
        step_timings["ocr_pages"] = round(time.time() - step_started_at, 3)

        print("  [3/4] Parsing HTML to DataFrames...")
        step_started_at = time.time()
        all_dfs, expected_headers, pages_with_tables = self._extract_transaction_dataframes(
            all_ocr_results,
            pdf_path,
        )
        if not all_dfs:
            print("  × No table data extracted")
            step_timings["parse_html_tables"] = round(time.time() - step_started_at, 3)
            self.last_run_stats = {
                "timings": step_timings,
                "pages_with_tables": [page + 1 for page in pages_with_tables],
                "page_count": len(page_images),
                "effective_dpi": effective_dpi,
                "layout_mode": self.layout_mode,
                "layout_enabled": False,
                "ocr_images": len(ocr_images),
                "ocr_metrics": ocr_metrics,
            }
            return pd.DataFrame()

        combined = self._finalize_extracted_tables(all_dfs, expected_headers)
        step_timings["parse_html_tables"] = round(time.time() - step_started_at, 3)
        elapsed = time.time() - started_at
        step_timings["total"] = round(elapsed, 3)
        self.last_run_stats = {
            "timings": step_timings,
            "pages_with_tables": [page + 1 for page in pages_with_tables],
            "page_count": len(page_images),
            "effective_dpi": effective_dpi,
            "layout_mode": self.layout_mode,
            "layout_enabled": False,
            "ocr_images": len(ocr_images),
            "ocr_metrics": ocr_metrics,
        }
        print(f"  → Processed {len(combined)} rows in {elapsed:.1f}s")
        return combined

    def parse_pdf(self, pdf_path: str) -> pd.DataFrame:
        """Parse a bank statement PDF into a single DataFrame."""
        t0 = time.time()
        step_timings: Dict[str, float] = {}
        effective_dpi = self._effective_pdf_dpi(pdf_path)

        print("  [1/8] Rendering PDF pages...")
        print(f"        → Rendering at {effective_dpi} DPI")
        step_started_at = time.time()
        try:
            page_images = pdf_to_images_pil(pdf_path, dpi=effective_dpi)
        except Exception as e:
            print(f"  ✗ Failed to render PDF: {e}")
            self.last_run_stats = {
                "timings": {"render_pages": round(time.time() - step_started_at, 3)},
                "pages_with_tables": [],
                "page_count": 0,
                "effective_dpi": effective_dpi,
            }
            return pd.DataFrame()
        step_timings["render_pages"] = round(time.time() - step_started_at, 3)
        print(f"        → {len(page_images)} pages rendered")

        if not page_images:
            print("  ✗ No pages found")
            self.last_run_stats = {
                "timings": step_timings,
                "pages_with_tables": [],
                "page_count": 0,
                "effective_dpi": effective_dpi,
            }
            return pd.DataFrame()

        if self.layout_detector is None:
            return self._parse_pdf_without_layout(
                pdf_path,
                page_images,
                effective_dpi,
                t0,
                step_timings,
            )

        print("  [2/8] Running layout detection...")
        t_layout = time.time()
        all_layout_results, _ = self._run_layout_detection(page_images)
        layout_elapsed = time.time() - t_layout
        step_timings["layout_detection"] = round(layout_elapsed, 3)
        print(f"        → Done in {layout_elapsed:.1f}s")

        print("  [3/8] Identifying main tables...")
        step_started_at = time.time()
        table_bboxes = self._identify_main_tables(all_layout_results, page_images)
        step_timings["identify_main_tables"] = round(time.time() - step_started_at, 3)
        pages_with_tables = [i for i, bbox in enumerate(table_bboxes) if bbox is not None]
        print(f"        → Tables on pages: {[p + 1 for p in pages_with_tables]}")

        if not pages_with_tables:
            print("  ✗ No tables found in document")
            self.last_run_stats = {
                "timings": step_timings,
                "pages_with_tables": [],
                "page_count": len(page_images),
                "effective_dpi": effective_dpi,
            }
            return pd.DataFrame()

        if PARSE_TESTING:
            selected_pages = select_parse_testing_pages(
                pages_with_tables,
                PARSE_TESTING_TABLE_PAGES_PER_SIDE,
            )
            print(
                "        → Parse testing enabled; limiting to pages: "
                f"{[p + 1 for p in selected_pages]}"
            )
            pages_with_tables = selected_pages

        print("  [4/8] Cropping table regions...")
        step_started_at = time.time()
        table_crops: List[Tuple[int, Image.Image]] = []
        debug_dir: str | None = None
        for page_idx in pages_with_tables:
            crop = crop_image_region(page_images[page_idx], table_bboxes[page_idx])
            table_crops.append((page_idx, crop))
            if SAVE_DEBUG_IMAGES and page_idx == 7:  # Page 8
                debug_dir = os.path.join("output", "debug", os.path.splitext(os.path.basename(pdf_path))[0])
                os.makedirs(debug_dir, exist_ok=True)
                crop.save(os.path.join(debug_dir, f"page_{page_idx + 1}_crop.png"))
        step_timings["crop_table_regions"] = round(time.time() - step_started_at, 3)

        # 5. Extract header from first validated table
        print("  [5/8] Computing header height from first table image...")
        step_started_at = time.time()
        header_source_index, header_h, header_img, header_text = self._select_header_source_table(table_crops)
        if header_source_index > 0:
            table_crops = table_crops[header_source_index:]

        first_page_idx, first_crop = table_crops[0]
        header_tokens = self._header_text_tokens(header_text)
        step_timings["select_header_source"] = round(time.time() - step_started_at, 3)

        if SAVE_DEBUG_IMAGES:
            debug_dir = os.path.join("output", "debug", os.path.splitext(os.path.basename(pdf_path))[0])
            os.makedirs(debug_dir, exist_ok=True)
            first_crop.save(os.path.join(debug_dir, f"page_{first_page_idx + 1}_crop.png"))

        if SAVE_DEBUG_IMAGES:
            header_img.save(os.path.join(debug_dir, "extracted_header.png"))

        print(f"        Header height: {header_h}px (from {first_crop.height}px crop)")

        print("  [6/8] Stitching headers onto continuation pages...")
        step_started_at = time.time()
        skip_header_stitching = self._should_skip_header_stitching(header_img, table_crops, header_tokens)
        if skip_header_stitching:
            print("        → Native continuation headers detected confidently; skipping stitch")
        else:
            print("        → Native headers not confirmed; stitching header onto continuation pages")

        stitched_images: List[Tuple[int, Image.Image]] = []
        for i, (page_idx, crop) in enumerate(table_crops):
            if i == 0:
                continue

            stitched = crop if skip_header_stitching else self._stitch_header(header_img, crop)
            normalized = self._normalize_ocr_image(stitched)
            stitched_images.append((page_idx, normalized))
            if SAVE_DEBUG_IMAGES and debug_dir:
                normalized.save(os.path.join(debug_dir, f"page_{page_idx + 1}_stitched.png"))
        step_timings["stitch_headers"] = round(time.time() - step_started_at, 3)

        ocr_images: List[Tuple[int, Image.Image]] = [
            (first_page_idx, self._normalize_ocr_image(first_crop))
        ] + stitched_images

        print(f"  [7/8] OCR-ing {len(ocr_images)} table images via shared pipeline...")
        t_ocr = time.time()
        all_ocr_results, ocr_metrics = self._ocr_tables_parallel(ocr_images)
        ocr_elapsed = time.time() - t_ocr
        step_timings["ocr_tables"] = round(ocr_elapsed, 3)
        print(f"        → Done in {ocr_elapsed:.1f}s")
        all_ocr_results.sort(key=lambda x: x[0])

        print("  [8/8] Parsing HTML to DataFrames...")
        step_started_at = time.time()
        all_dfs, expected_headers, parsed_pages = self._extract_transaction_dataframes(
            all_ocr_results,
            pdf_path,
        )

        if not all_dfs:
            print("  ✗ No table data extracted")
            step_timings["parse_html_tables"] = round(time.time() - step_started_at, 3)
            self.last_run_stats = {
                "timings": step_timings,
                "pages_with_tables": [p + 1 for p in parsed_pages],
                "page_count": len(page_images),
                "effective_dpi": effective_dpi,
                "layout_mode": self.layout_mode,
                "layout_enabled": True,
                "ocr_metrics": ocr_metrics,
            }
            return pd.DataFrame()

        combined = self._finalize_extracted_tables(all_dfs, expected_headers)
        step_timings["parse_html_tables"] = round(time.time() - step_started_at, 3)

        elapsed = time.time() - t0
        step_timings["total"] = round(elapsed, 3)
        self.last_run_stats = {
            "timings": step_timings,
            "pages_with_tables": [p + 1 for p in parsed_pages],
            "page_count": len(page_images),
            "effective_dpi": effective_dpi,
            "layout_mode": self.layout_mode,
            "layout_enabled": True,
            "ocr_images": len(ocr_images),
            "header_source_page": first_page_idx + 1,
            "skip_header_stitching": skip_header_stitching,
            "ocr_metrics": ocr_metrics,
        }
        print(f"  → Processed {len(combined)} rows in {elapsed:.1f}s")
        return combined

    def _identify_main_tables(
        self,
        all_layout_results: List[List[Dict]],
        page_images: List[Image.Image],
    ) -> List[Optional[List[int]]]:
        """Find the main table bbox on each page using single-table pages as anchors."""
        result: List[Optional[List[int]]] = [None] * len(all_layout_results)
        page_area = 1000 * 1000

        VALID_BANK_HEADERS = {'date', 'description', 'particulars', 'narration', 'debit', 'credit', 'amount', 'balance', 'withdrawal', 'deposit', 'value date', 'txn date'}

        # Extract tables and filter out tiny ones
        pages_tables = []
        all_pages_all_tables = []
        for regions in all_layout_results:
            tables = [r for r in regions if r.get("label") == "table"]
            all_pages_all_tables.append(tables)
            valid_tables = []
            for t in tables:
                area = (t["bbox_2d"][2] - t["bbox_2d"][0]) * (t["bbox_2d"][3] - t["bbox_2d"][1])
                if area >= page_area * self.MIN_TABLE_AREA_RATIO:
                    valid_tables.append(t)
            pages_tables.append(valid_tables)

        # Identify pages with exactly 1 table
        single_table_pages = [i for i, tables in enumerate(pages_tables) if len(tables) == 1]

        if not single_table_pages:
            print("        ! No distinct single-table anchors globally found. Initiating parallel OCR structural fallback constraint scanning...")

            # 1. Gather all candidates globally
            candidate_images = []
            candidate_map = {} # unique_idx -> (page_idx, table_idx, bbox)

            for i, tables in enumerate(all_pages_all_tables):
                if not tables:
                    continue
                for t_idx, t in enumerate(tables):
                    unique_idx = i * 1000 + t_idx
                    crop = crop_image_region(page_images[i], t["bbox_2d"])
                    candidate_images.append((unique_idx, crop))
                    candidate_map[unique_idx] = (i, t_idx, t["bbox_2d"])

            # 2. Parallel OCR evaluation utilizing built in optimizations natively
            ocr_results = {}
            if candidate_images:
                parallel_results = self._ocr_tables_parallel(candidate_images)
                for unique_idx, html in parallel_results:
                    if html:
                        ocr_results[unique_idx] = html.lower()[:500]
                    else:
                        ocr_results[unique_idx] = ""

            # 3. Constraint resolution parameters
            for i, tables in enumerate(all_pages_all_tables):
                if not tables:
                    continue
                found_valid = False
                for t_idx, t in enumerate(tables):
                    unique_idx = i * 1000 + t_idx
                    top_html = ocr_results.get(unique_idx, "")
                    if any(kw in top_html for kw in VALID_BANK_HEADERS):
                        result[i] = t["bbox_2d"]
                        found_valid = True
                        break

                if not found_valid:
                    # Still fallback securely to largest physical component physically if pure OCR validation returns false loops completely.
                    best = max(
                        tables,
                        key=lambda t: (t["bbox_2d"][2] - t["bbox_2d"][0]) * (t["bbox_2d"][3] - t["bbox_2d"][1])
                    )
                    result[i] = best["bbox_2d"]
            return result

        # Use single-table pages as the boundaries for the bank statement
        first_single = single_table_pages[0]
        last_single = single_table_pages[-1]

        for i, tables in enumerate(pages_tables):
            if not tables:
                continue

            if i < first_single - 1:
                result[i] = None
            elif i == first_single - 1:
                # Page before first single table: pick the LAST table on the page (max y1)
                best = max(tables, key=lambda t: t["bbox_2d"][1])
                result[i] = best["bbox_2d"]
            elif first_single <= i <= last_single:
                # Inside the core statement: pick the largest table
                best = max(
                    tables,
                    key=lambda t: (t["bbox_2d"][2] - t["bbox_2d"][0]) * (t["bbox_2d"][3] - t["bbox_2d"][1])
                )
                result[i] = best["bbox_2d"]
            elif i == last_single + 1:
                # Page after last single table: pick the FIRST table on the page (min y1)
                best = min(tables, key=lambda t: t["bbox_2d"][1])
                result[i] = best["bbox_2d"]
            else:
                result[i] = None

        return result

    def _get_header_height_cv2(self, image: Image.Image) -> int:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        # Threshold at 240 makes light gray/blue headers (e.g. 220) and dark lines/text into WHITE
        _, thresh = cv2.threshold(img_cv, 240, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        row_sums = np.sum(horizontal_lines, axis=1)

        line_y_coords = np.where(row_sums > (image.width * 255 * 0.5))[0]

        if len(line_y_coords) == 0:
            return -1

        clusters = []
        current_cluster = [line_y_coords[0]]
        for y in line_y_coords[1:]:
            if y - current_cluster[-1] < 6:
                current_cluster.append(y)
            else:
                clusters.append(current_cluster)
                current_cluster = [y]
        clusters.append(current_cluster)

        # 1. Shaded background boxes will appear as thick (>15px) clusters
        for i, cluster in enumerate(clusters):
            if len(cluster) > 15:
                if i <= 1:  # Header is the first box (or second if under a thin top border)
                    return cluster[-1] + 2

        # 2. Explicit drawn horizontal lines
        if len(clusters) >= 2:
            if np.mean(clusters[0]) < 20:
                return clusters[1][-1] + 2
            else:
                return clusters[0][-1] + 2

        elif len(clusters) == 1:
            if np.mean(clusters[0]) > 20:
                return clusters[0][-1] + 2

        return -1

    def _get_header_height_whitespace(self, image: Image.Image) -> int:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(img_cv, 240, 255, cv2.THRESH_BINARY_INV)
        row_sums = np.sum(thresh, axis=1)

        in_header_text = False
        for y, pixel_sum in enumerate(row_sums):
            if pixel_sum > (image.width * 255 * 0.02):
                in_header_text = True

            if in_header_text and pixel_sum == 0 and y > 20:
                return y + 5

        return -1

    def _calculate_header_height(self, table_crop: Image.Image) -> int:
        h = -1
        if HEADER_EXTRACTION_METHOD == 'cv2':
            h = self._get_header_height_cv2(table_crop)
        elif HEADER_EXTRACTION_METHOD == 'whitespace':
            h = self._get_header_height_whitespace(table_crop)
        elif HEADER_EXTRACTION_METHOD == 'ocr':
            h = self._get_header_height_ocr(None, table_crop.height)

        if h <= 0 and ENABLE_OCR_HEADER_FALLBACK:
            h = self._get_header_height_ocr(None, table_crop.height)

        if h <= 0:
            h = int(table_crop.height * 0.15)

        h = max(self.HEADER_MIN_PX, min(self.HEADER_MAX_PX, h))
        return min(h, table_crop.height)

    @staticmethod
    def _match_template_score(template: np.ndarray, target: np.ndarray) -> float:
        if (
            template.size == 0
            or target.size == 0
            or target.shape[0] < template.shape[0]
            or target.shape[1] < template.shape[1]
        ):
            return 0.0

        result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return float(max_val)

    @staticmethod
    def _match_template_sqdiff(template: np.ndarray, target: np.ndarray) -> Tuple[float, Tuple[int, int]]:
        if (
            template.size == 0
            or target.size == 0
            or target.shape[0] < template.shape[0]
            or target.shape[1] < template.shape[1]
        ):
            return 1.0, (0, 0)

        result = cv2.matchTemplate(target, template, cv2.TM_SQDIFF_NORMED)
        min_val, _, min_loc, _ = cv2.minMaxLoc(result)
        return float(min_val), tuple(int(v) for v in min_loc)

    def _has_native_header(self, header_img: Image.Image, table_crop: Image.Image) -> bool:
        search_height = min(
            table_crop.height,
            header_img.height + max(1, int(HEADER_MATCH_VERTICAL_TOLERANCE_PX)),
        )
        target_strip = table_crop.crop((0, 0, table_crop.width, search_height))

        if target_strip.width != header_img.width:
            target_strip = target_strip.resize(
                (header_img.width, target_strip.height),
                Image.Resampling.LANCZOS,
            )

        header_rgb = np.array(header_img.convert("RGB"))
        target_rgb = np.array(target_strip.convert("RGB"))

        gray_sqdiff, gray_loc = self._match_template_sqdiff(
            cv2.cvtColor(header_rgb, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY),
        )
        rgb_sqdiffs = [
            self._match_template_sqdiff(header_rgb[:, :, channel], target_rgb[:, :, channel])[0]
            for channel in range(3)
        ]
        rgb_sqdiff = max(rgb_sqdiffs)

        strong_sqdiff_match = (
            gray_sqdiff <= HEADER_MATCH_MAX_NORMED_SQDIFF
            and rgb_sqdiff <= HEADER_MATCH_MAX_NORMED_SQDIFF
            and gray_loc[1] <= max(1, int(HEADER_MATCH_VERTICAL_TOLERANCE_PX))
        )
        if strong_sqdiff_match:
            return True

        rgb_score = min(
            self._match_template_score(header_rgb[:, :, channel], target_rgb[:, :, channel])
            for channel in range(3)
        )

        header_gray = cv2.cvtColor(header_rgb, cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)
        gray_score = self._match_template_score(header_gray, target_gray)

        header_edges = cv2.Canny(header_gray, 50, 150)
        target_edges = cv2.Canny(target_gray, 50, 150)
        edge_score = self._match_template_score(header_edges, target_edges)

        strong_rgb_match = rgb_score >= HEADER_MATCH_RGB_THRESHOLD
        structural_match = (
            gray_score >= HEADER_MATCH_GRAY_THRESHOLD
            and edge_score >= HEADER_MATCH_EDGE_THRESHOLD
        )
        return bool(strong_rgb_match or structural_match)

    def _should_skip_header_stitching(
        self,
        header_img: Image.Image,
        table_crops: List[Tuple[int, Image.Image]],
        header_tokens: List[str],
    ) -> bool:
        continuation_crops = table_crops[1:]
        if len(continuation_crops) < 2:
            return False

        center_index = len(continuation_crops) // 2
        probe_indices = sorted(
            {
                center_index,
                min(len(continuation_crops) - 1, center_index + 1),
            }
        )

        probe_results: List[Tuple[int, bool]] = []
        for probe_index in probe_indices:
            page_idx, crop = continuation_crops[probe_index]
            visual_match = self._has_native_header(header_img, crop)
            text_match = self._header_text_matches(header_img, crop, header_tokens) if visual_match else False
            has_header = visual_match and text_match
            probe_results.append((page_idx, has_header))
            print(
                f"        → Header probe page {page_idx + 1}: "
                f"{'present' if has_header else 'missing'}"
            )

        return bool(probe_results) and all(result for _, result in probe_results)

    def _ocr_text(self, image: Image.Image) -> str:
        future = self.ocr_dispatcher.submit(image, task_type="text")
        result = future.result()
        if not result.content:
            return ""
        return str(result.content).strip()

    @staticmethod
    def _header_text_tokens(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _header_text_matches(
        self,
        header_img: Image.Image,
        table_crop: Image.Image,
        expected_tokens: List[str],
    ) -> bool:
        candidate_strip = table_crop.crop(
            (0, 0, table_crop.width, min(table_crop.height, header_img.height + 8))
        )
        if candidate_strip.width != header_img.width:
            candidate_strip = candidate_strip.resize(
                (header_img.width, candidate_strip.height),
                Image.Resampling.LANCZOS,
            )

        candidate_tokens = self._header_text_tokens(self._ocr_text(candidate_strip))

        if not expected_tokens or not candidate_tokens:
            return False

        expected_set = set(expected_tokens)
        candidate_set = set(candidate_tokens)
        shared = expected_set & candidate_set
        if len(shared) < HEADER_TEXT_MIN_SHARED_TOKENS:
            return False

        overlap_ratio = len(shared) / float(max(1, len(expected_set)))
        return overlap_ratio >= HEADER_TEXT_MIN_OVERLAP_RATIO

    def _get_header_height_ocr(self, html: Optional[str], crop_height: int) -> int:
        """Calculates the physical height of the table header based on visual lines."""
        if not html:
            return int(crop_height * 0.15)  # default fallback

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if not table:
            return int(crop_height * 0.15)

        header_trs = set()
        thead = table.find("thead")
        if thead:
            for tr in thead.find_all("tr"):
                header_trs.add(id(tr))
        else:
            trs = table.find_all("tr")
            if trs:
                header_trs.add(id(trs[0]))

        total_visual_lines = 0
        header_visual_lines = 0

        for tr in table.find_all("tr"):
            max_cell_lines = 1
            for cell in tr.find_all(["td", "th"]):
                text = cell.get_text(separator="\n")
                lines = [line for line in text.split("\n") if line.strip()]
                max_cell_lines = max(max_cell_lines, max(1, len(lines)))

            total_visual_lines += max_cell_lines
            if id(tr) in header_trs:
                header_visual_lines += max_cell_lines

        if total_visual_lines == 0:
            return int(crop_height * 0.15)

        pixels_per_line = float(crop_height) / total_visual_lines

        # Capture header lines + 0.1 lines padding for bottom border
        header_h = int(pixels_per_line * (header_visual_lines + 0.1))
        header_h = max(self.HEADER_MIN_PX, min(self.HEADER_MAX_PX, header_h))
        header_h = min(header_h, crop_height)

        return header_h

    @staticmethod
    def _stitch_header(header_img: Image.Image, table_crop: Image.Image) -> Image.Image:
        """Vertically stitch *header_img* on top of *table_crop*."""
        tw, _ = table_crop.size
        hw, hh = header_img.size

        if hw != tw:
            scale = tw / hw
            new_hh = max(1, int(hh * scale))
            header_resized = header_img.resize((tw, new_hh), Image.Resampling.LANCZOS)
        else:
            header_resized = header_img

        header_arr = np.array(header_resized.convert("RGB"))
        table_arr = np.array(table_crop.convert("RGB"))

        stitched_arr = np.vstack([header_arr, table_arr])
        return Image.fromarray(stitched_arr)

    def _normalize_ocr_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        max_side = max(1, self.OCR_MAX_IMAGE_SIDE)
        max_pixels = max(1, self.OCR_MAX_IMAGE_PIXELS)

        side_scale = min(1.0, max_side / max(width, height))
        pixel_scale = min(1.0, (max_pixels / float(width * height)) ** 0.5)
        scale = min(side_scale, pixel_scale)

        if scale >= 0.999:
            return image

        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _submit_ocr_task(self, image: Image.Image, *, task_type: str) -> Future[OCRTaskResult]:
        return self.ocr_dispatcher.submit(image, task_type=task_type)

    def _ocr_tables_parallel(
        self,
        stitched_images: List[Tuple[int, Image.Image]],
    ) -> Tuple[List[Tuple[int, Optional[str]]], Dict[str, float]]:
        if not stitched_images:
            return [], self._empty_ocr_metrics()

        batches = (
            chunk_items(stitched_images, OCR_BATCH_SIZE)
            if ENABLE_OCR_BATCHING
            else [list(stitched_images)]
        )

        results: List[Tuple[int, Optional[str]]] = []
        ocr_task_results: List[OCRTaskResult] = []

        for batch_index, batch in enumerate(batches, start=1):
            batch_workers = self._resolve_ocr_batch_workers(len(batch))
            if ENABLE_OCR_BATCHING:
                batch_pages = f"{batch[0][0] + 1}-{batch[-1][0] + 1}"
                print(
                    f"        Batch {batch_index}/{len(batches)}: pages {batch_pages} "
                    f"with {batch_workers} worker(s)"
                )
            else:
                print(
                    f"        Single pass: {len(batch)} page(s) with "
                    f"{batch_workers} worker(s)"
                )

            pending = {
                self._submit_ocr_task(img, task_type="table"): page_idx
                for page_idx, img in batch
            }
            for future in as_completed(pending):
                page_idx = pending[future]
                try:
                    task_result = future.result()
                    ocr_task_results.append(task_result)
                    results.append((page_idx, task_result.content))
                except Exception as exc:
                    print(f"        ✗ Page {page_idx + 1}: OCR failed ({exc})")
                    results.append((page_idx, None))

        results.sort(key=lambda x: x[0])
        return results, self._summarize_ocr_metrics(ocr_task_results)

    def _resolve_ocr_batch_workers(self, batch_size: int) -> int:
        return min(
            max(1, batch_size),
            max(1, self.ocr_max_workers),
            max(1, self.ocr_connection_pool_size),
        )

    @staticmethod
    def _empty_ocr_metrics() -> Dict[str, float]:
        return {
            "task_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "queue_wait_mean": 0.0,
            "queue_wait_max": 0.0,
            "build_request_mean": 0.0,
            "build_request_max": 0.0,
            "request_mean": 0.0,
            "request_max": 0.0,
            "total_mean": 0.0,
            "total_max": 0.0,
            "max_queue_size_at_submit": 0.0,
        }

    def _summarize_ocr_metrics(self, task_results: List[OCRTaskResult]) -> Dict[str, float]:
        if not task_results:
            return self._empty_ocr_metrics()

        queue_waits = [item.queue_wait_seconds for item in task_results]
        build_times = [item.build_request_seconds for item in task_results]
        request_times = [item.request_seconds for item in task_results]
        total_times = [item.total_seconds for item in task_results]
        status_codes = [item.status_code for item in task_results]
        submitted_queue_sizes = [item.queue_size_at_submit for item in task_results]

        return {
            "task_count": float(len(task_results)),
            "success_count": float(sum(1 for code in status_codes if code == 200)),
            "failure_count": float(sum(1 for code in status_codes if code != 200)),
            "queue_wait_mean": round(sum(queue_waits) / len(queue_waits), 6),
            "queue_wait_max": round(max(queue_waits), 6),
            "build_request_mean": round(sum(build_times) / len(build_times), 6),
            "build_request_max": round(max(build_times), 6),
            "request_mean": round(sum(request_times) / len(request_times), 6),
            "request_max": round(max(request_times), 6),
            "total_mean": round(sum(total_times) / len(total_times), 6),
            "total_max": round(max(total_times), 6),
            "max_queue_size_at_submit": float(max(submitted_queue_sizes)),
        }

    @staticmethod
    def _align_columns_with_headers(
        df: pd.DataFrame,
        expected_headers: List[str],
    ) -> pd.DataFrame:
        if df.empty or not expected_headers:
            return df

        generic_columns = [f"col_{idx}" for idx in range(len(df.columns))]
        if list(df.columns) != generic_columns:
            return df

        if len(df.columns) != len(expected_headers):
            return df

        return df.set_axis(make_columns_unique(list(expected_headers)), axis=1)

    @staticmethod
    def _find_date_column(df: pd.DataFrame) -> Optional[str]:
        if df.empty:
            return None

        named_date_columns = [
            column
            for column in df.columns
            if "date" in str(column).lower()
        ]
        if named_date_columns:
            return named_date_columns[0]

        best_column: Optional[str] = None
        best_score = 0
        for column in df.columns:
            values = df[column].fillna("").astype(str).str.strip()
            score = int(values.apply(is_date_like).sum())
            if score > best_score:
                best_score = score
                best_column = str(column)

        return best_column if best_score > 0 else str(df.columns[0])

    @staticmethod
    def _is_non_transaction_row(row_values: List[str]) -> bool:
        text = _row_text(row_values)
        if not text:
            return True
        return _is_non_transaction_text(text)

    @staticmethod
    def _should_merge_undated_row(row_values: List[str], date_column_index: int) -> bool:
        non_empty_indices = [idx for idx, value in enumerate(row_values) if value]
        non_empty_non_date = [idx for idx in non_empty_indices if idx != date_column_index]
        return 0 < len(non_empty_non_date) <= 2

    @staticmethod
    def _is_isolated_undated_row(
        rows_as_strings: pd.DataFrame,
        row_index: int,
        date_column: str,
    ) -> bool:
        if row_index <= 0 or row_index >= len(rows_as_strings) - 1:
            return False

        previous_has_date = is_date_like(rows_as_strings.iloc[row_index - 1][date_column])
        current_has_date = is_date_like(rows_as_strings.iloc[row_index][date_column])
        next_has_date = is_date_like(rows_as_strings.iloc[row_index + 1][date_column])
        return previous_has_date and not current_has_date and next_has_date

    def _merge_multiline_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        date_column = self._find_date_column(df)
        if date_column is None:
            return df
        date_column_index = list(df.columns).index(date_column)
        rows_as_strings = df.fillna("").astype(str)
        pair_candidates = sum(
            1
            for row_index in range(1, len(rows_as_strings) - 1)
            if self._is_isolated_undated_row(rows_as_strings, row_index, date_column)
        )

        enable_sparse_merge = pair_candidates >= 2

        merged_rows: List[Dict[str, str]] = []

        for row_index, row in rows_as_strings.iterrows():
            row_values = [value.strip() for value in row.tolist()]
            has_date = is_date_like(row[date_column])

            if has_date:
                merged_rows.append({column: str(row[column]).strip() for column in df.columns})
                continue

            if self._is_non_transaction_row(row_values):
                continue

            if not merged_rows:
                continue

            if not enable_sparse_merge:
                continue

            if not self._is_isolated_undated_row(rows_as_strings, row_index, date_column):
                continue

            previous = merged_rows[-1]
            previous_has_date = is_date_like(previous.get(date_column, ""))
            if not previous_has_date:
                continue

            if not self._should_merge_undated_row(row_values, date_column_index):
                continue

            for column in df.columns:
                current_value = str(row[column]).strip()
                if not current_value:
                    continue
                previous_value = previous.get(column, "").strip()
                if not previous_value:
                    previous[column] = current_value
                elif current_value not in previous_value:
                    previous[column] = f"{previous_value} {current_value}".strip()

        if not merged_rows:
            return df

        return pd.DataFrame(merged_rows, columns=df.columns)

    def _filter_valid_date_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        date_column = self._find_date_column(df)
        if date_column is None or date_column not in df.columns:
            return df

        mask = df[date_column].fillna("").astype(str).str.strip().apply(is_date_like)
        
        # Avoid dropping ALL rows if date detection completely failed
        if mask.sum() == 0:
            print("  Warning: No rows had a valid date. Skipping date filtering.")
            return df

        removed = int((~mask).sum())
        if removed > 0:
            print(f"  Removed {removed} row(s) without a valid date")
        return df.loc[mask].reset_index(drop=True)

    def _remove_header_rows(self, df: pd.DataFrame, headers: List[str]) -> pd.DataFrame:
        if not headers or df.empty:
            return df

        header_lower = {h.lower().strip() for h in headers if h.strip()}

        def is_header_row(row):
            non_empty = [str(v).strip().lower() for v in row if str(v).strip()]
            if not non_empty:
                return False
            return all(v in header_lower for v in non_empty)

        mask = df.apply(is_header_row, axis=1)
        removed = mask.sum()
        if removed > 0:
            print(f"  Removed {removed} repeated header row(s)")
        return df[~mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    input_path = sys.argv[1] if len(sys.argv) > 1 else "bank statements"
    default_config_path = Path(__file__).resolve().parent / "config.yaml"
    config_path = sys.argv[2] if len(sys.argv) > 2 else str(default_config_path)

    with BankStatementParser(config_path) as parser:
        pdfs = collect_pdf_paths(input_path)
        if not pdfs:
            print(f"No PDFs found at: {input_path}")
            raise SystemExit(1)

        for pdf in pdfs:
            df = parser.parse_pdf(pdf)

            if df.empty:
                print(f"  ? No data extracted from {pdf}")
                continue

            print(f"\nTotal transaction rows: {len(df)}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst 10 rows:")
            print(df.head(10).to_string())
            print(f"\nLast 10 rows:")
            print(df.tail(10).to_string())

            output_name = os.path.splitext(os.path.basename(pdf))[0]
            csv_path = f"./output/{output_name}_transactions.csv"
            os.makedirs("./output", exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f"\nSaved to: {csv_path}")
