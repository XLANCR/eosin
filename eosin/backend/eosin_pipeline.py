"""
Bank Statement PDF → Unified pandas DataFrame

Strategy (Header-Stitching):
  1. Render PDF pages to PIL images using SDK's pdf_to_images_pil()
  2. Run layout detection on ALL pages to find table bounding boxes
  3. Identify the main transaction table per page (using single-table page anchors)
  4. Crop the first table and OCR it to determine row count
  5. Crop the header region from page 1's table
  6. Stitch the header onto each continuation page's table crop
  7. OCR each final table page crop independently via vLLM
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
from PIL import Image, ImageFilter, ImageOps

# --- Configuration ---
HEADER_EXTRACTION_METHOD = 'cv2'  # 'cv2' (Lines), 'whitespace' (Gaps), or 'ocr' (HTML rows)
ENABLE_OCR_HEADER_FALLBACK = False
SAVE_DEBUG_IMAGES = False
ENABLE_OCR_BATCHING = False
OCR_BATCH_SIZE = 25
OCR_PIPELINE_WORKERS = None
OCR_PIPELINE_QUEUE_SIZE = 128
OCR_BACKEND_MODE = "page_http"
_OCR_DOCUMENT_MAX_IMAGES_PER_REQUEST_RAW = os.getenv("BANK_PARSER_OCR_DOCUMENT_MAX_IMAGES_PER_REQUEST", "").strip()
OCR_DOCUMENT_MAX_IMAGES_PER_REQUEST = (
    int(_OCR_DOCUMENT_MAX_IMAGES_PER_REQUEST_RAW)
    if _OCR_DOCUMENT_MAX_IMAGES_PER_REQUEST_RAW
    else None
)
OCR_BATCH_DRAIN_MAX_BATCH_SIZE = 32
OCR_BATCH_DRAIN_MAX_WAIT_SECONDS = 0.25
ENABLE_PAGE_OCR_RETRY = os.getenv("BANK_PARSER_ENABLE_PAGE_OCR_RETRY", "false").strip().lower() in {"1", "true", "yes", "on"}
PAGE_OCR_RETRY_ALL = os.getenv("BANK_PARSER_PAGE_OCR_RETRY_ALL", "false").strip().lower() in {"1", "true", "yes", "on"}
PAGE_OCR_RETRY_DPI = int(os.getenv("BANK_PARSER_PAGE_OCR_RETRY_DPI", "300"))
CAPTURE_RAW_OCR_DEBUG = os.getenv("BANK_PARSER_CAPTURE_RAW_OCR_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
PDF_RENDER_DPI_OVERRIDE = None
PDF_RENDER_DPI_AUTO = False
PDF_RENDER_TARGET_LONG_SIDE_PX = 2600
PDF_RENDER_DPI_SAFETY_FACTOR = 0.9
PDF_RENDER_DPI_MIN = 150
PDF_RENDER_DPI_MAX = 300
PARSE_TESTING = False
PARSE_TESTING_TABLE_PAGES_PER_SIDE = 3
RECOVER_LAYOUT_MISSED_PAGES = os.getenv(
    "BANK_PARSER_RECOVER_LAYOUT_MISSED_PAGES", "true"
).strip().lower() in {"1", "true", "yes", "on"}
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
from eosin.backend.transaction_reconstruction import (
    SOURCE_PAGE_COLUMN,
    SOURCE_ROW_COLUMN,
    SOURCE_TABLE_COLUMN,
    reconstruct_transactions,
)


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
BANK_HEADER_BRANCH_TERMS = {
    "branch",
    "branch name",
    "branch code",
    "sol id",
    "init br",
    "init. br",
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
WHITESPACE_RE = re.compile(r"\s+")
RUNAWAY_ZERO_RE = re.compile(r"0{12,}")
RUNAWAY_DIGIT_RE = re.compile(r"\d{24,}")
REPEATED_TOKEN_RE = re.compile(r"\b([A-Za-z]{2,})\b(?:\s+\1\b){3,}", re.IGNORECASE)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
PAGE_QUALITY_LONG_CELL_THRESHOLD = 96
PAGE_QUALITY_MIN_SCORE = 55
PAGE_QUALITY_LOW_CONFIDENCE_SCORE = 70
PAGE_DUPLICATE_WINDOW = 2


def is_date_like(value: object) -> bool:
    text = str(value).strip()
    return bool(text and DATE_VALUE_RE.match(text))


def _cell_is_amount_like(value: object) -> bool:
    text = str(value or "").strip()
    if not text or text == "-":
        return False
    normalized = text.replace(",", "")
    return bool(re.fullmatch(r"-?\d+(?:\.\d{1,2})?", normalized))


def _infer_headerless_bank_columns(rows: Sequence[Sequence[str]]) -> Optional[List[str]]:
    """Infer roles for GLM tables that put transaction rows in <thead>.

    This only names observed columns. It never moves, computes, or fills values.
    """
    if not rows:
        return None

    ncols = max(len(row) for row in rows)
    if ncols not in {5, 6, 7}:
        return None

    candidate_rows = [list(row) + [""] * (ncols - len(row)) for row in rows[: min(5, len(rows))]]
    date_hits = sum(1 for row in candidate_rows if is_date_like(row[0]))
    if date_hits < max(1, len(candidate_rows) // 2):
        return None

    def text_like(value: str) -> bool:
        text = str(value).strip()
        return bool(text and not is_date_like(text) and not _cell_is_amount_like(text))

    if ncols == 7:
        hdfc_text_hits = sum(1 for row in candidate_rows if text_like(row[1]))
        value_date_hits = sum(1 for row in candidate_rows if is_date_like(row[3]))
        hdfc_money_hits = sum(
            1
            for row in candidate_rows
            if any(_cell_is_amount_like(row[index]) for index in (4, 5, 6))
        )
        if hdfc_text_hits and value_date_hits and hdfc_money_hits:
            return [
                "Transaction Date",
                "Description",
                "Reference",
                "Value Date",
                "Debit",
                "Credit",
                "Balance",
            ]
        text_hits = sum(1 for row in candidate_rows if text_like(row[2]))
        money_hits = sum(1 for row in candidate_rows if any(_cell_is_amount_like(row[index]) for index in (3, 4, 5)))
        if text_hits and money_hits:
            return ["Tran Date", "col_1", "Particulars", "Debit", "Credit", "Balance", "Init. Br"]
    elif ncols == 6:
        hdfc_text_hits = sum(1 for row in candidate_rows if text_like(row[1]))
        value_date_hits = sum(1 for row in candidate_rows if is_date_like(row[3]))
        hdfc_money_hits = sum(
            1
            for row in candidate_rows
            if _cell_is_amount_like(row[4]) and _cell_is_amount_like(row[5])
        )
        if hdfc_text_hits and value_date_hits and hdfc_money_hits:
            return [
                "Transaction Date",
                "Description",
                "Reference",
                "Value Date",
                "Amount",
                "Balance",
            ]
        text_hits = sum(1 for row in candidate_rows if text_like(row[1]))
        money_hits = sum(1 for row in candidate_rows if any(_cell_is_amount_like(row[index]) for index in (2, 3, 4)))
        if text_hits and money_hits:
            return ["Tran Date", "Particulars", "Debit", "Credit", "Balance", "Init. Br"]
    elif ncols == 5:
        text_hits = sum(1 for row in candidate_rows if text_like(row[1]))
        money_hits = sum(1 for row in candidate_rows if any(_cell_is_amount_like(row[index]) for index in (2, 3, 4)))
        if text_hits and money_hits:
            return ["Date", "Description", "Debit", "Credit", "Balance"]

    return None


def _complete_blank_bank_headers(
    header_row: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> List[str]:
    completed = [str(header or "").strip() for header in header_row]
    if not completed or not rows:
        return completed

    ncols = len(completed)
    padded_rows = [list(row) + [""] * (ncols - len(row)) for row in rows[: min(5, len(rows))]]
    date_hits = sum(1 for row in padded_rows if row and is_date_like(row[0]))
    if date_hits < max(1, len(padded_rows) // 2):
        return completed

    normalized = [_normalize_header_text(header) for header in completed]
    if ncols >= 5 and not completed[0] and any(
        _header_contains_term(header, BANK_HEADER_DATE_TERMS) for header in normalized
    ):
        return completed

    has_text_header = any(_header_contains_term(header, BANK_HEADER_TEXT_TERMS) for header in normalized)
    money_hits = sum(1 for header in normalized if _header_contains_term(header, BANK_HEADER_MONEY_TERMS))
    if not has_text_header or money_hits < 2:
        return completed

    if not completed[0]:
        completed[0] = "Tran Date" if ncols >= 6 else "Date"
    for index, header in enumerate(completed):
        if not header:
            completed[index] = f"col_{index}"
    return completed


def _align_row_to_expected_headers(row: Sequence[str], expected_headers: Sequence[str]) -> List[str]:
    aligned = list(row)
    if len(expected_headers) < 6:
        return aligned

    normalized_headers = [_normalize_header_text(header) for header in expected_headers]
    has_blank_reference_slot = (
        normalized_headers[1].startswith("col_")
        or _header_contains_term(normalized_headers[1], BANK_HEADER_REFERENCE_TERMS)
    )
    has_text_after_slot = _header_contains_term(normalized_headers[2], BANK_HEADER_TEXT_TERMS)
    if (
        has_blank_reference_slot
        and has_text_after_slot
        and aligned
        and is_date_like(aligned[0])
        and len(aligned) > 1
        and str(aligned[1]).strip()
        and not _cell_is_amount_like(aligned[1])
    ):
        if len(aligned) == len(expected_headers) - 1:
            return [aligned[0], "", *aligned[1:]]
        if len(aligned) == len(expected_headers) and len(aligned) > 2 and not str(aligned[2]).strip():
            return [aligned[0], "", aligned[1], *aligned[3:]]
    return aligned


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


def _normalize_quality_text(value: object) -> str:
    return WHITESPACE_RE.sub(" ", str(value or "").strip().lower())


def _normalize_identity_text(value: object) -> str:
    return NON_ALNUM_RE.sub("", _normalize_quality_text(value))


def _cell_looks_runaway(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if len(text) >= PAGE_QUALITY_LONG_CELL_THRESHOLD:
        return True
    if RUNAWAY_ZERO_RE.search(text) or RUNAWAY_DIGIT_RE.search(text):
        return True
    if REPEATED_TOKEN_RE.search(text):
        return True
    return False


def _cell_has_runaway_artifact(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(
        RUNAWAY_ZERO_RE.search(text)
        or RUNAWAY_DIGIT_RE.search(text)
        or REPEATED_TOKEN_RE.search(text)
    )


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


def parse_span_attribute(value: object, default: int = 1, max_span: int = 50) -> int:
    """Parse colspan/rowspan values defensively from malformed model HTML."""
    if value is None:
        return default
    match = re.match(r"\s*(\d+)", str(value))
    if not match:
        return default
    span = int(match.group(1))
    return max(1, min(span, max_span))


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
        total_cols = sum(parse_span_attribute(c.get("colspan")) for c in th_cells)

        for c in tr.find_all("td"):
            total_cols += parse_span_attribute(c.get("colspan"))
            if c.get_text(strip=True):
                named_count += 1

        if named_count > best_score:
            best_score = named_count
            best_header_row = tr

    headers: List[str] = []
    if best_header_row:
        for cell in best_header_row.find_all(["th", "td"]):
            colspan = parse_span_attribute(cell.get("colspan"))
            text = cell.get_text(strip=True)
            headers.extend([text] + [""] * (colspan - 1))

    # Check if data rows have more columns than header
    data_ncols = 0
    for tr in table.find_all("tr"):
        td_cells = tr.find_all("td")
        if not td_cells:
            continue
        n = sum(parse_span_attribute(c.get("colspan")) for c in tr.find_all(["td", "th"]))
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


def _normalize_dedupe_cell(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def _rows_match_after_normalization(
    left_row: dict[str, str],
    right_row: dict[str, str],
    columns: Sequence[str],
) -> bool:
    saw_value = False
    for column in columns:
        left_value = _normalize_dedupe_cell(left_row.get(column, ""))
        right_value = _normalize_dedupe_cell(right_row.get(column, ""))
        if left_value or right_value:
            saw_value = True
        if left_value != right_value:
            return False
    return saw_value


def _rows_match_on_non_empty_overlap(
    left_row: dict[str, str],
    right_row: dict[str, str],
    columns: Sequence[str],
) -> bool:
    overlap = 0
    for column in columns:
        left_value = _normalize_dedupe_cell(left_row.get(column, ""))
        right_value = _normalize_dedupe_cell(right_row.get(column, ""))
        if not left_value or not right_value:
            continue
        if left_value != right_value:
            return False
        overlap += 1
    return overlap >= 3


def _normalize_header_cell_text(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())
    return normalized


def _semantic_header_key(value: object) -> str:
    normalized = _normalize_header_cell_text(value)
    if normalized in {"debt", "debtdr", "debtdr", "debit", "debitdr", "withdrawal", "withdrawals"}:
        return "debit"
    if normalized in {"credit", "creditcr", "cr", "deposit", "deposits"}:
        return "credit"
    if normalized in {"chqrefno", "chqrefnumber", "chequerefno", "refno", "referenceno", "referencenumber"}:
        return "reference"
    return normalized


def _header_cell_matches_expected(value: object, expected_values: Sequence[str]) -> bool:
    normalized_value = _normalize_header_cell_text(value)
    if not normalized_value:
        return False

    for expected in expected_values:
        normalized_expected = _normalize_header_cell_text(expected)
        if not normalized_expected:
            continue
        if normalized_value == normalized_expected:
            return True
        if normalized_expected.startswith(normalized_value) and len(normalized_value) >= max(4, len(normalized_expected) - 3):
            return True
        if normalized_value.startswith(normalized_expected) and len(normalized_expected) >= 4:
            return True

    return False


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
                colspan = parse_span_attribute(cell.get("colspan"))
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
            colspan = parse_span_attribute(cell.get("colspan"))
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

    inferred_headers = None if header_row else _infer_headerless_bank_columns(rows)
    if (
        inferred_headers
        and expected_headers
        and len(inferred_headers) != len(expected_headers)
        and "Reference" in inferred_headers
        and "Value Date" in inferred_headers
    ):
        expected_headers = None

    if expected_headers:
        ncols = len(expected_headers)
    elif inferred_headers:
        ncols = len(inferred_headers)
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

    if header_row and not expected_headers:
        header_row = _complete_blank_bank_headers(header_row, rows)

    normalized_rows: List[List[str]] = []
    header_lower = (
        {h.lower().strip() for h in expected_headers if h.strip()}
        if expected_headers
        else set()
    )

    for row in rows:
        if expected_headers:
            row = _align_row_to_expected_headers(row, expected_headers)

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
    elif inferred_headers:
        cols = list(inferred_headers)
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
    OCR_MAX_IMAGE_SIDE = int(os.getenv("BANK_PARSER_OCR_MAX_IMAGE_SIDE", "3500"))
    OCR_MAX_IMAGE_PIXELS = int(os.getenv("BANK_PARSER_OCR_MAX_IMAGE_PIXELS", "9000000"))

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
            backend_mode=self._resolve_ocr_backend_mode(),
            batch_drain_max_batch_size=self._resolve_ocr_batch_drain_max_batch_size(),
            batch_drain_max_wait_seconds=self._resolve_ocr_batch_drain_max_wait_seconds(),
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

    @staticmethod
    def _resolve_ocr_backend_mode() -> str:
        mode = str(OCR_BACKEND_MODE or "page_http").strip().lower()
        return mode or "page_http"

    @staticmethod
    def _resolve_ocr_batch_drain_max_batch_size() -> int:
        return max(1, int(OCR_BATCH_DRAIN_MAX_BATCH_SIZE))

    @staticmethod
    def _resolve_ocr_batch_drain_max_wait_seconds() -> float:
        return max(0.001, float(OCR_BATCH_DRAIN_MAX_WAIT_SECONDS))

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
        page_evaluations = self._extract_transaction_page_evaluations(all_ocr_results, pdf_path)
        return self._materialize_selected_tables(page_evaluations)

    def _extract_transaction_page_evaluations(
        self,
        all_ocr_results: List[Tuple[int, Optional[str]]],
        pdf_path: str,
    ) -> List[Dict[str, object]]:
        page_evaluations: List[Dict[str, object]] = []
        expected_headers: Optional[List[str]] = None

        for page_idx, html_content in all_ocr_results:
            evaluation = self._evaluate_page_ocr_result(
                page_idx=page_idx,
                html_content=html_content,
                expected_headers=expected_headers,
                pass_label="primary",
            )
            if CAPTURE_RAW_OCR_DEBUG and evaluation.get("raw_html"):
                debug_dir = os.path.join("output", "debug", os.path.splitext(os.path.basename(pdf_path))[0])
                os.makedirs(debug_dir, exist_ok=True)
                with open(os.path.join(debug_dir, f"page_{page_idx + 1}_raw.html"), "w", encoding="utf-8") as handle:
                    handle.write(str(evaluation["raw_html"]))

            raw_headers = evaluation.get("raw_headers")
            if expected_headers is None and raw_headers:
                expected_headers = list(raw_headers)

            page_evaluations.append(evaluation)

        return page_evaluations

    @staticmethod
    def _has_transaction_row_shape(dataframe: pd.DataFrame) -> bool:
        if dataframe.empty:
            return False
        amount_pattern = re.compile(
            r"^(?:[₹$€£]\s*)?[+-]?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d{1,2})?"
            r"(?:\s*(?:cr|dr))?$",
            re.IGNORECASE,
        )
        for _, row in dataframe.fillna("").astype(str).iterrows():
            values = [value.strip() for value in row.tolist() if value.strip()]
            has_date = any(is_date_like(value) for value in values)
            has_text = any(re.search(r"[A-Za-z]{3}", value) for value in values)
            has_amount = any(
                amount_pattern.fullmatch(value)
                and (not value.isdigit() or len(value) <= 10)
                for value in values
            )
            if has_date and has_text and has_amount:
                return True
        return False

    def _recover_missing_layout_pages(
        self,
        *,
        page_images: Sequence[Image.Image],
        table_bboxes: Sequence[Optional[List[int]]],
        page_evaluations: Sequence[Dict[str, object]],
    ) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
        evaluated_pages = {int(item["page_index"]) for item in page_evaluations}
        missing_pages = {
            page_idx
            for page_idx, bbox in enumerate(table_bboxes)
            if bbox is None and page_idx not in evaluated_pages
        }
        weak_pages = {
            int(item["page_index"])
            for item in page_evaluations
            if "money_in_non_amount_columns" in item.get("reasons", [])
        }
        recovery_pages = sorted(missing_pages | weak_pages)
        if not recovery_pages:
            return list(page_evaluations), self._empty_ocr_metrics()

        images: List[Tuple[int, Image.Image]] = []
        for page_idx in recovery_pages:
            image = page_images[page_idx]
            bbox = table_bboxes[page_idx]
            if page_idx in weak_pages and bbox is not None:
                x_min, y_min, _, y_max = (int(value) for value in bbox)
                image = crop_image_region(
                    image,
                    [x_min, y_min, int(image.width), y_max],
                )
            images.append((page_idx, self._normalize_ocr_image(image)))
        results, metrics = self._ocr_tables_parallel(images)
        recovered_by_page: Dict[int, Dict[str, object]] = {}
        for page_idx, html_content in results:
            pass_label = (
                "layout_miss_full_page"
                if page_idx in missing_pages
                else "layout_quality_right_edge"
            )
            evaluation = self._evaluate_page_ocr_result(
                page_idx=page_idx,
                html_content=html_content,
                expected_headers=None,
                pass_label=pass_label,
            )
            dataframe = evaluation.get("dataframe")
            if not isinstance(dataframe, pd.DataFrame) or not self._has_transaction_row_shape(dataframe):
                evaluation["selected"] = False
                evaluation["reasons"] = [
                    *evaluation.get("reasons", []),
                    "layout_miss_not_transaction_table",
                ]
            recovered_by_page[page_idx] = evaluation

        merged: List[Dict[str, object]] = []
        for primary in page_evaluations:
            page_idx = int(primary["page_index"])
            recovered = recovered_by_page.pop(page_idx, None)
            merged.append(
                self._select_best_page_ocr_evaluation([primary, recovered])
                if recovered is not None
                else primary
            )
        merged.extend(recovered_by_page.values())
        return sorted(merged, key=lambda item: int(item["page_index"])), metrics

    def _materialize_selected_tables(
        self,
        page_evaluations: Sequence[Dict[str, object]],
    ) -> Tuple[List[pd.DataFrame], Optional[List[str]], List[int]]:
        all_dfs: List[pd.DataFrame] = []
        expected_headers: Optional[List[str]] = None
        selected_pages: List[int] = []

        for evaluation in page_evaluations:
            page_idx = int(evaluation["page_index"])
            if not evaluation.get("selected"):
                print(f"        Page {page_idx + 1}: No usable table found in response")
                continue

            selected_pages.append(page_idx)
            selected_table_index = evaluation.get("selected_table_index")
            table_count = int(evaluation.get("table_count", 0))
            if isinstance(selected_table_index, int) and selected_table_index > 0 and table_count > 0:
                print(
                    f"        Page {page_idx + 1}: selected table "
                    f"{selected_table_index + 1}/{table_count}"
                )

            raw_headers = evaluation.get("raw_headers")
            if expected_headers is None:
                if raw_headers:
                    expected_headers = list(raw_headers)

            if expected_headers is not None:
                print(f"        Headers: {expected_headers}")
                all_dfs = [
                    self._align_columns_with_headers(existing_df, expected_headers)
                    for existing_df in all_dfs
                ]

            dataframe = evaluation.get("dataframe")
            if not isinstance(dataframe, pd.DataFrame):
                continue
            aligned_df = self._align_columns_with_headers(dataframe, expected_headers or [])
            if aligned_df.empty:
                continue
            aligned_df = aligned_df.copy()
            aligned_df[SOURCE_PAGE_COLUMN] = page_idx
            aligned_df[SOURCE_ROW_COLUMN] = range(len(aligned_df))
            aligned_df[SOURCE_TABLE_COLUMN] = (
                selected_table_index if isinstance(selected_table_index, int) else 0
            )
            all_dfs.append(aligned_df)
            added_row_count = len(aligned_df)

            if added_row_count > 0:
                print(f"        Page {page_idx + 1}: {added_row_count} rows")

        return all_dfs, expected_headers, selected_pages

    def _evaluate_page_ocr_result(
        self,
        *,
        page_idx: int,
        html_content: Optional[str],
        expected_headers: Optional[List[str]],
        pass_label: str,
    ) -> Dict[str, object]:
        evaluation: Dict[str, object] = {
            "page_index": int(page_idx),
            "pass_label": str(pass_label),
            "selected": False,
            "suspicious": True,
            "quality_score": 0,
            "reasons": [],
            "raw_html": html_content or "",
            "selected_table_index": None,
            "table_count": 0,
            "row_count": 0,
            "columns": [],
            "dataframe": None,
            "raw_headers": [],
            "retried": pass_label != "primary",
        }
        reasons: List[str] = []

        if not html_content:
            reasons.append("no_html")
            evaluation["reasons"] = reasons
            return evaluation

        table_htmls = extract_all_tables_from_response(html_content)
        evaluation["table_count"] = len(table_htmls)
        if not table_htmls:
            reasons.append("no_table")
            evaluation["reasons"] = reasons
            return evaluation

        selected = select_best_table_candidate(table_htmls, expected_headers=expected_headers)
        if selected is None:
            reasons.append("no_usable_table")
            evaluation["reasons"] = reasons
            return evaluation

        selected_table_index, _, dataframe, raw_headers = selected
        evaluation["selected"] = True
        evaluation["selected_table_index"] = selected_table_index
        evaluation["dataframe"] = dataframe
        evaluation["raw_headers"] = list(raw_headers)
        evaluation["row_count"] = int(len(dataframe))
        evaluation["columns"] = [str(column) for column in dataframe.columns]

        if any(str(column).endswith(("_2", "_3", "_4")) for column in dataframe.columns):
            reasons.append("duplicate_columns")

        header_like_rows = self._count_header_like_rows(dataframe, raw_headers)
        if header_like_rows > 0:
            reasons.append("header_like_rows")

        date_ratio = self._date_row_ratio(dataframe)
        if date_ratio < 0.5:
            reasons.append("low_date_coverage")

        money_ratio = self._money_signal_ratio(dataframe)
        if money_ratio < 0.3:
            reasons.append("low_money_coverage")

        duplicate_rows = self._count_duplicate_rows(dataframe)
        if duplicate_rows > 0:
            reasons.append("duplicate_rows")

        null_heavy_rows = self._count_null_heavy_rows(dataframe)
        if null_heavy_rows > 0:
            reasons.append("null_heavy_rows")

        non_amount_money_leaks = self._count_non_amount_money_leaks(dataframe)
        if non_amount_money_leaks > 0:
            reasons.append("money_in_non_amount_columns")

        dated_rows_missing_text = self._count_dated_rows_missing_text(dataframe)
        if dated_rows_missing_text > 0:
            reasons.append("dated_rows_missing_text")

        if self._has_long_cell(dataframe):
            reasons.append("long_cell")

        if self._has_runaway_tokens(dataframe):
            reasons.append("runaway_tokens")

        score = 100
        penalties = {
            "duplicate_columns": 30,
            "header_like_rows": 20,
            "low_date_coverage": 30,
            "low_money_coverage": 20,
            "duplicate_rows": 20,
            "null_heavy_rows": 20,
            "money_in_non_amount_columns": 30,
            "dated_rows_missing_text": 20,
            "long_cell": 20,
            "runaway_tokens": 20,
        }
        for reason in reasons:
            score -= penalties.get(reason, 0)
        score = max(0, min(100, score))

        evaluation["quality_score"] = score
        evaluation["suspicious"] = bool(reasons) or score < PAGE_QUALITY_MIN_SCORE
        evaluation["reasons"] = reasons
        return evaluation

    @staticmethod
    def _select_best_page_ocr_evaluation(
        evaluations: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        if not evaluations:
            raise ValueError("evaluations must not be empty")

        def sort_key(item: Dict[str, object]) -> Tuple[int, int, int]:
            selected = 1 if item.get("selected") else 0
            suspicious = 0 if not item.get("suspicious") else -1
            score = int(item.get("quality_score", 0))
            return selected, suspicious, score

        return max(evaluations, key=sort_key)

    @staticmethod
    def _should_retry_page_evaluation(evaluation: Dict[str, object]) -> bool:
        severe_reasons = {
            "no_html",
            "no_table",
            "no_usable_table",
            "duplicate_columns",
            "low_date_coverage",
            "low_money_coverage",
            "null_heavy_rows",
            "money_in_non_amount_columns",
            "dated_rows_missing_text",
            "long_cell",
            "runaway_tokens",
        }
        reasons = set(str(reason) for reason in evaluation.get("reasons", []))
        score = int(evaluation.get("quality_score", 0))
        return (not evaluation.get("selected")) or score < PAGE_QUALITY_MIN_SCORE or bool(reasons & severe_reasons)

    def _finalize_extracted_tables(
        self,
        all_dfs: List[pd.DataFrame],
        expected_headers: Optional[List[str]],
    ) -> pd.DataFrame:
        del expected_headers
        result = reconstruct_transactions(all_dfs)
        self._last_transaction_reconstruction = result.diagnostics
        return result.dataframe

    @staticmethod
    def _count_header_like_rows(df: pd.DataFrame, headers: Sequence[str]) -> int:
        if df.empty or not headers:
            return 0
        expected_headers = [header for header in headers if str(header).strip()]

        def is_header_row(row: pd.Series) -> bool:
            non_empty = [str(v).strip() for v in row if str(v).strip()]
            if not non_empty:
                return False
            matches = sum(
                1
                for value in non_empty
                if _header_cell_matches_expected(value, expected_headers)
            )
            return matches >= max(2, len(non_empty) - 1)

        return int(df.apply(is_header_row, axis=1).sum())

    def _date_row_ratio(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        date_column = self._find_date_column(df)
        if date_column is None or date_column not in df.columns:
            return 0.0
        mask = df[date_column].fillna("").astype(str).str.strip().apply(is_date_like)
        return float(mask.sum()) / float(max(1, len(df)))

    @staticmethod
    def _money_signal_ratio(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        candidate_columns = [
            column
            for column in df.columns
            if any(term in str(column).lower() for term in ("debit", "credit", "balance", "amount", "dr", "cr"))
        ]
        if not candidate_columns:
            return 0.0
        present = 0
        for _, row in df.fillna("").astype(str).iterrows():
            if any(str(row[column]).strip() not in {"", "-"} for column in candidate_columns):
                present += 1
        return float(present) / float(max(1, len(df)))

    @staticmethod
    def _row_signature(row: Dict[str, str], columns: Sequence[object]) -> Tuple[str, ...]:
        return tuple(_normalize_identity_text(row.get(str(column), "")) for column in columns)

    def _count_duplicate_rows(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        rows_as_strings = df.fillna("").astype(str)
        signatures = [
            self._row_signature(
                {column: str(row[column]).strip() for column in df.columns},
                df.columns,
            )
            for _, row in rows_as_strings.iterrows()
        ]
        return max(0, len(signatures) - len(set(signatures)))

    def _count_null_heavy_rows(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        count = 0
        for _, row in df.fillna("").astype(str).iterrows():
            meaningful = sum(1 for value in row.tolist() if str(value).strip() not in {"", "-"})
            if meaningful <= 2:
                count += 1
        return count

    def _count_non_amount_money_leaks(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0

        leak_columns = [
            column
            for column in df.columns
            if (
                self._header_term_hits(str(column), BANK_HEADER_REFERENCE_TERMS) > 0
                or self._header_term_hits(str(column), BANK_HEADER_BRANCH_TERMS) > 0
            )
            and self._header_term_hits(str(column), BANK_HEADER_MONEY_TERMS) == 0
        ]
        if not leak_columns:
            return 0

        count = 0
        for _, row in df.fillna("").astype(str).iterrows():
            for column in leak_columns:
                value = str(row[column]).strip()
                if not value:
                    continue
                if re.search(r"\d+\.\d{1,2}$", value.replace(",", "")):
                    count += 1
                    break
        return count

    def _count_dated_rows_missing_text(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        date_column = self._find_date_column(df)
        if date_column is None:
            return 0
        text_columns = [
            column
            for column in df.columns
            if self._header_term_hits(str(column), BANK_HEADER_TEXT_TERMS) > 0
        ]
        if not text_columns:
            return 0

        count = 0
        for _, row in df.fillna("").astype(str).iterrows():
            if not is_date_like(row.get(date_column, "")):
                continue
            has_text = any(str(row[column]).strip() for column in text_columns)
            if not has_text:
                count += 1
        return count

    def _has_long_cell(self, df: pd.DataFrame) -> bool:
        if df.empty:
            return False
        for value in df.fillna("").astype(str).to_numpy().flatten():
            if _cell_looks_runaway(value):
                return True
        return False

    def _has_runaway_tokens(self, df: pd.DataFrame) -> bool:
        if df.empty:
            return False
        for value in df.fillna("").astype(str).to_numpy().flatten():
            text = str(value).strip()
            if RUNAWAY_ZERO_RE.search(text) or REPEATED_TOKEN_RE.search(text):
                return True
        return False

    def _drop_low_quality_rows(
        self,
        df: pd.DataFrame,
        headers: Sequence[str],
    ) -> pd.DataFrame:
        if df.empty:
            return df

        expected_headers = [header for header in headers if str(header).strip()]
        kept_rows: List[Dict[str, str]] = []
        removed = 0
        date_column = self._find_date_column(df)

        for _, row in df.fillna("").astype(str).iterrows():
            current = {column: str(row[column]).strip() for column in df.columns}
            non_empty = [value for value in current.values() if value not in {"", "-"}]
            row_text = " ".join(non_empty)
            if not non_empty:
                removed += 1
                continue
            if expected_headers and self._count_header_like_rows(pd.DataFrame([current]), expected_headers) > 0:
                removed += 1
                continue
            if (
                self._is_non_transaction_row(list(current.values()))
                and "opening balance" not in row_text.lower()
                and "closing balance" not in row_text.lower()
                and date_column
                and is_date_like(current.get(date_column, ""))
            ):
                removed += 1
                continue
            if len(non_empty) <= 2 and not is_date_like(current.get(date_column or "", "")):
                removed += 1
                continue
            if is_date_like(current.get(date_column or "", "")) and len(non_empty) <= 1:
                removed += 1
                continue
            if any(_cell_has_runaway_artifact(value) for value in current.values()):
                removed += 1
                continue
            if RUNAWAY_ZERO_RE.search(row_text) or REPEATED_TOKEN_RE.search(row_text):
                removed += 1
                continue
            kept_rows.append(current)

        if removed > 0:
            print(f"  Removed {removed} low-quality row(s)")
        return pd.DataFrame(kept_rows, columns=df.columns) if kept_rows else df.iloc[0:0].copy()

    def _apply_bank_specific_repairs(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        repaired = df.copy()
        for column in repaired.columns:
            lowered = str(column).lower()
            series = repaired[column].fillna("").astype(str).str.strip()
            if "ref" in lowered or "chq" in lowered:
                repaired[column] = series.apply(
                    lambda value: "" if value == "0" or RUNAWAY_ZERO_RE.search(value) else value
                )
            elif "description" in lowered or "narration" in lowered or "particular" in lowered:
                repaired[column] = series.apply(
                    lambda value: "" if _cell_has_runaway_artifact(value) else value
                )
        return repaired

    @staticmethod
    def _merge_semantic_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        merged = df.copy()
        canonical_by_key: Dict[str, str] = {}
        drop_columns: List[str] = []

        for column in list(merged.columns):
            key = _semantic_header_key(column)
            if not key:
                continue
            canonical = canonical_by_key.get(key)
            if canonical is None:
                canonical_by_key[key] = str(column)
                continue

            canonical_series = merged[canonical].fillna("").astype(str).str.strip()
            duplicate_series = merged[column].fillna("").astype(str).str.strip()
            fill_mask = canonical_series.isin(["", "-"]) & ~duplicate_series.isin(["", "-"])
            merged.loc[fill_mask, canonical] = duplicate_series[fill_mask]
            drop_columns.append(str(column))

        if drop_columns:
            merged = merged.drop(columns=drop_columns)
        return merged

    def _build_document_quality_summary(
        self,
        df: pd.DataFrame,
        page_evaluations: Sequence[Dict[str, object]],
    ) -> Dict[str, object]:
        suspicious_pages = [
            int(item["page_index"]) + 1
            for item in page_evaluations
            if item.get("suspicious")
        ]
        retried_pages = sorted(
            {
                int(item["page_index"]) + 1
                for item in page_evaluations
                if item.get("retried")
            }
        )
        quality_scores = [int(item.get("quality_score", 0)) for item in page_evaluations if item.get("selected")]
        min_page_score = min(quality_scores) if quality_scores else 0
        date_ratio = self._date_row_ratio(df)
        money_ratio = self._money_signal_ratio(df)
        duplicate_rate = (
            float(self._count_duplicate_rows(df)) / float(max(1, len(df)))
            if not df.empty
            else 1.0
        )
        long_cell_remaining = self._has_long_cell(df)
        final_table_has_quality_issue = (
            df.empty
            or date_ratio < 0.9
            or money_ratio < 0.75
            or duplicate_rate > 0.0
            or long_cell_remaining
        )

        reasons: List[str] = []
        if suspicious_pages:
            reasons.append("suspicious_pages_present")
        if min_page_score < PAGE_QUALITY_LOW_CONFIDENCE_SCORE:
            reasons.append("low_page_score")
        if date_ratio < 0.9:
            reasons.append("low_date_coverage")
        if money_ratio < 0.75:
            reasons.append("low_money_coverage")
        if duplicate_rate > 0.0:
            reasons.append("duplicate_rows_remaining")
        if long_cell_remaining:
            reasons.append("long_cells_remaining")

        return {
            "low_confidence": bool(reasons),
            "reasons": reasons,
            "suspicious_pages": suspicious_pages,
            "retried_pages": retried_pages,
            "page_quality_scores": quality_scores,
            "min_page_score": min_page_score,
            "date_row_ratio": round(date_ratio, 4),
            "money_row_ratio": round(money_ratio, 4),
            "duplicate_row_rate": round(duplicate_rate, 4),
            "long_cells_remaining": bool(long_cell_remaining),
            "row_count": int(len(df)),
        }

    @staticmethod
    def _serialize_page_evaluations(
        page_evaluations: Sequence[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        serialized: List[Dict[str, object]] = []
        for item in page_evaluations:
            serialized.append(
                {
                    key: value
                    for key, value in item.items()
                    if key != "dataframe"
                }
            )
        return serialized

    def _parse_pdf_without_layout(
        self,
        pdf_path: str,
        page_images: List[Image.Image],
        effective_dpi: int,
        started_at: float,
        step_timings: Dict[str, float],
        *,
        layout_attempted: bool = False,
    ) -> pd.DataFrame:
        reason = "found no tables" if layout_attempted else "is disabled"
        print(f"  [2/4] Layout detection {reason}; OCR-ing full pages...")
        step_started_at = time.time()
        ocr_images = [
            (page_idx, self._normalize_ocr_image(page_image))
            for page_idx, page_image in enumerate(page_images)
        ]
        all_ocr_results, ocr_metrics = self._ocr_tables_parallel(ocr_images)
        step_timings["ocr_pages"] = round(time.time() - step_started_at, 3)

        print("  [3/4] Parsing HTML to DataFrames...")
        step_started_at = time.time()
        page_evaluations = self._extract_transaction_page_evaluations(all_ocr_results, pdf_path)
        retry_metrics = self._empty_ocr_metrics()
        if ENABLE_PAGE_OCR_RETRY:
            page_evaluations, retry_metrics = self._retry_suspicious_pages_without_layout(
                pdf_path=pdf_path,
                page_evaluations=page_evaluations,
            )
        all_dfs, expected_headers, pages_with_tables = self._materialize_selected_tables(page_evaluations)
        if not all_dfs:
            print("  × No table data extracted")
            step_timings["parse_html_tables"] = round(time.time() - step_started_at, 3)
            self.last_run_stats = {
                "timings": step_timings,
                "pages_with_tables": [page + 1 for page in pages_with_tables],
                "page_count": len(page_images),
                "effective_dpi": effective_dpi,
                "layout_mode": self.layout_mode,
                "layout_enabled": layout_attempted,
                "layout_fallback_full_page": layout_attempted,
                "ocr_images": len(ocr_images),
                "ocr_metrics": self._merge_ocr_metric_summaries(ocr_metrics, retry_metrics),
                "page_ocr": self._serialize_page_evaluations(page_evaluations),
            }
            return pd.DataFrame()

        combined = self._finalize_extracted_tables(all_dfs, expected_headers)
        quality_summary = self._build_document_quality_summary(combined, page_evaluations)
        step_timings["parse_html_tables"] = round(time.time() - step_started_at, 3)
        elapsed = time.time() - started_at
        step_timings["total"] = round(elapsed, 3)
        self.last_run_stats = {
            "timings": step_timings,
            "pages_with_tables": [page + 1 for page in pages_with_tables],
            "page_count": len(page_images),
            "effective_dpi": effective_dpi,
            "layout_mode": self.layout_mode,
            "layout_enabled": layout_attempted,
            "layout_fallback_full_page": layout_attempted,
            "ocr_images": len(ocr_images),
            "ocr_metrics": self._merge_ocr_metric_summaries(ocr_metrics, retry_metrics),
            "page_ocr": self._serialize_page_evaluations(page_evaluations),
            "quality_summary": quality_summary,
            "transaction_reconstruction": dict(self._last_transaction_reconstruction),
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
            print("        → No layout tables found; falling back to full-page OCR")
            return self._parse_pdf_without_layout(
                pdf_path,
                page_images,
                effective_dpi,
                t0,
                step_timings,
                layout_attempted=True,
            )

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
        debug_dir = (
            os.path.join("output", "debug", os.path.splitext(os.path.basename(pdf_path))[0])
            if SAVE_DEBUG_IMAGES
            else None
        )
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        for page_idx in pages_with_tables:
            crop = crop_image_region(page_images[page_idx], table_bboxes[page_idx])
            table_crops.append((page_idx, crop))
            if debug_dir:
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

        if debug_dir:
            first_crop.save(os.path.join(debug_dir, f"page_{first_page_idx + 1}_crop.png"))
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
            if debug_dir:
                normalized.save(os.path.join(debug_dir, f"page_{page_idx + 1}_stitched.png"))
        step_timings["stitch_headers"] = round(time.time() - step_started_at, 3)

        ocr_images: List[Tuple[int, Image.Image]] = [
            (first_page_idx, self._normalize_ocr_image(first_crop))
        ] + stitched_images
        if debug_dir:
            for page_idx, image in ocr_images:
                image.save(os.path.join(debug_dir, f"page_{page_idx + 1}_ocr_input.png"))

        print(f"  [7/8] OCR-ing {len(ocr_images)} table images via shared pipeline...")
        t_ocr = time.time()
        all_ocr_results, ocr_metrics = self._ocr_tables_parallel(ocr_images)
        ocr_elapsed = time.time() - t_ocr
        step_timings["ocr_tables"] = round(ocr_elapsed, 3)
        print(f"        → Done in {ocr_elapsed:.1f}s")
        all_ocr_results.sort(key=lambda x: x[0])

        print("  [8/8] Parsing HTML to DataFrames...")
        step_started_at = time.time()
        page_evaluations = self._extract_transaction_page_evaluations(all_ocr_results, pdf_path)
        recovery_metrics = self._empty_ocr_metrics()
        if RECOVER_LAYOUT_MISSED_PAGES and not PARSE_TESTING:
            recovery_started_at = time.time()
            page_evaluations, recovery_metrics = self._recover_missing_layout_pages(
                page_images=page_images,
                table_bboxes=table_bboxes,
                page_evaluations=page_evaluations,
            )
            step_timings["ocr_layout_missed_pages"] = round(
                time.time() - recovery_started_at,
                3,
            )
        retry_metrics = self._empty_ocr_metrics()
        if ENABLE_PAGE_OCR_RETRY:
            page_evaluations, retry_metrics = self._retry_suspicious_pages_with_layout(
                pdf_path=pdf_path,
                page_evaluations=page_evaluations,
                table_bboxes=table_bboxes,
                header_source_page_idx=first_page_idx,
                skip_header_stitching=skip_header_stitching,
            )
        all_dfs, expected_headers, parsed_pages = self._materialize_selected_tables(page_evaluations)

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
                "ocr_metrics": self._merge_ocr_metric_summaries(
                    ocr_metrics,
                    recovery_metrics,
                    retry_metrics,
                ),
                "page_ocr": self._serialize_page_evaluations(page_evaluations),
            }
            return pd.DataFrame()

        combined = self._finalize_extracted_tables(all_dfs, expected_headers)
        quality_summary = self._build_document_quality_summary(combined, page_evaluations)
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
            "ocr_images": len(ocr_images) + int(recovery_metrics.get("task_count", 0)),
            "header_source_page": first_page_idx + 1,
            "skip_header_stitching": skip_header_stitching,
            "ocr_metrics": self._merge_ocr_metric_summaries(
                ocr_metrics,
                recovery_metrics,
                retry_metrics,
            ),
            "page_ocr": self._serialize_page_evaluations(page_evaluations),
            "quality_summary": quality_summary,
            "transaction_reconstruction": dict(self._last_transaction_reconstruction),
            "debug_image_dir": debug_dir,
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
                parallel_results, _ = self._ocr_tables_parallel(candidate_images)
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

    def _submit_ocr_task(
        self,
        image: Image.Image,
        *,
        page_index: int = 0,
        task_type: str,
    ) -> Future[OCRTaskResult]:
        return self.ocr_dispatcher.submit(image, page_index=page_index, task_type=task_type)

    def _submit_ocr_document(
        self,
        stitched_images: Sequence[Tuple[int, Image.Image]],
        *,
        task_type: str,
    ) -> Future[OCRTaskResult]:
        page_indices = [page_idx for page_idx, _ in stitched_images]
        images = [image for _, image in stitched_images]
        return self.ocr_dispatcher.submit_document(
            images,
            page_indices=page_indices,
            task_type=task_type,
        )

    @staticmethod
    def _resolve_document_chunk_size() -> Optional[int]:
        configured = OCR_DOCUMENT_MAX_IMAGES_PER_REQUEST
        if configured is None:
            return None
        return max(1, int(configured))

    def _ocr_tables_parallel(
        self,
        stitched_images: List[Tuple[int, Image.Image]],
    ) -> Tuple[List[Tuple[int, Optional[str]]], Dict[str, float]]:
        if not stitched_images:
            return [], self._empty_ocr_metrics()

        results: List[Tuple[int, Optional[str]]] = []
        ocr_task_results: List[OCRTaskResult] = []
        submitted_tasks: dict[Future[OCRTaskResult], int] = {}

        print(
            f"        Page-level OCR: {len(stitched_images)} page(s) with "
            f"{self._resolve_ocr_batch_workers(len(stitched_images))} worker(s)"
        )
        for page_idx, image in stitched_images:
            future = self._submit_ocr_task(image, page_index=page_idx, task_type="table")
            submitted_tasks[future] = page_idx

        for future in as_completed(submitted_tasks):
            page_idx = submitted_tasks[future]
            try:
                task_result = future.result()
                ocr_task_results.append(task_result)
                results.append((page_idx, task_result.content))
            except Exception as exc:
                print(f"        ✗ Page {page_idx + 1}: OCR failed ({exc})")
                results.append((page_idx, None))

        results.sort(key=lambda x: x[0])
        return results, self._summarize_ocr_metrics(ocr_task_results)

    @staticmethod
    def _merge_ocr_metric_summaries(*metric_sets: Dict[str, float]) -> Dict[str, float]:
        valid_sets = [item for item in metric_sets if item]
        if not valid_sets:
            return {}

        summary: Dict[str, float] = {}
        additive_keys = {"task_count", "success_count", "failure_count"}
        max_keys = {
            "queue_wait_max",
            "build_request_max",
            "request_max",
            "total_max",
            "document_page_count_max",
            "backend_batch_size_max",
            "max_queue_size_at_submit",
        }
        weighted_mean_keys = {
            "queue_wait_mean",
            "build_request_mean",
            "request_mean",
            "total_mean",
            "document_page_count_mean",
            "backend_batch_size_mean",
        }

        total_tasks = sum(item.get("task_count", 0.0) for item in valid_sets)
        for key in additive_keys:
            summary[key] = round(sum(item.get(key, 0.0) for item in valid_sets), 6)
        for key in max_keys:
            summary[key] = round(max(item.get(key, 0.0) for item in valid_sets), 6)
        for key in weighted_mean_keys:
            if total_tasks <= 0:
                summary[key] = 0.0
            else:
                weighted_total = sum(item.get(key, 0.0) * item.get("task_count", 0.0) for item in valid_sets)
                summary[key] = round(weighted_total / total_tasks, 6)
        return summary

    @staticmethod
    def _render_specific_pages(pdf_path: str, page_indices: Sequence[int], dpi: int) -> Dict[int, Image.Image]:
        rendered: Dict[int, Image.Image] = {}
        if not page_indices:
            return rendered

        scale = float(dpi) / 72.0
        with fitz.open(pdf_path) as document:
            for page_index in sorted(set(int(page) for page in page_indices)):
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                mode = "RGB" if pixmap.n < 4 else "RGBA"
                image = Image.frombytes(mode, (pixmap.width, pixmap.height), pixmap.samples)
                rendered[page_index] = image.convert("RGB")
        return rendered

    def _normalize_retry_ocr_image(self, image: Image.Image) -> Image.Image:
        enhanced = ImageOps.autocontrast(image.convert("L"))
        enhanced = enhanced.filter(ImageFilter.SHARPEN)
        return self._normalize_ocr_image(enhanced.convert("RGB"))

    def _retry_suspicious_pages_with_layout(
        self,
        *,
        pdf_path: str,
        page_evaluations: Sequence[Dict[str, object]],
        table_bboxes: Sequence[Optional[List[int]]],
        header_source_page_idx: int,
        skip_header_stitching: bool,
    ) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
        retry_pages = [
            int(item["page_index"])
            for item in page_evaluations
            if (PAGE_OCR_RETRY_ALL or self._should_retry_page_evaluation(item))
            and item.get("page_index") in range(len(table_bboxes))
        ]
        if not retry_pages:
            return list(page_evaluations), self._empty_ocr_metrics()

        render_pages = set(retry_pages)
        if not skip_header_stitching:
            render_pages.add(int(header_source_page_idx))

        rendered_pages = self._render_specific_pages(pdf_path, sorted(render_pages), PAGE_OCR_RETRY_DPI)
        retry_header_img: Optional[Image.Image] = None
        if not skip_header_stitching and header_source_page_idx in rendered_pages:
            header_bbox = table_bboxes[header_source_page_idx]
            if header_bbox is not None:
                header_crop = crop_image_region(rendered_pages[header_source_page_idx], header_bbox)
                retry_header_height = self._calculate_header_height(header_crop)
                retry_header_img = header_crop.crop((0, 0, header_crop.width, retry_header_height))

        retry_images: List[Tuple[int, Image.Image]] = []
        for page_idx in retry_pages:
            bbox = table_bboxes[page_idx]
            page_image = rendered_pages.get(page_idx)
            if bbox is None or page_image is None:
                continue
            crop = crop_image_region(page_image, bbox)
            if page_idx != header_source_page_idx and not skip_header_stitching and retry_header_img is not None:
                crop = self._stitch_header(retry_header_img, crop)
            retry_images.append((page_idx, self._normalize_retry_ocr_image(crop)))

        if not retry_images:
            return list(page_evaluations), self._empty_ocr_metrics()

        retry_results, retry_metrics = self._ocr_tables_parallel(retry_images)
        retry_evaluations = self._extract_transaction_page_evaluations(retry_results, pdf_path)
        retry_map = {
            int(item["page_index"]): {**item, "pass_label": "retry_high_dpi", "retried": True}
            for item in retry_evaluations
        }

        merged_evaluations: List[Dict[str, object]] = []
        for primary in page_evaluations:
            page_idx = int(primary["page_index"])
            retry = retry_map.get(page_idx)
            if retry is None:
                merged_evaluations.append(primary)
                continue
            chosen = dict(self._select_best_page_ocr_evaluation([primary, retry]))
            chosen["retried"] = True
            merged_evaluations.append(chosen)
        return merged_evaluations, retry_metrics

    def _retry_suspicious_pages_without_layout(
        self,
        *,
        pdf_path: str,
        page_evaluations: Sequence[Dict[str, object]],
    ) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
        retry_pages = [
            int(item["page_index"])
            for item in page_evaluations
            if PAGE_OCR_RETRY_ALL or self._should_retry_page_evaluation(item)
        ]
        if not retry_pages:
            return list(page_evaluations), self._empty_ocr_metrics()

        rendered_pages = self._render_specific_pages(pdf_path, retry_pages, PAGE_OCR_RETRY_DPI)
        retry_images = [
            (page_idx, self._normalize_retry_ocr_image(image))
            for page_idx, image in rendered_pages.items()
        ]
        if not retry_images:
            return list(page_evaluations), self._empty_ocr_metrics()

        retry_results, retry_metrics = self._ocr_tables_parallel(retry_images)
        retry_evaluations = self._extract_transaction_page_evaluations(retry_results, pdf_path)
        retry_map = {
            int(item["page_index"]): {**item, "pass_label": "retry_high_dpi", "retried": True}
            for item in retry_evaluations
        }
        merged_evaluations: List[Dict[str, object]] = []
        for primary in page_evaluations:
            retry = retry_map.get(int(primary["page_index"]))
            if retry is None:
                merged_evaluations.append(primary)
                continue
            chosen = dict(self._select_best_page_ocr_evaluation([primary, retry]))
            chosen["retried"] = True
            merged_evaluations.append(chosen)
        return merged_evaluations, retry_metrics

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
            "document_page_count_mean": 0.0,
            "document_page_count_max": 0.0,
            "backend_batch_size_mean": 0.0,
            "backend_batch_size_max": 0.0,
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
        document_page_counts = [len(item.contents) for item in task_results]
        backend_batch_sizes = [item.batch_size for item in task_results]

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
            "document_page_count_mean": round(sum(document_page_counts) / len(document_page_counts), 6),
            "document_page_count_max": float(max(document_page_counts)),
            "backend_batch_size_mean": round(sum(backend_batch_sizes) / len(backend_batch_sizes), 6),
            "backend_batch_size_max": float(max(backend_batch_sizes)),
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

    def _dedupe_adjacent_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        rows_as_strings = df.fillna("").astype(str)
        kept_rows: List[dict[str, str]] = []
        removed = 0

        for _, row in rows_as_strings.iterrows():
            current = {column: str(row[column]).strip() for column in df.columns}
            duplicate_found = False
            for previous in kept_rows[-PAGE_DUPLICATE_WINDOW:]:
                if _rows_match_after_normalization(previous, current, df.columns):
                    duplicate_found = True
                    break
                if _rows_match_on_non_empty_overlap(previous, current, df.columns):
                    duplicate_found = True
                    break
            if duplicate_found:
                removed += 1
                continue
            kept_rows.append(current)

        if removed > 0:
            print(f"  Removed {removed} adjacent duplicate row(s)")

        return pd.DataFrame(kept_rows, columns=df.columns) if kept_rows else df.iloc[0:0].copy()

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

        expected_headers = [header for header in headers if str(header).strip()]

        def is_header_row(row):
            non_empty = [str(v).strip() for v in row if str(v).strip()]
            if not non_empty:
                return False
            matches = sum(
                1
                for value in non_empty
                if _header_cell_matches_expected(value, expected_headers)
            )
            return matches >= max(2, len(non_empty) - 1)

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
