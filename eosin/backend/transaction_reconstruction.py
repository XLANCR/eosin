from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import groupby
from math import ceil
from typing import Any, Sequence

import pandas as pd


SOURCE_PAGE_COLUMN = "__source_page"
SOURCE_ROW_COLUMN = "__source_row"
SOURCE_TABLE_COLUMN = "__source_table"
SOURCE_COLUMNS = (SOURCE_PAGE_COLUMN, SOURCE_ROW_COLUMN, SOURCE_TABLE_COLUMN)

_DATE_ATOM = (
    r"(?:"
    r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
    r"|\d{1,2}\s*[A-Za-z]{3,9}\s*[']?\d{2,4}"
    r"|\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4}"
    r")"
)
_DATE_CELL_RE = re.compile(
    rf"^\s*{_DATE_ATOM}(?:\s*\(\s*{_DATE_ATOM}\s*\))?"
    rf"(?:\s*\d{{1,2}}:\d{{2}}(?::\d{{2}})?\s*(?:[AP]M)?)?\s*$",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_GENERIC_HEADER_RE = re.compile(r"^(?:col(?:umn)?|unnamed)\D*\d+$", re.IGNORECASE)
_MONEY_CELL_RE = re.compile(
    r"^\s*(?:[₹$€£]\s*)?[+-]?(?:\d{1,3}(?:,\d{2,3})+|\d+)\.\d{1,2}"
    r"(?:\s*(?:cr|dr))?\s*$",
    re.IGNORECASE,
)
_SUMMARY_MARKERS = (
    ("opening_balance", "opening balance"),
    ("closing_balance", "closing balance"),
    ("transaction_total", "transaction total"),
    ("account_summary", "account summary"),
    ("account_details", "account details"),
    ("statement_summary", "statement summary"),
    ("legend", "legends used in the statement"),
)


@dataclass(frozen=True)
class ReconstructionResult:
    dataframe: pd.DataFrame
    diagnostics: dict[str, object]


def _normalized_header(value: object) -> str:
    return _NON_ALNUM_RE.sub("", str(value or "").lower())


def _is_date_value(value: object) -> bool:
    text = str(value or "").strip()
    return bool(text and _DATE_CELL_RE.fullmatch(text))


def _is_date_header(column: object) -> bool:
    return "date" in _normalized_header(column)


def _header_role(column: object) -> str:
    normalized = _normalized_header(column)
    if "date" in normalized:
        return "date"
    if any(token in normalized for token in ("description", "narration", "particular", "detail", "remark")):
        return "text"
    if any(token in normalized for token in ("reference", "refno", "chq", "cheque", "instrument")):
        return "reference"
    if any(token in normalized for token in ("debit", "withdraw")):
        return "debit"
    if any(token in normalized for token in ("credit", "deposit")):
        return "credit"
    if "balance" in normalized:
        return "balance"
    if "amount" in normalized:
        return "amount"
    if normalized in {"serialno", "slno", "srno", "no"} or str(column).strip() == "#":
        return "serial"
    return "other"


def _semantic_header_key(column: object) -> str:
    normalized = re.sub(r"\d+$", "", _normalized_header(column))
    if normalized in {"valuedt", "valuedate"}:
        return "value_date"
    if "date" in normalized:
        return "value_date" if "value" in normalized else "transaction_date"
    role = _header_role(column)
    if role in {"text", "reference", "debit", "credit", "balance", "amount", "serial"}:
        return role
    if normalized in {"debt", "debtdr", "debit", "debitdr", "withdrawal", "withdrawals"}:
        return "debit"
    if normalized in {"credit", "creditcr", "cr", "deposit", "deposits"}:
        return "credit"
    if normalized in {
        "chqrefno",
        "chqrefnumber",
        "chequerefno",
        "refno",
        "referenceno",
        "referencenumber",
    }:
        return "reference"
    return normalized


def _date_header_priority(column: object) -> tuple[int, int]:
    normalized = _normalized_header(column)
    is_value_date = "value" in normalized
    is_transaction_date = any(token in normalized for token in ("transaction", "txn", "tran", "post"))
    return (0 if is_transaction_date else 1 if not is_value_date else 2, len(normalized))


def _canonical_date_column(frames: Sequence[pd.DataFrame]) -> str:
    public_columns = [
        column
        for frame in frames
        for column in frame.columns
        if column not in SOURCE_COLUMNS
    ]
    named = list(dict.fromkeys(column for column in public_columns if _is_date_header(column)))
    if named:
        return str(min(named, key=_date_header_priority))
    return "Date"


def _resolve_date(row: pd.Series, public_columns: Sequence[object]) -> tuple[str, object | None]:
    named_date_columns = [column for column in public_columns if _is_date_header(column)]
    for column in named_date_columns:
        value = row.get(column, "")
        if _is_date_value(value):
            return str(value).strip(), column
    for column in public_columns:
        value = row.get(column, "")
        if _is_date_value(value):
            return str(value).strip(), column
    return "", None


def _clean_value(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _is_generic_header(column: object) -> bool:
    return bool(_GENERIC_HEADER_RE.fullmatch(str(column).strip()))


def _looks_like_money(value: object) -> bool:
    return bool(_MONEY_CELL_RE.fullmatch(_clean_value(value)))


def _generic_role_map(frame: pd.DataFrame) -> dict[object, str]:
    columns = [column for column in frame.columns if column not in SOURCE_COLUMNS]
    if len(columns) < 3 or not all(_is_generic_header(column) for column in columns):
        return {}

    date_counts = {column: int(frame[column].map(_is_date_value).sum()) for column in columns}
    minimum_date_count = max(2, ceil(len(frame) * 0.35))
    date_columns = [column for column in columns if date_counts[column] >= minimum_date_count]
    dated_rows = frame[date_columns].map(_is_date_value).any(axis=1) if date_columns else pd.Series(dtype=bool)
    if not date_columns or int(dated_rows.sum()) < 2:
        return {}

    candidate_rows = frame.loc[dated_rows]
    text_scores: dict[object, float] = {}
    money_columns: list[object] = []
    for column in columns:
        if column in date_columns:
            continue
        values = candidate_rows[column].map(_clean_value)
        present = values[values.ne("")]
        if present.empty:
            continue
        money_ratio = float(present.map(_looks_like_money).mean())
        alpha_ratio = float(present.map(lambda value: bool(re.search(r"[A-Za-z]{3}", value))).mean())
        if money_ratio >= 0.5:
            money_columns.append(column)
        if alpha_ratio >= 0.5:
            text_scores[column] = alpha_ratio * float(present.map(len).mean())

    if not text_scores or not money_columns:
        return {}
    description_column = max(text_scores, key=text_scores.get)
    mapping = {date_columns[0]: "Transaction Date", description_column: "Description"}
    if len(date_columns) > 1:
        mapping[date_columns[1]] = "Value Date"
    remaining = [column for column in columns if column not in mapping and column not in money_columns]
    if remaining:
        mapping[remaining[0]] = "Reference"
    if len(money_columns) >= 3:
        for column, role in zip(money_columns[-3:], ("Debit", "Credit", "Balance")):
            mapping[column] = role
    elif len(money_columns) == 2:
        mapping[money_columns[0]] = "Amount"
        mapping[money_columns[1]] = "Balance"
    else:
        mapping[money_columns[0]] = "Amount"
    return mapping


def _infer_generic_schemas(frames: Sequence[pd.DataFrame]) -> tuple[list[pd.DataFrame], int]:
    inferred: list[pd.DataFrame] = []
    count = 0
    for frame in frames:
        role_map = _generic_role_map(frame)
        inferred.append(frame.rename(columns=role_map) if role_map else frame.copy())
        count += bool(role_map)
    return inferred, count


def _shifted_financial_mapping(
    frame: pd.DataFrame,
) -> tuple[object, object, object] | None:
    public_columns = [column for column in frame.columns if column not in SOURCE_COLUMNS]
    balance_columns = [column for column in public_columns if _header_role(column) == "balance"]
    financial_columns = [
        column
        for column in public_columns
        if _header_role(column) in {"debit", "credit", "amount"}
    ]
    if len(balance_columns) != 1 or len(financial_columns) < 2:
        return None

    balance_column = balance_columns[0]
    if any(_money_decimal(value) is not None for value in frame[balance_column]):
        return None

    best_mapping: tuple[object, object, object] | None = None
    best_support = 0
    best_mapping_count = 0
    for amount_column in financial_columns:
        for shifted_balance_column in financial_columns:
            if amount_column == shifted_balance_column:
                continue
            support = 0
            previous_balance: Decimal | None = None
            for _, row in frame.iterrows():
                amount = _money_decimal(row.get(amount_column, ""))
                current_balance = _money_decimal(row.get(shifted_balance_column, ""))
                if amount is not None and current_balance is not None and previous_balance is not None:
                    support += int(abs(current_balance - previous_balance) == abs(amount))
                if current_balance is not None:
                    previous_balance = current_balance
            if support > best_support:
                best_mapping = (amount_column, shifted_balance_column, balance_column)
                best_support = support
                best_mapping_count = 1
            elif support == best_support and support > 0:
                best_mapping_count += 1
    return best_mapping if best_support >= 2 and best_mapping_count == 1 else None


def _repair_shifted_financial_frames(
    frames: Sequence[pd.DataFrame],
) -> tuple[list[pd.DataFrame], int, int]:
    repaired_frames: list[pd.DataFrame] = []
    repaired_frame_count = 0
    repaired_row_count = 0
    for frame in frames:
        mapping = _shifted_financial_mapping(frame)
        if mapping is None:
            repaired_frames.append(frame.copy())
            continue
        amount_column, shifted_balance_column, balance_column = mapping
        repaired = frame.copy()
        for row_index, row in repaired.iterrows():
            amount = _money_decimal(row.get(amount_column, ""))
            shifted_balance = _money_decimal(row.get(shifted_balance_column, ""))
            if amount is None or shifted_balance is None or _money_decimal(row.get(balance_column, "")) is not None:
                continue
            repaired.at[row_index, balance_column] = _clean_value(row.get(shifted_balance_column, ""))
            repaired.at[row_index, shifted_balance_column] = ""
            repaired_row_count += 1
        repaired_frames.append(repaired)
        repaired_frame_count += 1
    return repaired_frames, repaired_frame_count, repaired_row_count


def _row_values(row: pd.Series, public_columns: Sequence[object]) -> dict[object, str]:
    return {column: _clean_value(row.get(column, "")) for column in public_columns}


def _row_text(values: dict[object, str]) -> str:
    return " ".join(value.lower() for value in values.values() if value and value != "-")


def _is_repeated_header(values: dict[object, str]) -> bool:
    matches = 0
    meaningful = 0
    for column, value in values.items():
        if not value:
            continue
        meaningful += 1
        normalized_value = _normalized_header(value)
        normalized_column = _normalized_header(column)
        if normalized_value and (
            normalized_value == normalized_column
            or normalized_column.startswith(normalized_value)
            or normalized_value.startswith(normalized_column)
        ):
            matches += 1
    return meaningful >= 2 and matches >= max(2, meaningful - 1)


def _non_transaction_reason(values: dict[object, str]) -> str | None:
    if _is_repeated_header(values):
        return "repeated_header"
    text = _row_text(values)
    for reason, marker in _SUMMARY_MARKERS:
        if marker in text:
            return reason
    return None


def _role_values(values: dict[object, str], role: str) -> list[tuple[object, str]]:
    return [
        (column, value)
        for column, value in values.items()
        if _header_role(column) == role and value not in {"", "-"}
    ]


def _has_meaningful_text(values: dict[object, str]) -> bool:
    candidates = _role_values(values, "text") + _role_values(values, "reference")
    return any(sum(character.isalpha() for character in value) >= 3 for _, value in candidates)


def _has_money(values: dict[object, str]) -> bool:
    return any(
        _role_values(values, role)
        for role in ("debit", "credit", "amount", "balance")
    )


def _has_transaction_evidence(values: dict[object, str]) -> bool:
    return _has_meaningful_text(values) or _has_money(values)


def _has_independent_transaction_structure(values: dict[object, str]) -> bool:
    has_amount = any(_role_values(values, role) for role in ("debit", "credit", "amount"))
    has_balance = bool(_role_values(values, "balance"))
    return _has_meaningful_text(values) and (has_amount or has_balance)


def _has_continuation_content(values: dict[object, str]) -> bool:
    return _has_meaningful_text(values) or _has_money(values)


def _has_text_overlap(
    left: dict[object, str],
    right: dict[object, str],
) -> bool:
    left_values = [value for role in ("text", "reference") for _, value in _role_values(left, role)]
    right_values = [value for role in ("text", "reference") for _, value in _role_values(right, role)]
    for left_value in left_values:
        normalized_left = _NON_ALNUM_RE.sub("", left_value.lower())
        for right_value in right_values:
            normalized_right = _NON_ALNUM_RE.sub("", right_value.lower())
            if min(len(normalized_left), len(normalized_right)) < 6:
                continue
            if normalized_left in normalized_right or normalized_right in normalized_left:
                return True
    return False


def _money_decimal(value: object) -> Decimal | None:
    text = _clean_value(value).lower()
    if not text or text == "-":
        return None
    negative = text.startswith("-") or text.endswith("dr") or (text.startswith("(") and text.endswith(")"))
    normalized = re.sub(r"[^0-9.]", "", text)
    if not normalized or normalized.count(".") > 1:
        return None
    try:
        amount = Decimal(normalized)
    except InvalidOperation:
        return None
    return -amount if negative else amount


def _first_role_decimal(values: dict[object, str], role: str) -> Decimal | None:
    for _, value in _role_values(values, role):
        parsed = _money_decimal(value)
        if parsed is not None:
            return parsed
    return None


def _resolve_single_amount_direction(
    values: dict[object, str],
    previous_balance: Decimal | None,
) -> tuple[dict[object, str], bool]:
    if previous_balance is None or _role_values(values, "debit") or _role_values(values, "credit"):
        return values, False
    amount_values = _role_values(values, "amount")
    current_balance = _first_role_decimal(values, "balance")
    if len(amount_values) != 1 or current_balance is None:
        return values, False
    amount_column, amount_text = amount_values[0]
    amount = _money_decimal(amount_text)
    if amount is None:
        return values, False

    delta = current_balance - previous_balance
    target_role = "credit" if delta == amount else "debit" if delta == -amount else None
    target_columns = [column for column in values if _header_role(column) == target_role]
    if not target_columns:
        return values, False
    resolved = dict(values)
    resolved[target_columns[0]] = amount_text
    resolved[amount_column] = ""
    return resolved, True


def _correct_explicit_amount_direction(
    values: dict[object, str],
    previous_balance: Decimal | None,
) -> tuple[dict[object, str], bool]:
    if previous_balance is None:
        return values, False
    current_balance = _first_role_decimal(values, "balance")
    money_cells = [
        (column, role, text, _money_decimal(text))
        for role in ("debit", "credit")
        for column, text in _role_values(values, role)
    ]
    money_cells = [cell for cell in money_cells if cell[3] not in {None, Decimal("0")}]
    if current_balance is None or len(money_cells) != 1:
        return values, False

    source_column, source_role, amount_text, amount = money_cells[0]
    if amount is None:
        return values, False
    delta = current_balance - previous_balance
    target_role = "credit" if delta == abs(amount) else "debit" if delta == -abs(amount) else None
    if target_role is None or target_role == source_role:
        return values, False
    target_columns = [column for column in values if _header_role(column) == target_role]
    if not target_columns:
        return values, False
    resolved = dict(values)
    resolved[source_column] = ""
    resolved[target_columns[0]] = amount_text
    return resolved, True


def _derive_missing_balance(
    values: dict[object, str],
    previous_balance: Decimal | None,
) -> tuple[dict[object, str], bool]:
    if previous_balance is None or _role_values(values, "balance"):
        return values, False
    balance_columns = [column for column in values if _header_role(column) == "balance"]
    debit_values = _role_values(values, "debit")
    credit_values = _role_values(values, "credit")
    if not balance_columns or bool(debit_values) == bool(credit_values):
        return values, False
    amount_values = debit_values or credit_values
    if len(amount_values) != 1:
        return values, False
    amount = _money_decimal(amount_values[0][1])
    if amount is None or amount < 0:
        return values, False
    current_balance = (
        previous_balance - amount
        if debit_values
        else previous_balance + amount
    )
    resolved = dict(values)
    resolved[balance_columns[0]] = f"{current_balance:.2f}"
    return resolved, True


def _repair_shifted_amount_balance(
    values: dict[object, str],
    previous_balance: Decimal | None,
) -> tuple[dict[object, str], bool]:
    if previous_balance is None or _role_values(values, "balance"):
        return values, False
    balance_columns = [column for column in values if _header_role(column) == "balance"]
    money_cells = [
        (column, role, text, _money_decimal(text))
        for role in ("debit", "credit")
        for column, text in _role_values(values, role)
    ]
    money_cells = [cell for cell in money_cells if cell[3] is not None]
    if not balance_columns or len(money_cells) != 2:
        return values, False

    candidate_repair: dict[object, str] | None = None
    for balance_index, balance_cell in enumerate(money_cells):
        amount_cell = money_cells[1 - balance_index]
        candidate_balance = balance_cell[3]
        amount = amount_cell[3]
        if candidate_balance is None or amount is None or amount < 0:
            continue
        delta = candidate_balance - previous_balance
        target_role = "credit" if delta == amount else "debit" if delta == -amount else None
        target_columns = [column for column in values if _header_role(column) == target_role]
        if not target_columns:
            continue
        resolved = dict(values)
        resolved[money_cells[0][0]] = ""
        resolved[money_cells[1][0]] = ""
        resolved[target_columns[0]] = amount_cell[2]
        resolved[balance_columns[0]] = balance_cell[2]
        if candidate_repair is not None and candidate_repair != resolved:
            return values, False
        candidate_repair = resolved
    return (candidate_repair, True) if candidate_repair is not None else (values, False)


def _merge_continuation(target: dict[object, str], fragment: dict[object, str]) -> bool:
    updates: dict[object, str] = {}
    for column, value in fragment.items():
        if not value or value == "-" or _header_role(column) in {"date", "serial"}:
            continue
        existing = target.get(column, "").strip()
        role = _header_role(column)
        if not existing or existing == "-":
            updates[column] = value
            continue
        if existing == value:
            continue
        if role in {"text", "reference"}:
            if value not in existing:
                updates[column] = f"{existing} {value}".strip()
            continue
        return False
    target.update(updates)
    return True


def _ordered_combined_frame(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    sort_columns = [column for column in (SOURCE_PAGE_COLUMN, SOURCE_TABLE_COLUMN, SOURCE_ROW_COLUMN) if column in combined]
    if sort_columns:
        combined = combined.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return combined


def _coalesce_semantic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    merged = frame.copy()
    canonical_by_key: dict[str, object] = {}
    drop_columns: list[object] = []
    for column in [item for item in merged.columns if item not in SOURCE_COLUMNS]:
        key = _semantic_header_key(column)
        if not key:
            continue
        canonical = canonical_by_key.get(key)
        if canonical is None:
            canonical_by_key[key] = column
            continue
        canonical_values = merged[canonical].map(_clean_value)
        duplicate_values = merged[column].map(_clean_value)
        canonical_blank = canonical_values.isin(["", "-"])
        duplicate_present = ~duplicate_values.isin(["", "-"])
        conflicts = (~canonical_blank) & duplicate_present & (canonical_values != duplicate_values)
        merged.loc[canonical_blank & duplicate_present, canonical] = duplicate_values[
            canonical_blank & duplicate_present
        ]
        if not bool(conflicts.any()):
            drop_columns.append(column)
    return merged.drop(columns=drop_columns) if drop_columns else merged


def _replayed_source_pages(
    frame: pd.DataFrame,
    public_columns: Sequence[object],
) -> set[int]:
    if SOURCE_PAGE_COLUMN not in frame.columns:
        return set()
    seen_signatures: set[tuple[str, Decimal]] = set()
    latest_date: pd.Timestamp | None = None
    replayed: set[int] = set()
    for page_value, page_rows in frame.groupby(SOURCE_PAGE_COLUMN, sort=True):
        signatures: list[tuple[str, Decimal]] = []
        dates: list[pd.Timestamp] = []
        for _, row in page_rows.iterrows():
            resolved_date, _ = _resolve_date(row, public_columns)
            balance = _first_role_decimal(_row_values(row, public_columns), "balance")
            parsed_date = pd.to_datetime(resolved_date, dayfirst=True, errors="coerce")
            if not resolved_date or balance is None or pd.isna(parsed_date):
                continue
            timestamp = pd.Timestamp(parsed_date)
            signatures.append((timestamp.date().isoformat(), balance))
            dates.append(timestamp)
        overlap = sum(signature in seen_signatures for signature in signatures)
        overlap_threshold = max(2, ceil(len(signatures) * 0.67))
        is_replay = (
            len(page_rows) <= 5
            and len(signatures) >= 2
            and overlap >= overlap_threshold
            and latest_date is not None
            and max(dates) < latest_date
        )
        if is_replay:
            replayed.add(int(page_value))
            continue
        seen_signatures.update(signatures)
        if dates:
            page_latest = max(dates)
            latest_date = page_latest if latest_date is None else max(latest_date, page_latest)
    return replayed


def _replayed_source_rows(
    frame: pd.DataFrame,
    public_columns: Sequence[object],
    replayed_pages: set[int],
) -> set[tuple[int, int, int]]:
    if SOURCE_PAGE_COLUMN not in frame.columns:
        return set()
    seen_signatures: set[tuple[str, Decimal]] = set()
    latest_date: pd.Timestamp | None = None
    replayed: set[tuple[int, int, int]] = set()
    for page_value, page_rows in frame.groupby(SOURCE_PAGE_COLUMN, sort=True):
        page_index = int(page_value)
        if page_index in replayed_pages:
            continue
        records: list[tuple[pd.Timestamp, tuple[str, Decimal], tuple[int, int, int]]] = []
        for _, row in page_rows.iterrows():
            values = _row_values(row, public_columns)
            resolved_date, _ = _resolve_date(row, public_columns)
            parsed_date = pd.to_datetime(resolved_date, dayfirst=True, errors="coerce")
            amounts = [
                _money_decimal(value)
                for role in ("debit", "credit", "amount")
                for _, value in _role_values(values, role)
            ]
            amounts = [abs(amount) for amount in amounts if amount is not None]
            if not resolved_date or pd.isna(parsed_date) or len(amounts) != 1:
                continue
            timestamp = pd.Timestamp(parsed_date)
            source_id = (
                page_index,
                int(row.get(SOURCE_TABLE_COLUMN, 0)),
                int(row.get(SOURCE_ROW_COLUMN, 0)),
            )
            records.append((timestamp, (timestamp.date().isoformat(), amounts[0]), source_id))

        page_latest = max((record[0] for record in records), default=None)
        candidates = [
            record
            for record in records[:3]
            if latest_date is not None
            and record[0] < latest_date
            and record[1] in seen_signatures
        ]
        if (
            latest_date is not None
            and page_latest is not None
            and page_latest > latest_date
            and len(candidates) >= 2
        ):
            replayed.update(record[2] for record in candidates)

        seen_signatures.update(
            record[1]
            for record in records
            if record[2] not in replayed
        )
        if page_latest is not None:
            latest_date = page_latest if latest_date is None else max(latest_date, page_latest)
    return replayed


def _runaway_repeated_source_rows(
    frame: pd.DataFrame,
    public_columns: Sequence[object],
) -> set[tuple[int, int, int]]:
    repeated: set[tuple[int, int, int]] = set()

    def signature(item: tuple[object, pd.Series]) -> tuple[object, ...]:
        _, row = item
        values = _row_values(row, public_columns)
        return (
            int(row.get(SOURCE_PAGE_COLUMN, -1)),
            int(row.get(SOURCE_TABLE_COLUMN, 0)),
            *(" ".join(values[column].casefold().split()) for column in public_columns),
        )

    ordered_items = list(frame.iterrows())
    indexed_items = list(enumerate(ordered_items))
    for _, grouped_items in groupby(indexed_items, key=lambda item: signature(item[1])):
        positioned_items = list(grouped_items)
        items = [item for _, item in positioned_items]
        if len(items) < 3:
            continue
        first_row = items[0][1]
        values = _row_values(first_row, public_columns)
        resolved_date, _ = _resolve_date(first_row, public_columns)
        amount_values = [
            _money_decimal(value)
            for role in ("debit", "credit", "amount")
            for _, value in _role_values(values, role)
        ]
        amount_values = [amount for amount in amount_values if amount not in {None, Decimal("0")}]
        if not resolved_date or len(amount_values) != 1 or _first_role_decimal(values, "balance") is None:
            continue
        first_position = positioned_items[0][0]
        exclude_first = False
        if first_position > 0:
            previous_row = ordered_items[first_position - 1][1]
            same_source_table = (
                int(previous_row.get(SOURCE_PAGE_COLUMN, -1))
                == int(first_row.get(SOURCE_PAGE_COLUMN, -1))
                and int(previous_row.get(SOURCE_TABLE_COLUMN, 0))
                == int(first_row.get(SOURCE_TABLE_COLUMN, 0))
            )
            previous_balance = _first_role_decimal(
                _row_values(previous_row, public_columns),
                "balance",
            )
            current_balance = _first_role_decimal(values, "balance")
            exclude_first = same_source_table and previous_balance == current_balance
        repeated.update(
            (
                int(row.get(SOURCE_PAGE_COLUMN, -1)),
                int(row.get(SOURCE_TABLE_COLUMN, 0)),
                int(row.get(SOURCE_ROW_COLUMN, 0)),
            )
            for _, row in items[0 if exclude_first else 1 :]
        )
    return repeated


def reconstruct_transactions(frames: Sequence[pd.DataFrame]) -> ReconstructionResult:
    non_empty_frames = [frame.copy() for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not non_empty_frames:
        diagnostics = {
            "selected_source_rows": 0,
            "emitted_transaction_sources": 0,
            "absorbed_continuation_sources": 0,
            "excluded_non_transaction_sources": 0,
            "rejected_unclassified_sources": 0,
            "generic_schema_frames_inferred": 0,
            "balance_resolved_amount_sources": 0,
            "derived_balance_sources": 0,
            "shifted_schema_frames_repaired": 0,
            "shifted_amount_balance_repairs": 0,
            "replayed_source_pages": [],
            "replayed_source_rows": [],
            "runaway_repeated_source_rows": [],
            "conservation_ok": True,
        }
        return ReconstructionResult(dataframe=pd.DataFrame(), diagnostics=diagnostics)

    inferred_frames, inferred_schema_count = _infer_generic_schemas(non_empty_frames)
    repaired_frames, shifted_schema_frame_count, shifted_schema_row_count = (
        _repair_shifted_financial_frames(inferred_frames)
    )
    combined = _coalesce_semantic_columns(_ordered_combined_frame(repaired_frames))
    public_columns = [column for column in combined.columns if column not in SOURCE_COLUMNS]
    canonical_date_column = _canonical_date_column([combined])
    if canonical_date_column not in combined.columns:
        combined[canonical_date_column] = ""
        public_columns.append(canonical_date_column)
    replayed_pages = _replayed_source_pages(combined, public_columns)
    replayed_rows = _replayed_source_rows(combined, public_columns, replayed_pages)
    runaway_repeated_rows = _runaway_repeated_source_rows(combined, public_columns)

    emitted_rows: list[dict[object, str]] = []
    emitted_source_count = 0
    absorbed_continuations = 0
    exclusions: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    pending_fragments: list[dict[object, str]] = []
    last_row_was_attachable = False
    inherited_date = ""
    previous_balance: Decimal | None = None
    balance_resolved_amounts = 0
    derived_balances = 0
    shifted_amount_balance_repairs = shifted_schema_row_count

    ordered_rows = list(combined.iterrows())
    for position, (_, row) in enumerate(ordered_rows):
        values = _row_values(row, public_columns)
        source_page = int(row.get(SOURCE_PAGE_COLUMN, -1))
        source_id = (
            source_page,
            int(row.get(SOURCE_TABLE_COLUMN, 0)),
            int(row.get(SOURCE_ROW_COLUMN, 0)),
        )
        if source_page in replayed_pages:
            exclusions["replayed_source_page"] += 1
            last_row_was_attachable = False
            continue
        if source_id in replayed_rows:
            exclusions["replayed_source_row"] += 1
            last_row_was_attachable = False
            continue
        if source_id in runaway_repeated_rows:
            exclusions["runaway_repeated_row"] += 1
            last_row_was_attachable = False
            continue
        reason = _non_transaction_reason(values)
        if reason:
            exclusions[reason] += 1
            last_row_was_attachable = False
            continue

        values, amount_was_resolved = _resolve_single_amount_direction(values, previous_balance)
        balance_resolved_amounts += int(amount_was_resolved)
        values, explicit_direction_was_resolved = _correct_explicit_amount_direction(
            values,
            previous_balance,
        )
        balance_resolved_amounts += int(explicit_direction_was_resolved)
        values, shifted_values_were_repaired = _repair_shifted_amount_balance(
            values,
            previous_balance,
        )
        shifted_amount_balance_repairs += int(shifted_values_were_repaired)
        values, balance_was_derived = _derive_missing_balance(values, previous_balance)
        derived_balances += int(balance_was_derived)

        resolved_date, _ = _resolve_date(row, public_columns)
        if resolved_date:
            if not _has_transaction_evidence(values):
                rejections["dated_without_transaction_evidence"] += 1
                last_row_was_attachable = False
                continue
            current = dict(values)
            current[canonical_date_column] = resolved_date
            for fragment in pending_fragments:
                if _merge_continuation(current, fragment):
                    absorbed_continuations += 1
                else:
                    rejections["conflicting_leading_fragment"] += 1
            pending_fragments.clear()
            emitted_rows.append(current)
            emitted_source_count += 1
            current_balance = _first_role_decimal(current, "balance")
            if current_balance is not None:
                previous_balance = current_balance
            inherited_date = resolved_date
            last_row_was_attachable = True
            continue

        if _has_money(values) and position + 1 < len(ordered_rows):
            next_row = ordered_rows[position + 1][1]
            next_page = int(next_row.get(SOURCE_PAGE_COLUMN, -1))
            next_date, _ = _resolve_date(next_row, public_columns)
            next_values = _row_values(next_row, public_columns)
            if (
                next_page == source_page
                and next_page not in replayed_pages
                and next_date
                and _has_text_overlap(values, next_values)
            ):
                pending_fragments.append(values)
                last_row_was_attachable = False
                continue

        if _has_independent_transaction_structure(values) and inherited_date:
            current = dict(values)
            current[canonical_date_column] = inherited_date
            emitted_rows.append(current)
            emitted_source_count += 1
            current_balance = _first_role_decimal(current, "balance")
            if current_balance is not None:
                previous_balance = current_balance
            last_row_was_attachable = True
            continue

        if _has_continuation_content(values):
            if emitted_rows and last_row_was_attachable and _merge_continuation(emitted_rows[-1], values):
                absorbed_continuations += 1
                continue
            if not emitted_rows:
                pending_fragments.append(values)
                continue
            rejections["ambiguous_or_conflicting_continuation"] += 1
            last_row_was_attachable = False
            continue

        rejections["no_transaction_evidence"] += 1
        last_row_was_attachable = False

    if pending_fragments:
        rejections["unattached_leading_fragment"] += len(pending_fragments)

    dataframe = pd.DataFrame(emitted_rows, columns=public_columns)
    if not dataframe.empty:
        dataframe = dataframe.replace(r"^\s*$", pd.NA, regex=True).dropna(axis=1, how="all").fillna("")

    selected = len(combined)
    excluded_count = sum(exclusions.values())
    rejected_count = sum(rejections.values())
    diagnostics = {
        "selected_source_rows": selected,
        "emitted_transaction_sources": emitted_source_count,
        "absorbed_continuation_sources": absorbed_continuations,
        "excluded_non_transaction_sources": excluded_count,
        "rejected_unclassified_sources": rejected_count,
        "generic_schema_frames_inferred": inferred_schema_count,
        "balance_resolved_amount_sources": balance_resolved_amounts,
        "derived_balance_sources": derived_balances,
        "shifted_schema_frames_repaired": shifted_schema_frame_count,
        "shifted_amount_balance_repairs": shifted_amount_balance_repairs,
        "replayed_source_pages": sorted(replayed_pages),
        "replayed_source_rows": [
            {"page": page, "row": row, "table": table}
            for page, table, row in sorted(replayed_rows)
        ],
        "runaway_repeated_source_rows": [
            {"page": page, "row": row, "table": table}
            for page, table, row in sorted(runaway_repeated_rows)
        ],
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "rejection_reasons": dict(sorted(rejections.items())),
        "conservation_ok": selected
        == emitted_source_count + absorbed_continuations + excluded_count + rejected_count,
    }
    return ReconstructionResult(dataframe=dataframe, diagnostics=diagnostics)
