from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
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
    return any(len(_NON_ALNUM_RE.sub("", value.lower())) >= 3 for _, value in candidates)


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
            "conservation_ok": True,
        }
        return ReconstructionResult(dataframe=pd.DataFrame(), diagnostics=diagnostics)

    inferred_frames, inferred_schema_count = _infer_generic_schemas(non_empty_frames)
    combined = _coalesce_semantic_columns(_ordered_combined_frame(inferred_frames))
    public_columns = [column for column in combined.columns if column not in SOURCE_COLUMNS]
    canonical_date_column = _canonical_date_column(inferred_frames)
    if canonical_date_column not in combined.columns:
        combined[canonical_date_column] = ""
        public_columns.append(canonical_date_column)

    emitted_rows: list[dict[object, str]] = []
    emitted_source_count = 0
    absorbed_continuations = 0
    exclusions: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    pending_fragments: list[dict[object, str]] = []
    last_row_was_attachable = False
    inherited_date = ""

    for _, row in combined.iterrows():
        values = _row_values(row, public_columns)
        reason = _non_transaction_reason(values)
        if reason:
            exclusions[reason] += 1
            last_row_was_attachable = False
            continue

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
            inherited_date = resolved_date
            last_row_was_attachable = True
            continue

        if _has_independent_transaction_structure(values) and inherited_date:
            current = dict(values)
            current[canonical_date_column] = inherited_date
            emitted_rows.append(current)
            emitted_source_count += 1
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
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "rejection_reasons": dict(sorted(rejections.items())),
        "conservation_ok": selected
        == emitted_source_count + absorbed_continuations + excluded_count + rejected_count,
    }
    return ReconstructionResult(dataframe=dataframe, diagnostics=diagnostics)
