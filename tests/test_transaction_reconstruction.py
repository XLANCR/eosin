from __future__ import annotations

import pandas as pd

from eosin.backend.transaction_reconstruction import (
    SOURCE_PAGE_COLUMN,
    SOURCE_ROW_COLUMN,
    SOURCE_TABLE_COLUMN,
    reconstruct_transactions,
)


def source_frame(page_index: int, rows: list[dict[str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows).fillna("")
    frame[SOURCE_PAGE_COLUMN] = page_index
    frame[SOURCE_ROW_COLUMN] = range(len(frame))
    frame[SOURCE_TABLE_COLUMN] = 0
    return frame


def test_coalesces_date_columns_before_classifying_transactions() -> None:
    frames = [
        source_frame(
            0,
            [
                {
                    "Tran Date": "01/01/2024",
                    "Date (Value Date)": "",
                    "Description": "First payment",
                    "Debit": "10.00",
                    "Balance": "90.00",
                }
            ],
        ),
        source_frame(
            1,
            [
                {
                    "Tran Date": "",
                    "Date (Value Date)": "02/01/2024",
                    "Description": "Second payment",
                    "Debit": "20.00",
                    "Balance": "70.00",
                }
            ],
        ),
    ]

    result = reconstruct_transactions(frames)

    assert len(result.dataframe) == 2
    assert list(result.dataframe["Tran Date"]) == ["01/01/2024", "02/01/2024"]
    assert result.diagnostics["selected_source_rows"] == 2
    assert result.diagnostics["emitted_transaction_sources"] == 2


def test_accepts_transaction_date_with_time_suffix() -> None:
    frame = source_frame(
        0,
        [
            {
                "TRANSACTION DATE": "02 Jan 2024 11:18 AM",
                "TRANSACTION DETAILS": "UPI transfer",
                "AMOUNT": "500.00",
                "BALANCE": "507.75",
            }
        ],
    )

    result = reconstruct_transactions([frame])

    assert len(result.dataframe) == 1
    assert result.dataframe.iloc[0]["TRANSACTION DATE"] == "02 Jan 2024 11:18 AM"


def test_accepts_month_name_date_without_space_before_year() -> None:
    frame = source_frame(
        0,
        [
            {
                "Txn Date": "10 Dec2021",
                "Description": "UPI transfer",
                "Debit": "3.50",
                "Balance": "5,29,776.23",
            }
        ],
    )

    result = reconstruct_transactions([frame])

    assert len(result.dataframe) == 1


def test_uses_generic_date_column_when_named_date_column_is_empty() -> None:
    frame = source_frame(
        0,
        [
            {
                "col_1": "30/01/2022",
                "Date": "",
                "Description": "Cash withdrawal",
                "Amount": "100.00",
                "Balance": "900.00",
            }
        ],
    )

    result = reconstruct_transactions([frame])

    assert len(result.dataframe) == 1
    assert result.dataframe.iloc[0]["Date"] == "30/01/2022"
    assert SOURCE_PAGE_COLUMN not in result.dataframe.columns
    assert SOURCE_ROW_COLUMN not in result.dataframe.columns
    assert SOURCE_TABLE_COLUMN not in result.dataframe.columns


def test_merges_text_continuation_across_page_boundary() -> None:
    frames = [
        source_frame(
            0,
            [
                {
                    "Date": "01/01/2024",
                    "Description": "UPI PAYMENT",
                    "Reference": "",
                    "Debit": "10.00",
                    "Balance": "90.00",
                }
            ],
        ),
        source_frame(
            1,
            [
                {
                    "Date": "",
                    "Description": "TO MERCHANT",
                    "Reference": "ABC123",
                    "Debit": "",
                    "Balance": "",
                },
                {
                    "Date": "02/01/2024",
                    "Description": "SALARY",
                    "Reference": "",
                    "Debit": "",
                    "Balance": "1090.00",
                },
            ],
        ),
    ]

    result = reconstruct_transactions(frames)

    assert len(result.dataframe) == 2
    assert result.dataframe.iloc[0]["Description"] == "UPI PAYMENT TO MERCHANT"
    assert result.dataframe.iloc[0]["Reference"] == "ABC123"
    assert result.diagnostics["absorbed_continuation_sources"] == 1


def test_inherits_grouped_date_only_for_full_transaction_row() -> None:
    frame = source_frame(
        0,
        [
            {
                "Date": "03/01/2024",
                "Description": "FIRST PAYMENT",
                "Debit": "10.00",
                "Balance": "90.00",
            },
            {
                "Date": "",
                "Description": "SECOND PAYMENT",
                "Debit": "20.00",
                "Balance": "70.00",
            },
        ],
    )

    result = reconstruct_transactions([frame])

    assert len(result.dataframe) == 2
    assert list(result.dataframe["Date"]) == ["03/01/2024", "03/01/2024"]
    assert result.diagnostics["emitted_transaction_sources"] == 2


def test_rejects_ambiguous_undated_row_without_emitting_it() -> None:
    frame = source_frame(0, [{"Date": "", "Description": "Unstructured note"}])

    result = reconstruct_transactions([frame])

    assert result.dataframe.empty
    assert result.diagnostics["rejected_unclassified_sources"] == 1


def test_excludes_summaries_and_repeated_headers() -> None:
    frame = source_frame(
        0,
        [
            {"Date": "Date", "Description": "Description", "Debit": "Debit", "Balance": "Balance"},
            {"Date": "01/01/2024", "Description": "OPENING BALANCE", "Debit": "", "Balance": "100.00"},
            {"Date": "01/01/2024", "Description": "UPI PAYMENT", "Debit": "10.00", "Balance": "90.00"},
            {"Date": "", "Description": "TRANSACTION TOTAL DR/CR", "Debit": "10.00", "Balance": "90.00"},
            {"Date": "01/01/2024", "Description": "CLOSING BALANCE", "Debit": "", "Balance": "90.00"},
        ],
    )

    result = reconstruct_transactions([frame])

    assert len(result.dataframe) == 1
    assert result.dataframe.iloc[0]["Description"] == "UPI PAYMENT"
    assert result.diagnostics["excluded_non_transaction_sources"] == 4
    assert result.diagnostics["exclusion_reasons"] == {
        "closing_balance": 1,
        "opening_balance": 1,
        "repeated_header": 1,
        "transaction_total": 1,
    }


def test_preserves_identical_legitimate_transactions() -> None:
    row = {
        "Date": "04/01/2024",
        "Description": "UPI PAYMENT",
        "Debit": "50.00",
        "Balance": "500.00",
    }
    frame = source_frame(0, [row, row.copy()])

    result = reconstruct_transactions([frame])

    assert len(result.dataframe) == 2
    assert result.diagnostics["emitted_transaction_sources"] == 2


def test_accounts_for_every_selected_source_row() -> None:
    frame = source_frame(
        0,
        [
            {"Date": "05/01/2024", "Description": "UPI", "Debit": "5.00", "Balance": "95.00"},
            {"Date": "", "Description": "CONTINUED TEXT", "Debit": "", "Balance": ""},
            {"Date": "", "Description": "ACCOUNT SUMMARY", "Debit": "", "Balance": ""},
            {"Date": "", "Description": "?", "Debit": "", "Balance": ""},
        ],
    )

    result = reconstruct_transactions([frame])
    diagnostics = result.diagnostics

    accounted = (
        diagnostics["emitted_transaction_sources"]
        + diagnostics["absorbed_continuation_sources"]
        + diagnostics["excluded_non_transaction_sources"]
        + diagnostics["rejected_unclassified_sources"]
    )
    assert diagnostics["selected_source_rows"] == accounted == 4
    assert diagnostics["conservation_ok"] is True
