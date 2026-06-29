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


def test_coalesces_equivalent_schema_families_without_sparse_columns() -> None:
    frames = [
        source_frame(
            0,
            [
                {
                    "Date": "01/01/2024",
                    "Narration": "FIRST PAYMENT",
                    "Chg/Ref.No.": "REF1",
                    "Value Dt": "01/01/2024",
                    "Withdrawal Anti.": "10.00",
                    "Deposit Amt.": "",
                    "Closing Balance": "90.00",
                }
            ],
        ),
        source_frame(
            1,
            [
                {
                    "Transaction Date": "02/01/2024",
                    "Description": "SECOND PAYMENT",
                    "Reference": "REF2",
                    "Value Date": "02/01/2024",
                    "Debit": "",
                    "Credit": "5.00",
                    "Balance": "95.00",
                }
            ],
        ),
    ]

    result = reconstruct_transactions(frames)

    assert list(result.dataframe.columns) == [
        "Date",
        "Narration",
        "Chg/Ref.No.",
        "Value Dt",
        "Withdrawal Anti.",
        "Deposit Amt.",
        "Closing Balance",
    ]
    assert list(result.dataframe["Date"]) == ["01/01/2024", "02/01/2024"]
    assert list(result.dataframe["Narration"]) == ["FIRST PAYMENT", "SECOND PAYMENT"]


def test_resolves_single_amount_direction_from_running_balance() -> None:
    frames = [
        source_frame(
            0,
            [
                {
                    "Date": "01/01/2024",
                    "Narration": "OPENING TRANSACTION",
                    "Withdrawal": "",
                    "Deposit": "100.00",
                    "Closing Balance": "100.00",
                }
            ],
        ),
        source_frame(
            1,
            [
                {
                    "Transaction Date": "02/01/2024",
                    "Description": "PURCHASE",
                    "Amount": "10.00",
                    "Balance": "90.00",
                },
                {
                    "Transaction Date": "03/01/2024",
                    "Description": "REFUND",
                    "Amount": "50.00",
                    "Balance": "140.00",
                },
            ],
        ),
    ]

    result = reconstruct_transactions(frames)

    assert "Amount" not in result.dataframe.columns
    assert list(result.dataframe["Withdrawal"]) == ["", "10.00", ""]
    assert list(result.dataframe["Deposit"]) == ["100.00", "", "50.00"]
    assert result.diagnostics["balance_resolved_amount_sources"] == 2


def test_derives_missing_balance_from_explicit_ledger_values() -> None:
    frame = source_frame(
        0,
        [
            {"Date": "01/01/2024", "Description": "START", "Credit": "100.00", "Balance": "100.00"},
            {"Date": "02/01/2024", "Description": "PURCHASE", "Debit": "10.00", "Balance": ""},
            {"Date": "03/01/2024", "Description": "REFUND", "Credit": "5.00", "Balance": ""},
        ],
    )

    result = reconstruct_transactions([frame])

    assert list(result.dataframe["Balance"]) == ["100.00", "90.00", "95.00"]
    assert result.diagnostics["derived_balance_sources"] == 2


def test_merges_undated_fragment_forward_into_next_dated_row() -> None:
    frames = [
        source_frame(
            0,
            [{"Date": "01/01/2024", "Description": "START", "Credit": "100.00", "Balance": "100.00"}],
        ),
        source_frame(
            1,
            [
                {"Date": "", "Description": "PAYMENT", "Credit": "20.00", "Balance": ""},
                {"Date": "02/01/2024", "Description": "MERCHANT PAYMENT", "Credit": "", "Balance": "120.00"},
            ],
        ),
    ]

    result = reconstruct_transactions(frames)

    assert len(result.dataframe) == 2
    assert result.dataframe.iloc[1]["Description"] == "MERCHANT PAYMENT"
    assert result.dataframe.iloc[1]["Credit"] == "20.00"
    assert result.diagnostics["absorbed_continuation_sources"] == 1


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


def test_infers_transaction_roles_for_fully_generic_ocr_table() -> None:
    frame = source_frame(
        2,
        [
            {
                "col_0": "27/11/20",
                "col_1": "IMPS PAYMENT TO MERCHANT",
                "col_2": "000003212146353",
                "col_3": "27/11/20",
                "col_4": "700.00",
                "col_5": "",
                "col_6": "26,345.34",
            },
            {
                "col_0": "02/12/20",
                "col_1": "SALARY CREDIT",
                "col_2": "000000000000002",
                "col_3": "02/12/20",
                "col_4": "",
                "col_5": "24,690.00",
                "col_6": "51,035.34",
            },
        ],
    )

    result = reconstruct_transactions([frame])

    assert len(result.dataframe) == 2
    assert list(result.dataframe["Transaction Date"]) == ["27/11/20", "02/12/20"]
    assert "Date" not in result.dataframe.columns
    assert list(result.dataframe["Description"]) == [
        "IMPS PAYMENT TO MERCHANT",
        "SALARY CREDIT",
    ]
    assert result.dataframe.iloc[0]["Debit"] == "700.00"
    assert result.dataframe.iloc[1]["Credit"] == "24,690.00"
    assert result.diagnostics["generic_schema_frames_inferred"] == 1


def test_does_not_infer_transaction_roles_for_generic_account_profile() -> None:
    frame = source_frame(
        0,
        [
            {"col_0": "Account Number", "col_1": "123456789", "col_2": ""},
            {"col_0": "Customer Name", "col_1": "Example Customer", "col_2": ""},
            {"col_0": "Opening Balance", "col_1": "1,000.00", "col_2": ""},
        ],
    )

    result = reconstruct_transactions([frame])

    assert result.dataframe.empty
    assert result.diagnostics["generic_schema_frames_inferred"] == 0


def test_does_not_infer_generic_schema_from_sparse_statement_dates() -> None:
    rows = [
        {"col_0": "Statement From", "col_1": "01/01/2024", "col_2": "1,000.00"},
        {"col_0": "Statement To", "col_1": "31/01/2024", "col_2": "900.00"},
    ]
    rows.extend(
        {"col_0": f"Account metadata {index}", "col_1": "", "col_2": ""}
        for index in range(8)
    )

    result = reconstruct_transactions([source_frame(0, rows)])

    assert result.dataframe.empty
    assert result.diagnostics["generic_schema_frames_inferred"] == 0


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


def test_keeps_grouped_undated_transaction_before_next_dated_row() -> None:
    frame = source_frame(
        0,
        [
            {"Date": "03/01/2024", "Description": "FIRST", "Debit": "10.00", "Balance": "90.00"},
            {"Date": "", "Description": "SECOND", "Debit": "20.00", "Balance": "70.00"},
            {"Date": "04/01/2024", "Description": "THIRD", "Credit": "30.00", "Balance": "100.00"},
        ],
    )

    result = reconstruct_transactions([frame])

    assert len(result.dataframe) == 3
    assert list(result.dataframe["Date"]) == ["03/01/2024", "03/01/2024", "04/01/2024"]


def test_rejects_ambiguous_undated_row_without_emitting_it() -> None:
    frame = source_frame(0, [{"Date": "", "Description": "Unstructured note"}])

    result = reconstruct_transactions([frame])

    assert result.dataframe.empty
    assert result.diagnostics["rejected_unclassified_sources"] == 1


def test_rejects_dated_statement_period_row_without_money_or_description() -> None:
    frame = source_frame(
        0,
        [
            {
                "Transaction Date": "01/09/2020",
                "Description": "01/09/2020",
                "Reference": "To",
                "Value Date": "01/02/2021",
                "Debit": "",
                "Credit": "",
                "Balance": "",
            }
        ],
    )

    result = reconstruct_transactions([frame])

    assert result.dataframe.empty
    assert result.diagnostics["rejection_reasons"] == {
        "dated_without_transaction_evidence": 1
    }


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


def test_excludes_short_replayed_source_page_after_date_regression() -> None:
    frames = [
        source_frame(
            0,
            [
                {"Date": "01/01/2024", "Description": "A", "Debit": "10.00", "Balance": "90.00"},
                {"Date": "01/01/2024", "Description": "B", "Debit": "20.00", "Balance": "70.00"},
            ],
        ),
        source_frame(
            1,
            [
                {"Date": "02/01/2024", "Description": "C", "Credit": "30.00", "Balance": "100.00"},
            ],
        ),
        source_frame(
            2,
            [
                {"Date": "01/01/2024", "Description": "A OCR REPLAY", "Debit": "10.00", "Balance": "90.00"},
                {"Date": "01/01/2024", "Description": "B OCR REPLAY", "Debit": "20.00", "Balance": "70.00"},
            ],
        ),
    ]

    result = reconstruct_transactions(frames)

    assert len(result.dataframe) == 3
    assert result.diagnostics["exclusion_reasons"]["replayed_source_page"] == 2
    assert result.diagnostics["replayed_source_pages"] == [2]


def test_excludes_replayed_prefix_before_page_resumes_forward() -> None:
    frames = [
        source_frame(
            0,
            [
                {"Date": "01/01/2024", "Description": "FIRST", "Debit": "10.00", "Balance": "90.00"},
                {"Date": "02/01/2024", "Description": "SECOND", "Debit": "20.00", "Balance": "70.00"},
                {"Date": "03/01/2024", "Description": "THIRD", "Credit": "30.00", "Balance": "100.00"},
            ],
        ),
        source_frame(
            1,
            [
                {"Date": "01/01/2024", "Description": "FIRST REPLAY", "Debit": "10.00", "Balance": ""},
                {"Date": "02/01/2024", "Description": "SECOND REPLAY", "Debit": "20.00", "Balance": ""},
                {"Date": "04/01/2024", "Description": "FOURTH", "Credit": "30.00", "Balance": "130.00"},
            ],
        ),
    ]

    result = reconstruct_transactions(frames)

    assert len(result.dataframe) == 4
    assert result.diagnostics["exclusion_reasons"]["replayed_source_row"] == 2
    assert result.diagnostics["replayed_source_rows"] == [
        {"page": 1, "row": 0, "table": 0},
        {"page": 1, "row": 1, "table": 0},
    ]


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
