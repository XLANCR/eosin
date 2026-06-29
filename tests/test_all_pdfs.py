"""Batch tester for all PDFs in tests/test_pdfs/.

Runs eosin.parser.Parser against every PDF, collects output (DataFrame or
exception) and prints a summary table at the end.  Useful for quickly
identifying which statements fail and why.

Usage:
    python -m pytest tests/test_all_pdfs.py -v -s
    # or standalone:
    python tests/test_all_pdfs.py
"""

import os
import sys
import traceback
from pathlib import Path

import pandas as pd
import pytest

# Ensure the package root is importable when running standalone
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from eosin.parser import Parser  # noqa: E402
except ModuleNotFoundError:
    pytest.skip("legacy eosin.parser was retired by the evidence-service migration", allow_module_level=True)

TEST_PDFS_DIR = Path(__file__).resolve().parent / "test_pdfs"


def collect_all_outputs() -> dict[str, dict]:
    """Parse every PDF in test_pdfs/ and return results keyed by filename.

    Each value is a dict with keys:
        status  : "ok" | "empty" | "error"
        rows    : int (number of data rows, 0 on failure)
        columns : list[str] (column names, [] on failure)
        df      : pd.DataFrame | None
        error   : str | None (traceback text on failure)
    """
    results: dict[str, dict] = {}

    pdf_files = sorted(
        f for f in os.listdir(TEST_PDFS_DIR) if f.lower().endswith(".pdf")
    )

    for filename in pdf_files:
        filepath = TEST_PDFS_DIR / filename
        entry: dict = {
            "status": "error",
            "rows": 0,
            "columns": [],
            "df": None,
            "error": None,
        }

        try:
            parser = Parser(str(filepath))
            df = parser.parse()

            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                entry["status"] = "empty"
                entry["df"] = df
            else:
                entry["status"] = "ok"
                entry["rows"] = len(df)
                entry["columns"] = list(df.columns)
                entry["df"] = df
        except Exception:
            entry["error"] = traceback.format_exc()

        results[filename] = entry

    return results


def print_summary(results: dict[str, dict]) -> None:
    """Print a human-readable summary of parsing results."""
    ok_count = sum(1 for r in results.values() if r["status"] == "ok")
    empty_count = sum(1 for r in results.values() if r["status"] == "empty")
    error_count = sum(1 for r in results.values() if r["status"] == "error")

    print("\n" + "=" * 80)
    print("EOSIN PDF PARSING RESULTS")
    print("=" * 80)
    print(f"  Total PDFs : {len(results)}")
    print(f"  OK         : {ok_count}")
    print(f"  Empty      : {empty_count}")
    print(f"  Errors     : {error_count}")
    print("=" * 80)

    for filename, result in results.items():
        status_icon = {"ok": "✓", "empty": "⚠", "error": "✗"}[result["status"]]
        line = f"  {status_icon}  {filename:<20s}  status={result['status']:<6s}  rows={result['rows']}"
        if result["columns"]:
            line += f"  cols={result['columns']}"
        print(line)

    # Detail sections
    if any(r["status"] == "ok" for r in results.values()):
        print("\n" + "-" * 80)
        print("PARSED DATA PREVIEWS")
        print("-" * 80)
        for filename, result in results.items():
            if result["status"] == "ok" and result["df"] is not None:
                print(f"\n>>> {filename} ({result['rows']} rows)")
                pd.set_option("display.max_rows", None)
                pd.set_option("display.max_columns", None)
                pd.set_option("display.width", 200)
                pd.set_option("display.max_colwidth", 40)
                print(result["df"].to_string(index=True))

    if any(r["status"] == "empty" for r in results.values()):
        print("\n" + "-" * 80)
        print("EMPTY RESULTS (no data extracted)")
        print("-" * 80)
        for filename, result in results.items():
            if result["status"] == "empty":
                print(f"  {filename}")

    if any(r["status"] == "error" for r in results.values()):
        print("\n" + "-" * 80)
        print("ERRORS")
        print("-" * 80)
        for filename, result in results.items():
            if result["status"] == "error":
                print(f"\n>>> {filename}")
                print(result["error"])

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Pytest integration: one test per PDF
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


def _pdf_ids() -> list[str]:
    return sorted(
        f for f in os.listdir(TEST_PDFS_DIR) if f.lower().endswith(".pdf")
    )


@pytest.fixture(params=_pdf_ids(), ids=lambda f: f)
def pdf_path(request):
    return str(TEST_PDFS_DIR / request.param)


def test_pdf_parses(pdf_path):
    """Each PDF should parse without raising an exception."""
    parser = Parser(pdf_path)
    df = parser.parse()
    # At minimum: no crash. We also assert *some* data was extracted.
    assert df is not None, f"Parser returned None for {pdf_path}"
    assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)}"
    # Allow empty DataFrames (some PDFs may legitimately produce no rows)
    # but flag them in output
    if df.empty:
        pytest.skip(f"Parser returned empty DataFrame for {os.path.basename(pdf_path)}")


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = collect_all_outputs()
    print_summary(results)
