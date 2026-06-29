# Transaction Reconstruction and Modal Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve every supported bank transaction through page/schema reconstruction and let one Modal GPU container safely admit 35 PDFs while retaining the two-parser throughput optimum.

**Architecture:** Add a pure, provenance-aware transaction reconstruction module between page-level HTML parsing and public dataframe projection. The existing parser supplies source-tagged page frames; the reconstructor normalizes schema, identifies dated anchors, assembles proven continuations, excludes explicit non-transactions, and returns conservation diagnostics. Modal admission is widened independently: 35 inputs per container feed a bounded two-parser pool with request-length queue waiting and measured startup phases.

**Tech Stack:** Python 3.12, pandas, pytest, FastAPI, Modal, vLLM, existing 55-document endpoint artifacts and sidecar CSVs.

---

## File Map

- Create `eosin/backend/transaction_reconstruction.py`: pure reconstruction types, classification, continuation assembly, projection, and diagnostics.
- Modify `eosin/backend/eosin_pipeline.py`: attach request-local source provenance, invoke the reconstructor, and expose diagnostics.
- Modify `tests/test_eosin_pipeline.py`: integration coverage and removal of destructive-finalizer expectations.
- Create `tests/test_transaction_reconstruction.py`: focused TDD cases for row preservation and classification.
- Create `scripts/replay_cached_ocr_quality.py`: deterministic 55-document replay and quality comparison from captured endpoint responses.
- Create `tests/test_replay_cached_ocr_quality.py`: fixture loading and aggregate-report tests.
- Modify `eosin/backend/bank_parser_service.py`: long bounded parser-slot waiting and queue-wait diagnostics.
- Modify `tests/test_bank_parser_service_pool.py`: 35-request/two-parser admission and timeout tests.
- Modify `modal_app.py`: 35-input Modal defaults and startup phase instrumentation.
- Modify `.env.example`: production admission and startup tuning defaults.
- Modify `tests/test_repo_readiness.py`: configuration and startup instrumentation assertions.
- Modify parent `../AGENTS.md`: durable transaction-conservation and 35-request single-container contracts.

### Task 1: Pure transaction reconstruction

**Files:**
- Create: `eosin/backend/transaction_reconstruction.py`
- Create: `tests/test_transaction_reconstruction.py`

- [ ] **Step 1: Write failing date/schema tests**

Add tests that construct page dataframes with internal source columns and assert:

```python
def test_coalesces_date_columns_before_classifying_transactions():
    frames = [
        source_frame(0, [{"Tran Date": "01/01/2024", "Description": "A", "Debit": "10", "Balance": "90"}]),
        source_frame(1, [{"Tran Date": "", "Date (Value Date)": "02/01/2024", "Description": "B", "Debit": "20", "Balance": "70"}]),
    ]
    result = reconstruct_transactions(frames)
    assert len(result.dataframe) == 2
    assert result.diagnostics["emitted_transaction_sources"] == 2


def test_accepts_transaction_date_with_time_suffix():
    frame = source_frame(0, [{"TRANSACTION DATE": "02 Jan 2024 11:18 AM", "DETAILS": "UPI", "AMOUNT": "500", "BALANCE": "507.75"}])
    result = reconstruct_transactions([frame])
    assert len(result.dataframe) == 1
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_transaction_reconstruction.py
```

Expected: import failure because `transaction_reconstruction` does not exist.

- [ ] **Step 3: Implement source types, semantic roles, and date resolution**

Implement immutable result and source records:

```python
@dataclass(frozen=True)
class ReconstructionResult:
    dataframe: pd.DataFrame
    diagnostics: dict[str, object]


def reconstruct_transactions(frames: Sequence[pd.DataFrame]) -> ReconstructionResult:
    ...
```

Use internal columns `__source_page`, `__source_row`, and `__source_table`. Resolve a per-row date by scanning semantic date columns first, then generic columns with date-shaped values. Accept optional time suffixes and parenthesized value dates. Choose a canonical public date column by observed date coverage, not column order.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run `pytest -q tests/test_transaction_reconstruction.py`.

Expected: date/schema tests pass.

- [ ] **Step 5: Write failing continuation and exclusion tests**

Cover:

```python
def test_merges_text_continuation_across_page_boundary(): ...
def test_inherits_grouped_date_only_for_full_transaction_row(): ...
def test_rejects_ambiguous_undated_row_without_emitting_it(): ...
def test_excludes_opening_closing_totals_and_repeated_headers(): ...
def test_preserves_identical_legitimate_transactions(): ...
def test_accounts_for_every_selected_source_row(): ...
```

Assert that text-only adjacent fragments extend narration/reference fields, complementary blank fields are filled, conflicting monetary values are not merged, and diagnostics satisfy the conservation equation.

- [ ] **Step 6: Run tests and verify RED**

Run `pytest -q tests/test_transaction_reconstruction.py`.

Expected: continuation/classification assertions fail while date/schema tests stay green.

- [ ] **Step 7: Implement classification and continuation assembly**

Implement these outcomes: `transaction`, `continuation`, `excluded_non_transaction`, and `rejected_unclassified`.

Rules:

- an explicit resolved date plus narration/reference/amount/balance evidence creates an anchor;
- normalized header rows and explicit summary markers are excluded;
- text-only adjacent fragments merge into the previous transaction, including across pages;
- leading fragments without a previous anchor may merge into the next explicit anchor;
- an undated row may inherit the previous date only when it has independent transaction structure (text plus amount/balance);
- conflicting monetary fields reject a continuation instead of overwriting evidence;
- fuzzy transaction deduplication is never performed;
- diagnostics include selected, emitted-source, continuation, exclusion-by-reason, rejection-by-reason, and conservation counts.

- [ ] **Step 8: Run focused tests and commit**

Run:

```bash
pytest -q tests/test_transaction_reconstruction.py
git add eosin/backend/transaction_reconstruction.py tests/test_transaction_reconstruction.py
git commit -m "fix: preserve transactions during table reconstruction"
```

Expected: all focused tests pass.

### Task 2: Integrate reconstruction into the parser

**Files:**
- Modify: `eosin/backend/eosin_pipeline.py`
- Modify: `tests/test_eosin_pipeline.py`

- [ ] **Step 1: Write failing integration tests**

Add tests proving `_materialize_selected_tables()` annotates source page/row/table provenance and `_finalize_extracted_tables()`:

- coalesces date columns before row decisions;
- returns all dated transactions from the synthetic SBI/Axis schema-drift shape;
- retains identical dated rows;
- stores reconstruction diagnostics;
- strips internal provenance columns from public output.

Replace the old test requiring normalized adjacent transaction deletion with a test requiring both transactions to survive.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest -q tests/test_eosin_pipeline.py -k "finalize or materialize or reconstruction"
```

Expected: new provenance/diagnostic assertions fail and the changed duplicate-row assertion fails.

- [ ] **Step 3: Attach provenance and invoke the reconstructor**

In `_materialize_selected_tables()`, copy each aligned dataframe and add source page, source row, and selected-table indices. In `_finalize_extracted_tables()`, call `reconstruct_transactions()`, store `result.diagnostics` on the parser instance, and return `result.dataframe`.

Add `transaction_reconstruction` to `last_run_stats` for both layout and no-layout paths. Keep `page_ocr` and existing public columns compatible.

- [ ] **Step 4: Run focused and complete parser tests**

Run:

```bash
pytest -q tests/test_transaction_reconstruction.py tests/test_eosin_pipeline.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit integration**

```bash
git add eosin/backend/eosin_pipeline.py tests/test_eosin_pipeline.py
git commit -m "refactor: use provenance-aware transaction finalization"
```

### Task 3: Add cached OCR acceptance replay

**Files:**
- Create: `scripts/replay_cached_ocr_quality.py`
- Create: `tests/test_replay_cached_ocr_quality.py`

- [ ] **Step 1: Write failing replay tests**

Build a temporary captured-response fixture shaped like the load runner output:

```json
{
  "request": {"pdf_path": "Bank/sample.pdf"},
  "payload": {
    "debug": {"page_ocr": [{"page_index": 0, "raw_html": "<table>...</table>"}]}
  }
}
```

Assert the replay tool loads raw page HTML, runs the current parser finalizer without network access, finds the sidecar CSV, computes row counts and financial-key precision/recall/F1, and writes per-document plus aggregate JSON.

- [ ] **Step 2: Run test and verify RED**

Run `pytest -q tests/test_replay_cached_ocr_quality.py`.

Expected: import failure because the replay module does not exist.

- [ ] **Step 3: Implement the replay tool**

Reuse `BankStatementParser._evaluate_page_ocr_result()`, `_materialize_selected_tables()`, and `_finalize_extracted_tables()`; reuse normalization helpers from `compare_parser_to_ground_truth.py`. Add CLI arguments:

```text
--responses-dir PATH
--corpus-dir PATH
--output PATH
--baseline PATH (optional)
--fail-on-regression
```

The output must include reconstruction diagnostics, row-count deltas, financial-key metrics, and per-document regression reasons. It must never call Modal.

- [ ] **Step 4: Verify and commit**

Run:

```bash
pytest -q tests/test_replay_cached_ocr_quality.py
python scripts/replay_cached_ocr_quality.py \
  --responses-dir /tmp/eosin-production-endpoint-c2-20260629/20260629T090619Z/server-responses \
  --corpus-dir "bank statements" \
  --output /tmp/eosin-reconstruction-after.json
git add scripts/replay_cached_ocr_quality.py tests/test_replay_cached_ocr_quality.py
git commit -m "test: add cached OCR reconstruction benchmark"
```

Expected: 55 documents evaluated without network access. Review aggregate and worst-document output before proceeding; if financial-key F1 regresses, return to Task 1 rather than weakening acceptance.

### Task 4: Admit 35 PDFs per Modal container

**Files:**
- Modify: `eosin/backend/bank_parser_service.py`
- Modify: `tests/test_bank_parser_service_pool.py`
- Modify: `modal_app.py`
- Modify: `.env.example`
- Modify: `tests/test_repo_readiness.py`

- [ ] **Step 1: Write failing parser-pool admission tests**

Use two blocking fake parsers and 35 worker threads. Assert two parse immediately, the other 33 wait, all complete after parsers are released, and no request receives the former 30-second timeout. Assert `parser_queue_wait` and queue-depth diagnostics are present.

Also test an explicitly short configured timeout still raises `TimeoutError`, preserving bounded failure behavior.

- [ ] **Step 2: Run tests and verify RED**

Run `pytest -q tests/test_bank_parser_service_pool.py`.

Expected: queue diagnostics/default-wait assertions fail.

- [ ] **Step 3: Implement long bounded waiting and diagnostics**

Change the production default parser-slot timeout to the function/request timeout (`1800` seconds). Measure queue wait around `Queue.get()`, capture queue depth at admission, and add these values to response timings/debug without shared mutable request state.

Keep parser pool size at two and retain `finally`-based parser return.

- [ ] **Step 4: Write failing Modal configuration tests**

Assert:

- safe input cap is at least 35;
- defaults are `EOSIN_MODAL_MAX_INPUTS=35` and `EOSIN_MODAL_TARGET_INPUTS=35`;
- target cannot exceed max;
- `.env.example` exposes 35 inputs and 1800-second parser wait;
- startup logs separate vLLM health wait, volume commit, parser-service construction, and total enter time.

- [ ] **Step 5: Run tests and verify RED**

Run `pytest -q tests/test_repo_readiness.py -k "modal or startup or pool"`.

- [ ] **Step 6: Implement Modal admission and startup instrumentation**

Set the bounded Modal input default and safe cap to 35 while preserving environment overrides. Clamp target to max. Add phase timers without changing vLLM execution mode. Time cache commits separately and time parser-service construction in `enter()`.

Update `.env.example`:

```dotenv
EOSIN_MODAL_MAX_INPUTS=35
EOSIN_MODAL_TARGET_INPUTS=35
BANK_PARSER_POOL_WAIT_TIMEOUT=1800
```

- [ ] **Step 7: Run focused tests and commit**

Run:

```bash
pytest -q tests/test_bank_parser_service_pool.py tests/test_repo_readiness.py
git add eosin/backend/bank_parser_service.py tests/test_bank_parser_service_pool.py modal_app.py .env.example tests/test_repo_readiness.py
git commit -m "perf: queue 35 pdfs per modal gpu container"
```

### Task 5: Verification, startup experiment, DOX, and deployment

**Files:**
- Modify: `../AGENTS.md`
- Modify only if measurements justify it: `modal_app.py`, `.env.example`, `tests/test_repo_readiness.py`

- [ ] **Step 1: Run the full local verification suite**

Run:

```bash
pytest -q
python -m py_compile eosin/backend/transaction_reconstruction.py eosin/backend/eosin_pipeline.py eosin/backend/bank_parser_service.py modal_app.py scripts/replay_cached_ocr_quality.py
git diff --check
```

Expected: zero failures and clean diff checks.

- [ ] **Step 2: Review cached 55-document acceptance**

Compare `/tmp/eosin-reconstruction-after.json` with the current captured baseline. Confirm conservation for every document, no known catastrophic loss, no aggregate financial-key F1 regression, and improved row coverage on named failures.

- [ ] **Step 3: Perform code and security review**

Review for transaction invention, conflicting-value overwrite, unbounded memory growth, thread safety, proxy-auth regression, secret leakage, and accidental cache persistence. Address all critical/high findings and rerun focused tests.

- [ ] **Step 4: Update DOX and commit local completion**

Record durable contracts in `../AGENTS.md`:

- production finalization must conserve and classify every selected source row;
- only dated transactions and proven continuations reach public output;
- fuzzy transaction deduplication is prohibited;
- one Modal container admits 35 PDFs and drains them through a two-parser pool;
- cached Modal OCR is permitted only for deterministic experiments.

Stage only the exact AGENTS change in the outer repository. Commit Eosin changes separately from the outer DOX change.

- [ ] **Step 5: Deploy and measure cache-free behavior**

Deploy the Eosin branch once local gates pass. Warm nothing manually before the cold measurement. Record vLLM health wait, cache commit, parser construction, total readiness, then submit 35 PDFs concurrently and verify one-container admission, zero parser-slot 503s, GPU utilization, and request completion.

- [ ] **Step 6: Run startup compile-thread experiment**

Only after correctness is fixed, compare current `TORCHINDUCTOR_COMPILE_THREADS=1` with a bounded CPU-derived value using identical deploy/cold-start/warmed-throughput workloads. Keep a change only if startup improves without reducing warmed pages/second or causing compilation instability. Do not enable eager mode or GPU snapshots.

- [ ] **Step 7: Run final 55-document endpoint gate**

Run the cache-free endpoint load and compare all responses to sidecar CSVs. Save summaries outside git unless a durable report is requested. Verify 55/55 completion, reconstruction conservation, response schema, row coverage, financial-key metrics, latency, and throughput.

- [ ] **Step 8: Final review and commits**

Run fresh verification, inspect every diff and commit, ensure no credentials or corpus data are staged, and make any final narrowly scoped conventional commit required by measured startup changes or documentation.
