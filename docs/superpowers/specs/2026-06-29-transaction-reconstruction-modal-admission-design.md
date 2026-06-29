# Transaction Reconstruction and Modal Admission Design

## Objective

Return one clean final table containing all supported transaction rows without
silently dropping valid transactions, while allowing one Modal GPU container to
accept and drain a burst of at least 35 PDFs before autoscaling creates another
container.

Production requests remain cache-free. Captured Modal OCR responses may be used
as deterministic test fixtures during development and benchmarking.

## Current Failure Evidence

Replaying the 55-document endpoint responses through the current finalizer
accounted for 12,295 selected OCR rows:

- repeated-header removal removed 14 rows;
- low-quality filtering removed 220 rows;
- multiline merging removed 1,407 rows;
- adjacent fuzzy deduplication removed 850 rows;
- final date filtering removed 12 rows;
- 9,792 rows remained.

The largest losses are deterministic post-processing failures, not missing OCR:

- SBI: 299 candidate transaction rows became 7;
- Axis: 338 candidate transaction rows became 61;
- Kotak: 142 candidate transaction rows became 7.

Three defects combine to cause these losses:

1. The multiline merger keeps rows only when one selected date column matches a
   narrow date expression. Every other row is implicitly discarded unless it is
   merged as a sparse continuation.
2. Page-level schema drift creates multiple date columns, but semantic duplicate
   columns are coalesced only after multiline processing.
3. The adjacent-row deduplicator treats three matching non-empty cells as proof
   of duplication, which removes legitimate repeated transactions.

## Required Output Contract

The endpoint returns only final transaction rows. It does not return repeated
headers, account metadata, opening or closing balance summaries, transaction
totals, or unclassified OCR noise.

A source row may have one of four outcomes:

1. `transaction`: emitted as a final transaction row;
2. `continuation`: absorbed into exactly one transaction row;
3. `excluded_non_transaction`: removed for an explicit header or summary reason;
4. `rejected_unclassified`: not emitted because it cannot safely be interpreted
   as a transaction or continuation.

For every parsed document, diagnostics must satisfy this conservation rule:

```text
selected_source_rows
  = emitted_transaction_sources
  + absorbed_continuation_sources
  + excluded_non_transaction_sources
  + rejected_unclassified_sources
```

No row may disappear without one of these recorded outcomes.

## Reconstruction Pipeline

### 1. Attach source provenance

Each selected page table receives request-local provenance before concatenation:

- source page index;
- source row index;
- selected table index;
- original column/value mapping.

Provenance remains internal and is removed from the public dataframe. Aggregate
row-accounting diagnostics are returned under the existing `debug` payload.

### 2. Normalize schema before row decisions

Schema normalization runs before filtering or continuation assembly.

- Semantically equivalent date columns are coalesced per row.
- Semantically equivalent debit, credit, amount, balance, narration, and
  reference columns are coalesced only when one side is blank.
- Values are never overwritten when two non-empty candidates conflict; the
  original columns remain available to classification diagnostics.
- Generic columns containing date values may supply a date candidate when a
  named date column is empty.

Date recognition accepts the formats already supported plus optional time text
and parenthesized value dates. A date prefix is extracted without rewriting the
source cell exposed in the final dataframe.

### 3. Classify explicit transaction anchors

A row with a resolved transaction date is a transaction anchor when it also has
transaction evidence such as narration/reference text, an amount, or a balance.
A dated row matching a known summary marker remains a non-transaction.

Rows that repeat the normalized table header are excluded as repeated headers.
Rows containing explicit opening balance, closing balance, transaction-total,
legend, or account-summary markers are excluded with the corresponding reason.

### 4. Assemble undated continuations

An undated row is absorbed only when all applicable evidence supports one
adjacent transaction:

- it is adjacent in source order, including across a page boundary;
- it contains transaction fields such as narration, reference, amount, or
  balance content;
- it does not match a header, summary, or account-metadata marker;
- merging does not overwrite a conflicting non-empty value;
- the direction of attachment is unambiguous.

Text-only fragments extend the matching narration/reference field. Complementary
fields fill blanks. Conflicting monetary values prevent continuation merging.

An undated row with full transaction structure may inherit the nearest explicit
date only when the source layout groups multiple transactions under one date and
the row has independent monetary or balance evidence. Ambiguous rows are recorded
as `rejected_unclassified`; they are never emitted with an invented date.

### 5. Preserve legitimate repeated transactions

Fuzzy transaction deduplication is removed. Two transactions with identical
dates, descriptions, and amounts are allowed because bank statements may contain
legitimate repeated payments.

Repeated headers continue to be removed structurally. OCR duplication is handled
by page-quality diagnostics and retry selection rather than destructive
document-level row overlap heuristics.

### 6. Build the public dataframe

Only classified transactions are projected to the final dataframe. Absorbed
continuations enrich their transaction. Internal provenance columns are removed,
empty public columns are dropped, and original transaction ordering is retained.

## Modal Single-Container Admission

The Modal class accepts 35 concurrent PDF requests per container. The parser pool
remains at the measured throughput optimum of two parser instances, so one GPU
continuously batches page OCR from up to two active documents while the remaining
requests wait in a bounded in-container admission queue.

- `target_inputs=35` and `max_inputs=35` prevent scale-out below 36 concurrent
  requests.
- `max_containers` remains a safety ceiling for bursts above one container's
  admitted capacity.
- Parser-slot waiting uses the request/function timeout instead of the current
  30-second timeout, preventing healthy queued requests from returning 503.
- Queue depth, wait time, active parser count, and timeout count are exposed in
  request diagnostics/metrics.
- PDF admission remains uncapped by bytes or pages as previously required.

## vLLM Startup Strategy

The 35-request admission policy amortizes one compiled vLLM startup across a
meaningful workload. The default remains compiled mode because prior testing
showed eager fast boot reduced warmed throughput.

Startup optimization is evaluated independently from parser correctness:

1. measure subprocess-to-health, cache-commit, parser-pool construction, and full
   enter-to-ready phases separately;
2. compare the current single TorchInductor compile thread with a bounded value
   based on available CPUs;
3. determine whether cache-volume commits can be skipped when no cache files
   changed or moved off the readiness-critical path;
4. retain a change only when it improves startup without reducing warmed
   pages/second or correctness.

GPU snapshots remain out of scope because the current deployment explicitly
disabled them after runtime instability.

## Testing and Acceptance

### TDD regression tests

Synthetic tests cover:

- duplicate date columns populated on different pages;
- transaction dates containing times;
- text continuations within a page and across page boundaries;
- grouped undated transactions with independent money/balance evidence;
- opening/closing balances, totals, repeated headers, and noise exclusions;
- identical legitimate transactions remaining distinct;
- complete row-accounting conservation;
- 35 admitted requests waiting safely behind a two-parser pool.

### Cached 55-document experiment

The captured Modal page OCR is replayed without GPU calls. The experiment records
per-document source rows, transaction rows, continuation merges, exclusions,
rejections, row-count difference from sidecar CSV, financial-key precision,
recall, F1, and regressions against the current endpoint output.

Acceptance requires:

- no known catastrophic transaction loss;
- all source rows accounted for;
- no reduction in aggregate financial-key F1;
- no document loses a previously matched financial transaction without an
  explicit reviewed reason;
- material improvement in row-count coverage on the known SBI, Axis, Kotak,
  HDFC, AU, UCO, and ICICI failures.

### Live endpoint verification

After local tests and cached acceptance pass:

1. deploy once;
2. run a cache-free smoke test;
3. submit 35 PDFs concurrently to one cold container;
4. verify no parser-slot 503s, container admission behavior, GPU utilization,
   pages/second, latency distribution, startup phases, row accounting, and
   response schema;
5. rerun the full 55-document endpoint comparison before declaring completion.

## Non-Goals

- Returning every OCR `<tr>` as a public row.
- Inventing transaction dates or monetary values.
- Replacing GLM OCR or the downstream Almond evidence-first parser.
- Enabling production statement/result caching.
- Enabling GPU memory snapshots.
