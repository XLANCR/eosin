# Eosin: A Comprehensive Bank Statement Cell Parsing Tool

**Eosin** is a tool built to tackle one of the trickiest problems in data extraction—parsing bank statements. If you've ever looked at bank statements from different institutions, you know how wildly they can vary in structure, format, and content. Eosin aims to make sense of that chaos by using clever techniques to extract data from these complex PDFs, no matter how inconsistent or irregular they are.

## Why the Name Eosin?

In biology, _eosin_ is a dye that helps differentiate cells under a microscope. In a similar way, this package is designed to differentiate and extract data from the messy structures of bank statements. While creating a parser for a single, specific statement format is easy, Eosin is built to handle the hardest version of the problem—working with all kinds of inconsistent formats and messy data.

## Production Modal endpoint

Use the deployed Modal web endpoint as `https://<workspace>--bank-parser.modal.run`. When proxy auth is enabled, callers must send `Modal-Key` and `Modal-Secret`; Modal reports this as `requires_proxy_auth`.

Production secrets are provisioned with `modal secret create eosin-tailscale` for Tailscale access and `modal secret create eosin-metrics-push` for push metrics credentials. pull-based monitoring is intentionally disabled for the Modal worker; use the configured push endpoint instead.

Eosin does not impose application-level PDF byte-size or page-count limits.
Uploads must still have a PDF filename and signature and must open successfully
with PyMuPDF. Production work remains protected by Modal proxy authentication,
bounded container concurrency, parser-pool admission timeout, maximum container
count, and function timeout. `/health` is liveness; `/ready` checks
parser-service readiness and returns `503` when the service cannot accept work.

Bank statements use `POST /v2/extract/bank-statement-evidence`. Invoices and
receipts use `POST /v2/extract/document-evidence` with multipart
`document_type=invoice|receipt`; those types run full-page text OCR instead of
bank-table OCR. The API and Celery worker carry this explicit type through the
request, and only normalized document fields are persisted.

### Modal runtime shape

- A deployment is capped at one GPU container. That container accepts up to 35 queued inputs; it does not create one container per PDF.
- Two parser instances bound CPU/layout work and feed up to 16 concurrent page OCR requests into the selected inference server's continuous batching. The batch-drain window is 50 ms.
- `EOSIN_MODAL_INFERENCE_BACKEND=sglang|vllm` selects the OpenAI-compatible GLM-OCR server. SGLang is the production default and uses the GLM-OCR NEXTN speculative-decoding recipe; vLLM remains the explicit rollback path with MTP-3, `max_num_seqs=196`, prefix caching, async scheduling, and chunked prefill. Deploy backend comparisons under a separate app name before changing production.
- Cold startup launches the inference server before parser imports so model startup and CPU preparation overlap. Cache volumes are committed only when explicitly seeding them; unchanged volumes are not committed on every production start.
- GPU memory snapshots remain disabled. Controlled SGLang snapshot tests reduced cold restore time but regressed warmed AU document throughput, so the experimental lifecycle is not part of the production path. Production statement OCR and parser outputs are never persisted as caches.
- Single-page OCR keeps the GLM-OCR 7,000-token default. `BANK_PARSER_OCR_PAGE_MAX_TOKENS` is an experiment control; a 4,096-token AU run preserved 6-page parser quality but regressed the harder 16-page document, so do not lower the production default without corpus-level evidence.
- Do not poll `/health` to keep a serverless container warm. Use direct parser traffic and push-based metrics; readiness checks are for deployment/load-balancer control only.

### Direct Modal benchmark

Use `scripts/benchmark_modal_parser.py` for bounded cold-start and warmed-throughput checks. It invokes the Modal class directly, caps concurrency at 35, cancels calls that exceed the request timeout, and writes local OCR artifacts that `scripts/replay_cached_ocr_quality.py` can evaluate. It never polls a public health endpoint.

```bash
python scripts/benchmark_modal_parser.py \
  --app-name eosin-glm-ocr \
  --class-name BankParserModalApp \
  --corpus-root "bank statements" \
  --pdf "bank statements/UCO Bank/782482386-Bank-Statement.pdf" \
  --concurrency 1 \
  --request-timeout-seconds 600 \
  --output-json /tmp/eosin-modal-benchmark.json
```

### What Makes Bank Statements So Hard to Parse?

Bank statements are notorious for being a nightmare to automate due to:

1. **Inconsistent Headers**: Each statement has its own unique headers, which can even change across pages.
2. **Cell Size Variations**: Adjacent cells aren’t always the same size.
3. **Irregular Rows and Columns**: Rows and columns often don’t follow consistent heights and widths.
4. **No Reliable Borders**: Borders may or may not exist, so we can’t rely on them.
5. **Multiline Dates**: Dates might be crammed into one line or spread across two or more.
6. **Date Format Chaos**: There’s no consistent way dates are presented—every statement seems to have its own idea.
7. **Missing Data**: Some rows might have empty columns, especially for certain transactions.
8. **Random Rows**: There are often irrelevant or random rows of data that throw everything off.
9. **Alignment Problems**: Text inside cells might be aligned in any direction—center, left, or right.
10. **Varying Headers**: Table headers can change or overlap as you go from page to page.
11. **No Consistent Row Spacing**: Nearby rows might be squeezed together or spaced far apart.
12. **Currency Format Mess**: Currency symbols and formats can be completely different between statements.
13. **Unreadable Statements**: Some statements are just hard to read—even for a human.

### The Assumptions We Make

To manage all these headaches, we made a few assumptions:

- **Dates Are Key**: We treat the date header as the most reliable thing on the page. We use it to figure out the structure of the table and align everything else around it.
- **Smart Date Parsing**: Eosin will try to pull together broken or spread-out dates and align them. If it still doesn’t make sense, we’ll ignore it and move on.
- **Headers Don’t Overlap**: We assume headers don’t interfere with each other, making them useful to anchor the rest of the data.
- **Spacing is Reasonably Consistent Across Pages**: While row and column spacing might be all over the place on one page, we assume it doesn’t change too wildly across the different pages.
