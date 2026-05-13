# Eosin GPU Backend

Standalone GPU backend for GLM-OCR bank-statement parsing. The primary deployment target is Modal: each active container runs a FastAPI parser API and a local `vllm serve` process on an L40S GPU.

## What Runs

- `modal_app.py`: Modal app, image build, vLLM process startup, ASGI endpoint.
- `eosin/backend/bank_parser_api.py`: FastAPI routes for parsing and health checks.
- `eosin/backend/bank_parser_service.py`: parser lifecycle wrapper.
- `eosin/backend/eosin_pipeline.py`: PDF preprocessing, layout detection, table stitching, OCR fanout, and result shaping.
- `eosin/backend/ocr_pipeline.py`: OCR execution backends for document-level HTTP mode and batch-drain throughput mode.
- `scripts/load_test_bank_parser.py`: concurrent PDF load tester with JSONL, CSV, and summary reports.

## Setup

Create a local virtualenv and install the project dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Authenticate Modal once on the machine that deploys this service:

```bash
modal setup
```

Create local config from the template:

```bash
cp .env.example .env
```

Edit `.env` for your deployment. `.env` is intentionally ignored by git; `.env.example` is the committed reference.

## Modal Deploy

Deploy the API:

```bash
modal deploy modal_app.py
```

Check the deployed health endpoint:

```bash
curl -sS --max-time 180 "$EOSIN_PARSER_BASE_URL/health"
```

Check the merged parser plus vLLM metrics endpoint:

```bash
curl -sS --max-time 180 "$EOSIN_PARSER_BASE_URL/metrics"
```

The default production shape in `.env.example` is:

- GPU: `L40S`
- CPU: `8`
- Memory: `32768` MiB
- Modal min containers: `0`
- Modal max containers: `1`
- Modal concurrent inputs per container: `64`
- Idle scaledown window: `120` seconds
- vLLM model: `zai-org/GLM-OCR`
- vLLM max model length: `16384`
- vLLM max sequences: `32`
- vLLM max batched tokens: `16384`
- vLLM GPU memory utilization: `0.85`
- vLLM speculative config: `{"method": "mtp", "num_speculative_tokens": 1}`
- Modal memory/GPU snapshots: disabled in the production path after snapshot retries and NCCL heartbeat log storms
- vLLM sleep/wake snapshot path: disabled in the production path
- vLLM snapshot warmup mode: `multimodal`
- Hugging Face Xet high-performance transfers: enabled
- TorchInductor compile threads for snapshot compatibility: `1`

Modal uses `@modal.concurrent(max_inputs=..., target_inputs=...)`; older `allow_concurrent_inputs` examples are not used by the current SDK.

The Modal setup now follows the official `vllm_inference` and `ministral3_inference` patterns more closely:

- one Volume for Hugging Face weights and one for vLLM compile/runtime caches
- optional model revision pinning through `EOSIN_MODAL_VLLM_MODEL_REVISION`
- explicit fast-boot toggle through `EOSIN_MODAL_VLLM_FAST_BOOT`
- no Modal memory snapshot or vLLM sleep/wake path in production
- best-effort Hugging Face and vLLM cache volume commits after successful vLLM startup

Cost guardrails are intentional. `EOSIN_MODAL_MIN_CONTAINERS` defaults to `0`; any always-on deployment must also set `EOSIN_MODAL_ALLOW_ALWAYS_ON=true`. Modal memory/GPU snapshots are disabled because the GLM-OCR/vLLM snapshot path produced retry loops and repeated NCCL heartbeat errors in Modal logs while keeping paid GPU containers alive.

The `vllm_throughput` example is only partially applicable here because this repo exposes an online OCR API rather than an offline batch job. The repo keeps the HTTP-serving shape, but applies the relevant throughput controls:

- high GPU memory utilization
- chunked prefill
- async scheduling
- speculative decoding
- prefix caching
- server load tracking metrics
- denser document-level batching in the parser’s `batch_document` OCR backend

## Secrets

This repo intentionally does not commit secrets.

Create the Modal secrets before deploy:

```bash
modal secret create eosin-tailscale TAILSCALE_AUTHKEY="tskey-auth-..."
modal secret create eosin-metrics-push BANK_PARSER_METRICS_PUSH_AUTH_VALUE="<shared-ingest-token>"
```

The Tailscale auth key should be reusable and ephemeral. If you exposed one in chat or source control, rotate it before deploy.

## Caller Integration

Configure the caller project to send PDFs to the deployed parser endpoint:

```bash
export EOSIN_PARSER_BASE_URL="https://<workspace>--bank-parser.modal.run"
```

The parser accepts `POST /parse/bank-statement` with multipart form field `file`. Authentication is intentionally not enforced yet; add it before exposing the endpoint beyond trusted callers.

## OCR Modes

The parser supports internal OCR execution modes while keeping the same public API:

- `page_http`
  production quality default; sends one OCR request per final table page crop and merges page outputs deterministically
- `page_batch`
  reserved throughput mode; still keeps one page per OCR task and batches at the scheduler layer
- `document_http`
  experimental compatibility mode; sends larger multi-page OCR requests
- `batch_document`
  experimental compatibility mode; batches document-level OCR jobs

Current production default: `page_http`

Use `page_http` as the quality control configuration. Only use document modes for explicit experiments, because large multi-page GLM-OCR requests were the source of repeated-row and column-corruption failures.
Use `BANK_PARSER_OCR_DOCUMENT_MAX_TOKENS_CAP` to cap the response budget for multi-page document OCR requests; the runtime now scales `max_tokens` up from the single-page base request and clamps it at this ceiling. The default cap is intentionally conservative to limit repeated-row overgeneration on long statements.
Use `BANK_PARSER_OCR_DOCUMENT_MAX_IMAGES_PER_REQUEST` to split very large PDFs into a few ordered multimodal subrequests when one giant image set degrades GLM-OCR quality. Chunk OCR requests are submitted together so vLLM can schedule them concurrently.

Layout fallback is controlled with `BANK_PARSER_LAYOUT_MODE`:

- `required`
  fail if the PP-DocLayout detector cannot initialize
- `auto`
  fall back to the non-layout extraction path when layout initialization fails
- `disabled`
  always use the non-layout extraction path

For Modal, the default is `auto` so a transient layout-detector issue does not take the whole worker out of service.

## Load Testing

Put local PDFs under `bank statements/` using any nested folder layout. That directory is ignored and must not be committed.

Run a small validation:

```bash
python scripts/load_test_bank_parser.py \
  --endpoint "$EOSIN_PARSER_BASE_URL" \
  --corpus-dir "bank statements" \
  --target-requests 10 \
  --concurrency 10 \
  --timeout 1800
```

Run a heavier sweep:

```bash
python scripts/load_test_bank_parser.py \
  --endpoint "$EOSIN_PARSER_BASE_URL" \
  --corpus-dir "bank statements" \
  --target-requests 100 \
  --concurrency 32 \
  --timeout 1800
```

Reports are written to `load-test-results/<timestamp>/` and include:

- request-level JSONL and CSV logs
- raw server responses under `server-responses/`
- parsed pandas DataFrame markdown files under `dataframes/`
- success and failure counts
- latency percentiles
- throughput
- PDF counts and uploaded byte counts
- server-side parser timings
- OCR queue wait and vLLM request timing summaries
- raw server responses
- markdown exports of parsed DataFrames

`load-test-results/` is ignored so previous benchmark logs stay local.

## Local Docker

The Docker Compose path is kept for non-Modal GPU hosts:

```bash
docker compose -f docker-compose.vllm.yml up --build
```

The local API is expected at:

```bash
curl -sS http://localhost:8090/health
```

## Operational Notes

- Each PDF completes preprocessing atomically before OCR starts for that PDF.
- Layout detection is intentionally bounded to one concurrent layout job.
- OCR work is done per final table page crop in the production `page_http` path.
- `batch_document` can wait up to `BANK_PARSER_OCR_BATCH_DRAIN_MAX_WAIT_SECONDS` before flushing ready PDFs together, but it is experimental.
- Hugging Face cache is persisted in the Modal volume `eosin-hf-cache`.
- vLLM compile/runtime caches are persisted in the Modal volume `eosin-vllm-cache`.
- Modal memory/GPU snapshots are disabled in production; do not re-enable them without an isolated cost-capped experiment.
- pull-based monitoring is intentionally disabled for the Modal worker.

## Monitoring

The repo now includes a local-first metrics stack under [monitoring/README.md](/home/nol/Documents/work/eosin/monitoring/README.md).

Start it with:

```bash
docker compose -f monitoring/docker-compose.metrics.yml up -d
```

The worker pushes metrics to VictoriaMetrics during real request handling. Do not scrape the Modal worker endpoint from Prometheus, VictoriaMetrics, or blackbox probes. That keeps the serverless container warm and creates unnecessary cost.

While requests are in flight, the worker also pushes periodic snapshots so long OCR runs still produce GPU and vLLM time series instead of only emitting a single sample at request completion.
