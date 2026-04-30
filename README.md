# Eosin GPU Backend

Standalone GPU backend for GLM-OCR bank-statement parsing. The primary deployment target is Modal: each warm container runs a FastAPI parser API and a local `vllm serve` process on an A100 80GB GPU.

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

- GPU: `A100-80GB`
- CPU: `12`
- Memory: `40960` MiB
- Modal max containers: `1`
- Modal concurrent inputs per container: `128`
- Idle scaledown window: `180` seconds
- vLLM model: `zai-org/GLM-OCR`
- vLLM max model length: `22480`
- vLLM max sequences: `192`
- vLLM max batched tokens: `32768`
- vLLM GPU memory utilization: `0.90`
- vLLM speculative config: `{"method": "mtp", "num_speculative_tokens": 1}`

Modal uses `@modal.concurrent(max_inputs=..., target_inputs=...)`; older `allow_concurrent_inputs` examples are not used by the current SDK.

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

The parser now supports two internal OCR execution modes while keeping the same public API:

- `document_http`
  sends one OCR request per PDF after preprocessing completes for that PDF
- `batch_document`
  holds ready PDFs briefly, then flushes multiple document OCR jobs together so vLLM sees denser concurrent work

Current throughput default: `batch_document`

Use `document_http` as the control configuration. Use `batch_document` plus larger `BANK_PARSER_OCR_BATCH_DRAIN_MAX_BATCH_SIZE` values when you want to push throughput harder on a single container.

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
- OCR work is now done per PDF, not per page, in both document backends.
- `batch_document` can wait up to `BANK_PARSER_OCR_BATCH_DRAIN_MAX_WAIT_SECONDS` before flushing ready PDFs together.
- The current default drain batch size is `64`; practical sweep points are `32`, `64`, `96`, and `128`.
- Hugging Face cache is persisted in the Modal volume `eosin-hf-cache`.
- vLLM compile/runtime caches are persisted in the Modal volume `eosin-vllm-cache`.
- pull-based monitoring is intentionally disabled for the Modal worker.

## Monitoring

The repo now includes a local-first metrics stack under [monitoring/README.md](/home/nol/Documents/work/eosin/monitoring/README.md).

Start it with:

```bash
docker compose -f monitoring/docker-compose.metrics.yml up -d
```

The worker pushes metrics to VictoriaMetrics during real request handling. Do not scrape the Modal worker endpoint from Prometheus, VictoriaMetrics, or blackbox probes. That keeps the serverless container warm and creates unnecessary cost.
