# Noel/Prometheus Verification Report

Date: 2026-05-06
Branch: `Noel/prometheus`
Commit: `ee196f113b3490ac01c7e9ff5b383f9fbf9a16e2`

## Scope

This pass focused on:

- restoring the main checkout to `Noel/prometheus`
- verifying the branch locally
- checking whether the live Modal endpoint is actually serving this branch
- running small live load tests before attempting an expensive 100-request run
- fixing monitoring-stack regressions found during review

## Local Verification

Commands:

```bash
python -m py_compile modal_app.py eosin/__init__.py eosin/backend/*.py scripts/load_test_bank_parser.py tests/test_repo_readiness.py tests/test_ocr_backends.py tests/test_metrics_push.py tests/test_eosin_pipeline.py
python -m pytest tests/test_repo_readiness.py tests/test_ocr_backends.py tests/test_metrics_push.py tests/test_eosin_pipeline.py -q
```

Result:

- `25 passed, 1 skipped`
- no local import/runtime regressions found in the exercised paths

## Code Fixes Applied In This Pass

### 1. Metrics push now continues during long OCR requests

Problem:

- the push-only monitoring path only scheduled a metrics push when a request finished
- long OCR runs could therefore miss the GPU/vLLM time series that mattered most

Fix:

- added a periodic active-request push loop in `eosin/backend/metrics.py`
- pushes now continue while requests are in flight
- added test coverage in `tests/test_metrics_push.py`

### 2. Removed stale pull-based monitoring artifacts from the repo surface

Problem:

- the repo README and compose file described a push-only monitoring stack
- but the tree still contained old `vmagent` and `blackbox` scrape configs
- the Grafana dashboard and alert rules still referenced probe-based metrics

Fix:

- removed the stale `monitoring/vmagent/*` files
- removed `monitoring/blackbox-exporter/config.yml`
- updated `monitoring/vmalert/eosin.rules.yml`
- updated `monitoring/grafana/dashboards/eosin-overview.json`
- tightened readiness tests so these pull-based leftovers do not regress back in

### 3. Aligned the committed OCR backend default with the README

Problem:

- the README already described `batch_document` as the throughput-oriented default
- but `.env.example` and `modal_app.py` still defaulted to `document_http`

Fix:

- changed the committed default to `batch_document`
- updated readiness tests to enforce the same default going forward

## Live Endpoint Checks

Target endpoint from local `.env`:

- `https://noelalex404--bank-parser.modal.run`

Observed behavior:

- `/health` returned `200` with `{"status":"ok"}`
- `/metrics` returned `404`

Interpretation:

- the live endpoint is almost certainly not serving the current `Noel/prometheus` build
- on this branch, `/metrics` is implemented in `eosin/backend/bank_parser_api.py`
- a healthy `Noel/prometheus` deployment should not return `404` for `/metrics`

This is the main blocker for any serious throughput investigation on the live deployment.

## Live Load Tests Run

### Smoke run

Results directory:

- `load-test-results/20260506T044126Z`

Configuration:

- requests: `3`
- concurrency: `3`

Summary:

- success rate: `100%`
- client mean latency: `181.11s`
- server mean total: `56.91s`
- mean OCR stage: `36.74s`
- mean client minus server gap: `124.20s`

Interpretation:

- the parser itself completed in under a minute on average
- the additional `~124s` per request occurred outside server-reported timings
- this strongly suggests cold-start and/or admission delay before request execution began

### Calibration run

Results directory:

- `load-test-results/20260506T044941Z`

Configuration:

- requests: `10`
- concurrency: `10`

Summary:

- success rate: `100%`
- test duration: `382.03s`
- requests/sec: `0.026176`
- client mean latency: `357.53s`
- server mean total: `116.75s`
- mean OCR stage: `80.86s`
- mean client minus server gap: `240.78s`

OCR-specific summary:

- mean OCR task count per PDF: `13.2`
- mean OCR queue wait mean: `0.225s`
- max OCR queue wait max: `35.43s`
- mean OCR request mean: `72.51s`
- max OCR request max: `91.24s`

Interpretation:

- the OCR step remains the dominant stage by a large margin
- queue wait is not the main limiter overall
- the server-side OCR request time roughly doubled between the 3-request and 10-request runs
- the client-side non-server delay also grew sharply
- this endpoint is not yet demonstrating the throughput shape expected from the intended optimized `Noel/prometheus` deployment

## Why I Did Not Run The Full 100-Request Test

I stopped short of the expensive 100-request run for one reason:

- the live endpoint does not appear to be running the current branch, because `/metrics` is still `404`

Running a 100-request benchmark against the wrong build would produce misleading throughput numbers and burn time/cost without giving trustworthy data.

## Production Issues Still Open

### 1. Deployment mismatch

Severity: high

- local branch has `/metrics`
- public endpoint returns `404`
- production-style benchmarking should wait until the deployed service is confirmed to be this branch

### 2. Tailscale auth key is still present in local `.env`

Severity: high

- this should be rotated
- keep it in Modal secrets, not as durable local config

### 3. Current endpoint still shows very large admission/cold-start overhead

Severity: high

- even successful requests spend much more time outside server-reported work than inside it
- this must be re-measured after the correct branch is deployed

### 4. OCR remains the dominant runtime cost

Severity: medium

- this is expected directionally
- but with the current deployment mismatch, it is too early to decide whether the remaining bottleneck is request shape, vLLM scheduling, or stale server config

## Recommended Next Step

1. redeploy `Noel/prometheus` and confirm `/metrics` returns `200`
2. confirm pushed metrics are visible in local VictoriaMetrics/Grafana
3. rerun the load matrix on the correct build:
   - `3 @ 3`
   - `10 @ 10`
   - `25 @ 25`
   - `50 @ 50`
   - only then `100 @ 100`
4. use the new push-based GPU/vLLM time series to decide whether to keep `document_http` or switch default deployment mode to `batch_document`

Until step 1 is true, any larger stress run is operationally expensive and analytically weak.
