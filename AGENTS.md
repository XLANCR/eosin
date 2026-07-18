# Eosin OCR Service DOX

## Purpose

- Own the Eosin GLM-OCR service, Modal deployment definition, request-local bank/document evidence extraction, local comparison tools, and service tests.

## Ownership

- `modal_app.py` owns Modal app/container/runtime configuration and web/RPC entry points.
- `eosin/backend/` owns OCR execution, quality metadata, bounded parser pooling, and evidence payload construction.
- `scripts/` owns explicit development deployment and local evaluation workflows.
- `tests/` owns service, OCR-backend, deployment-contract, and comparison safeguards.
- `bank statements/` contains local acceptance inputs and ground truth; do not publish its contents automatically.

## Local Contracts

- Modal development and production use separate accounts. The `noelalex404` account is development-only. Production access is denied by default and requires explicit per-task user authorization plus verification that the active workspace is the separate production account.
- Development uses Modal environment `dev`, app `eosin-glm-ocr-dev`, and web label `bank-parser-dev`; after relevant Eosin changes, leave this dev deployment on the latest local revision.
- Proxy credentials and Modal tokens stay process-local and must never be committed or written to artifacts.
- Eosin returns request-local OCR evidence and quality metadata; it does not persist statement-derived OCR or parser output in production.
- If every requested OCR task fails, the service must fail closed with bounded status/error telemetry instead of returning a successful empty payload.

## Work Guidance

- Deploy development through `scripts/deploy_dev.ps1`. With explicit production authorization, deploy environment `main`, app `eosin-glm-ocr`, and web label `bank-parser` only after confirming the active workspace is not `noelalex404`.
- Keep the measured parser-pool and OCR concurrency defaults unless a controlled benchmark improves throughput, latency, quality, queueing, and cost together.
- Compare parser output to GT with `scripts/compare_parser_to_ground_truth.py`; production endpoints require an explicit opt-in flag.

## Verification

- Run focused service checks with `py -3.14 -m pytest -q tests/test_bank_parser_service_pool.py tests/test_ocr_backends.py tests/test_compare_parser_to_ground_truth.py`.
- For live acceptance, retain the exact PDF hash, OCR page-quality metadata, service timings, queue metrics, retry/error counts, and materialized request-local fixture.

## Child DOX Index
