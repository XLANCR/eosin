$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "1"
$env:MODAL_ENVIRONMENT = "dev"
$env:EOSIN_MODAL_APP_NAME = "eosin-glm-ocr-dev"
$env:EOSIN_MODAL_WEB_LABEL = "bank-parser-dev"
$env:EOSIN_MODAL_TAILSCALE_ENABLE = "false"
$env:EOSIN_MODAL_METRICS_PUSH_ENABLE = "false"

modal deploy --env dev (Join-Path $PSScriptRoot "..\modal_app.py")
