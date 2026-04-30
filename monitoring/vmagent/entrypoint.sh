#!/bin/sh
set -eu

BASE_URL="${EOSIN_PARSER_BASE_URL%/}"

if [ -z "$BASE_URL" ]; then
  echo "EOSIN_PARSER_BASE_URL must be set" >&2
  exit 1
fi

mkdir -p /tmp/vmagent-targets

cat > /tmp/vmagent-targets/eosin-targets.json <<EOF
[
  {
    "targets": ["$BASE_URL"],
    "labels": {
      "service": "eosin-bank-parser",
      "environment": "modal",
      "mode": "pull"
    }
  }
]
EOF

cat > /tmp/vmagent-targets/blackbox-health-targets.json <<EOF
[
  {
    "targets": ["$BASE_URL/health"],
    "labels": {
      "service": "eosin-bank-parser",
      "probe": "health"
    }
  }
]
EOF

cat > /tmp/vmagent-targets/blackbox-metrics-targets.json <<EOF
[
  {
    "targets": ["$BASE_URL/metrics"],
    "labels": {
      "service": "eosin-bank-parser",
      "probe": "metrics"
    }
  }
]
EOF

exec /vmagent-prod \
  --promscrape.config=/etc/vmagent/promscrape.yml \
  --remoteWrite.url=http://victoriametrics:8428/api/v1/write \
  --httpListenAddr=:8429
