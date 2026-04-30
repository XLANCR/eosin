# Monitoring Stack

This stack is now **push-only** for the Modal worker.

It exists to persist and visualize metrics that the worker pushes during real request handling. It does **not** scrape or probe the serverless Modal endpoint, because doing that keeps the worker warm and defeats scale-to-zero.

## What it includes

- `victoriametrics`: persistent metrics storage
- `grafana`: dashboards and datasource provisioning
- `alertmanager`: alert routing
- `vmalert`: recording and alert rules over stored metrics

## Serverless rule

Do not point Prometheus-style scrapers at the Modal worker endpoint.

Pull-based monitoring is intentionally disabled for the worker. The Modal container should emit metrics only while it is processing real traffic, then scale back to zero normally.

## Quick start

Start the local stack:

```bash
docker compose -f monitoring/docker-compose.metrics.yml up -d
```

Open Grafana at `http://localhost:3001`.

## Services

- VictoriaMetrics: `http://localhost:8428`
- Grafana: `http://localhost:3001`
- Alertmanager: `http://localhost:9093`
- vmalert: `http://localhost:8880`

## Persistence

All long-lived state is stored in named Docker volumes:

- `vm_data`
- `grafana_data`
- `alertmanager_data`

## Metrics ingest

The Modal worker pushes Prometheus-formatted snapshots directly to VictoriaMetrics using:

- direct HTTP, or
- a Tailscale SOCKS5 transport

For this repo, the recommended production path is Tailscale userspace networking inside the Modal worker, with the worker pushing to your VictoriaMetrics endpoint over the tailnet.

## Current scope

This stack is for:

- storing pushed worker metrics
- dashboarding parser, vLLM, and GPU data
- comparing runs without keeping the worker alive

It is not an uptime checker for the Modal GPU worker. Use Modal’s own control plane or deployment dashboard for that.
