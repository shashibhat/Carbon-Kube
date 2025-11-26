#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROM_PORT=${PROM_PORT:-9090}

kubectl port-forward svc/monitoring-kube-prometheus-prometheus ${PROM_PORT}:9090 >/tmp/pf-prom.log 2>&1 &
PF=$!
sleep 5
python3 "${REPO_ROOT}/scripts/export_results.py" --prom "http://localhost:${PROM_PORT}" --out "${REPO_ROOT}/results"
kill $PF || true