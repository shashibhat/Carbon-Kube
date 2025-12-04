#!/usr/bin/env bash
set -euo pipefail
MODE=${1:-baseline}
NAMESPACE=${NAMESPACE:-default}
CHART=${CHART_PATH:-./charts/carbon-kube}
if [ "$MODE" = "baseline" ]; then
  helm upgrade carbon-kube "$CHART" --install --namespace "$NAMESPACE" \
    --set controllers.enabled=false \
    --set scheduler.enabled=false \
    --set keplerAttr.enabled=false \
    --set mutator.enabled=false
else
  helm upgrade carbon-kube "$CHART" --install --namespace "$NAMESPACE" \
    --set controllers.enabled=true \
    --set keplerAttr.enabled=true
fi
