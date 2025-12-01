#!/usr/bin/env bash
set -euo pipefail
NAMESPACE=${NAMESPACE:-default}
HELM_RELEASE=${HELM_RELEASE:-carbon-kube}
VALUES_PATH=${VALUES_PATH:-../charts/carbon-kube}
RESULTS_DIR=${RESULTS_DIR:-results}
EXPERIMENT_ID=${EXPERIMENT_ID:-exp1}
SPARK_APP_NAME=${SPARK_APP_NAME:-spark-pi}
PROMETHEUS_NS=${PROMETHEUS_NS:-default}
PROMETHEUS_SVC=${PROMETHEUS_SVC:-monitoring-kube-prometheus-prometheus}
PROM_PORT=${PROM_PORT:-9090}
BASELINE_DURATION=${BASELINE_DURATION:-90}
CARBON_DURATION=${CARBON_DURATION:-90}
ELECTRICITYMAPS_SECRET=${ELECTRICITYMAPS_SECRET:-electricitymaps}
ELECTRICITYMAPS_KEY=${ELECTRICITYMAPS_KEY:-auth-token}
ELECTRICITYMAPS_BASEURL=${ELECTRICITYMAPS_BASEURL:-https://api.electricitymap.org/v3}
ZONES=${ZONES:-{US-WECC,US-CAL-CISO}}
CK_IMAGE_REGISTRY=${CK_IMAGE_REGISTRY:-602695720187.dkr.ecr.us-west-2.amazonaws.com}
CK_IMAGE_REPO=${CK_IMAGE_REPO:-carbon-kube}
CK_IMAGE_TAG=${CK_IMAGE_TAG:-latest}
mkdir -p "${RESULTS_DIR}"
wait_for_spark_app() {
  while true; do
    state=$(kubectl get sparkapplications "${SPARK_APP_NAME}" -o jsonpath='{.status.applicationState.state}' 2>/dev/null || echo UNKNOWN)
    if [[ "${state}" == "COMPLETED" || "${state}" == "FAILED" ]]; then
      break
    fi
    sleep 10
  done
}
kubectl -n "${PROMETHEUS_NS}" port-forward "svc/${PROMETHEUS_SVC}" "${PROM_PORT}:9090" >/tmp/pf-prom.log 2>&1 &
PF_PID=$!
sleep 5
PROM_URL="http://localhost:${PROM_PORT}"

# Ensure chart dependencies are fetched
echo "[setup] Fetching chart dependencies"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm repo add grafana https://grafana.github.io/helm-charts >/dev/null 2>&1 || true
helm repo update >/dev/null 2>&1 || true
helm dependency build "${VALUES_PATH}" >/dev/null 2>&1 || true
echo "[baseline] Installing chart with carbon-aware disabled"
helm upgrade "${HELM_RELEASE}" "${VALUES_PATH}" --install --atomic --force --namespace "${NAMESPACE}" \
  --set mutator.enabled=false --set taintController.enabled=false --set poller.enabled=true \
  --set poller.electricityMaps.secretName=${ELECTRICITYMAPS_SECRET} \
  --set poller.electricityMaps.secretKey=${ELECTRICITYMAPS_KEY} \
  --set poller.electricityMaps.baseUrl=${ELECTRICITYMAPS_BASEURL} \
  --set poller.electricityMaps.zones="${ZONES}" \
  --set image.registry="${CK_IMAGE_REGISTRY}" \
  --set image.repository="${CK_IMAGE_REPO}" \
  --set image.tag="${CK_IMAGE_TAG}" \
  --set gpu.nvidia.enabled=false \
  --set monitoring.enabled=false \
  --set monitoring.prometheus.enabled=false \
  --set monitoring.grafana.enabled=false
echo "[baseline] Waiting ${BASELINE_DURATION}s"
sleep ${BASELINE_DURATION}
echo "[baseline] Submitting Spark app: ${SPARK_APP_NAME}"
kubectl delete sparkapplication "${SPARK_APP_NAME}" --ignore-not-found=true
kubectl apply -f https://raw.githubusercontent.com/kubeflow/spark-operator/master/examples/spark-pi.yaml
wait_for_spark_app
echo "[baseline] Collecting metrics"
curl -s "${PROM_URL}/api/v1/query?query=co2_saved_kg_total" > "${RESULTS_DIR}/${EXPERIMENT_ID}_baseline_co2.json"
curl -s "${PROM_URL}/api/v1/query?query=migrations_total" > "${RESULTS_DIR}/${EXPERIMENT_ID}_baseline_migrations.json"
curl -s "${PROM_URL}/api/v1/query?query=latency_increase_percent" > "${RESULTS_DIR}/${EXPERIMENT_ID}_baseline_latency.json"
kubectl get carbonscores -o yaml > "${RESULTS_DIR}/${EXPERIMENT_ID}_baseline_carbonscores.yaml" || true
kubectl get pods -o wide > "${RESULTS_DIR}/${EXPERIMENT_ID}_baseline_pods.txt"
kubectl get nodes -o wide > "${RESULTS_DIR}/${EXPERIMENT_ID}_baseline_nodes.txt"
echo "[carbon] Enabling mutator and taintController"
helm upgrade "${HELM_RELEASE}" "${VALUES_PATH}" --install --atomic --force --namespace "${NAMESPACE}" \
  --set mutator.enabled=true --set taintController.enabled=true --set poller.enabled=true \
  --set poller.electricityMaps.secretName=${ELECTRICITYMAPS_SECRET} \
  --set poller.electricityMaps.secretKey=${ELECTRICITYMAPS_KEY} \
  --set poller.electricityMaps.baseUrl=${ELECTRICITYMAPS_BASEURL} \
  --set poller.electricityMaps.zones="${ZONES}" \
  --set image.registry="${CK_IMAGE_REGISTRY}" \
  --set image.repository="${CK_IMAGE_REPO}" \
  --set image.tag="${CK_IMAGE_TAG}" \
  --set gpu.nvidia.enabled=false \
  --set monitoring.enabled=false \
  --set scheduler.image.repository=registry.k8s.io/kube-scheduler \
  --set scheduler.image.tag=v1.28.4 \
  --set monitoring.prometheus.enabled=false \
  --set monitoring.grafana.enabled=false
echo "[carbon] Waiting ${CARBON_DURATION}s"
sleep ${CARBON_DURATION}
echo "[carbon] Submitting Spark app: ${SPARK_APP_NAME}"
kubectl delete sparkapplication "${SPARK_APP_NAME}" --ignore-not-found=true
kubectl apply -f https://raw.githubusercontent.com/kubeflow/spark-operator/master/examples/spark-pi.yaml
wait_for_spark_app
echo "[carbon] Collecting metrics"
curl -s "${PROM_URL}/api/v1/query?query=co2_saved_kg_total" > "${RESULTS_DIR}/${EXPERIMENT_ID}_carbon_co2.json"
curl -s "${PROM_URL}/api/v1/query?query=migrations_total" > "${RESULTS_DIR}/${EXPERIMENT_ID}_carbon_migrations.json"
curl -s "${PROM_URL}/api/v1/query?query=latency_increase_percent" > "${RESULTS_DIR}/${EXPERIMENT_ID}_carbon_latency.json"
kubectl get carbonscores -o yaml > "${RESULTS_DIR}/${EXPERIMENT_ID}_carbon_carbonscores.yaml" || true
kubectl get pods -o wide > "${RESULTS_DIR}/${EXPERIMENT_ID}_carbon_pods.txt"
kubectl get nodes -o wide > "${RESULTS_DIR}/${EXPERIMENT_ID}_carbon_nodes.txt"
echo "[analysis] Generating comparison and figures"
python3 scripts/analyze_results.py || true
python3 scripts/export_compare.py --results "${RESULTS_DIR}" || true
EVD_DIR="${RESULTS_DIR}/ieee"
mkdir -p "${EVD_DIR}/baseline" "${EVD_DIR}/carbon" "${EVD_DIR}/tables" "${EVD_DIR}/figures" "${EVD_DIR}/artifacts"
cp "${RESULTS_DIR}/${EXPERIMENT_ID}_baseline_"*.json "${EVD_DIR}/baseline/" || true
cp "${RESULTS_DIR}/${EXPERIMENT_ID}_carbon_"*.json "${EVD_DIR}/carbon/" || true
cp "${RESULTS_DIR}/compare.csv" "${EVD_DIR}/tables/" || true
cp "${RESULTS_DIR}/compare.png" "${RESULTS_DIR}/compare.pdf" "${RESULTS_DIR}/plot_"*.png "${EVD_DIR}/figures/" || true
cp "${RESULTS_DIR}/${EXPERIMENT_ID}_"*pods.txt "${RESULTS_DIR}/${EXPERIMENT_ID}_"*nodes.txt "${RESULTS_DIR}/${EXPERIMENT_ID}_"*carbonscores.yaml "${EVD_DIR}/artifacts/" || true
echo "[evidence] Bundle assembled at ${EVD_DIR}"
cat > "${EVD_DIR}/evidence.md" <<EOF
# Experimental Evidence

## Testbed

- Platform: AWS EKS
- Regions: ${ZONES}
- Workload: ${SPARK_APP_NAME}

## Methodology

- Baseline run with carbon-aware features disabled.
- Carbon-aware run with mutator and taint controller enabled.
- Metrics captured from Prometheus.

## Results

- Tables: tables/compare.csv
- Figures: figures/
- Artifacts: artifacts/

## Scoring Formula

score = clamp(w_c*Carbon + w_cost*Cost + w_sla*SLA + w_dg*DataGravity - penalties, 0, 100)
EOF
kill ${PF_PID} || true
