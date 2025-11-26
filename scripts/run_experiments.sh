#!/usr/bin/env bash
set -euo pipefail

# -------------------------------
# CONFIG
# -------------------------------
NAMESPACE=${NAMESPACE:-default}
HELM_RELEASE=${HELM_RELEASE:-carbon-kube}
VALUES_PATH="../deploy/helm"
RESULTS_DIR=${RESULTS_DIR:-results}
EXPERIMENT_ID=${EXPERIMENT_ID:-exp1}
SPARK_APP_NAME=${SPARK_APP_NAME:-spark-pi}

# Prometheus access config
PROMETHEUS_NS=${PROMETHEUS_NS:-default}
PROMETHEUS_SVC=${PROMETHEUS_SVC:-monitoring-kube-prometheus-prometheus}
PROM_PORT=${PROM_PORT:-9090}

# Experiment duration
BASELINE_DURATION=${BASELINE_DURATION:-90}
CARBON_DURATION=${CARBON_DURATION:-90}

# ElectricityMaps config (required for your poller)
ELECTRICITYMAPS_SECRET=electricitymaps
ELECTRICITYMAPS_KEY=auth-token
ELECTRICITYMAPS_BASEURL="https://api.electricitymap.org/v3"
ZONES="{US-WECC,US-CAL-CISO}"

# Your image
CK_IMAGE_REPO="602695720187.dkr.ecr.us-west-2.amazonaws.com/carbon-kube"
CK_IMAGE_TAG="latest"

mkdir -p "${RESULTS_DIR}"

echo "[INFO] Starting experiment: ${EXPERIMENT_ID}"
echo "[INFO] Using zones=${ZONES}"
echo "[INFO] Results → ${RESULTS_DIR}/"

# -------------------------------
# Helper: wait for Spark to finish
# -------------------------------
wait_for_spark_app() {
  echo "[INFO] Waiting for SparkApplication ${SPARK_APP_NAME} to complete..."
  while true; do
    state=$(kubectl get sparkapplications "${SPARK_APP_NAME}" -o jsonpath='{.status.applicationState.state}' 2>/dev/null || echo "UNKNOWN")
    echo "[DEBUG] Spark state: ${state}"
    if [[ "${state}" == "COMPLETED" || "${state}" == "FAILED" ]]; then
      echo "[INFO] SparkApplication finished: ${state}"
      break
    fi
    sleep 10
  done
}

# -------------------------------
# Start Prometheus port-forward
# -------------------------------
echo "[INFO] Port-forwarding Prometheus..."
kubectl -n "${PROMETHEUS_NS}" port-forward "svc/${PROMETHEUS_SVC}" "${PROM_PORT}:9090" >/tmp/pf-prom.log 2>&1 &
PF_PID=$!
sleep 5
PROM_URL="http://localhost:${PROM_PORT}"

# -------------------------------
# 1. BASELINE RUN
# -------------------------------
echo "[INFO] ===== BASELINE RUN (carbon-kube disabled) ====="

helm upgrade "${HELM_RELEASE}" "${VALUES_PATH}" \
  --install \
  --namespace "${NAMESPACE}" \
  --set mutator.enabled=false \
  --set taintController.enabled=false \
  --set poller.enabled=true \
  --set poller.electricityMaps.secretName=${ELECTRICITYMAPS_SECRET} \
  --set poller.electricityMaps.secretKey=${ELECTRICITYMAPS_KEY} \
  --set poller.electricityMaps.baseUrl=${ELECTRICITYMAPS_BASEURL} \
  --set poller.electricityMaps.zones="${ZONES}" \
  --set image.repository="${CK_IMAGE_REPO}" \
  --set image.tag="${CK_IMAGE_TAG}"

echo "[INFO] Sleeping ${BASELINE_DURATION}s (baseline stabilization)..."
sleep ${BASELINE_DURATION}

echo "[INFO] Launching Spark job (baseline)..."
kubectl delete sparkapplication "${SPARK_APP_NAME}" --ignore-not-found=true
kubectl apply -f https://raw.githubusercontent.com/kubeflow/spark-operator/master/examples/spark-pi.yaml
wait_for_spark_app

# Capture baseline metrics
echo "[INFO] Collecting baseline metrics..."
curl -s "${PROM_URL}/api/v1/query?query=co2_saved_kg_total" \
  > "${RESULTS_DIR}/${EXPERIMENT_ID}_baseline_co2.json"
curl -s "${PROM_URL}/api/v1/query?query=migrations_total" \
  > "${RESULTS_DIR}/${EXPERIMENT_ID}_baseline_migrations.json"
curl -s "${PROM_URL}/api/v1/query?query=latency_increase_percent" \
  > "${RESULTS_DIR}/${EXPERIMENT_ID}_baseline_latency.json"

kubectl get carbonscores -o yaml > "${RESULTS_DIR}/${EXPERIMENT_ID}_baseline_carbonscores.yaml"
kubectl get pods -o wide         > "${RESULTS_DIR}/${EXPERIMENT_ID}_baseline_pods.txt"
kubectl get nodes -o wide        > "${RESULTS_DIR}/${EXPERIMENT_ID}_baseline_nodes.txt"

# -------------------------------
# 2. CARBON-AWARE RUN
# -------------------------------
echo "[INFO] ===== CARBON-AWARE RUN (carbon-kube enabled) ====="

helm upgrade "${HELM_RELEASE}" "${VALUES_PATH}" \
  --install \
  --namespace "${NAMESPACE}" \
  --set mutator.enabled=true \
  --set taintController.enabled=true \
  --set poller.enabled=true \
  --set poller.electricityMaps.secretName=${ELECTRICITYMAPS_SECRET} \
  --set poller.electricityMaps.secretKey=${ELECTRICITYMAPS_KEY} \
  --set poller.electricityMaps.baseUrl=${ELECTRICITYMAPS_BASEURL} \
  --set poller.electricityMaps.zones="${ZONES}" \
  --set image.repository="${CK_IMAGE_REPO}" \
  --set image.tag="${CK_IMAGE_TAG}"

echo "[INFO] Waiting ${CARBON_DURATION}s to let taints/mutator influence scheduling..."
sleep ${CARBON_DURATION}

echo "[INFO] Launching Spark job (carbon-aware)..."
kubectl delete sparkapplication "${SPARK_APP_NAME}" --ignore-not-found=true
kubectl apply -f https://raw.githubusercontent.com/kubeflow/spark-operator/master/examples/spark-pi.yaml
wait_for_spark_app

echo "[INFO] Collecting carbon-aware metrics..."
curl -s "${PROM_URL}/api/v1/query?query=co2_saved_kg_total" \
  > "${RESULTS_DIR}/${EXPERIMENT_ID}_carbon_co2.json"
curl -s "${PROM_URL}/api/v1/query?query=migrations_total" \
  > "${RESULTS_DIR}/${EXPERIMENT_ID}_carbon_migrations.json"
curl -s "${PROM_URL}/api/v1/query?query=latency_increase_percent" \
  > "${RESULTS_DIR}/${EXPERIMENT_ID}_carbon_latency.json"

kubectl get carbonscores -o yaml > "${RESULTS_DIR}/${EXPERIMENT_ID}_carbon_carbonscores.yaml"
kubectl get pods -o wide         > "${RESULTS_DIR}/${EXPERIMENT_ID}_carbon_pods.txt"
kubectl get nodes -o wide        > "${RESULTS_DIR}/${EXPERIMENT_ID}_carbon_nodes.txt"

# -------------------------------
# Cleanup
# -------------------------------
echo "[INFO] Stopping Prometheus port-forward"
kill ${PF_PID} || true

echo "[INFO] Experiment complete. Data stored in: ${RESULTS_DIR}/"