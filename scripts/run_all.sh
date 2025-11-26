#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE=${ENV_FILE:-"${SCRIPT_DIR}/aws.env"}
export JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION=1

if [[ ! -f "$ENV_FILE" ]]; then
  AWS_PROFILE=${AWS_PROFILE:-default}
  AWS_REGION=${AWS_REGION:-us-west-2}
  export AWS_PROFILE
  ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
  echo "AWS_PROFILE=$AWS_PROFILE" > "$ENV_FILE"
  echo "AWS_REGION=$AWS_REGION" >> "$ENV_FILE"
  echo "AWS_ACCOUNT_ID=$ACCOUNT_ID" >> "$ENV_FILE"
  echo "EKS_CLUSTER_NAME=CarbonKubeStack-Cluster" >> "$ENV_FILE"
  echo "ECR_REPO_NAME=carbon-kube" >> "$ENV_FILE"
  echo "HELM_RELEASE=carbon-kube" >> "$ENV_FILE"
  echo "GRAFANA_PORT=3000" >> "$ENV_FILE"
  echo "PROMETHEUS_PORT=9090" >> "$ENV_FILE"
fi

set -a
source "$ENV_FILE"
set +a
export AWS_PROFILE

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed or not in PATH. Please install Docker before running." >&2
  exit 1
fi

# aws ecr create-repository --region "$AWS_REGION" --repository-name "$ECR_REPO_NAME" || true
# aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# docker build -t "$ECR_REPO_NAME:latest" "$REPO_ROOT"
# docker tag "$ECR_REPO_NAME:latest" "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest"
# docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME:latest"

# pushd "$REPO_ROOT/deploy/cdk" >/dev/null
# if command -v pip3 >/dev/null 2>&1; then
#   pip3 install -r requirements.txt
# else
#   python3 -m pip install -r requirements.txt
# fi
# if ! command -v cdk >/dev/null 2>&1; then
#   if command -v brew >/dev/null 2>&1; then
#     brew install aws-cdk
#   elif command -v npm >/dev/null 2>&1; then
#     npm install -g aws-cdk
#   else
#     echo "AWS CDK CLI not found. Install with 'brew install aws-cdk' or 'npm install -g aws-cdk'." >&2
#     exit 1
#   fi
# fi
# cdk bootstrap aws://602695720187/us-west-2 --profile default
# cdk deploy --app "python3 app.py" --require-approval never
#popd >/dev/null

aws eks update-kubeconfig --name "$EKS_CLUSTER_NAME" --region "$AWS_REGION"

helm upgrade "$HELM_RELEASE" "$REPO_ROOT/deploy/helm" --install \
  --set image.repository="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME"

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts || true
helm install monitoring prometheus-community/kube-prometheus-stack || true

helm repo add spark https://googlecloudplatform.github.io/spark-on-k8s-operator || true
helm install spark spark/spark-operator || true
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/spark-on-k8s-operator/master/examples/spark-pi.yaml || true

mkdir -p results

kubectl port-forward svc/monitoring-kube-prometheus-prometheus "$PROMETHEUS_PORT:9090" >/tmp/pf-prom.log 2>&1 &
PF_PID=$!
sleep 5

helm upgrade "$HELM_RELEASE" "$REPO_ROOT/deploy/helm" \
  --set mutator.enabled=false --set taintController.enabled=false --set poller.enabled=true
sleep 120
curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=co2_saved_kg_total" > results/baseline_co2.json
curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=migrations_total" > results/baseline_migrations.json
curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=latency_increase_percent" > results/baseline_latency.json

helm upgrade "$HELM_RELEASE" "$REPO_ROOT/deploy/helm" \
  --set mutator.enabled=true --set taintController.enabled=true --set poller.enabled=true
sleep 120
curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=co2_saved_kg_total" > results/carbon_co2.json
curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=migrations_total" > results/carbon_migrations.json
curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=latency_increase_percent" > results/carbon_latency.json

kubectl get carbonscores -o json > results/carbon-intensity.json || true
kubectl get pods -o wide > results/pod-placement.txt

kill "$PF_PID" || true

python3 "$REPO_ROOT/scripts/export_compare.py" --results "$REPO_ROOT/results"