#!/usr/bin/env bash
set -euo pipefail

REGION=${REGION:-us-west-2}
ACCOUNT_ID=${ACCOUNT_ID:-}
CLUSTER_NAME=${CLUSTER_NAME:-CarbonKubeStack-Cluster}

if [[ -z "$ACCOUNT_ID" ]]; then
  ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
fi

aws ecr create-repository --region "$REGION" --repository-name carbon-kube || true
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

docker build -t carbon-kube:latest .
docker tag carbon-kube:latest "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/carbon-kube:latest"
docker push "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/carbon-kube:latest"

pushd deploy/cdk
pip3 install -r requirements.txt
cdk bootstrap
cdk deploy --require-approval never
popd

aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION"

helm upgrade carbon-kube ./deploy/helm --install \
  --set image.repository="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/carbon-kube"

helm repo add spark https://googlecloudplatform.github.io/spark-on-k8s-operator || true
helm install spark spark/spark-operator || true
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/spark-on-k8s-operator/master/examples/spark-pi.yaml || true

mkdir -p results

kubectl port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090 >/tmp/pf-prom.log 2>&1 &
PF_PID=$!
sleep 5

helm upgrade carbon-kube ./deploy/helm \
  --set mutator.enabled=false --set taintController.enabled=false --set poller.enabled=true
sleep 120
curl -s "http://localhost:9090/api/v1/query?query=co2_saved_kg_total" > results/baseline_co2.json
curl -s "http://localhost:9090/api/v1/query?query=migrations_total" > results/baseline_migrations.json
curl -s "http://localhost:9090/api/v1/query?query=latency_increase_percent" > results/baseline_latency.json

helm upgrade carbon-kube ./deploy/helm \
  --set mutator.enabled=true --set taintController.enabled=true --set poller.enabled=true
sleep 120
curl -s "http://localhost:9090/api/v1/query?query=co2_saved_kg_total" > results/carbon_co2.json
curl -s "http://localhost:9090/api/v1/query?query=migrations_total" > results/carbon_migrations.json
curl -s "http://localhost:9090/api/v1/query?query=latency_increase_percent" > results/carbon_latency.json

kubectl get carbonscores -o json > results/carbon-intensity.json || true
kubectl get pods -o wide > results/pod-placement.txt

kill "$PF_PID" || true