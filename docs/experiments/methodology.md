# Experimental Methodology

## Environment
- Cluster: AWS EKS v1.28
- Monitoring: kube‑prometheus‑stack, Grafana dashboard `grafana/carbon-kube-dashboard.json`
- Image: single multi‑component image `carbon-kube:latest` in ECR

## Workloads
- Spark operator installed via Helm
- Example job: `spark-pi` from upstream examples

## Modes
- Baseline: `mutator.enabled=false`, `taintController.enabled=false`
- Carbon‑Aware: `mutator.enabled=true`, `taintController.enabled=true`

## Procedure
- Deploy cluster and monitoring
- Install Carbon‑Kube chart with ECR image
- Run spark workloads
- Collect Prometheus snapshots for both modes
- Export comparison CSV/figures

## Commands
- Install monitoring:
  - `helm repo add prometheus-community https://prometheus-community.github.io/helm-charts`
  - `helm install monitoring prometheus-community/kube-prometheus-stack`
- Install Carbon‑Kube:
  - `helm upgrade carbon-kube ./deploy/helm --install --set image.repository=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/carbon-kube`
- Baseline:
  - `helm upgrade carbon-kube ./deploy/helm --set mutator.enabled=false --set taintController.enabled=false`
- Carbon‑Aware:
  - `helm upgrade carbon-kube ./deploy/helm --set mutator.enabled=true --set taintController.enabled=true`
- Export:
  - `PROM_PORT=9090 bash scripts/export_results.sh`
  - `python3 scripts/export_compare.py --results results`

## Data Artifacts
- `results/metrics.csv`, `results/metrics.png`
- `results/compare.csv`, `results/compare.png`, `results/compare.pdf`
- `results/pod-placement.txt`, `results/carbon-intensity.json`