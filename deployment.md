
# Carbon‑Kube Deployment Guide  
### Production Deployment on AWS EKS (GitHub‑Style)

This document provides **complete, reproducible steps** to deploy Carbon‑Kube on AWS EKS using Docker, Helm, and AWS CDK.  
It also includes instructions for **IEEE‑grade experimentation and metrics collection**.

---

# 1. Prerequisites

### Tools
```
awscli
kubectl 1.28+
helm 3+
docker
cdk v2
node/npm (for CDK CLI)
python3
```

### AWS Requirements
- IAM permissions for EKS, EC2, ECR, CloudFormation  
- Default VPC or CDK-created VPC  

---

# 2. Build & Push Docker Image

```
REGION=us-west-2
ACCOUNT_ID=<your-account-id>

aws ecr create-repository --region $REGION --repository-name carbon-kube
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

docker build -t carbon-kube .
docker tag carbon-kube:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/carbon-kube:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/carbon-kube:latest
```

Update Helm values:

```yaml
image:
  repository: <acct>.dkr.ecr.us-west-2.amazonaws.com/carbon-kube
  tag: latest
```

---

# 3. Deploy EKS via AWS CDK

```
cd deploy/cdk
pip install -r requirements.txt
cdk bootstrap
cdk deploy

aws eks update-kubeconfig --name <cluster-name> --region $REGION
```

This provisions:

- EKS cluster  
- Node groups  
- IAM roles  
- Applies Carbon‑Kube Helm chart  

---

# 4. Configure Carbon‑Kube via Helm

```
helm upgrade carbon-kube ./deploy/helm --install
```

Components deployed:
- Poller  
- Scheduler Mutator  
- Taint Controller  
- Metrics server  
- CRD  

Validate:

```
kubectl get pods | grep carbon-kube
```

---

# 5. Install Monitoring (Prometheus + Grafana)

```
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack
```

Port‑forward Grafana:

```
kubectl port-forward svc/monitoring-grafana 3000:80
```

Import the dashboard:

```
grafana/carbon-kube-dashboard.json
```

---

# 6. Running Workloads

Example: Spark job

```
helm repo add spark https://googlecloudplatform.github.io/spark-on-k8s-operator
helm install spark spark/spark-operator
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/spark-on-k8s-operator/master/examples/spark-pi.yaml
```

---

# 7. Baseline vs Carbon‑Aware Experiments

### Baseline Mode
```
helm upgrade carbon-kube ./deploy/helm \
  --set mutator.enabled=false \
  --set taintController.enabled=false
```

### Carbon‑Aware Mode
```
helm upgrade carbon-kube ./deploy/helm \
  --set mutator.enabled=true \
  --set taintController.enabled=true
```

Record:
- CO₂ saved  
- Job latency impact  
- AZ distribution  
- Migration events  

---

# 8. Exporting Data for IEEE Paper

### Prometheus Export
```
curl http://localhost:9090/api/v1/query?query=co2_saved_kg_total > co2.json
```

### CarbonScore CRD
```
kubectl get carbonscores -o json > carbon-intensity.json
```

### Node allocation
```
kubectl get pods -o wide > pod-placement.txt
```

---

# 9. Cleanup

```
helm uninstall carbon-kube
helm uninstall monitoring
helm uninstall spark
cd deploy/cdk
cdk destroy
```

---

# 10. Summary

This guide provides all required steps to:

- Deploy Carbon‑Kube in production  
- Run scientific evaluations  
- Capture Prometheus metrics  
- Prepare CO₂ reduction data for IEEE publication  

Need help generating **evaluation.md**, **IEEE LaTeX template**, or **automation scripts**?  
Ask anytime!
Port‑forward Prometheus:

```
kubectl port-forward svc/monitoring-kube-prometheus-prometheus 9090:9090
```

