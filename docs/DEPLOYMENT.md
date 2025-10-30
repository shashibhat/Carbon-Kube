# Carbon-Kube Deployment Guide

This guide provides comprehensive instructions for deploying Carbon-Kube in various environments, from local development to production multi-cloud setups.

## 📋 Prerequisites

### System Requirements

- **Kubernetes**: v1.20+ (tested with v1.28)
- **Helm**: v3.8+
- **kubectl**: Compatible with your cluster version
- **Docker**: v20.10+ (for building images)
- **Go**: v1.21+ (for development)
- **Python**: v3.9+ (for scripts and testing)

### API Keys (Optional but Recommended)

- **Electricity Maps API**: For real-time carbon intensity data
- **NOAA API**: For weather-based carbon estimation
- **AWS API**: For cloud provider carbon data

## 🚀 Quick Start Deployment

### 1. Basic Installation

```bash
# Clone the repository
git clone https://github.com/your-org/carbon-kube.git
cd carbon-kube

# Create namespace
kubectl create namespace carbon-kube

# Install with default values
helm install carbon-kube ./charts/carbon-kube \
  --namespace carbon-kube \
  --wait
```

### 2. Installation with API Keys

```bash
# Install with API configuration
helm install carbon-kube ./charts/carbon-kube \
  --namespace carbon-kube \
  --set electricityMaps.apiKey="your-electricity-maps-key" \
  --set noaa.apiKey="your-noaa-key" \
  --set aws.accessKeyId="your-aws-key" \
  --set aws.secretAccessKey="your-aws-secret" \
  --wait
```

### 3. Verify Installation

```bash
# Check pod status
kubectl get pods -n carbon-kube

# Check scheduler registration
kubectl get schedulers

# View carbon intensity data
kubectl get configmap carbon-intensity-data -n carbon-kube -o yaml
```

## 🔧 Configuration Options

### Helm Values Configuration

Create a custom `values.yaml` file:

```yaml
# values-production.yaml
global:
  debug: false
  namespace: carbon-kube

scheduler:
  replicas: 3
  image:
    tag: "v1.0.0"
  resources:
    requests:
      cpu: 100m
      memory: 128Mi
    limits:
      cpu: 500m
      memory: 512Mi

carbonPoller:
  schedule: "*/5 * * * *"  # Every 5 minutes
  resources:
    requests:
      cpu: 50m
      memory: 64Mi
    limits:
      cpu: 200m
      memory: 256Mi

rlTuner:
  enabled: true
  schedule: "0 */6 * * *"  # Every 6 hours
  model:
    algorithm: "SAC"
    learning_rate: 0.0003
    batch_size: 64

electricityMaps:
  apiKey: "your-api-key"
  zones: ["US-CA", "US-TX", "EU-DE", "EU-FR"]

threshold: 250  # gCO2/kWh

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
    adminPassword: "secure-password"

rbac:
  create: true

serviceAccount:
  create: true
```

Deploy with custom configuration:

```bash
helm install carbon-kube ./charts/carbon-kube \
  --namespace carbon-kube \
  --values values-production.yaml \
  --wait
```

## 🏗️ Environment-Specific Deployments

### Development Environment

```bash
# Development setup with debug enabled
helm install carbon-kube ./charts/carbon-kube \
  --namespace carbon-kube-dev \
  --set global.debug=true \
  --set scheduler.replicas=1 \
  --set carbonPoller.schedule="*/1 * * * *" \
  --set rlTuner.enabled=false \
  --create-namespace
```

### Staging Environment

```bash
# Staging setup with monitoring
helm install carbon-kube ./charts/carbon-kube \
  --namespace carbon-kube-staging \
  --set scheduler.replicas=2 \
  --set monitoring.prometheus.enabled=true \
  --set monitoring.grafana.enabled=true \
  --set electricityMaps.apiKey="${ELECTRICITY_MAPS_API_KEY}" \
  --create-namespace
```

### Production Environment

```bash
# Production setup with high availability
helm install carbon-kube ./charts/carbon-kube \
  --namespace carbon-kube \
  --values values-production.yaml \
  --set scheduler.replicas=3 \
  --set scheduler.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].labelSelector.matchLabels.app=carbon-kube-scheduler \
  --set scheduler.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].topologyKey=kubernetes.io/hostname \
  --create-namespace
```

## ☁️ Multi-Cloud Deployment

### AWS EKS Deployment

#### 1. Infrastructure Setup with CDK

```bash
# Install CDK dependencies
cd cdk
pip install -r requirements.txt

# Bootstrap CDK (one-time setup)
cdk bootstrap

# Deploy primary cluster
cdk deploy CarbonKubeStack-Primary \
  --parameters ClusterName=carbon-kube-primary \
  --parameters Region=us-west-2

# Deploy secondary cluster
cdk deploy CarbonKubeStack-Secondary \
  --parameters ClusterName=carbon-kube-secondary \
  --parameters Region=us-east-1
```

#### 2. Application Deployment

```bash
# Configure kubectl for primary cluster
aws eks update-kubeconfig --region us-west-2 --name carbon-kube-primary

# Deploy to primary cluster
helm install carbon-kube ./charts/carbon-kube \
  --namespace carbon-kube \
  --set electricityMaps.zones=["US-CA","US-NW"] \
  --set aws.region="us-west-2" \
  --create-namespace

# Configure kubectl for secondary cluster
aws eks update-kubeconfig --region us-east-1 --name carbon-kube-secondary

# Deploy to secondary cluster
helm install carbon-kube ./charts/carbon-kube \
  --namespace carbon-kube \
  --set electricityMaps.zones=["US-NY","US-NE"] \
  --set aws.region="us-east-1" \
  --create-namespace
```

### Google GKE Deployment

```bash
# Create GKE cluster
gcloud container clusters create carbon-kube-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10

# Get credentials
gcloud container clusters get-credentials carbon-kube-cluster --zone us-central1-a

# Deploy Carbon-Kube
helm install carbon-kube ./charts/carbon-kube \
  --namespace carbon-kube \
  --set electricityMaps.zones=["US-MIDW"] \
  --set gcp.project="your-project-id" \
  --create-namespace
```

### Azure AKS Deployment

```bash
# Create resource group
az group create --name carbon-kube-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group carbon-kube-rg \
  --name carbon-kube-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group carbon-kube-rg --name carbon-kube-cluster

# Deploy Carbon-Kube
helm install carbon-kube ./charts/carbon-kube \
  --namespace carbon-kube \
  --set electricityMaps.zones=["US-MIDA"] \
  --set azure.subscriptionId="your-subscription-id" \
  --create-namespace
```

## 🔒 Security Configuration

### RBAC Setup

Carbon-Kube requires specific permissions. The Helm chart creates appropriate RBAC resources:

```yaml
# Scheduler permissions
- apiGroups: [""]
  resources: ["nodes", "pods", "configmaps"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["events"]
  verbs: ["create"]

# Poller permissions
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]

# RL Tuner permissions
- apiGroups: [""]
  resources: ["pods", "events"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list", "watch", "update", "patch"]
```

### Secret Management

#### Using Kubernetes Secrets

```bash
# Create secrets manually
kubectl create secret generic carbon-kube-secrets \
  --from-literal=electricity-maps-key="your-key" \
  --from-literal=noaa-key="your-key" \
  --from-literal=aws-access-key-id="your-key" \
  --from-literal=aws-secret-access-key="your-secret" \
  -n carbon-kube
```

#### Using External Secrets Operator

```yaml
# external-secret.yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: carbon-kube-secrets
  namespace: carbon-kube
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: carbon-kube-secrets
    creationPolicy: Owner
  data:
  - secretKey: electricity-maps-key
    remoteRef:
      key: carbon-kube/api-keys
      property: electricity-maps
```

### Network Policies

Enable network policies for enhanced security:

```yaml
# values.yaml
networkPolicy:
  enabled: true
  ingress:
    - from:
      - namespaceSelector:
          matchLabels:
            name: kube-system
      - namespaceSelector:
          matchLabels:
            name: monitoring
  egress:
    - to: []
      ports:
      - protocol: TCP
        port: 443  # HTTPS for API calls
      - protocol: TCP
        port: 53   # DNS
      - protocol: UDP
        port: 53   # DNS
```

## 📊 Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus-values.yaml
prometheus:
  prometheusSpec:
    additionalScrapeConfigs:
    - job_name: 'carbon-kube-scheduler'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - carbon-kube
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: carbon-kube-scheduler
```

### Grafana Dashboard Import

```bash
# Import dashboards via ConfigMap
kubectl create configmap carbon-kube-dashboards \
  --from-file=charts/carbon-kube/dashboards/ \
  -n monitoring

# Or import via Grafana UI
# Navigate to Grafana -> Dashboards -> Import
# Upload the JSON files from charts/carbon-kube/dashboards/
```

## 🔄 Upgrade and Rollback

### Upgrading Carbon-Kube

```bash
# Upgrade to new version
helm upgrade carbon-kube ./charts/carbon-kube \
  --namespace carbon-kube \
  --values values-production.yaml \
  --set image.tag="v1.1.0"

# Check upgrade status
helm status carbon-kube -n carbon-kube

# View upgrade history
helm history carbon-kube -n carbon-kube
```

### Rolling Back

```bash
# Rollback to previous version
helm rollback carbon-kube 1 -n carbon-kube

# Rollback to specific revision
helm rollback carbon-kube 2 -n carbon-kube
```

## 🧪 Testing Deployment

### Smoke Tests

```bash
# Test scheduler functionality
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
  namespace: carbon-kube
spec:
  schedulerName: carbon-kube-scheduler
  containers:
  - name: test
    image: nginx:alpine
    resources:
      requests:
        cpu: 100m
        memory: 128Mi
EOF

# Check if pod was scheduled
kubectl get pod test-pod -n carbon-kube -o wide

# Clean up
kubectl delete pod test-pod -n carbon-kube
```

### Integration Tests

```bash
# Run integration test suite
cd test/integration
python -m pytest test_e2e.py -v
```

### Performance Tests

```bash
# Run performance benchmarks
cd test/integration
python -m pytest test_e2e.py::TestCarbonKubePerformance -v
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Scheduler Not Registering

```bash
# Check scheduler logs
kubectl logs -n carbon-kube deployment/carbon-kube-scheduler

# Verify scheduler configuration
kubectl get configmap scheduler-config -n carbon-kube -o yaml

# Check RBAC permissions
kubectl auth can-i get nodes --as=system:serviceaccount:carbon-kube:carbon-kube-scheduler
```

#### 2. Carbon Data Not Updating

```bash
# Check poller logs
kubectl logs -n carbon-kube cronjob/carbon-kube-poller

# Verify API keys
kubectl get secret carbon-kube-secrets -n carbon-kube -o yaml

# Check ConfigMap data
kubectl get configmap carbon-intensity-data -n carbon-kube -o yaml
```

#### 3. RL Tuner Not Training

```bash
# Check RL tuner logs
kubectl logs -n carbon-kube cronjob/carbon-kube-rl-tuner

# Verify model storage
kubectl get pvc -n carbon-kube

# Check training data
kubectl exec -n carbon-kube deployment/carbon-kube-scheduler -- ls -la /app/models/
```

### Debug Mode

Enable debug logging:

```bash
helm upgrade carbon-kube ./charts/carbon-kube \
  --namespace carbon-kube \
  --set global.debug=true \
  --reuse-values
```

### Collecting Diagnostics

```bash
# Collect all relevant information
kubectl get all -n carbon-kube > carbon-kube-diagnostics.txt
kubectl describe pods -n carbon-kube >> carbon-kube-diagnostics.txt
kubectl get events -n carbon-kube --sort-by='.lastTimestamp' >> carbon-kube-diagnostics.txt
kubectl logs -n carbon-kube deployment/carbon-kube-scheduler >> carbon-kube-diagnostics.txt
```

## 📈 Scaling and Performance

### Horizontal Scaling

```yaml
# Enable HPA
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

### Vertical Scaling

```yaml
# Adjust resource limits
scheduler:
  resources:
    requests:
      cpu: 200m
      memory: 256Mi
    limits:
      cpu: 1000m
      memory: 1Gi
```

### Performance Tuning

```yaml
# Optimize for high-throughput clusters
scheduler:
  config:
    profiles:
    - schedulerName: carbon-kube-scheduler
      plugins:
        score:
          enabled:
          - name: CarbonAwareScore
            weight: 70
        filter:
          enabled:
          - name: CarbonAwareFilter
      pluginConfig:
      - name: CarbonAwareScore
        args:
          scoringStrategy: "exponential"
          cacheTimeout: "30s"
```

## 🔄 Backup and Recovery

### Configuration Backup

```bash
# Backup Helm values
helm get values carbon-kube -n carbon-kube > carbon-kube-values-backup.yaml

# Backup ConfigMaps
kubectl get configmap -n carbon-kube -o yaml > carbon-kube-configmaps-backup.yaml

# Backup Secrets (be careful with sensitive data)
kubectl get secret -n carbon-kube -o yaml > carbon-kube-secrets-backup.yaml
```

### Disaster Recovery

```bash
# Restore from backup
kubectl apply -f carbon-kube-configmaps-backup.yaml
kubectl apply -f carbon-kube-secrets-backup.yaml

# Reinstall with backed up values
helm install carbon-kube ./charts/carbon-kube \
  --namespace carbon-kube \
  --values carbon-kube-values-backup.yaml
```

This deployment guide provides comprehensive instructions for deploying Carbon-Kube in various environments. For additional support, refer to the troubleshooting section or open an issue in the project repository.