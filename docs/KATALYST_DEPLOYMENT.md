# Carbon-Kube + Katalyst Deployment Guide

This guide provides comprehensive instructions for deploying Carbon-Kube with Katalyst-core integration on Kubernetes clusters.

## Prerequisites

### Software Requirements
- Kubernetes 1.24+
- Helm 3.8+
- AWS CLI 2.0+ (for AWS deployments)
- CDK CLI 2.0+ (for AWS CDK deployments)
- Python 3.9+
- Go 1.19+

### Cluster Requirements
- Minimum 3 worker nodes
- 8 CPU cores and 16GB RAM per node
- Container runtime: containerd or CRI-O
- CNI plugin: Calico, Flannel, or Cilium
- Storage class for persistent volumes

## Deployment Options

### Option 1: AWS CDK Deployment (Recommended)

#### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-org/Carbon-Kube.git
cd Carbon-Kube

# Install CDK dependencies
cd cdk
pip install -r requirements.txt

# Configure AWS credentials
aws configure
```

#### 2. Deploy Infrastructure
```bash
# Bootstrap CDK (first time only)
cdk bootstrap

# Deploy primary stack with Katalyst integration
cdk deploy CarbonKubePrimary

# Deploy secondary stack for multi-region testing
cdk deploy CarbonKubeSecondary
```

#### 3. Verify Deployment
```bash
# Get cluster credentials
aws eks update-kubeconfig --region us-west-2 --name carbon-kube-green

# Check Katalyst components
kubectl get pods -n katalyst-system
kubectl get pods -n carbon-kube

# Verify CRDs
kubectl get crd | grep carbon.katalyst.io
```

### Option 2: Helm Deployment

#### 1. Install Katalyst-core
```bash
# Add Katalyst Helm repository
helm repo add kubewharf https://kubewharf.github.io/charts
helm repo update

# Create namespace
kubectl create namespace katalyst-system

# Install Katalyst-core
helm install katalyst-core kubewharf/katalyst-core \
  --namespace katalyst-system \
  --set global.imageTag=v0.4.0 \
  --set katalystCore.enabled=true \
  --set qosManager.enabled=true \
  --set resourceManager.enabled=true \
  --set nodeResourceManager.enabled=true \
  --set scheduler.enabled=true \
  --set webhook.enabled=true
```

#### 2. Install Carbon-Kube with Katalyst Integration
```bash
# Create namespace
kubectl create namespace carbon-kube

# Install Carbon-Kube with Katalyst integration
helm install carbon-kube ./charts/carbon-kube \
  --namespace carbon-kube \
  --set katalyst.enabled=true \
  --set katalyst.carbonQoSController.enabled=true \
  --set katalyst.enhancedScheduler.enabled=true \
  --set katalyst.enhancedRLTuner.enabled=true \
  --set katalyst.crds.enabled=true
```

#### 3. Configure Integration
```bash
# Apply sample QoS profiles
kubectl apply -f examples/katalyst/carbon-qos-profiles.yaml

# Apply node topology configurations
kubectl apply -f examples/katalyst/carbon-node-topology.yaml

# Apply optimization policies
kubectl apply -f examples/katalyst/carbon-optimization-policy.yaml
```

## Configuration

### Carbon QoS Profiles

Create custom QoS profiles based on carbon intensity:

```yaml
apiVersion: carbon.katalyst.io/v1alpha1
kind: CarbonQoSProfile
metadata:
  name: green-guaranteed
  namespace: carbon-kube
spec:
  carbonThreshold: 100.0
  cpuGuarantee: 0.8
  memoryGuarantee: 0.8
  priority: high
  energyEfficiency:
    pueThreshold: 1.2
    powerCapEnabled: true
    thermalManagement: true
  topologyAwareness:
    numaAffinity: true
    gpuLocality: true
    networkOptimization: true
```

### Enhanced Scheduler Configuration

Configure the enhanced scheduler with custom weights:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: katalyst-enhanced-scheduler-config
  namespace: carbon-kube
data:
  config.yaml: |
    apiVersion: kubescheduler.config.k8s.io/v1beta3
    kind: KubeSchedulerConfiguration
    profiles:
    - schedulerName: carbon-katalyst-scheduler
      plugins:
        filter:
          enabled:
          - name: CarbonKatalystPlugin
        score:
          enabled:
          - name: CarbonKatalystPlugin
      pluginConfig:
      - name: CarbonKatalystPlugin
        args:
          weights:
            carbon: 40
            qos: 30
            topology: 20
            energy: 10
          carbonThresholds:
            green: 100
            yellow: 300
            red: 500
```

### RL Tuner Configuration

Configure the enhanced RL tuner for Katalyst integration:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: katalyst-rl-tuner-config
  namespace: carbon-kube
data:
  config.yaml: |
    environment:
      observation_space_size: 20
      action_space_size: 8
      reward_weights:
        carbon_reduction: 0.4
        qos_satisfaction: 0.3
        resource_efficiency: 0.2
        topology_optimization: 0.1
    training:
      algorithm: "SAC"
      learning_rate: 0.0003
      batch_size: 256
      buffer_size: 1000000
      episodes: 1000
    evaluation:
      episodes: 100
      frequency: 50
```

## Monitoring and Observability

### Prometheus Metrics

The integration exposes additional metrics for monitoring:

```yaml
# Carbon QoS Controller metrics
carbon_qos_profiles_total
carbon_qos_reconciliations_total
carbon_qos_node_topology_updates_total

# Enhanced Scheduler metrics
carbon_scheduler_decisions_total
carbon_scheduler_score_distribution
carbon_scheduler_qos_compatibility_checks_total

# Enhanced RL Tuner metrics
carbon_rl_training_episodes_total
carbon_rl_optimization_actions_total
carbon_rl_reward_distribution
```

### Grafana Dashboards

Import the provided Grafana dashboards:

1. **Carbon-Kube + Katalyst Overview**: `monitoring/grafana/katalyst-overview.json`
2. **QoS Performance**: `monitoring/grafana/qos-performance.json`
3. **Scheduler Analytics**: `monitoring/grafana/scheduler-analytics.json`
4. **RL Optimization**: `monitoring/grafana/rl-optimization.json`

### Alerting Rules

Configure Prometheus alerting rules:

```yaml
groups:
- name: carbon-katalyst-alerts
  rules:
  - alert: HighCarbonIntensity
    expr: carbon_intensity_current > 400
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High carbon intensity detected"
      
  - alert: QoSViolation
    expr: carbon_qos_satisfaction_ratio < 0.8
    for: 10m
    labels:
      severity: critical
    annotations:
      summary: "QoS satisfaction below threshold"
```

## Testing and Validation

### Functional Testing

Run the comprehensive test suite:

```bash
# Unit tests
go test ./pkg/katalyst/...
python -m pytest scripts/tests/test_katalyst_rl_tuner.py

# Integration tests
kubectl apply -f tests/katalyst/integration-tests.yaml
kubectl wait --for=condition=complete job/katalyst-integration-test --timeout=300s

# End-to-end tests
./scripts/run-e2e-tests.sh --with-katalyst
```

### Performance Validation

Validate the integration performance:

```bash
# Deploy test workloads
kubectl apply -f examples/katalyst/test-workloads/

# Monitor carbon savings
kubectl get carbonoptimizationpolicy -o wide

# Check QoS compliance
kubectl get carbonqosprofile -o wide

# Verify topology optimization
kubectl get carbonnodetopology -o wide
```

## Troubleshooting

### Common Issues

#### 1. Katalyst Components Not Starting
```bash
# Check Katalyst core status
kubectl get pods -n katalyst-system
kubectl logs -n katalyst-system deployment/katalyst-core

# Verify RBAC permissions
kubectl auth can-i create pods --as=system:serviceaccount:katalyst-system:katalyst-core
```

#### 2. Carbon QoS Controller Issues
```bash
# Check controller logs
kubectl logs -n carbon-kube deployment/carbon-qos-controller

# Verify CRD installation
kubectl get crd carbonqosprofiles.carbon.katalyst.io

# Check service account permissions
kubectl describe serviceaccount carbon-qos-controller -n carbon-kube
```

#### 3. Enhanced Scheduler Not Scheduling
```bash
# Check scheduler logs
kubectl logs -n carbon-kube deployment/carbon-enhanced-scheduler

# Verify scheduler configuration
kubectl get configmap katalyst-enhanced-scheduler-config -n carbon-kube -o yaml

# Check plugin registration
kubectl get events --field-selector reason=FailedScheduling
```

#### 4. RL Tuner Training Issues
```bash
# Check tuner job logs
kubectl logs -n carbon-kube job/katalyst-rl-tuner

# Verify training data
kubectl exec -n carbon-kube deployment/carbon-poller -- cat /data/carbon_intensity.json

# Check model persistence
kubectl get pvc katalyst-rl-models -n carbon-kube
```

### Debug Commands

```bash
# Get comprehensive status
kubectl get all -n katalyst-system
kubectl get all -n carbon-kube

# Check custom resources
kubectl get carbonqosprofiles,carbonnodetopologies,carbonoptimizationpolicies --all-namespaces

# Verify integration health
kubectl get events --sort-by='.lastTimestamp' | grep -i carbon
kubectl get events --sort-by='.lastTimestamp' | grep -i katalyst

# Export configuration for analysis
kubectl get configmap -n carbon-kube -o yaml > carbon-kube-config.yaml
kubectl get configmap -n katalyst-system -o yaml > katalyst-config.yaml
```

## Scaling and Production Considerations

### High Availability

- Deploy multiple replicas of critical components
- Use pod disruption budgets
- Configure anti-affinity rules
- Implement health checks and readiness probes

### Security

- Enable RBAC with least privilege principles
- Use network policies to restrict traffic
- Implement pod security policies
- Regular security scanning of container images

### Performance Optimization

- Tune resource requests and limits
- Configure horizontal pod autoscaling
- Optimize RL training parameters
- Use node affinity for performance-critical workloads

### Backup and Recovery

- Regular backup of CRD configurations
- Export and version control QoS profiles
- Backup RL model artifacts
- Document recovery procedures

## Support and Community

- **Documentation**: [Carbon-Kube Docs](https://carbon-kube.io/docs)
- **Issues**: [GitHub Issues](https://github.com/your-org/Carbon-Kube/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/Carbon-Kube/discussions)
- **Slack**: [#carbon-kube](https://kubernetes.slack.com/channels/carbon-kube)

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to the Carbon-Kube + Katalyst integration.