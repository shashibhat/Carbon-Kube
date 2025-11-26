# Carbon-Kube with Katalyst Integration Examples

This directory contains comprehensive examples and configurations for deploying and testing Carbon-Kube with Katalyst integration. These examples demonstrate carbon-aware scheduling, QoS management, and optimization policies.

## 📁 Files Overview

### Core Configuration Files

- **`carbon-qos-profiles.yaml`** - Defines various QoS profiles for different workload types
- **`carbon-node-topology.yaml`** - Node topology configurations for carbon-aware scheduling
- **`carbon-optimization-policy.yaml`** - Optimization policies for different environments
- **`namespaces.yaml`** - Namespace configurations with carbon optimization labels

### Test Workloads

- **`test-workloads.yaml`** - Comprehensive test workloads demonstrating different QoS profiles
- **`monitoring-setup.yaml`** - Complete monitoring stack with Prometheus, Grafana, and Alertmanager

## 🚀 Quick Start

### 1. Deploy Namespaces

```bash
kubectl apply -f namespaces.yaml
```

### 2. Deploy Carbon QoS Profiles

```bash
kubectl apply -f carbon-qos-profiles.yaml
```

### 3. Deploy Node Topology Configurations

```bash
kubectl apply -f carbon-node-topology.yaml
```

### 4. Deploy Optimization Policies

```bash
kubectl apply -f carbon-optimization-policy.yaml
```

### 5. Deploy Test Workloads

```bash
kubectl apply -f test-workloads.yaml
```

### 6. Set Up Monitoring (Optional)

```bash
kubectl apply -f monitoring-setup.yaml
```

## 📊 QoS Profiles Explained

### Green Guaranteed
- **Use Case**: Critical production workloads
- **Carbon Threshold**: 200 gCO2/kWh
- **CPU/Memory**: Guaranteed resources
- **Priority**: High
- **Energy Efficiency**: High

### Mixed Burstable
- **Use Case**: Web applications with variable load
- **Carbon Threshold**: 350 gCO2/kWh
- **CPU/Memory**: Burstable resources
- **Priority**: Medium
- **Energy Efficiency**: Medium

### Dirty Best Effort
- **Use Case**: Background processing, batch jobs
- **Carbon Threshold**: 500 gCO2/kWh
- **CPU/Memory**: Best effort
- **Priority**: Low
- **Energy Efficiency**: Low

### AI/ML Optimized
- **Use Case**: GPU-intensive training and inference
- **Carbon Threshold**: 300 gCO2/kWh
- **CPU/Memory**: High guaranteed resources
- **Priority**: High
- **Energy Efficiency**: High
- **Special**: GPU topology awareness

### Batch Flexible
- **Use Case**: Flexible batch processing
- **Carbon Threshold**: 400 gCO2/kWh
- **CPU/Memory**: Flexible allocation
- **Priority**: Low
- **Energy Efficiency**: Medium

## 🏗️ Test Workloads

### Production Workloads
- **Green Guaranteed App**: Critical nginx service with high carbon efficiency
- **Mixed Burstable App**: Apache web server with variable resource needs

### Development Workloads
- **Dev Environment**: Node.js development server with flexible scheduling

### AI/ML Workloads
- **AI/ML Training Job**: TensorFlow GPU training job with topology awareness

### Batch Processing
- **Dirty Best Effort App**: Background processing with high migration tolerance
- **Batch Flexible CronJob**: Scheduled data processing with carbon optimization

### Edge Computing
- **Edge Sensor Collector**: DaemonSet for distributed sensor data collection

## 🎯 Optimization Policies

### Production Optimization
- **Carbon Budget**: 100 kg CO2/day
- **Strategy**: Balanced
- **Migration**: Conservative (10/hour)
- **Scheduling**: Green nodes preferred

### Development Optimization
- **Carbon Budget**: 20 kg CO2/day
- **Strategy**: Aggressive
- **Migration**: Liberal (20/hour)
- **Scheduling**: Flexible

### AI/ML Optimization
- **Carbon Budget**: 500 kg CO2/day
- **Strategy**: Conservative
- **Migration**: Limited (5/hour)
- **Scheduling**: GPU topology aware

### Batch Optimization
- **Carbon Budget**: 50 kg CO2/day
- **Strategy**: Aggressive
- **Migration**: High (50/hour)
- **Scheduling**: Highly flexible

### Edge Optimization
- **Carbon Budget**: 30 kg CO2/day
- **Strategy**: Balanced
- **Migration**: Disabled
- **Scheduling**: Location aware

### Global Optimization
- **Carbon Budget**: 1000 kg CO2/day
- **Strategy**: Balanced
- **Migration**: Moderate (25/hour)
- **Scheduling**: Cluster-wide optimization

## 📈 Monitoring and Observability

The monitoring setup includes:

### Prometheus Metrics
- Carbon intensity tracking
- QoS profile violations
- Energy efficiency scores
- Pod migration rates
- Node carbon intensity

### Grafana Dashboards
- Real-time carbon intensity
- Carbon budget usage
- QoS profile distribution
- Energy efficiency trends
- Migration patterns

### Alerting Rules
- High carbon intensity alerts
- Carbon budget exceeded warnings
- QoS profile violations
- Scheduler plugin failures
- Energy efficiency degradation

## 🧪 Testing Scenarios

### Scenario 1: Carbon Intensity Changes
1. Deploy workloads with different QoS profiles
2. Simulate carbon intensity changes
3. Observe workload migrations and scheduling decisions

### Scenario 2: Resource Pressure
1. Deploy resource-intensive workloads
2. Monitor QoS profile enforcement
3. Verify carbon-aware resource allocation

### Scenario 3: Node Failures
1. Simulate node failures
2. Observe carbon-aware rescheduling
3. Verify topology awareness in placement

### Scenario 4: Batch Job Scheduling
1. Deploy flexible batch jobs
2. Monitor carbon-optimized scheduling
3. Verify energy-efficient execution timing

## 🔧 Customization

### Adding New QoS Profiles
1. Define profile in `carbon-qos-profiles.yaml`
2. Update workload annotations
3. Configure optimization policies

### Custom Node Topology
1. Update `carbon-node-topology.yaml`
2. Add node-specific carbon data
3. Configure hardware topology

### Environment-Specific Policies
1. Create new optimization policy
2. Apply to target namespace
3. Configure carbon budgets and strategies

## 📚 Best Practices

### QoS Profile Selection
- Use **Green Guaranteed** for critical services
- Use **Mixed Burstable** for web applications
- Use **Dirty Best Effort** for batch processing
- Use **AI/ML Optimized** for GPU workloads
- Use **Batch Flexible** for delay-tolerant jobs

### Carbon Budget Planning
- Set realistic daily/monthly/annual budgets
- Consider workload criticality
- Plan for peak usage periods
- Monitor and adjust based on actual usage

### Migration Policies
- Conservative for critical workloads
- Aggressive for flexible workloads
- Consider migration costs vs. carbon savings
- Exclude stateful services from frequent migrations

### Monitoring Setup
- Deploy comprehensive monitoring stack
- Set up appropriate alerting thresholds
- Create custom dashboards for your environment
- Regular review of carbon efficiency metrics

## 🆘 Troubleshooting

### Common Issues

1. **Workloads not scheduled**
   - Check QoS profile compatibility
   - Verify node carbon intensity data
   - Review scheduler plugin logs

2. **High migration rates**
   - Adjust migration policy thresholds
   - Review carbon intensity fluctuations
   - Check workload migration tolerance

3. **QoS violations**
   - Review resource requests/limits
   - Check node capacity
   - Verify QoS profile configuration

4. **Monitoring gaps**
   - Verify ServiceMonitor configurations
   - Check Prometheus targets
   - Review metric collection intervals

### Debug Commands

```bash
# Check QoS profiles
kubectl get carbonqosprofile -A

# Check node topology
kubectl get carbonnodetopology -A

# Check optimization policies
kubectl get carbonoptimizationpolicy -A

# View scheduler logs
kubectl logs -n kube-system -l app=carbon-katalyst-scheduler

# Check QoS controller logs
kubectl logs -n carbon-kube -l app=carbon-qos-controller

# Monitor carbon metrics
kubectl port-forward -n monitoring svc/prometheus 9090:9090
```

## 📖 Additional Resources

- [Katalyst Documentation](https://katalyst.kubewharf.io/)
- [Carbon-Kube Architecture](../../docs/ARCHITECTURE.md)
- [Deployment Guide](../../docs/KATALYST_DEPLOYMENT.md)
- [Integration Guide](../../docs/KATALYST_INTEGRATION.md)

## 🤝 Contributing

To contribute additional examples or improvements:

1. Follow the existing YAML structure
2. Include comprehensive labels and annotations
3. Add documentation for new scenarios
4. Test configurations before submitting
5. Update this README with new examples