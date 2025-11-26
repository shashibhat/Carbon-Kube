# NVIDIA Operator Subchart for Carbon-Kube

This subchart provides NVIDIA GPU Operator integration for Carbon-Kube with enhanced carbon-aware GPU monitoring and scheduling capabilities.

## Features

- **DCGM Integration**: Real-time GPU metrics collection for carbon monitoring
- **MIG Support**: Multi-Instance GPU support with configurable strategies
- **Carbon-Aware Scheduling**: GPU power consumption and utilization tracking
- **Prometheus Integration**: GPU metrics exported for monitoring and alerting
- **Comprehensive GPU Support**: Support for H100, A100, V100, T4, RTX series, and more

## Configuration

### Basic GPU Support

To enable GPU support in Carbon-Kube:

```yaml
gpu:
  nvidia:
    enabled: true
```

### Advanced Configuration

For full GPU configuration with DCGM and MIG support, use `values-gpu.yaml`:

```bash
helm install carbon-kube ./charts/carbon-kube -f ./charts/carbon-kube/charts/nvidia-operator/values-gpu.yaml
```

### Key Configuration Options

#### DCGM Exporter
```yaml
dcgmExporter:
  enabled: true
  service:
    port: 9400
  config:
    # Carbon-aware metrics collection
    data: |
      DCGM_FI_DEV_POWER_USAGE,      gauge, Power draw (in W).
      DCGM_FI_DEV_GPU_UTIL,         gauge, GPU utilization (in %).
      DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION, counter, Total energy consumption.
```

#### MIG Configuration
```yaml
migManager:
  enabled: true
  config:
    data:
      config.yaml: |
        version: v1
        mig-configs:
          all-1g.10gb:
            - devices: all
              mig-enabled: true
              mig-devices:
                "1g.10gb": 7
        default-config: "all-disabled"
```

#### Device Plugin
```yaml
devicePlugin:
  enabled: true
  config:
    data:
      any: |-
        version: v1
        sharing:
          timeSlicing:
            resources:
            - name: nvidia.com/gpu
              replicas: 1
        migStrategy: single
```

## GPU Metrics

The subchart exposes the following carbon-relevant GPU metrics:

- `DCGM_FI_DEV_POWER_USAGE`: Real-time power consumption (Watts)
- `DCGM_FI_DEV_GPU_UTIL`: GPU utilization percentage
- `DCGM_FI_DEV_MEM_COPY_UTIL`: Memory utilization percentage
- `DCGM_FI_DEV_GPU_TEMP`: GPU temperature (Celsius)
- `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION`: Cumulative energy consumption (mJ)
- `DCGM_FI_DEV_SM_CLOCK`: SM clock frequency (MHz)
- `DCGM_FI_DEV_MEM_CLOCK`: Memory clock frequency (MHz)

## Prerequisites

- Kubernetes 1.20+
- NVIDIA GPU nodes with compatible drivers
- Helm 3.0+
- Prometheus Operator (for ServiceMonitor)

## Installation

### 1. Enable GPU Support

```bash
# Install with GPU support enabled
helm install carbon-kube ./charts/carbon-kube \
  --set gpu.nvidia.enabled=true \
  --set gpu.monitoring.dcgm.enabled=true
```

### 2. Full GPU Configuration

```bash
# Install with comprehensive GPU configuration
helm install carbon-kube ./charts/carbon-kube \
  -f ./charts/carbon-kube/charts/nvidia-operator/values-gpu.yaml
```

### 3. Verify Installation

```bash
# Check GPU operator pods
kubectl get pods -n gpu-operator

# Check DCGM metrics
kubectl port-forward -n gpu-operator svc/dcgm-exporter 9400:9400
curl http://localhost:9400/metrics | grep DCGM_FI_DEV_POWER_USAGE
```

## Carbon-Aware GPU Scheduling

The subchart integrates with Carbon-Kube's emission plugin to provide:

1. **Real-time Power Monitoring**: DCGM provides actual GPU power consumption
2. **TDP-based Estimation**: Fallback to Thermal Design Power calculations
3. **Utilization-based Scaling**: Power consumption scaled by GPU utilization
4. **Carbon Intensity Integration**: GPU power consumption weighted by grid carbon intensity

## Supported GPU Types

| GPU Model | TDP (Watts) | DCGM Support | MIG Support |
|-----------|-------------|--------------|-------------|
| H100 SXM  | 700W        | ✅           | ✅          |
| H100 PCIe | 350W        | ✅           | ✅          |
| A100 SXM  | 400W        | ✅           | ✅          |
| A100 PCIe | 250W        | ✅           | ✅          |
| V100 SXM2 | 300W        | ✅           | ❌          |
| V100 PCIe | 250W        | ✅           | ❌          |
| T4        | 70W         | ✅           | ❌          |
| RTX 4090  | 450W        | ✅           | ❌          |
| L4        | 72W         | ✅           | ❌          |
| L40       | 300W        | ✅           | ❌          |

## Troubleshooting

### DCGM Metrics Not Available

```bash
# Check DCGM exporter logs
kubectl logs -n gpu-operator -l app=dcgm-exporter

# Verify GPU nodes
kubectl get nodes -l accelerator=nvidia

# Check GPU device plugin
kubectl logs -n gpu-operator -l app=nvidia-device-plugin-daemonset
```

### MIG Configuration Issues

```bash
# Check MIG manager logs
kubectl logs -n gpu-operator -l app=nvidia-mig-manager

# Verify MIG configuration
kubectl get configmap -n gpu-operator mig-parted-config -o yaml
```

### Carbon Metrics Integration

```bash
# Check emission plugin logs
kubectl logs -n kube-system -l app=carbon-kube-scheduler

# Verify GPU power calculations
kubectl logs -n kube-system -l app=carbon-kube-scheduler | grep "GPU power"
```

## Contributing

This subchart is part of the Carbon-Kube project. For contributions and issues, please refer to the main Carbon-Kube repository.