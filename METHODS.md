# Carbon-Kube Methodology

## Emissions Calculation

### Core Equation

The carbon emissions for workloads are calculated using the following equation:

```
E = ∫ P_IT · PUE · MOER dt
```

Where:
- **E**: Total carbon emissions (gCO₂)
- **P_IT**: IT power consumption (W)
- **PUE**: Power Usage Effectiveness
- **MOER**: Marginal Operating Emissions Rate (gCO₂/kWh)
- **dt**: Time differential

### Power Consumption Model

#### CPU Power Calculation
```
P_CPU = P_base + (P_max - P_base) × utilization_ratio
```

#### GPU Power Calculation
```
P_GPU = TDP × dcgm_util
```

Where:
- **TDP**: Thermal Design Power of the GPU
- **dcgm_util**: GPU utilization from DCGM metrics (0.0-1.0)

### Power Usage Effectiveness (PUE) Parameters

#### CPU Workloads
- **Default PUE**: 1.2 (modern data centers)
- **Range**: 1.1 - 1.8 depending on data center efficiency

#### GPU Workloads
- **GPU PUE**: 1.5 (higher due to additional cooling requirements)
- **Justification**: GPU workloads generate more heat and require enhanced cooling infrastructure

### Marginal Operating Emissions Rate (MOER)

MOER values are sourced from:
- **Primary**: Electricity Maps API (real-time grid carbon intensity)
- **Fallback**: Regional averages from EPA eGRID data
- **Update Frequency**: Every 15 minutes for real-time optimization

### Scoring Algorithm

The emission score for node selection is calculated as:

```
Score = (1 - normalized_emission_factor) × 100
```

Where `normalized_emission_factor` is:
```
emission_factor = (P_IT × PUE × MOER) / reference_emission
normalized_emission_factor = min(emission_factor / max_emission_factor, 1.0)
```

### GPU-Specific Considerations

1. **Multi-Instance GPU (MIG)**: Power is distributed proportionally across MIG slices
2. **Dynamic Voltage and Frequency Scaling (DVFS)**: Accounted for through real-time DCGM metrics
3. **Memory Power**: Included in TDP calculations for HBM-equipped GPUs (H100, A100)

### Baseline Comparisons

#### Static Scheduling
- Uses average regional MOER values
- No real-time carbon intensity consideration
- PUE assumed constant at 1.2

#### HPA (Horizontal Pod Autoscaler)
- CPU-based scaling only
- No carbon awareness
- Standard Kubernetes resource allocation

#### No Reinforcement Learning
- Greedy scheduling based on current carbon intensity
- No predictive optimization
- No learning from historical patterns

### Validation Methodology

#### Confidence Intervals
- **Method**: scipy.stats t-interval
- **Sample Size**: N=10 seeded runs
- **Confidence Level**: 95%

#### Statistical Testing
- **Hypothesis Test**: Welch's t-test (unequal variances)
- **Significance Level**: p < 0.05
- **Effect Size**: Cohen's d for practical significance

### Workload Profiles

#### BERT Fine-tuning
- **GPU Memory**: 16-24 GB
- **Typical Duration**: 2-8 hours
- **Power Pattern**: Sustained high utilization

#### Llama Inference (vLLM)
- **GPU Memory**: 40-80 GB (for 70B models)
- **Typical Duration**: Continuous serving
- **Power Pattern**: Variable based on request load

#### RAPIDS TPC-DS
- **GPU Memory**: 8-32 GB
- **Typical Duration**: 30 minutes - 2 hours
- **Power Pattern**: Burst compute with memory-intensive phases

#### Flink NexMark GPU
- **GPU Memory**: 4-16 GB
- **Typical Duration**: Continuous streaming
- **Power Pattern**: Steady-state with periodic spikes

### Expected Outcomes

- **GPU Carbon Savings**: 20%+ reduction compared to static scheduling
- **Latency Impact**: <5% increase in job completion time
- **Memory Fragmentation**: Monitored via DCGM metrics
- **Migration Overhead**: <2% of total execution time