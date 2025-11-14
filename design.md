
# Carbon‑Kube Design Document  
### Version 1.0 — November 13, 2025  
Author: Grok (based on analysis of original repository)

---

# 1. Overview

Carbon‑Kube is a **carbon‑aware Kubernetes scheduler extension** designed to minimize the CO₂ footprint of compute-heavy workloads (Spark, Flink, Ray, ML pipelines). This new version is **rewritten from scratch** with modern Google‑style engineering standards:

- Strong modularity  
- gRPC + Protobuf configs  
- Go for scheduler logic  
- Python asyncio for carbon data ingestion  
- Reinforcement learning tuning loop  
- Event‑driven design  
- 80%+ test coverage  
- Bazel monorepo

Carbon‑Kube integrates with Katalyst, Karpenter, and standard Kubernetes API servers.

---

# 2. Architecture

```
Carbon APIs → Poller → CarbonScore CRD → Mutator → Node Tainter → Karpenter → Workload Migration
                                             ↑                               ↓
                                             └───────────── RL Tuner ───────┘
```

### Components

| Component | Language | Description |
|----------|----------|-------------|
| Poll Service | Python (asyncio) | Fetches carbon intensity from ElectricityMaps/NOAA |
| Scheduler Mutator | Go | Adjusts node ranking during pod scheduling |
| Node Tainter Controller | Go | Applies taints when zones exceed carbon thresholds |
| RL Tuner | Python | Optimizes migration thresholds using Q-learning |
| Metrics Exporter | Go | Exposes CO₂ + scheduling metrics |
| Deployment (CDK + Helm) | Python/YAML | Deploys full system on AWS EKS |

---

# 3. Data Model (Protobuf)

```proto
syntax = "proto3";

message CarbonScore {
  string zone = 1;
  float intensity_g_per_kwh = 2;
  google.protobuf.Timestamp forecast_time = 3;
  float cpu_multiplier = 4;
}

message Config {
  float migration_threshold = 1;
  repeated string green_zones = 2;
  bool rl_enabled = 3;
}
```

---

# 4. Component Details

## 4.1 Poll Service

- Python asyncio  
- Uses aiohttp  
- Fetches carbon intensity per region  
- Writes CarbonScore CRD via Kubernetes API  

Pseudo:

```python
scores = await asyncio.gather(*[fetch(zone) for zone in zones])
publish_to_crd(scores)
```

---

## 4.2 Scheduler Mutator (Go)

Implements Katalyst mutator interface.

```
node.Score = base - (emission_score * weight)
```

Penalty applies only if:

```
emission_score > migration_threshold
```

---

## 4.3 Node Tainter Controller

- Watches CarbonScore CRD  
- If zone intensity > threshold: taint nodes  
- If lower: remove taint  

---

## 4.4 RL Tuner

- Q‑learning  
- State: (carbon_intensity, latency_risk)  
- Actions: ±10% threshold adjustments  
- Reward: `− emissions + latency_penalty`

---

# 5. Testing Strategy

- Go unit tests for mutator/taint controller  
- Python unit tests with pytest  
- Integration tests using KIND/Minikube  
- Load tests with K6  
- Chaos tests with Litmus  

---

# 6. Deployment Plan

- Bazel monorepo  
- Build Go & Python binaries  
- Dockerize with multi‑arch builds  
- Helm charts for complete deployment  
- AWS CDK for EKS provisioning  
- Prometheus/Grafana for monitoring  

---

# 7. Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Carbon API rate limits | Caching + batching |
| Scheduling latency | Optimize mutator path |
| RL instability | Low learning rate, A/B testing |

---

# 8. Conclusion

Carbon‑Kube’s redesigned architecture provides a scalable, production-ready foundation for carbon-aware scheduling across large Kubernetes clusters.  
Its modularity, observability-first design, and formal RL tuning loop make it suitable both for enterprise use and academic research (IEEE/ACM papers).

