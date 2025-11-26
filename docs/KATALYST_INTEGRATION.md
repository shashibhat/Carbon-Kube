# Carbon-Kube + Katalyst Integration Design

## 🎯 Overview

This document outlines the integration design between Carbon-Kube and Katalyst-core, combining carbon-aware scheduling with advanced resource management, QoS-based resource models, and topology-aware allocation for optimal sustainability and performance.

## 🏗️ Integration Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Katalyst-Enhanced Kubernetes                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   Carbon-Kube   │    │   Katalyst QoS  │    │ Katalyst Agents │         │
│  │   Scheduler     │◄──►│   Controller    │◄──►│  (Node/VPA)     │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                       │                       │                 │
│           ▼                       ▼                       ▼                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ Carbon Metrics  │    │ QoS Metrics     │    │ Resource Metrics│         │
│  │ & Thresholds    │    │ & Profiles      │    │ & Topology      │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Integration Points

#### 1. **Carbon-Aware QoS Model**
```yaml
apiVersion: node.katalyst.io/v1alpha1
kind: NodeResourceTopology
metadata:
  name: carbon-aware-qos
spec:
  qosClasses:
    - name: "green-guaranteed"
      carbonIntensityThreshold: 100  # gCO2/kWh
      resourceGuarantee: 100%
      priority: 1000
    - name: "mixed-burstable" 
      carbonIntensityThreshold: 300
      resourceGuarantee: 80%
      priority: 500
    - name: "dirty-besteffort"
      carbonIntensityThreshold: 500
      resourceGuarantee: 50%
      priority: 100
```

#### 2. **Enhanced Scheduler Plugin**
The Carbon-Kube scheduler will be enhanced to work with Katalyst's scheduling framework:

```go
// Enhanced plugin interface for Katalyst integration
type CarbonKatalystPlugin struct {
    katalystClient    katalyst.Interface
    qosManager       *QoSManager
    topologyManager  *TopologyManager
    carbonManager    *CarbonManager
}

// Katalyst-aware scoring with QoS and topology considerations
func (p *CarbonKatalystPlugin) Score(ctx context.Context, state *framework.CycleState, 
    pod *v1.Pod, nodeName string) (int64, *framework.Status) {
    
    // Get carbon intensity for node zone
    carbonScore := p.carbonManager.GetCarbonScore(nodeName)
    
    // Get QoS requirements from Katalyst
    qosProfile := p.qosManager.GetQoSProfile(pod)
    
    // Get topology information
    topology := p.topologyManager.GetNodeTopology(nodeName)
    
    // Calculate composite score
    return p.calculateCompositeScore(carbonScore, qosProfile, topology)
}
```

## 🔄 Integration Components

### 1. **Carbon-Aware QoS Controller**

```go
// pkg/katalyst/carbon_qos_controller.go
package katalyst

import (
    katalystv1alpha1 "github.com/kubewharf/katalyst-api/pkg/apis/node/v1alpha1"
    "github.com/kubewharf/katalyst-core/pkg/controller"
)

type CarbonQoSController struct {
    katalystClient katalyst.Interface
    carbonPoller   *CarbonPoller
    qosProfiles    map[string]*QoSProfile
}

type QoSProfile struct {
    Name                    string
    CarbonThreshold        float64  // gCO2/kWh
    ResourceGuarantee      float64  // Percentage
    Priority               int32
    TolerateHighCarbon     bool
    MigrationPolicy        string   // "aggressive", "conservative", "disabled"
    EnergyEfficiencyTarget float64  // Target PUE
}

func (c *CarbonQoSController) ReconcileQoSProfiles(ctx context.Context) error {
    // Get current carbon intensity data
    carbonData := c.carbonPoller.GetLatestData()
    
    // Update QoS profiles based on carbon intensity
    for zone, intensity := range carbonData {
        profile := c.determineOptimalQoSProfile(intensity)
        if err := c.updateNodeQoSProfile(zone, profile); err != nil {
            return err
        }
    }
    
    return nil
}
```

### 2. **Enhanced Carbon Data Model**

```go
// pkg/katalyst/carbon_topology.go
type CarbonNodeTopology struct {
    NodeName         string
    Zone             string
    CarbonIntensity  float64
    EnergySource     EnergySourceBreakdown
    QoSClass         string
    ResourceProfile  ResourceProfile
    TopologyInfo     TopologyInfo
}

type EnergySourceBreakdown struct {
    Renewable    float64 // Percentage
    Nuclear      float64
    Gas          float64
    Coal         float64
    LastUpdated  time.Time
}

type ResourceProfile struct {
    CPUEfficiency    float64 // Performance per watt
    MemoryEfficiency float64
    NetworkLatency   time.Duration
    StorageIOPS      int64
}

type TopologyInfo struct {
    NUMANodes        []NUMANode
    GPUs             []GPUInfo
    NetworkDevices   []NetworkDevice
    PowerZones       []PowerZone
}
```

### 3. **Carbon-Aware Resource Allocation**

```python
# scripts/katalyst_carbon_allocator.py
class CarbonAwareResourceAllocator:
    """
    Integrates with Katalyst's VPA and resource allocation to optimize
    for both performance and carbon efficiency.
    """
    
    def __init__(self):
        self.katalyst_client = KatalystClient()
        self.carbon_poller = CarbonPoller()
        self.ml_optimizer = RLTuner()
    
    def calculate_optimal_allocation(self, workload_profile: WorkloadProfile) -> ResourceAllocation:
        """Calculate optimal resource allocation considering carbon and QoS."""
        
        # Get current carbon intensity
        carbon_data = self.carbon_poller.get_current_intensity()
        
        # Get Katalyst resource recommendations
        katalyst_recommendation = self.katalyst_client.get_vpa_recommendation(
            workload_profile.namespace, 
            workload_profile.name
        )
        
        # Apply carbon-aware adjustments
        carbon_adjusted = self._apply_carbon_adjustments(
            katalyst_recommendation, 
            carbon_data,
            workload_profile.carbon_budget
        )
        
        # Use RL to optimize the final allocation
        optimized_allocation = self.ml_optimizer.optimize_allocation(
            carbon_adjusted,
            workload_profile.sla_requirements
        )
        
        return optimized_allocation
    
    def _apply_carbon_adjustments(self, base_allocation, carbon_data, carbon_budget):
        """Apply carbon-aware adjustments to base resource allocation."""
        
        adjustments = {}
        
        for zone, intensity in carbon_data.items():
            if intensity > carbon_budget.threshold:
                # Reduce resource allocation in high-carbon zones
                adjustments[zone] = {
                    'cpu_scale': 0.8,
                    'memory_scale': 0.9,
                    'priority_boost': False
                }
            elif intensity < carbon_budget.green_threshold:
                # Increase allocation in green zones
                adjustments[zone] = {
                    'cpu_scale': 1.2,
                    'memory_scale': 1.1,
                    'priority_boost': True
                }
        
        return self._apply_adjustments(base_allocation, adjustments)
```

### 4. **Katalyst CRD Extensions**

```yaml
# Carbon-aware Custom Resource Definitions for Katalyst
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: carbonprofiles.carbon.katalyst.io
spec:
  group: carbon.katalyst.io
  versions:
  - name: v1alpha1
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              carbonBudget:
                type: object
                properties:
                  dailyLimit:
                    type: number
                    description: "Daily CO2 budget in kg"
                  threshold:
                    type: number
                    description: "Carbon intensity threshold in gCO2/kWh"
                  greenThreshold:
                    type: number
                    description: "Green energy threshold in gCO2/kWh"
              qosMapping:
                type: object
                properties:
                  greenGuaranteed:
                    type: object
                    properties:
                      maxCarbonIntensity: 100
                      resourceGuarantee: 100
                  mixedBurstable:
                    type: object
                    properties:
                      maxCarbonIntensity: 300
                      resourceGuarantee: 80
              migrationPolicy:
                type: object
                properties:
                  enabled: true
                  strategy: "carbon-aware"
                  cooldownPeriod: "5m"
                  maxMigrationsPerHour: 10
---
apiVersion: carbon.katalyst.io/v1alpha1
kind: CarbonProfile
metadata:
  name: default-carbon-profile
  namespace: default
spec:
  carbonBudget:
    dailyLimit: 100.0  # 100kg CO2 per day
    threshold: 300     # 300 gCO2/kWh
    greenThreshold: 100 # 100 gCO2/kWh
  qosMapping:
    greenGuaranteed:
      maxCarbonIntensity: 100
      resourceGuarantee: 100
    mixedBurstable:
      maxCarbonIntensity: 300
      resourceGuarantee: 80
    dirtyBestEffort:
      maxCarbonIntensity: 500
      resourceGuarantee: 50
  migrationPolicy:
    enabled: true
    strategy: "carbon-aware"
    cooldownPeriod: "5m"
    maxMigrationsPerHour: 10
```

## 🔧 Enhanced Scheduler Configuration

### Katalyst-Enhanced Scheduler Config

```yaml
# config/katalyst-carbon-scheduler.yaml
apiVersion: kubescheduler.config.k8s.io/v1beta3
kind: KubeSchedulerConfiguration
profiles:
- schedulerName: katalyst-carbon-scheduler
  plugins:
    filter:
      enabled:
      - name: CarbonAwareFilter
      - name: KatalystQoSFilter
      - name: KatalystTopologyFilter
    score:
      enabled:
      - name: CarbonAwareScore
        weight: 30
      - name: KatalystQoSScore
        weight: 25
      - name: KatalystTopologyScore
        weight: 20
      - name: NodeResourcesFit
        weight: 15
      - name: NodeAffinity
        weight: 10
  pluginConfig:
  - name: CarbonAwareScore
    args:
      carbonThreshold: 300
      greenBonus: 50
      dirtyPenalty: 100
  - name: KatalystQoSScore
    args:
      qosWeights:
        guaranteed: 100
        burstable: 50
        besteffort: 10
  - name: KatalystTopologyScore
    args:
      numaAware: true
      deviceAware: true
      powerAware: true
```

## 📊 Enhanced Monitoring Integration

### Katalyst + Carbon Metrics

```yaml
# monitoring/katalyst-carbon-servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: katalyst-carbon-metrics
spec:
  selector:
    matchLabels:
      app: katalyst-carbon
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
---
# Additional Prometheus rules for Katalyst integration
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: katalyst-carbon-rules
spec:
  groups:
  - name: katalyst.carbon.rules
    rules:
    - record: katalyst:carbon_efficiency_ratio
      expr: |
        (
          sum(rate(katalyst_workload_performance_score[5m])) by (qos_class, zone)
          /
          sum(rate(carbon_intensity_gco2_kwh[5m])) by (zone)
        )
    
    - record: katalyst:qos_carbon_compliance
      expr: |
        (
          count(katalyst_qos_guarantee_met == 1) by (qos_class)
          /
          count(katalyst_qos_guarantee_met) by (qos_class)
        ) * 100
    
    - alert: KatalystCarbonBudgetExceeded
      expr: katalyst:daily_carbon_consumption > katalyst:daily_carbon_budget
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "Daily carbon budget exceeded"
        description: "Carbon consumption {{ $value }}kg exceeds daily budget"
```

## 🚀 Deployment Integration

### Enhanced Helm Chart for Katalyst

```yaml
# charts/carbon-kube/templates/katalyst-integration.yaml
{{- if .Values.katalyst.enabled }}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "carbon-kube.fullname" . }}-katalyst-controller
spec:
  replicas: {{ .Values.katalyst.controller.replicas }}
  selector:
    matchLabels:
      app: katalyst-carbon-controller
  template:
    metadata:
      labels:
        app: katalyst-carbon-controller
    spec:
      serviceAccountName: {{ include "carbon-kube.serviceAccountName" . }}
      containers:
      - name: controller
        image: "{{ .Values.katalyst.controller.image.repository }}:{{ .Values.katalyst.controller.image.tag }}"
        env:
        - name: KATALYST_NAMESPACE
          value: {{ .Release.Namespace }}
        - name: CARBON_THRESHOLD
          value: "{{ .Values.carbon.threshold }}"
        - name: QOS_PROFILES_CONFIG
          value: "/etc/config/qos-profiles.yaml"
        volumeMounts:
        - name: qos-config
          mountPath: /etc/config
        resources:
          {{- toYaml .Values.katalyst.controller.resources | nindent 12 }}
      volumes:
      - name: qos-config
        configMap:
          name: {{ include "carbon-kube.fullname" . }}-qos-profiles
{{- end }}
```

### Enhanced Values Configuration

```yaml
# charts/carbon-kube/values.yaml additions
katalyst:
  enabled: true
  controller:
    replicas: 1
    image:
      repository: carbon-kube/katalyst-controller
      tag: "v1.0.0"
    resources:
      limits:
        cpu: 500m
        memory: 512Mi
      requests:
        cpu: 100m
        memory: 128Mi
  
  qosProfiles:
    greenGuaranteed:
      carbonThreshold: 100
      resourceGuarantee: 100
      priority: 1000
      migrationEnabled: false
    
    mixedBurstable:
      carbonThreshold: 300
      resourceGuarantee: 80
      priority: 500
      migrationEnabled: true
    
    dirtyBestEffort:
      carbonThreshold: 500
      resourceGuarantee: 50
      priority: 100
      migrationEnabled: true
  
  topology:
    numaAware: true
    deviceAware: true
    powerZoneAware: true
  
  resourceAllocation:
    vpaIntegration: true
    hpaIntegration: true
    carbonAwareScaling: true
```

## 🧪 Testing Integration

### Katalyst Integration Tests

```python
# test/integration/test_katalyst_integration.py
import pytest
from kubernetes import client
from katalyst_client import KatalystClient

class TestKatalystIntegration:
    
    def test_carbon_qos_profile_creation(self):
        """Test that carbon-aware QoS profiles are created correctly."""
        katalyst_client = KatalystClient()
        
        # Create a carbon profile
        carbon_profile = {
            "apiVersion": "carbon.katalyst.io/v1alpha1",
            "kind": "CarbonProfile",
            "metadata": {"name": "test-profile"},
            "spec": {
                "carbonBudget": {"threshold": 300},
                "qosMapping": {
                    "greenGuaranteed": {"maxCarbonIntensity": 100}
                }
            }
        }
        
        result = katalyst_client.create_carbon_profile(carbon_profile)
        assert result.metadata.name == "test-profile"
    
    def test_scheduler_katalyst_integration(self):
        """Test that scheduler works with Katalyst QoS classes."""
        # Deploy a pod with QoS requirements
        pod_spec = {
            "metadata": {
                "annotations": {
                    "katalyst.io/qos-class": "green-guaranteed",
                    "carbon-kube.io/carbon-budget": "100"
                }
            },
            "spec": {
                "containers": [{
                    "name": "test",
                    "image": "nginx",
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "128Mi"},
                        "limits": {"cpu": "500m", "memory": "512Mi"}
                    }
                }]
            }
        }
        
        # Verify pod is scheduled to appropriate node
        scheduled_pod = self.wait_for_pod_scheduled(pod_spec)
        node_carbon = self.get_node_carbon_intensity(scheduled_pod.spec.node_name)
        
        assert node_carbon <= 100  # Should be on green node
    
    def test_carbon_aware_vpa_recommendations(self):
        """Test VPA integration with carbon awareness."""
        # Create VPA with carbon constraints
        vpa_spec = {
            "metadata": {"annotations": {"carbon-kube.io/carbon-aware": "true"}},
            "spec": {
                "targetRef": {"kind": "Deployment", "name": "test-app"},
                "updatePolicy": {"updateMode": "Auto"}
            }
        }
        
        # Verify VPA recommendations consider carbon intensity
        recommendations = self.get_vpa_recommendations("test-app")
        assert "carbon-efficiency" in recommendations.metadata.annotations
```

## 🔄 Migration Strategy

### Phased Integration Approach

#### Phase 1: Core Integration (Week 1-2)
1. **Katalyst Client Integration**
   - Add Katalyst API client to Carbon-Kube
   - Implement basic QoS profile management
   - Update scheduler plugin for Katalyst compatibility

#### Phase 2: Enhanced Scheduling (Week 3-4)
2. **Advanced Scheduler Features**
   - Topology-aware carbon scheduling
   - QoS-based resource allocation
   - Enhanced scoring algorithms

#### Phase 3: Resource Management (Week 5-6)
3. **VPA/HPA Integration**
   - Carbon-aware vertical scaling
   - QoS-constrained horizontal scaling
   - Resource efficiency optimization

#### Phase 4: Advanced Features (Week 7-8)
4. **ML and Automation**
   - Enhanced RL tuner with Katalyst metrics
   - Automated QoS profile optimization
   - Predictive carbon-aware scaling

## 📈 Expected Benefits

### Performance Improvements
- **15-25% better resource utilization** through Katalyst's advanced allocation
- **10-20% reduced scheduling latency** with topology awareness
- **Enhanced QoS compliance** with carbon constraints

### Sustainability Gains
- **20-30% additional carbon reduction** through QoS-aware optimization
- **Improved energy efficiency** with topology-aware placement
- **Better carbon budget management** with integrated monitoring

### Operational Excellence
- **Unified management** of performance and sustainability
- **Enhanced observability** with integrated metrics
- **Simplified deployment** on Katalyst-enhanced clusters

This integration creates a powerful platform that combines the best of both worlds: Katalyst's advanced resource management with Carbon-Kube's sustainability focus, delivering optimal performance while minimizing environmental impact.