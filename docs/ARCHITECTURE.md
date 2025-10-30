# Carbon-Kube Architecture

This document provides a detailed overview of Carbon-Kube's architecture, component interactions, and design decisions.

## 🏗️ System Overview

Carbon-Kube is designed as a distributed system with three main components that work together to provide carbon-aware Kubernetes scheduling:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Carbon-Kube System                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   Carbon Data   │    │   Scheduler     │    │   RL Tuner      │        │
│  │     Poller      │    │     Plugin      │    │   Component     │        │
│  │                 │    │                 │    │                 │        │
│  │ • Electricity   │───▶│ • Filter Nodes  │◀───│ • Optimize      │        │
│  │   Maps API      │    │ • Score Nodes   │    │   Thresholds    │        │
│  │ • NOAA Weather  │    │ • Make Decisions│    │ • Learn from    │        │
│  │ • AWS Carbon    │    │                 │    │   Migrations    │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│           │                       │                       │                │
│           ▼                       ▼                       ▼                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   ConfigMaps    │    │   Pod Scheduling│    │   Threshold     │        │
│  │  (Carbon Data)  │    │   Decisions     │    │  Optimization   │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🧩 Component Architecture

### 1. Scheduler Plugin (`pkg/emissionplugin/`)

The scheduler plugin is implemented in Go and integrates with the Kubernetes Scheduler Framework.

#### Key Interfaces

```go
type Plugin interface {
    Name() string
    Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status
    Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status)
}
```

#### Responsibilities

- **Node Filtering**: Exclude nodes with carbon intensity above threshold
- **Node Scoring**: Assign scores based on carbon intensity (lower = better)
- **Configuration Management**: Read thresholds and carbon data from ConfigMaps
- **Metrics Collection**: Expose Prometheus metrics for monitoring

#### Data Flow

```
Pod Scheduling Request
         │
         ▼
┌─────────────────┐
│   Filter Phase  │ ──── Check carbon threshold
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Score Phase   │ ──── Calculate carbon scores
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Node Selection │ ──── Select lowest carbon node
└─────────────────┘
```

### 2. Carbon Data Poller (`scripts/poller.py`)

The poller is a Python service that fetches carbon intensity data from multiple sources.

#### Architecture

```python
class CarbonIntensityPoller:
    def __init__(self, config: Dict[str, Any])
    
    # Data source methods
    async def fetch_electricity_maps_data(self, zones: List[str]) -> Dict[str, Any]
    async def fetch_noaa_weather_data(self, regions: List[str]) -> Dict[str, Any]
    async def fetch_aws_carbon_data(self, regions: List[str]) -> Dict[str, Any]
    
    # Processing methods
    def calculate_renewable_proxy(self, weather_data: Dict) -> float
    def estimate_carbon_from_weather(self, weather_data: Dict) -> float
    
    # Output methods
    async def update_configmap(self, carbon_data: Dict[str, Any]) -> None
```

#### Data Sources

1. **Electricity Maps API**
   - Real-time carbon intensity data
   - Renewable energy percentage
   - Grid composition information

2. **NOAA Weather API**
   - Wind speed and direction
   - Solar irradiance data
   - Temperature and humidity

3. **AWS Carbon Footprint API**
   - Regional carbon intensity
   - Instance-specific emissions
   - Sustainability metrics

#### Fallback Strategy

```
Primary: Electricity Maps API
    │
    ├─ Success ──▶ Use real-time data
    │
    └─ Failure ──▶ Secondary: NOAA Weather
                      │
                      ├─ Success ──▶ Estimate from weather
                      │
                      └─ Failure ──▶ Tertiary: AWS Carbon API
                                        │
                                        ├─ Success ──▶ Use AWS data
                                        │
                                        └─ Failure ──▶ Use cached data
```

### 3. RL Tuner (`scripts/rl_tuner.py`)

The RL Tuner uses reinforcement learning to optimize migration thresholds dynamically.

#### Architecture

```python
class RLTuner:
    def __init__(self, config: Dict[str, Any])
    
    # Environment setup
    def setup_environment(self) -> CarbonMigrationEnv
    
    # Data collection
    async def collect_migration_events(self) -> List[MigrationEvent]
    def add_to_replay_buffer(self, events: List[MigrationEvent]) -> None
    
    # Model training
    def train_model(self) -> None
    def save_model(self, path: str) -> None
    def load_model(self, path: str) -> None
    
    # Threshold optimization
    def get_recommended_threshold(self, state: RLState) -> float
    async def update_threshold_configmap(self, threshold: float) -> None
```

#### Reinforcement Learning Environment

```python
class CarbonMigrationEnv(gym.Env):
    """
    State Space: [current_carbon, avg_carbon_1h, migration_rate, energy_consumption]
    Action Space: Continuous threshold adjustment (-50 to +50 gCO2/kWh)
    Reward Function: -carbon_emissions - migration_cost + energy_savings
    """
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]
    def reset(self) -> np.ndarray
    def render(self, mode: str = 'human') -> None
```

#### Learning Algorithm

- **Algorithm**: Soft Actor-Critic (SAC)
- **Network Architecture**: 2-layer MLP (64 units each)
- **Experience Replay**: 10,000 transitions
- **Update Frequency**: Every 50 episodes
- **Learning Rate**: 0.0003

## 🔄 Data Flow Architecture

### 1. Carbon Data Pipeline

```
External APIs ──▶ Poller ──▶ ConfigMap ──▶ Scheduler Plugin
     │              │           │              │
     │              │           │              ▼
     │              │           │         Scheduling
     │              │           │         Decisions
     │              │           │              │
     │              │           │              ▼
     │              │           │         Pod Placement
     │              │           │              │
     │              │           │              ▼
     │              │           └──────▶ RL Tuner ◀─┘
     │              │                        │
     │              │                        ▼
     │              │                  Threshold
     │              │                  Optimization
     │              │                        │
     │              └────────────────────────┘
     │
     └─ Metrics ──▶ Prometheus ──▶ Grafana
```

### 2. Configuration Management

```
Helm Values ──▶ ConfigMaps ──▶ Components
     │              │              │
     │              │              ├─ Scheduler Plugin
     │              │              ├─ Carbon Poller
     │              │              └─ RL Tuner
     │              │
     │              └─ Secrets ──▶ API Keys
     │
     └─ Service Accounts ──▶ RBAC Permissions
```

### 3. Monitoring Pipeline

```
Components ──▶ Metrics ──▶ Prometheus ──▶ Grafana
     │            │           │            │
     │            │           │            ├─ Overview Dashboard
     │            │           │            ├─ Scheduler Dashboard
     │            │           │            └─ RL Tuner Dashboard
     │            │           │
     │            │           └─ Alerts ──▶ AlertManager
     │            │
     │            ├─ carbon_kube_scheduler_*
     │            ├─ carbon_kube_poller_*
     │            └─ carbon_kube_rl_tuner_*
     │
     └─ Logs ──▶ Kubernetes Events ──▶ Log Aggregation
```

## 🏛️ Design Patterns

### 1. Plugin Architecture

The scheduler uses the Strategy pattern to allow different scoring and filtering algorithms:

```go
type ScoringStrategy interface {
    Score(carbonIntensity float64, threshold float64) int64
}

type LinearScoringStrategy struct{}
type ExponentialScoringStrategy struct{}
type ThresholdScoringStrategy struct{}
```

### 2. Observer Pattern

Components observe ConfigMap changes for dynamic reconfiguration:

```python
class ConfigMapWatcher:
    def __init__(self, callback: Callable[[Dict], None])
    def watch(self, namespace: str, name: str) -> None
    def on_change(self, event: Dict) -> None
```

### 3. Circuit Breaker

API calls use circuit breaker pattern for resilience:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int, timeout: int)
    async def call(self, func: Callable, *args, **kwargs) -> Any
    def is_open(self) -> bool
    def reset(self) -> None
```

## 🔧 Configuration Architecture

### 1. Hierarchical Configuration

```
Default Values (values.yaml)
         │
         ▼
Environment Variables
         │
         ▼
ConfigMaps (Runtime)
         │
         ▼
Component Configuration
```

### 2. Configuration Schema

```yaml
# Global configuration
global:
  debug: boolean
  namespace: string
  
# Component-specific configuration
scheduler:
  replicas: integer
  image: string
  config:
    carbonWeight: integer (0-100)
    performanceWeight: integer (0-100)
    
carbonPoller:
  schedule: cron
  sources:
    electricityMaps:
      enabled: boolean
      zones: array[string]
    noaa:
      enabled: boolean
      regions: array[string]
      
rlTuner:
  enabled: boolean
  schedule: cron
  model:
    algorithm: string
    learning_rate: float
    batch_size: integer
```

## 🚀 Deployment Architecture

### 1. Kubernetes Resources

```
Namespace: carbon-kube
├── Deployments
│   └── carbon-kube-scheduler
├── CronJobs
│   ├── carbon-kube-poller
│   └── carbon-kube-rl-tuner
├── ConfigMaps
│   ├── carbon-kube-config
│   ├── carbon-intensity-data
│   └── scheduler-config
├── Secrets
│   └── carbon-kube-secrets
├── Services
│   └── carbon-kube-scheduler
├── ServiceAccounts
│   ├── carbon-kube-scheduler
│   ├── carbon-kube-poller
│   └── carbon-kube-rl-tuner
└── RBAC
    ├── ClusterRoles
    └── ClusterRoleBindings
```

### 2. Multi-Cloud Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      AWS        │    │      GCP        │    │     Azure       │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │     EKS     │ │    │ │     GKE     │ │    │ │     AKS     │ │
│ │             │ │    │ │             │ │    │ │             │ │
│ │ Carbon-Kube │ │    │ │ Carbon-Kube │ │    │ │ Carbon-Kube │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │  Monitoring │ │    │ │  Monitoring │ │    │ │  Monitoring │ │
│ │   Stack     │ │    │ │   Stack     │ │    │ │   Stack     │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Central       │
                    │  Monitoring     │
                    │   Dashboard     │
                    └─────────────────┘
```

## 🔒 Security Architecture

### 1. Authentication & Authorization

```
API Requests ──▶ Service Account ──▶ RBAC ──▶ Kubernetes API
     │               │                │
     │               │                └─ Minimal Permissions
     │               │
     │               └─ Pod Security Context
     │
     └─ TLS Encryption ──▶ Secure Communication
```

### 2. Secret Management

```
External Secrets ──▶ Kubernetes Secrets ──▶ Environment Variables
     │                      │                      │
     │                      │                      └─ Components
     │                      │
     │                      └─ Volume Mounts ──▶ File System
     │
     └─ Rotation Policy ──▶ Automated Updates
```

### 3. Network Security

```
Internet ──▶ Ingress ──▶ Services ──▶ Pods
    │           │           │          │
    │           │           │          └─ Network Policies
    │           │           │
    │           │           └─ Service Mesh (Optional)
    │           │
    │           └─ TLS Termination
    │
    └─ Rate Limiting ──▶ DDoS Protection
```

## 📊 Performance Architecture

### 1. Scalability Design

- **Horizontal Scaling**: Multiple scheduler replicas
- **Vertical Scaling**: Resource limits and requests
- **Auto-scaling**: HPA based on CPU/memory metrics
- **Load Balancing**: Service-level load distribution

### 2. Caching Strategy

```
API Responses ──▶ In-Memory Cache ──▶ Redis (Optional)
     │                 │                  │
     │                 │                  └─ Distributed Cache
     │                 │
     │                 └─ TTL: 5 minutes
     │
     └─ Cache Miss ──▶ Fallback to API
```

### 3. Performance Metrics

- **Scheduler Latency**: P50, P95, P99 percentiles
- **Throughput**: Pods scheduled per second
- **Resource Usage**: CPU, memory, network I/O
- **Cache Hit Rate**: Percentage of cached responses

## 🔄 Event-Driven Architecture

### 1. Event Sources

```
Kubernetes Events ──▶ Event Bus ──▶ Event Handlers
     │                   │              │
     │                   │              ├─ Pod Lifecycle
     │                   │              ├─ Node Changes
     │                   │              └─ ConfigMap Updates
     │                   │
     │                   └─ Event Filtering ──▶ Relevant Events
     │
     └─ Custom Events ──▶ Migration Events ──▶ RL Tuner
```

### 2. Event Processing

```
Event ──▶ Validation ──▶ Processing ──▶ Action
  │           │             │            │
  │           │             │            ├─ Update State
  │           │             │            ├─ Trigger Training
  │           │             │            └─ Send Metrics
  │           │             │
  │           │             └─ Async Processing
  │           │
  │           └─ Schema Validation
  │
  └─ Event Sourcing ──▶ Audit Trail
```

## 🎯 Future Architecture Considerations

### 1. Microservices Evolution

- Split monolithic components into smaller services
- Service mesh integration (Istio/Linkerd)
- Event-driven communication (NATS/Kafka)

### 2. Edge Computing Support

- Edge node carbon awareness
- Distributed scheduling decisions
- Offline operation capabilities

### 3. Advanced ML Integration

- Multi-agent reinforcement learning
- Federated learning across clusters
- Real-time model serving

### 4. Sustainability Metrics

- Carbon accounting and reporting
- Sustainability dashboards
- Carbon budget enforcement

This architecture provides a solid foundation for carbon-aware Kubernetes scheduling while maintaining flexibility for future enhancements and scaling requirements.