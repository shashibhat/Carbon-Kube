# Carbon-Kube: Carbon-Aware Kubernetes Scheduler

Carbon-Kube is an intelligent Kubernetes scheduler that optimizes workload placement based on real-time carbon intensity data, helping reduce the environmental impact of cloud computing through carbon-aware scheduling decisions.

## 🌱 Overview

Carbon-Kube integrates with multiple data sources to make informed scheduling decisions:

- **Real-time Carbon Data**: Fetches carbon intensity from Electricity Maps API
- **Weather-based Fallback**: Uses NOAA weather data to estimate renewable energy availability
- **Machine Learning Optimization**: Employs reinforcement learning to optimize migration thresholds
- **Multi-cloud Support**: Supports deployment across AWS, GCP, and Azure with CDK

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Carbon Data   │    │   Scheduler     │    │   RL Tuner      │
│     Poller      │───▶│     Plugin      │◀───│   Component     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ConfigMaps    │    │   Pod Scheduling│    │   Threshold     │
│  (Carbon Data)  │    │   Decisions     │    │  Optimization   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Components

1. **Scheduler Plugin** (`pkg/emissionplugin/`): Go-based Kubernetes scheduler plugin
2. **Carbon Data Poller** (`scripts/poller.py`): Python service for fetching carbon intensity data
3. **RL Tuner** (`scripts/rl_tuner.py`): Reinforcement learning component for threshold optimization
4. **Helm Charts** (`charts/carbon-kube/`): Kubernetes deployment manifests
5. **AWS CDK Stack** (`cdk/`): Multi-cloud infrastructure deployment

## 🚀 Quick Start

### Prerequisites

- Kubernetes cluster (v1.20+)
- Helm 3.x
- kubectl configured
- API keys for Electricity Maps and NOAA (optional)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/carbon-kube.git
   cd carbon-kube
   ```

2. **Install using Helm**:
   ```bash
   # Add required API keys
   helm install carbon-kube ./charts/carbon-kube \
     --set electricityMaps.apiKey="your-electricity-maps-key" \
     --set noaa.apiKey="your-noaa-key" \
     --namespace carbon-kube \
     --create-namespace
   ```

3. **Verify installation**:
   ```bash
   kubectl get pods -n carbon-kube
   kubectl logs -n carbon-kube deployment/carbon-kube-scheduler
   ```

### Configuration

The system can be configured through Helm values or ConfigMaps:

```yaml
# values.yaml
global:
  debug: true
  
scheduler:
  replicas: 2
  
carbonPoller:
  schedule: "*/5 * * * *"  # Every 5 minutes
  
rlTuner:
  schedule: "0 */6 * * *"  # Every 6 hours
  enabled: true
  
electricityMaps:
  apiKey: "your-api-key"
  zones: ["US-CA", "US-TX", "EU-DE"]
  
threshold: 250  # gCO2/kWh
```

## 📊 Monitoring

Carbon-Kube provides comprehensive monitoring through Prometheus metrics and Grafana dashboards.

### Metrics

Key metrics exposed:

- `carbon_kube_scheduler_decisions_total`: Total scheduling decisions
- `carbon_kube_scheduler_latency_seconds`: Scheduler latency
- `carbon_kube_carbon_intensity_current`: Current carbon intensity by region
- `carbon_kube_migrations_total`: Pod migration statistics
- `carbon_rl_tuner_reward_average`: RL tuner performance

### Dashboards

Three pre-built Grafana dashboards:

1. **Carbon-Kube Overview**: High-level system metrics and carbon intensity
2. **Scheduler Performance**: Detailed scheduler metrics and performance
3. **RL Tuner**: Machine learning model performance and optimization

Access dashboards:
```bash
kubectl port-forward -n carbon-kube svc/grafana 3000:80
# Open http://localhost:3000
```

## 🧪 Testing

### Unit Tests

Run Go unit tests:
```bash
cd pkg/emissionplugin
go test -v ./...
```

Run Python unit tests:
```bash
cd scripts
python -m pytest test_poller.py test_rl_tuner.py -v
```

### Integration Tests

Run end-to-end tests:
```bash
cd test/integration
python -m pytest test_e2e.py -v
```

### Performance Tests

Load test the scheduler:
```bash
cd test/integration
python -m pytest test_e2e.py::TestCarbonKubePerformance -v
```

## 🔧 Development

### Building from Source

1. **Build Go components**:
   ```bash
   make build-go
   ```

2. **Build Python components**:
   ```bash
   make build-python
   ```

3. **Build Docker images**:
   ```bash
   make build-images
   ```

### Local Development

1. **Set up development environment**:
   ```bash
   make dev-setup
   ```

2. **Run components locally**:
   ```bash
   # Terminal 1: Run scheduler
   make run-scheduler
   
   # Terminal 2: Run poller
   make run-poller
   
   # Terminal 3: Run RL tuner
   make run-rl-tuner
   ```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run tests: `make test`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## 🌍 Multi-Cloud Deployment

### AWS Deployment

Deploy using AWS CDK:

```bash
cd cdk
pip install -r requirements.txt
cdk bootstrap
cdk deploy CarbonKubeStack-Primary CarbonKubeStack-Secondary
```

### GCP Deployment

```bash
# Configure GCP credentials
gcloud auth application-default login

# Deploy using Terraform (coming soon)
cd terraform/gcp
terraform init
terraform apply
```

### Azure Deployment

```bash
# Configure Azure credentials
az login

# Deploy using ARM templates (coming soon)
cd arm-templates
az deployment group create --resource-group carbon-kube --template-file main.json
```

## 📈 Performance Tuning

### Scheduler Optimization

1. **Adjust scoring weights**:
   ```yaml
   scheduler:
     config:
       carbonWeight: 70    # Carbon intensity weight (0-100)
       performanceWeight: 30  # Performance weight (0-100)
   ```

2. **Configure filtering thresholds**:
   ```yaml
   threshold: 300  # Only schedule on nodes with <300 gCO2/kWh
   ```

### RL Tuner Configuration

1. **Model parameters**:
   ```yaml
   rlTuner:
     model:
       learning_rate: 0.0003
       batch_size: 64
       gamma: 0.99
   ```

2. **Training frequency**:
   ```yaml
   rlTuner:
     schedule: "0 */4 * * *"  # Train every 4 hours
     environment:
       update_frequency: 50   # Update after 50 episodes
   ```

## 🔒 Security

### API Key Management

Store API keys securely using Kubernetes Secrets:

```bash
kubectl create secret generic carbon-kube-secrets \
  --from-literal=electricity-maps-key="your-key" \
  --from-literal=noaa-key="your-key" \
  -n carbon-kube
```

### RBAC Configuration

Carbon-Kube uses minimal required permissions:

- **Scheduler**: Read nodes, pods, ConfigMaps; Create events
- **Poller**: Read/Write ConfigMaps
- **RL Tuner**: Read events, pods; Write ConfigMaps

### Network Policies

Enable network policies for enhanced security:

```yaml
networkPolicy:
  enabled: true
  ingress:
    - from:
      - namespaceSelector:
          matchLabels:
            name: kube-system
```

## 📚 API Reference

### Scheduler Plugin API

The scheduler plugin implements the Kubernetes Scheduler Framework:

- `Filter(pod, node)`: Filters nodes based on carbon intensity threshold
- `Score(pod, node)`: Scores nodes based on carbon intensity (0-100)
- `Name()`: Returns plugin name

### Carbon Data Format

Carbon intensity data structure:

```json
{
  "US-CA": {
    "carbon_intensity": 250.5,
    "timestamp": 1640995200,
    "source": "electricity_maps",
    "renewable_percentage": 45.2
  }
}
```

### Configuration Schema

Complete configuration reference available in [`charts/carbon-kube/values.yaml`](charts/carbon-kube/values.yaml).

## 🐛 Troubleshooting

### Common Issues

1. **Scheduler not making decisions**:
   ```bash
   # Check scheduler logs
   kubectl logs -n carbon-kube deployment/carbon-kube-scheduler
   
   # Verify ConfigMap data
   kubectl get configmap carbon-intensity-data -n carbon-kube -o yaml
   ```

2. **Carbon data not updating**:
   ```bash
   # Check poller logs
   kubectl logs -n carbon-kube cronjob/carbon-kube-poller
   
   # Verify API keys
   kubectl get secret carbon-kube-secrets -n carbon-kube -o yaml
   ```

3. **RL tuner not training**:
   ```bash
   # Check RL tuner logs
   kubectl logs -n carbon-kube cronjob/carbon-kube-rl-tuner
   
   # Verify model storage
   kubectl get pvc -n carbon-kube
   ```

### Debug Mode

Enable debug logging:

```yaml
global:
  debug: true
```

### Support

- 📧 Email: support@carbon-kube.io
- 💬 Slack: [#carbon-kube](https://kubernetes.slack.com/channels/carbon-kube)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/carbon-kube/issues)

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kubernetes Scheduler Framework](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/)
- [Electricity Maps API](https://www.electricitymap.org/api)
- [NOAA Weather API](https://www.weather.gov/documentation/services-web-api)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [Katalyst](https://github.com/kubewharf/katalyst-core)

## 🗺️ Roadmap

- [ ] Support for more carbon intensity data sources
- [ ] Integration with cloud provider carbon APIs
- [ ] Advanced ML models (Deep Q-Learning, A3C)
- [ ] Real-time workload migration
- [ ] Carbon budgeting and reporting
- [ ] Integration with Kubernetes Resource Recommender
- [ ] Support for edge computing scenarios

---

**Carbon-Kube** - Making Kubernetes greener, one pod at a time! 🌱