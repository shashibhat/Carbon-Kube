# Changelog

All notable changes to Carbon-Kube will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive CI/CD pipeline with GitHub Actions
- Security scanning with Trivy and Gosec
- Grafana dashboard for carbon metrics visualization
- Python unit tests for carbon intensity poller
- Chart testing configuration for Helm validation

### Changed
- Enhanced documentation with deployment guides
- Improved error handling in scheduler plugin
- Updated dependencies to latest stable versions

### Fixed
- Resolved ConfigMap creation race conditions
- Fixed metric collection edge cases
- Corrected zone mapping for international regions

## [1.0.0] - 2024-01-15

### Added
- **Katalyst Integration**: Full integration with Kubewharf Katalyst for enhanced carbon-aware scheduling
  - Carbon QoS Controller for workload classification
  - Enhanced Scheduler with carbon-aware node scoring
  - RL Tuner for dynamic optimization
  - Custom Resource Definitions (CRDs) for carbon policies

- **Core Components**:
  - Carbon Emission Plugin for Kubernetes scheduler
  - Multi-source carbon intensity poller (Electricity Maps, NOAA, AWS)
  - Prometheus metrics collection and monitoring
  - Helm charts for easy deployment
  - AWS CDK infrastructure as code

- **Carbon-Aware Features**:
  - Real-time carbon intensity data fetching
  - Zone-based carbon scoring for nodes
  - Workload migration for carbon optimization
  - QoS profile-based resource management
  - Carbon budget tracking and enforcement

- **Monitoring & Observability**:
  - Comprehensive Prometheus metrics
  - Grafana dashboards for visualization
  - Alerting rules for carbon thresholds
  - Performance monitoring for scheduler plugins

- **Multi-Cloud Support**:
  - AWS regions with carbon intensity mapping
  - Electricity Maps API integration
  - NOAA weather data fallback
  - Extensible architecture for other cloud providers

- **Documentation**:
  - Complete deployment guides
  - Architecture documentation
  - API reference
  - Example configurations and test workloads

### Technical Details
- **Languages**: Go 1.21, Python 3.11
- **Dependencies**: 
  - Kubernetes 1.28+
  - Katalyst v0.7.0+
  - Prometheus
  - Grafana
- **Deployment**: Helm 3.12+, AWS CDK 2.x
- **Testing**: Unit tests, integration tests, end-to-end validation

### Performance
- Scheduler plugin latency: <10ms p95
- Carbon data refresh: 5-minute intervals
- Memory footprint: <100MB per component
- CPU usage: <0.1 cores per component

### Security
- RBAC policies for minimal permissions
- Secret management for API tokens
- Network policies for component isolation
- Security scanning in CI/CD pipeline

## [0.9.0] - 2024-01-10

### Added
- Initial Katalyst integration planning
- Enhanced scheduler plugin architecture
- Carbon QoS profile definitions
- RL tuner integration framework

### Changed
- Refactored plugin architecture for extensibility
- Improved carbon intensity calculation algorithms
- Enhanced error handling and retry mechanisms

### Fixed
- Memory leaks in long-running processes
- Race conditions in ConfigMap updates
- Timezone handling in carbon data processing

## [0.8.0] - 2024-01-05

### Added
- AWS Carbon Footprint API integration
- Multi-region carbon intensity support
- Enhanced monitoring and alerting
- Performance optimizations

### Changed
- Improved API rate limiting and caching
- Enhanced logging and debugging capabilities
- Updated documentation and examples

### Fixed
- API timeout handling
- Metric collection edge cases
- Configuration validation issues

## [0.7.0] - 2024-01-01

### Added
- NOAA weather data integration as fallback
- Enhanced carbon intensity forecasting
- Workload migration capabilities
- Comprehensive test suite

### Changed
- Improved scheduler plugin performance
- Enhanced carbon scoring algorithms
- Better error handling and recovery

### Fixed
- ConfigMap synchronization issues
- Metric labeling inconsistencies
- Documentation gaps

## [0.6.0] - 2023-12-20

### Added
- Prometheus metrics integration
- Grafana dashboard templates
- Alerting rules for carbon thresholds
- Performance monitoring

### Changed
- Optimized carbon intensity data processing
- Improved plugin registration and lifecycle
- Enhanced configuration management

### Fixed
- Memory usage optimization
- API response parsing edge cases
- Scheduler integration stability

## [0.5.0] - 2023-12-15

### Added
- Multi-source carbon data aggregation
- Zone-based carbon intensity mapping
- Workload carbon footprint calculation
- Basic monitoring and logging

### Changed
- Refactored data collection architecture
- Improved carbon scoring methodology
- Enhanced configuration flexibility

### Fixed
- Data staleness detection
- API error handling
- Configuration validation

## [0.4.0] - 2023-12-10

### Added
- Electricity Maps API integration
- Real-time carbon intensity fetching
- ConfigMap-based data storage
- Basic scheduler plugin framework

### Changed
- Improved data collection reliability
- Enhanced error handling
- Better configuration management

### Fixed
- API rate limiting issues
- Data format inconsistencies
- Plugin registration problems

## [0.3.0] - 2023-12-05

### Added
- Basic carbon intensity poller
- Kubernetes scheduler plugin skeleton
- Initial Helm chart structure
- Basic documentation

### Changed
- Improved project structure
- Enhanced build and deployment scripts
- Better error handling

### Fixed
- Build configuration issues
- Dependency management
- Basic functionality bugs

## [0.2.0] - 2023-12-01

### Added
- Project structure and architecture
- Basic Go modules and dependencies
- Initial Kubernetes integration
- Development environment setup

### Changed
- Refined project goals and scope
- Improved development workflow
- Enhanced documentation structure

### Fixed
- Initial setup and configuration issues
- Basic compilation and runtime errors

## [0.1.0] - 2023-11-25

### Added
- Initial project creation
- Basic project structure
- README and initial documentation
- License and contributing guidelines

---

## Release Notes

### Version 1.0.0 Highlights

This major release introduces comprehensive Katalyst integration, making Carbon-Kube a production-ready solution for carbon-aware Kubernetes scheduling. Key improvements include:

1. **Enhanced Intelligence**: Integration with Katalyst's RL tuner enables dynamic optimization based on real-world performance data.

2. **Production Ready**: Comprehensive monitoring, alerting, and operational tools make this suitable for production deployments.

3. **Multi-Cloud**: Support for multiple cloud providers and carbon data sources ensures broad applicability.

4. **Extensible**: Plugin architecture and CRD-based configuration enable easy customization and extension.

### Migration Guide

For users upgrading from v0.x versions:

1. **Backup Configuration**: Export existing ConfigMaps and custom configurations
2. **Update Dependencies**: Ensure Kubernetes 1.28+ and install Katalyst
3. **Deploy New Charts**: Use Helm charts with Katalyst integration
4. **Migrate Policies**: Convert existing policies to new CRD format
5. **Update Monitoring**: Deploy new Grafana dashboards and alerting rules

### Breaking Changes

- Scheduler plugin name changed from `CarbonPlugin` to `CarbonEmissionPlugin`
- ConfigMap structure updated to include Katalyst metadata
- Metric names standardized with `carbon_` prefix
- API endpoints restructured for better organization

### Support

For questions, issues, or contributions:
- GitHub Issues: https://github.com/carbon-kube/carbon-kube/issues
- Documentation: https://carbon-kube.github.io/docs/
- Community: https://carbon-kube.github.io/community/