# Carbon-Kube Project Summary

## Core Thesis

Carbon-Kube is a Kubernetes-native scheduler extension for data pipelines (Spark/Flink/ETL) that treats jobs as DAG stages with external data dependencies, enforces SLA and carbon budgets per tenant through a single policy CRD, uses forecast-aware temporal shifting to choose low-carbon start times, and uses measured pod energy (Kepler/DCGM) plus hardware-aware placement to ground and optimize decisions.

## Feature Overview

- DAG- and data-gravity-aware scheduling on Kubernetes
- Unified SLA, carbon, and budget policy as code
- Forecast-aware temporal shifting under strict envelopes
- Real per-pod energy via Kepler and hardware-aware placement
- Budget enforcement with tenant fairness
- Comprehensive Helm packaging, RBAC, ServiceMonitors
- Reproducible evaluation harness and unit tests

## Core Abstractions (CRDs)

### CarbonJobSpec

- apiVersion: `carbonkube.io/v1`
- kind: `CarbonJobSpec`
- Fields
  - DAG identity: `dagId`, `stageId`, `upstreamStages[]`
  - Time hints: `estimatedRuntimeSeconds`, `estimatedCpuSeconds?`, `deadline?`
  - External dependencies: `dataSources[]` with `type`, `resource`, `region`, `avgIngressGBPerJob?`, `avgReadGBPerJob?`
  - Optional policy binding: `policyRef`
- Status
  - `dag.isCriticalPath`, `dag.topoDepth`, `dag.normalizedImportance`

### CarbonPolicy

- apiVersion: `carbonkube.io/v1`
- kind: `CarbonPolicy`
- Fields
  - Target: `namespaceSelector`, `workloadSelector` with label matching
  - Priority class: `criticality` (Critical | LatencySensitive | Batch | BestEffort)
  - SLA envelope: `maxDelaySeconds`, `maxSlowdownPercent`, `deadlineMode`, `defaultRelativeDeadlineSeconds`
  - Carbon knobs: `aggressiveness`, `maxCarbonIntensity?`, `minRenewableFraction?`
  - Budget: `tenantId`, `monthlyCarbonBudgetKg`, `perJobBudgetKg`, `burstAllowancePercent`, `fairness{minShare,maxShare,overagePolicy}`
  - Conflict resolution: `onBudgetExhaustion`, `onSlaRisk`, `allowOverrideAnnotations`

## Control Plane Components

### PolicyResolver

- Resolves applicable `CarbonPolicy` for each `CarbonJobSpec` (or annotated workload) via selectors
- Annotates jobs with policy context
  - `carbonkube.io/policy-name`
  - `carbonkube.io/criticality`
  - `carbonkube.io/tenant`
  - `carbonkube.io/carbon-aggressiveness`

### DAGController

- Builds per-DAG graphs from `CarbonJobSpec`
- Computes topological depth, longest path, normalized importance
- Annotates jobs and sets status
  - `carbonkube.io/dag-id`, `carbonkube.io/stage-id`
  - `carbonkube.io/dag-importance`, `carbonkube.io/dag-critical`

### TemporalPlanner

- Inputs per job/stage
  - Submission time `t0`
  - Estimated runtime `T`
  - Deadline `D`
  - Policy envelope: `maxDelaySeconds`, `maxSlowdownPercent`, `aggressiveness`, `criticality`
  - Carbon intensity forecast `C(t)` from provider
- Feasible window
  - `t_min = t0`
  - `t_max = min(t0 + Δ_max, D − T, t_asap + s_max·T/100)`
  - Critical jobs or `t_max ≤ t_min` → schedule ASAP
- Carbon cost approximation
  - `CO2(t) ≈ Σ_{k=0}^{n−1} C(t + kΔ)`
- Interpolation with aggressiveness `a ∈ [0,1]`
  - `t_final = t_asap + a · (t_carbon − t_asap)`
  - Clamp to `[t_min, t_max]`, snap to slot
- Annotates
  - `carbonkube.io/scheduled-at`

### BudgetEnforcer

- Reads per-pod energy (`kepler_container_joules_total`) via Prometheus
- Converts joules → kWh and multiplies by zone carbon intensity
  - `CO₂_kg = (joules / 3,600,000) * (carbon_intensity_gco2_per_kwh / 1,000)`
- Aggregates per tenant (`tenantId`) and writes ConfigMap state
  - `usedCarbonMonthlyKg`, `remainingBudgetKg`, `overBudget`, `throttled`
- Influences scheduling
  - Over budget with `overagePolicy=degrade`: increase budget penalty for non-Critical jobs
  - With `overagePolicy=reject`: admission denial for BestEffort/Batch

### Kepler Attribution

- Prometheus-scraped per-container/pod energy (joules)
- Metrics used by BudgetEnforcer and dashboards
- Optional attribution service that exposes derived metrics from Prometheus queries

## Forecast Providers

- Prometheus
  - `carbon_intensity_gco2_per_kwh{zone=...}` for current and query_range for near-term forecast
- Electricity Maps (optional plugin)
  - HTTP API `v3/carbon-intensity/forecast`
  - Ingested into Prometheus or consumed directly by TemporalPlanner

## Scheduler Plugin Scoring

For pod `p` on node `n`:

`Score(p,n) = w_c · S_carbon + w_s · S_slaUrgency + w_d · S_dagImportance − w_g · P_dataGravity + w_h · S_hardwareEff − w_b · P_budget`

- Carbon term `S_carbon`
  - Uses near-term CI average for node’s zone
  - Lower CI → higher normalized score
- SLA urgency `S_slaUrgency`
  - `slack = deadline − (now + remainingRuntime)`
  - Less slack → higher score
- DAG importance `S_dagImportance`
  - From `normalizedImportance` in [0,1]
  - If `dag-critical=true`, add boost
- Data gravity penalty `P_dataGravity`
  - For each `dataSource`: if `node.region == dataSource.region` → 0
  - Else: `extraGB = avgIngressGBPerJob + avgReadGBPerJob`
  - `penalty += extraGB * (cost_coef + carbon_coef * CI_diff + latency_coef * latency_factor(region_pair))`
- Hardware efficiency `S_hardwareEff`
  - Node labels: `arch`, GPU type
  - Derived via Kepler/DCGM profiling as expected Joules/task on HW type
  - Higher efficiency → higher score
- Budget penalty `P_budget`
  - If tenant is near/over budget: increase penalty for low-priority jobs

Weights `w_x`

- Base values from env/config
- Modified per policy
  - Higher `carbon.aggressiveness` → higher `w_c`, lower `w_s`
  - Critical jobs → high `w_s`, low `w_c`, and `w_b=0`

## Webhook

- Mutating Admission webhook injects labels/annotations used by the scheduler plugin
  - `preferredRegion`
  - `carbonPriorityScore`
  - Ensures presence of policy context annotations when missing

## Helm and Deployment

- Chart: `charts/carbon-kube` with templates for controllers, webhook, scheduler integration, ServiceMonitors, RBAC, ConfigMaps, Secrets
- Values include toggles for controllers, kepler attribution, and provider envs
- RBAC permits access to CRDs, pods, nodes, ConfigMaps, and Prometheus-scraped metrics

## Build and Tests

- Go builds for controllers, kepler-attr, and plugin components
- Unit tests for planner window and carbon cost series; budget helper functions
- Helm lint and template validation

## Metrics and Dashboards

- Prometheus Rules provide derived signals such as intensity averages, migrations rate, and CO₂ estimates
- Grafana dashboards visualize carbon intensity, savings, migrations, and latencies

## Configuration

- Forecast provider selection
  - `CARBONKUBE_FORECAST_PROVIDER` = `prometheus` or `electricitymaps`
  - `CARBONKUBE_PROMETHEUS_URL`
  - `CARBONKUBE_ELECTRICITYMAPS_URL`, `CARBONKUBE_ELECTRICITYMAPS_TOKEN`
- Data gravity coefficients
  - `CARBONKUBE_DG_COST_COEF`, `CARBONKUBE_DG_CARBON_COEF`, `CARBONKUBE_DG_LATENCY_COEF`
- Weights
  - `CARBONKUBE_WC`, `CARBONKUBE_WS`, `CARBONKUBE_WD`, `CARBONKUBE_WG`, `CARBONKUBE_WH`, `CARBONKUBE_WB`

## Security and Best Practices

- No secrets committed; tokens provided via Kubernetes Secret and environment variables
- Strict RBAC scopes for controllers and webhook
- Avoids logging sensitive tokens; uses envs and Secret refs in templates

## Example Annotations Used

- Policy: `carbonkube.io/policy-name`, `carbonkube.io/criticality`, `carbonkube.io/tenant`, `carbonkube.io/carbon-aggressiveness`
- DAG: `carbonkube.io/dag-id`, `carbonkube.io/stage-id`, `carbonkube.io/dag-importance`, `carbonkube.io/dag-critical`
- Temporal: `carbonkube.io/scheduled-at`
- Runtime hints: `carbonkube.io/deadline`, `carbonkube.io/estimated-runtime-seconds`

## Key Code Paths

- CRDs: `charts/carbon-kube/crds/`
- API types: `api/v1/`
- Controllers: `controllers/`
- Scheduler plugin: `pkg/emissionplugin/plugin.go`
- Forecast providers: `pkg/providers/forecast.go`
- Helm templates: `charts/carbon-kube/templates/`
- Build system: `Makefile`, `Dockerfile`
