## Scope
- Align CRDs (`CarbonJobSpec`, `CarbonPolicy`) with PROJECT_SUMMARY.md.
- Implement controllers: PolicyResolver, DAGController, TemporalPlanner, BudgetEnforcer, KeplerAttribution aggregator.
- Update scheduler plugin scoring to honor DAG, carbon forecasts, budgets, hardware perf/W.
- Extend Helm chart, RBAC, ServiceMonitors, Docker/Make, and tests.
- Update README with formulas and end-to-end usage.

## Current State (summary)
- CRDs exist in `charts/carbon-kube/crds/*`; Go API types in `api/v1/*`.
- Controllers use `client-go` and a custom mutating webhook; no `controller-runtime`.
- Scheduler plugin present under `pkg/emissionplugin` and `pkg/katalyst/*`.
- Build via `Makefile` and `Dockerfile`; Helm chart under `charts/carbon-kube/*`.

## Implementation Steps
### 1) CRD Schema Alignment
- Update `charts/carbon-kube/crds/carbonkube.io_carbonjobs.yaml` to include fields: `dagId`, `stageId`, `upstreamStages[]`, `estimatedRuntimeSeconds`, `estimatedCpuSeconds?`, `deadline?`, `dataSources[]` (kafka/db/s3 with resource/region/size hints), `policyRef?`.
- Add `status.dag` with `isCriticalPath`, `topoDepth`, `normalizedImportance`.
- Update `charts/carbon-kube/crds/carbonkube.io_carbonpolicies.yaml` to include: `target.namespaceSelector`, `target.workloadSelector`, `criticality`, `sla{maxDelaySeconds,maxSlowdownPercent,deadlineMode,defaultRelativeDeadlineSeconds}`, `carbon{aggressiveness,maxCarbonIntensity?,minRenewableFraction?}`, `budget{enabled,tenantId,monthlyCarbonBudgetKg,perJobBudgetKg,burstAllowancePercent,fairness{minShare,maxShare,overagePolicy}}`, `conflictResolution{onBudgetExhaustion,onSlaRisk,allowOverrideAnnotations}`.
- Keep versions `v1alpha1`; preserve existing metadata.

### 2) Go API Types
- Update `api/v1/carbonjobspec_types.go` structs to match CRD schema and add `Status.DAG` fields.
- Update `api/v1/carbonpolicy_types.go` to match policy schema; include `Fairness` and `ConflictResolution` sections.
- Ensure DeepCopy methods compile with current pattern (manual or codegen already present).

### 3) PolicyResolver Controller
- Implement `controllers/policy_resolver_controller.go` using `client-go/dynamic`:
  - Watch `CarbonJobSpec` and `CarbonPolicy`.
  - Resolve applicable `CarbonPolicy` via selectors and attach annotations:
    - `carbonkube.io/policy-name`, `carbonkube.io/criticality`, `carbonkube.io/tenant`, `carbonkube.io/carbon-aggressiveness`.
  - Write resolved context also to a namespaced `ConfigMap` cache for quick lookup.
- Expose Prometheus metrics for resolve hits/misses.

### 4) DAGController
- Implement `controllers/dag_controller.go`:
  - Maintain an in-memory graph per `dagId` built from `CarbonJobSpec`.
  - Compute `topoDepth`, longest path distances, `normalizedImportance∈[0,1]`.
  - Patch `status.dag` and add annotations:
    - `carbonkube.io/dag-id`, `carbonkube.io/stage-id`, `carbonkube.io/dag-importance`, `carbonkube.io/dag-critical`.
- Persist summary to a `ConfigMap` for cross-controller consumption.

### 5) TemporalPlanner
- Implement `controllers/temporal_planner_controller.go`:
  - Inputs: `t0`, `T`, `D`, policy (`maxDelaySeconds`, `maxSlowdownPercent`, `aggressiveness`, `criticality`), carbon intensity forecast `C(t)` per region.
  - Compute feasible window:
    - `t_min = t0`
    - `t_max = min(t0 + Δ_max, D − T, t_asap + s_max·T/100)`
    - If `criticality == Critical` or `t_max ≤ t_min` then schedule ASAP.
  - Carbon cost approximation:
    - `CO2(t) ≈ Σ_{k=0}^{n−1} C(t + kΔ)`
  - Interpolate start:
    - `t_final = t_asap + a·(t_carbon − t_asap)`
    - Clamp to `[t_min, t_max]`, snap to slot.
  - Annotate: `carbonkube.io/scheduled-at`.
- Provide forecast source: pluggable provider that reads Prometheus/HTTP; default simple provider with mock curve if empty.

### 6) BudgetEnforcer
- Implement `controllers/budget_enforcer_controller.go`:
  - Periodic loop: read energy/CO2 metrics from KeplerAttribution (Prometheus).
  - Aggregate per `tenantId` and maintain `status.tenants` map in a `ConfigMap`:
    - `usedCarbonMonthlyKg`, `remainingBudgetKg`, `overBudget`, `throttled`.
  - Influence scheduling:
    - If over budget and `overagePolicy == degrade`, lower effective `aggressiveness` for non-Critical jobs; add budget penalty term for scheduler.
    - If `reject`, add admission rule to deny BestEffort/Batch jobs.
- Export metrics for dashboards.

### 7) KeplerAttribution Aggregator
- Add `cmd/kepler-attr/main.go`:
  - Scrape Kepler per-pod energy (joules).
  - Map pods → tenant/namespace/job via annotations.
  - Convert to kWh, multiply by regional CI → kgCO2.
  - Export metrics:
    - `carbonkube_tenant_joules_total{tenant=...}`
    - `carbonkube_tenant_co2_kg_total{tenant=...}`
    - `carbonkube_namespace_co2_kg_total{namespace=...}`
- Helm: DaemonSet/Deployment, RBAC, ServiceMonitor.

### 8) Scheduler Plugin Updates
- Update `pkg/emissionplugin/plugin.go` scoring:
  - Inputs from annotations: `dag-importance`, `scheduled-at`, policy `criticality`, budget penalty, data-gravity cost, hardware perf/W.
  - Score per node/region:
    - `score = w1·(1 − C_norm) + w2·renewable_fraction + w3·dag_importance − w4·budget_penalty − w5·data_move_cost + w6·perf_per_watt_gain`
  - Normalize to framework’s expected range; honor bypass for Critical workloads.
- Respect `scheduled-at` by preferring nodes available near planned start.

### 9) Webhook Enhancements
- In `controllers/scheduler_webhook.go`, ensure policy context injection if missing and validate annotations consistency.
- Add admission check for `reject` overage policy for BestEffort/Batch.

### 10) Helm/RBAC/Deployments
- Extend `charts/carbon-kube/templates`:
  - Deployments for new controllers: `policy-resolver`, `dag-controller`, `temporal-planner`, `budget-enforcer`, `kepler-attr`.
  - ServiceAccounts, Roles/RoleBindings, ClusterRoles as needed to watch CRDs and patch resources.
  - ConfigMaps for DAG and tenant state; ServiceMonitors for metrics.
  - Values toggles for each component.

### 11) Build & Tests
- Makefile: add targets for new binaries/controllers; update lint/test aggregates.
- Dockerfile: add stages for controllers and kepler-attr.
- Go unit tests:
  - TemporalPlanner window calculations and `t_final` interpolation.
  - BudgetEnforcer policy reactions and penalties.
- Helm tests: `helm lint` and `helm template` in CI.

### 12) README Update (formulas + usage)
- Document CRDs with examples.
- Include formulas:
  - Feasible window: `t_min = t0`, `t_max = min(t0 + Δ_max, D − T, t_asap + s_max·T/100)`
  - Carbon cost: `CO2(t) ≈ Σ C(t + kΔ)`
  - Aggressive interpolation: `t_final = t_asap + a·(t_carbon − t_asap)`
  - Scheduler score: `score = w1·(1 − C_norm) + w2·R + w3·I − w4·B − w5·M + w6·P`
- Usage: install Helm chart, apply CRDs, run scheduler plugin; monitoring via Grafana.

## Verification Plan
- Build: `go build ./...` and image builds succeed locally.
- Tests: `go test ./...` green; Python tests (poller/RL) remain green.
- Helm: `helm lint` and `helm template` produce valid manifests.
- Smoke deploy on Kind/minikube (optional), submit example DAG and observe annotations and scheduling.

## Assumptions
- Continue using `client-go` watcher pattern; do not introduce `controller-runtime`.
- Carbon forecast source is pluggable; default to a simple provider if none configured.
- Kepler is available in-cluster; otherwise aggregator runs with a mock to keep build/tests passing.

## Deliverables
- Updated CRDs and Go API types.
- New controllers and aggregator binaries.
- Updated scheduler plugin scoring.
- Extended Helm chart/RBAC/ServiceMonitors.
- Updated Makefile/Dockerfile and unit tests.
- Updated README with formulas and examples.