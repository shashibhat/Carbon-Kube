## Answer

* The code uses Kubernetes scheduler, not Karpenter.

* A custom `kube-scheduler` profile (`carbon-kube-scheduler`) is deployed with custom plugins.

## Evidence

* `charts/carbon-kube/templates/deployment.yaml:47` runs `/usr/local/bin/kube-scheduler` with a scheduler config.

* `charts/carbon-kube/templates/configmap.yaml:32-35` defines `apiVersion: kubescheduler.config.k8s.io/v1beta3`, `kind: KubeSchedulerConfiguration`, with profile `schedulerName: carbon-kube-scheduler` and custom plugins.

* `docs/DEPLOYMENT.md:443,575` and multiple test workloads set `schedulerName: carbon-kube-scheduler`.

* Custom plugins implemented: `pkg/katalyst/enhanced_scheduler_plugin.go` and `gpu-extension/scheduler/gpu_scheduler_plugin.go` (Filter/Score integrations).

* No references to `karpenter.sh` or `karpenter.k8s.aws` anywhere.

* Descheduler library present: `third_party/kubewharf/katalyst-core/pkg/controller/tide/tide.go:42` imports `sigs.k8s.io/descheduler`.

## Proposed Next Steps (on approval)

1. Add a brief clarification in `README.md` that the project uses a customized Kubernetes scheduler (with plugins) and not Karpenter.
2. Update the architecture diagram caption around `Karpenter / Rescheduler` to reflect descheduler/Katalyst components used here.
3. Cross-link to `charts/carbon-kube/templates/configmap.yaml` showing the enabled plugins for discoverability.

## Verification (no changes)

* Point to and open the referenced files to review the configs and plugin registrations.

* If desired, list all `schedulerName:` occurrences to confirm consistent usage across examples and tests.

