//go:build katalyst
package katalyst

import (
    "context"
    v1 "k8s.io/api/core/v1"
    "k8s.io/kubernetes/pkg/scheduler/framework"
)

type CarbonScorePlugin struct {}

func (p *CarbonScorePlugin) Name() string { return "CarbonKubeScorePlugin" }

func (p *CarbonScorePlugin) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
    ann := pod.Annotations
    base := int64(50)
    carbonPenalty := int64(0)
    dataGravityPenalty := int64(0)
    if ann != nil {
        if v, ok := ann["carbonkube.io/carbonPriorityScore"]; ok && v != "" {
            base = 100
        }
        if v, ok := ann["carbonkube.io/dataGravityPenalty"]; ok && v != "" {
            dataGravityPenalty = 10
        }
    }
    s := base - carbonPenalty - dataGravityPenalty
    if s < 0 { s = 0 }
    if s > 100 { s = 100 }
    return s, framework.NewStatus(framework.Success, "")
}

func (p *CarbonScorePlugin) ScoreExtensions() framework.ScoreExtensions { return nil }

var _ framework.ScorePlugin = &CarbonScorePlugin{}
