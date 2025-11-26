package emission

import (
    "context"
    "encoding/json"
    "log"
    "time"

    v1 "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/types"
    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/rest"
)

// NodeTainter is a stub controller that periodically reads carbon scores
// and determines which zones should be tainted.
//
// In a real implementation, this would use controller-runtime with a
// Kubernetes client to patch Node objects and add/remove taints.
type NodeTainter struct {
    scoreClient CarbonScoreClient
    cfg         Config
    kube        *kubernetes.Clientset
}

// NewNodeTainter constructs a new NodeTainter.
func NewNodeTainter(client CarbonScoreClient, cfg Config) *NodeTainter {
    // Try to init kube client (in-cluster). Best-effort.
    var ks *kubernetes.Clientset
    if cfg.MigrationThreshold > 0 {
        if rc, err := rest.InClusterConfig(); err == nil {
            if cs, err := kubernetes.NewForConfig(rc); err == nil {
                ks = cs
            }
        }
    }
    return &NodeTainter{scoreClient: client, cfg: cfg, kube: ks}
}

// Run is a simple loop that logs which zones would be tainted based on
// the current scores. It is intentionally side-effect free so you can
// drop in real kube client logic.
func (t *NodeTainter) Run(ctx context.Context, interval time.Duration) error {
    ticker := time.NewTicker(interval)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-ticker.C:
            scores, err := t.scoreClient.GetScores(ctx)
            if err != nil {
                log.Printf("taint controller: get scores error: %v", err)
                continue
            }
            for _, s := range scores {
                if t.kube == nil {
                    // log-only if kube client unavailable
                    if s.IntensityGPerKwh > t.cfg.MigrationThreshold {
                        log.Printf("would taint zone=%s (intensity=%.2f)", s.Zone, s.IntensityGPerKwh)
                    } else {
                        log.Printf("would untaint zone=%s (intensity=%.2f)", s.Zone, s.IntensityGPerKwh)
                    }
                    continue
                }
                // List nodes in this zone
                nodes, err := t.kube.CoreV1().Nodes().List(ctx, metav1.ListOptions{LabelSelector: "topology.kubernetes.io/zone=" + s.Zone})
                if err != nil {
                    log.Printf("list nodes: %v", err)
                    continue
                }
                for _, node := range nodes.Items {
                    if s.IntensityGPerKwh > t.cfg.MigrationThreshold {
                        // add taint
                        taint := v1.Taint{Key: "carbon-kube/high-intensity", Value: s.Zone, Effect: v1.TaintEffectNoSchedule}
                        patch := buildTaintPatch(&node, taint, true)
                        if _, err := t.kube.CoreV1().Nodes().Patch(ctx, node.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{}); err != nil {
                            log.Printf("patch taint add: %v", err)
                        } else {
                            log.Printf("tainted node=%s zone=%s", node.Name, s.Zone)
                        }
                    } else {
                        // remove taint
                        taint := v1.Taint{Key: "carbon-kube/high-intensity"}
                        patch := buildTaintPatch(&node, taint, false)
                        if _, err := t.kube.CoreV1().Nodes().Patch(ctx, node.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{}); err != nil {
                            log.Printf("patch taint remove: %v", err)
                        } else {
                            log.Printf("untainted node=%s zone=%s", node.Name, s.Zone)
                        }
                    }
                }
            }
        }
    }
}

// buildTaintPatch returns a strategic merge patch to add/remove a taint.
func buildTaintPatch(node *v1.Node, taint v1.Taint, add bool) []byte {
    var newTaints []v1.Taint
    if add {
        newTaints = append(node.Spec.Taints, taint)
    } else {
        for _, t := range node.Spec.Taints {
            if t.Key != taint.Key {
                newTaints = append(newTaints, t)
            }
        }
    }
    patch := map[string]interface{}{
        "spec": map[string]interface{}{
            "taints": newTaints,
        },
    }
    b, _ := json.Marshal(patch)
    return b
}
