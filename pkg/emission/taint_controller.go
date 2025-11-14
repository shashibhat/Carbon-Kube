package emission

import (
    "context"
    "log"
    "time"
)

// NodeTainter is a stub controller that periodically reads carbon scores
// and determines which zones should be tainted.
//
// In a real implementation, this would use controller-runtime with a
// Kubernetes client to patch Node objects and add/remove taints.
type NodeTainter struct {
    scoreClient CarbonScoreClient
    cfg         Config
}

// NewNodeTainter constructs a new NodeTainter.
func NewNodeTainter(client CarbonScoreClient, cfg Config) *NodeTainter {
    return &NodeTainter{
        scoreClient: client,
        cfg:         cfg,
    }
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
                if s.IntensityGPerKwh > t.cfg.MigrationThreshold {
                    log.Printf("would taint nodes in zone=%s (intensity=%.2f)", s.Zone, s.IntensityGPerKwh)
                } else {
                    log.Printf("would untaint nodes in zone=%s (intensity=%.2f)", s.Zone, s.IntensityGPerKwh)
                }
            }
        }
    }
}
