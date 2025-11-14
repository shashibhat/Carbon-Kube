package emission

import (
    "context"
    "fmt"
)

// Pod is a simplified Pod representation for scoring purposes.
type Pod struct {
    Name            string
    Namespace       string
    CPUMilliRequest int64
}

// Node is a simplified Node representation whose Score is adjusted.
type Node struct {
    Name   string
    Zone   string
    Score  float64
    Labels map[string]string
}

// EmissionMutator adjusts node scores based on carbon intensity.
type EmissionMutator struct {
    scoreClient CarbonScoreClient
    cfg         Config
}

// NewEmissionMutator constructs a new EmissionMutator.
func NewEmissionMutator(client CarbonScoreClient, cfg Config) *EmissionMutator {
    return &EmissionMutator{
        scoreClient: client,
        cfg:         cfg,
    }
}

// Mutate updates node.Score based on estimated emissions. High-emission
// zones are penalized by subtracting the emission cost.
func (m *EmissionMutator) Mutate(ctx context.Context, pod Pod, nodes []Node) error {
    scores, err := m.scoreClient.GetScores(ctx)
    if err != nil {
        return fmt.Errorf("get scores: %w", err)
    }

    for i := range nodes {
        node := &nodes[i]
        score := lookupZoneScore(scores, node.Zone)
        emission := calculateEmission(pod, score)
        if emission > float64(m.cfg.MigrationThreshold) {
            node.Score -= emission
        }
    }
    return nil
}

func lookupZoneScore(scores []CarbonScore, zone string) CarbonScore {
    for _, s := range scores {
        if s.Zone == zone {
            return s
        }
    }
    // Default neutral intensity when no score is available.
    return CarbonScore{
        Zone:             zone,
        IntensityGPerKwh: 400.0,
        CpuMultiplier:    1.0,
    }
}

func calculateEmission(pod Pod, score CarbonScore) float64 {
    cpuCores := float64(pod.CPUMilliRequest) / 1000.0
    return cpuCores * float64(score.IntensityGPerKwh) * float64(score.CpuMultiplier)
}
