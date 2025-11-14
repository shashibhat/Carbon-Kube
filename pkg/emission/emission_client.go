package emission

import (
    "context"
    "errors"
    "sync"
    "time"
)

// CarbonScoreClient provides carbon intensity scores.
type CarbonScoreClient interface {
    GetScores(ctx context.Context) ([]CarbonScore, error)
}

// memoryScoreClient is an in-memory implementation, useful for tests and
// for wiring in a CRD-backed implementation later.
type memoryScoreClient struct {
    mu     sync.RWMutex
    scores []CarbonScore
}

// NewMemoryScoreClient constructs a new in-memory score client.
func NewMemoryScoreClient(initial []CarbonScore) *memoryScoreClient {
    return &memoryScoreClient{scores: initial}
}

// UpdateScores replaces the internal slice (not concurrency-safe w.r.t readers).
func (c *memoryScoreClient) UpdateScores(scores []CarbonScore) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.scores = scores
}

// GetScores returns the current slice of scores.
func (c *memoryScoreClient) GetScores(ctx context.Context) ([]CarbonScore, error) {
    select {
    case <-ctx.Done():
        return nil, ctx.Err()
    default:
    }

    c.mu.RLock()
    defer c.mu.RUnlock()
    if len(c.scores) == 0 {
        return nil, errors.New("no scores available")
    }
    out := make([]CarbonScore, len(c.scores))
    copy(out, c.scores)
    return out, nil
}

// NewTestScore returns a deterministic CarbonScore for zone.
func NewTestScore(zone string, intensity float32) CarbonScore {
    return CarbonScore{
        Zone:               zone,
        IntensityGPerKwh:   intensity,
        CpuMultiplier:      1.0,
        ForecastUnixSecond: time.Now().Unix(),
    }
}
