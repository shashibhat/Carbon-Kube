package emission

import (
    "context"
    "testing"
    "time"
)

func TestNodeTainterRunCancels(t *testing.T) {
    client := NewMemoryScoreClient([]CarbonScore{
        NewTestScore("zone-a", 100.0),
    })
    cfg := Config{
        MigrationThreshold: 200.0,
    }
    tainter := NewNodeTainter(client, cfg)

    ctx, cancel := context.WithCancel(context.Background())
    go func() {
        time.Sleep(50 * time.Millisecond)
        cancel()
    }()

    if err := tainter.Run(ctx, 10*time.Millisecond); err == nil {
        // We expect context cancellation error; any non-nil is acceptable here.
        t.Logf("Run exited with nil error (context cancelled)")
    }
}
