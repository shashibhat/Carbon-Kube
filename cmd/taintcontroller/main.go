package main

import (
    "context"
    "log"
    "time"

    "github.com/example/carbon-kube/pkg/emission"
)

// Node taint controller entrypoint. This uses the in-memory score client
// and logs which zones would be tainted.
func main() {
    cfg := emission.Config{
        MigrationThreshold: 200.0,
    }
    client := emission.NewMemoryScoreClient([]emission.CarbonScore{
        emission.NewTestScore("us-west-2a", 150.0),
        emission.NewTestScore("us-west-2b", 450.0),
    })
    tainter := emission.NewNodeTainter(client, cfg)

    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    log.Println("starting node taint controller demo")
    if err := tainter.Run(ctx, 15*time.Second); err != nil {
        log.Printf("taint controller exited: %v", err)
    }
}
