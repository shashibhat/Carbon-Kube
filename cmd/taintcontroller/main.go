package main

import (
    "context"
    "log"
    "time"
    "os"

    "github.com/example/carbon-kube/pkg/emission"
)

// Node taint controller entrypoint. This uses the in-memory score client
// and logs which zones would be tainted.
func main() {
    cfg := emission.Config{
        MigrationThreshold: 200.0,
    }
    ns := os.Getenv("POD_NAMESPACE")
    if ns == "" { ns = "default" }
    client, err := emission.NewKubeCarbonScoreClient(ns)
    if err != nil { log.Fatalf("kube score client: %v", err) }
    tainter := emission.NewNodeTainter(client, cfg)

    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    log.Println("starting node taint controller demo")
    if err := tainter.Run(ctx, 15*time.Second); err != nil {
        log.Printf("taint controller exited: %v", err)
    }
}
