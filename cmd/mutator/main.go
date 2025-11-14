package main

import (
    "context"
    "log"
    "net/http"
    "time"

    "github.com/example/carbon-kube/pkg/emission"
)

// Demo mutator binary. In production this would be registered as a
// scheduler plugin (e.g., with Katalyst), but here we simulate scoring
// on a timer and expose metrics.
func main() {
    mux := http.NewServeMux()
    emission.RegisterMetrics(mux)

    go func() {
        log.Println("metrics server listening on :9090")
        if err := http.ListenAndServe(":9090", mux); err != nil {
            log.Fatalf("metrics server failed: %v", err)
        }
    }()

    cfg := emission.Config{
        MigrationThreshold: 200.0,
        GreenZones:         []string{"us-west-2a"},
        RLEnabled:          true,
    }

    client := emission.NewMemoryScoreClient([]emission.CarbonScore{
        emission.NewTestScore("us-west-2a", 100.0),
        emission.NewTestScore("us-west-2b", 600.0),
    })
    mut := emission.NewEmissionMutator(client, cfg)

    pod := emission.Pod{
        Name:            "demo-pod",
        Namespace:       "default",
        CPUMilliRequest: 500,
    }
    nodes := []emission.Node{
        {Name: "node-a", Zone: "us-west-2a", Score: 10},
        {Name: "node-b", Zone: "us-west-2b", Score: 10},
    }

    for {
        ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
        if err := mut.Mutate(ctx, pod, nodes); err != nil {
            log.Printf("mutate error: %v", err)
        } else {
            emission.RecordMigration()
            log.Printf("mutated nodes: %+v", nodes)
        }
        cancel()
        time.Sleep(30 * time.Second)
    }
}
