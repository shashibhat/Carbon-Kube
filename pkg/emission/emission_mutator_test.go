package emission

import (
    "context"
    "testing"
)

func TestMutatePenalizesHighEmissionZone(t *testing.T) {
    ctx := context.Background()

    client := NewMemoryScoreClient([]CarbonScore{
        NewTestScore("green-zone", 100.0),
        NewTestScore("brown-zone", 800.0),
    })

    cfg := Config{
        MigrationThreshold: 200.0,
        GreenZones:         []string{"green-zone"},
        RLEnabled:          false,
    }

    m := NewEmissionMutator(client, cfg)

    pod := Pod{
        Name:            "job-1",
        Namespace:       "default",
        CPUMilliRequest: 1000, // 1 core
    }

    nodes := []Node{
        {Name: "n1", Zone: "green-zone", Score: 10},
        {Name: "n2", Zone: "brown-zone", Score: 10},
    }

    if err := m.Mutate(ctx, pod, nodes); err != nil {
        t.Fatalf("Mutate() returned error: %v", err)
    }

    if nodes[0].Score >= nodes[1].Score {
        t.Fatalf("expected brown-zone node to be penalized, got n1=%f n2=%f", nodes[0].Score, nodes[1].Score)
    }
}
