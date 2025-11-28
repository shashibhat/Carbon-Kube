package main

import (
    "context"
    "log"
    "net/http"
    "time"
    "os"

    "github.com/shashibhat/Carbon-Kube/pkg/emission"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/rest"
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

    ns := os.Getenv("POD_NAMESPACE")
    if ns == "" {
        ns = "default"
    }
    client, err := emission.NewKubeCarbonScoreClient(ns)
    if err != nil {
        log.Fatalf("kube score client: %v", err)
    }
    mut := emission.NewEmissionMutator(client, cfg)

    pod := emission.Pod{Name: "demo-pod", Namespace: ns, CPUMilliRequest: 500}

    rc, err := rest.InClusterConfig()
    if err != nil {
        log.Fatalf("kube config: %v", err)
    }
    ks, err := kubernetes.NewForConfig(rc)
    if err != nil {
        log.Fatalf("kube client: %v", err)
    }

    for {
        ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
        k8sNodes, err := ks.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
        if err != nil {
            log.Printf("list nodes: %v", err)
            cancel()
            time.Sleep(30 * time.Second)
            continue
        }
        nodes := make([]emission.Node, 0, len(k8sNodes.Items))
        for _, n := range k8sNodes.Items {
            zone := n.Labels["topology.kubernetes.io/zone"]
            nodes = append(nodes, emission.Node{Name: n.Name, Zone: zone, Score: 10, Labels: n.Labels})
        }
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
