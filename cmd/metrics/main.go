package main

import (
    "log"
    "net/http"

    "github.com/example/carbon-kube/pkg/emission"
)

// Standalone metrics exporter.
func main() {
    mux := http.NewServeMux()
    emission.RegisterMetrics(mux)

    log.Println("metrics server listening on :9090")
    if err := http.ListenAndServe(":9090", mux); err != nil {
        log.Fatalf("metrics server failed: %v", err)
    }
}
