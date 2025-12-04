package main

import (
    "context"
    "os"
    "os/signal"
    "syscall"
    "time"
    "k8s.io/client-go/rest"
    "github.com/shashibhat/Carbon-Kube/controllers"
)

func main() {
    cfg, err := rest.InClusterConfig()
    if err != nil {
        return
    }
    ns := os.Getenv("NAMESPACE")
    if ns == "" {
        ns = "default"
    }
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    go func() { c, _ := controllers.NewPolicyResolver(cfg); _ = c.Start(ctx, ns) }()
    go func() { c, _ := controllers.NewDAGController(cfg); _ = c.Start(ctx, ns) }()
    go func() { c, _ := controllers.NewTemporalPlanner(cfg); _ = c.Start(ctx, ns) }()
    go func() { c, _ := controllers.NewBudgetEnforcer(cfg); _ = c.Start(ctx, ns) }()
    sig := make(chan os.Signal, 1)
    signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
    <-sig
    time.Sleep(1 * time.Second)
}
