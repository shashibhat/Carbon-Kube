package controllers

import (
    "context"
    "os"
    "time"
    "strconv"
    v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    unstructured "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
    "k8s.io/apimachinery/pkg/runtime/schema"
    "k8s.io/apimachinery/pkg/watch"
    "k8s.io/client-go/dynamic"
    "k8s.io/client-go/rest"
    "k8s.io/klog/v2"
    "github.com/shashibhat/Carbon-Kube/pkg/providers"
)

type TemporalPlanner struct {
    Dyn dynamic.Interface
}

var tpJobGVR = schema.GroupVersionResource{Group: "carbonkube.io", Version: "v1", Resource: "carbonjobs"}

func NewTemporalPlanner(cfg *rest.Config) (*TemporalPlanner, error) {
    dyn, err := dynamic.NewForConfig(cfg)
    if err != nil {
        return nil, err
    }
    return &TemporalPlanner{Dyn: dyn}, nil
}

func (t *TemporalPlanner) Start(ctx context.Context, namespace string) error {
    backoff := time.Second
    for {
        w, err := t.Dyn.Resource(tpJobGVR).Namespace(namespace).Watch(ctx, v1.ListOptions{})
        if err != nil {
            klog.Errorf("TemporalPlanner watch error: %v", err)
            select {
            case <-time.After(backoff):
                backoff = time.Duration(minInt64(int64(backoff*2), int64(30*time.Second)))
                continue
            case <-ctx.Done():
                return ctx.Err()
            }
        }
        ch := w.ResultChan()
        for {
            select {
            case <-ctx.Done():
                return ctx.Err()
            case e, ok := <-ch:
                if !ok {
                    klog.Warning("TemporalPlanner watch channel closed; restarting")
                    time.Sleep(backoff)
                    break
                }
                if e.Type == watch.Added || e.Type == watch.Modified {
                    u := e.Object.DeepCopyObject().(*unstructured.Unstructured)
                    spec := u.UnstructuredContent()["spec"].(map[string]interface{})
                    anns := map[string]string{}
                    if md, ok := u.UnstructuredContent()["metadata"].(map[string]interface{}); ok {
                        if a, ok := md["annotations"].(map[string]interface{}); ok {
                            for k, v := range a { if s, ok := v.(string); ok { anns[k] = s } }
                        }
                    }
                    ns := u.GetNamespace()
                    t0 := time.Now().UTC()
                    T := seconds(spec["estimatedRuntimeSeconds"]) * time.Second
                    deadline := parseDeadline(spec["deadline"], t0, anns)
                    agg := clamp01(parseFloat(anns["carbonkube.io/carbon-aggressiveness"]))
                    critical := anns["carbonkube.io/criticality"] == "Critical"
                    asap := t0
                    maxDelaySec := parseIntAnnotation(anns, "carbonkube.io/maxDelaySeconds", func() int { if critical { return 0 } ; return 0 })
                    maxSlowdownPct := parseIntAnnotation(anns, "carbonkube.io/maxSlowdownPercent", func() int { return 0 })
                    tDelay := t0.Add(time.Duration(maxDelaySec) * time.Second)
                    tSlow := t0.Add(time.Duration(maxSlowdownPct) * T / 100)
                    tMin, tMax := computeScheduleWindow(t0, T, tDelay, deadline, tSlow)
                    final := asap
                    if !critical && tMax.After(tMin) {
                        slot := 15 * time.Minute
                        region := firstRegion(spec)
                        if region != "" {
                            fp := selectProvider()
                            points, errF := fp.Forecast(region, tMin, tMax.Add(T), slot)
                            if errF == nil && len(points) > 0 {
                                tCarbon := argminCarbonSeries(tMin, tMax, slot, T, points)
                                final = clampTime(asap.Add(time.Duration(agg)*tCarbon.Sub(asap)), tMin, tMax)
                            } else {
                                klog.Warningf("TemporalPlanner forecast unavailable: err=%v len=%d", errF, len(points))
                            }
                        } else {
                            klog.Warning("TemporalPlanner no region found; falling back to ASAP")
                        }
                    }
                    if md, ok := u.UnstructuredContent()["metadata"].(map[string]interface{}); ok {
                        a := map[string]interface{}{}
                        if v, ok := md["annotations"].(map[string]interface{}); ok { a = v }
                        a["carbonkube.io/scheduled-at"] = final.Format(time.RFC3339)
                        md["annotations"] = a
                    }
                    if _, err := t.Dyn.Resource(tpJobGVR).Namespace(ns).Update(ctx, u, v1.UpdateOptions{}); err != nil {
                        klog.Errorf("TemporalPlanner update failed: %v", err)
                    }
                }
            }
        }
    }
}

func seconds(v interface{}) time.Duration {
    switch x := v.(type) {
    case int64:
        return time.Duration(x)
    case int:
        return time.Duration(x)
    case float64:
        return time.Duration(int64(x))
    default:
        return 0
    }
}

func parseDeadline(v interface{}, t0 time.Time, anns map[string]string) time.Time {
    if s, ok := v.(string); ok && s != "" {
        if tt, err := time.Parse(time.RFC3339, s); err == nil {
            return tt
        }
    }
    mode := anns["carbonkube.io/deadlineMode"]
    if mode == "Relative" {
        if drs := anns["carbonkube.io/defaultRelativeDeadlineSeconds"]; drs != "" {
            if v, err := strconv.Atoi(drs); err == nil { return t0.Add(time.Duration(v) * time.Second) }
        }
    } else if mode == "Absolute" {
        return t0.Add(24 * time.Hour)
    }
    return t0.Add(24 * time.Hour)
}

func parseFloat(s string) float64 {
    if f, err := strconv.ParseFloat(s, 64); err == nil {
        return f
    }
    return 0
}

func minTime(a, b, c time.Time) time.Time {
    m := a
    if b.Before(m) {
        m = b
    }
    if c.Before(m) {
        m = c
    }
    return m
}

func clampTime(x, lo, hi time.Time) time.Time {
    if x.Before(lo) {
        return lo
    }
    if x.After(hi) {
        return hi
    }
    return x
}

// argminCarbonSeries selects the start time within [tMin,tMax] that minimizes carbon cost.

func argminCarbonSeries(tMin time.Time, tMax time.Time, slot time.Duration, T time.Duration, points []providers.ForecastPoint) time.Time {
    best := tMin
    bestC := 1e9
    for t := tMin; !t.After(tMax); t = t.Add(slot) {
        c := carbonCostSeries(t, T, slot, points)
        if c < bestC {
            best = t
            bestC = c
        }
    }
    return best
}

func carbonCostSeries(start time.Time, T time.Duration, slot time.Duration, points []providers.ForecastPoint) float64 {
    n := int(T / slot)
    total := 0.0
    for k := 0; k < n; k++ {
        tt := start.Add(time.Duration(k) * slot)
        total += lookup(points, tt)
    }
    return total
}

func lookup(points []providers.ForecastPoint, ts time.Time) float64 {
    closest := 0.0
    minDiff := int64(1<<62)
    for _, p := range points {
        d := abs64(p.Timestamp.Unix() - ts.Unix())
        if d < minDiff {
            minDiff = d
            closest = p.Value
        }
    }
    return closest
}

func abs64(x int64) int64 {
    if x < 0 { return -x }
    return x
}

func firstRegion(spec map[string]interface{}) string {
    if v, ok := spec["dataSources"].([]interface{}); ok {
        for _, it := range v {
            if m, ok := it.(map[string]interface{}); ok {
                if r, ok := m["region"].(string); ok && r != "" {
                    return r
                }
            }
        }
    }
    return ""
}

func selectProvider() providers.ForecastProvider {
    mode := os.Getenv("CARBONKUBE_FORECAST_PROVIDER")
    if mode == "prometheus" {
        base := os.Getenv("CARBONKUBE_PROMETHEUS_URL")
        return providers.NewPrometheusProvider(base, "carbon_intensity_gco2_per_kwh", "zone")
    }
    base := os.Getenv("CARBONKUBE_ELECTRICITYMAPS_URL")
    token := os.Getenv("CARBONKUBE_ELECTRICITYMAPS_TOKEN")
    return providers.NewElectricityMapsProvider(base, token)
}

// Helpers for tests and clarity
func parseIntAnnotation(anns map[string]string, key string, def func() int) int {
    s := anns[key]
    if s == "" { return def() }
    v, err := strconv.Atoi(s)
    if err != nil { return def() }
    return v
}

func clamp01(f float64) float64 { if f < 0 { return 0 } ; if f > 1 { return 1 } ; return f }

func computeScheduleWindow(t0 time.Time, T time.Duration, tDelay time.Time, deadline time.Time, tSlow time.Time) (time.Time, time.Time) {
    tMin := t0
    tMax := minTime(tDelay, deadline.Add(-T), tSlow)
    return tMin, tMax
}

func computeFinalStartTime(asap time.Time, tMin time.Time, tMax time.Time, slot time.Duration, T time.Duration, points []providers.ForecastPoint, agg float64, critical bool) time.Time {
    if critical || !tMax.After(tMin) || len(points) == 0 { return asap }
    tCarbon := argminCarbonSeries(tMin, tMax, slot, T, points)
    return clampTime(asap.Add(time.Duration(clamp01(agg))*tCarbon.Sub(asap)), tMin, tMax)
}

func minInt64(a, b int64) int64 { if a < b { return a } ; return b }
