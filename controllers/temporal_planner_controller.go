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
    w, err := t.Dyn.Resource(tpJobGVR).Namespace(namespace).Watch(ctx, v1.ListOptions{})
    if err != nil {
        return err
    }
    ch := w.ResultChan()
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case e, ok := <-ch:
            if !ok {
                return nil
            }
            if e.Type == watch.Added || e.Type == watch.Modified {
                u := e.Object.DeepCopyObject().(*unstructured.Unstructured)
                spec := u.UnstructuredContent()["spec"].(map[string]interface{})
                ns := u.GetNamespace()
                t0 := time.Now().UTC()
                T := seconds(spec["estimatedRuntimeSeconds"]) * time.Second
                deadline := parseDeadline(spec["deadline"], t0, spec)
                policyAgg := 1.0
                if v, ok := u.GetAnnotations()["carbonkube.io/carbon-aggressiveness"]; ok {
                    policyAgg = parseFloat(v)
                }
                critical := false
                if v, ok := u.GetAnnotations()["carbonkube.io/criticality"]; ok {
                    critical = v == "Critical"
                }
                asap := t0
                dMax := seconds(spec["maxDelaySeconds"]) * time.Second
                sMax := seconds(spec["maxSlowdownPercent"]) // interpreted as percent
                tMin := t0
                tMax := minTime(t0.Add(dMax), deadline.Add(-T), asap.Add(time.Duration(sMax)*T/100))
                final := asap
                if !critical && tMax.After(tMin) {
                    slot := 15 * time.Minute
                    region := firstRegion(spec)
                    fp := selectProvider()
                    points, errF := fp.Forecast(region, tMin, tMax.Add(T), slot)
                    if errF == nil && len(points) > 0 {
                        tCarbon := argminCarbonSeries(tMin, tMax, slot, T, points)
                        final = clampTime(asap.Add(time.Duration(policyAgg)*tCarbon.Sub(asap)), tMin, tMax)
                    }
                }
                if md, ok := u.UnstructuredContent()["metadata"].(map[string]interface{}); ok {
                    anns := map[string]interface{}{}
                    if a, ok := md["annotations"].(map[string]interface{}); ok {
                        anns = a
                    }
                    anns["carbonkube.io/scheduled-at"] = final.Format(time.RFC3339)
                    md["annotations"] = anns
                }
                _, _ = t.Dyn.Resource(tpJobGVR).Namespace(ns).Update(ctx, u, v1.UpdateOptions{})
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

func parseDeadline(v interface{}, t0 time.Time, spec map[string]interface{}) time.Time {
    if s, ok := v.(string); ok && s != "" {
        if tt, err := time.Parse(time.RFC3339, s); err == nil {
            return tt
        }
    }
    if sla, ok := spec["sla"].(map[string]interface{}); ok {
        if mode, ok := sla["deadlineMode"].(string); ok && mode == "Relative" {
            if dr, ok := sla["defaultRelativeDeadlineSeconds"].(int64); ok {
                return t0.Add(time.Duration(dr) * time.Second)
            }
        }
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

func argminCarbon(asap, tMin, tMax time.Time, slot time.Duration, T time.Duration) time.Time {
    best := tMin
    bestC := 1e9
    for t := tMin; !t.After(tMax); t = t.Add(slot) {
        c := carbonCostSeries(t, T, slot, []providers.ForecastPoint{})
        if c < bestC {
            best = t
            bestC = c
        }
    }
    return best
}

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
