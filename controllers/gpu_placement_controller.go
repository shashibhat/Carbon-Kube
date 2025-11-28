package controllers

import (
    "context"
    "encoding/json"
    "fmt"
    "strconv"
    v1 "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/watch"
    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/rest"
)

type GPUPlacementController struct {
    Client kubernetes.Interface
}

func NewGPUPlacementController(cfg *rest.Config) (*GPUPlacementController, error) {
    client, err := kubernetes.NewForConfig(cfg)
    if err != nil {
        return nil, err
    }
    return &GPUPlacementController{Client: client}, nil
}

type zonalIntensity struct {
    Zone string `json:"zone"`
    Intensity float64 `json:"intensity"`
}

func (c *GPUPlacementController) Start(ctx context.Context, namespace string) error {
    w, err := c.Client.CoreV1().Pods(namespace).Watch(ctx, metav1.ListOptions{})
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
                p := e.Object.(*v1.Pod)
                if p.Spec.NodeName != "" {
                    continue
                }
                if !requiresGPU(p) {
                    continue
                }
                zone := c.selectZone(ctx)
                score := c.zoneCarbonScore(ctx, zone)
                ann := p.Annotations
                if ann == nil {
                    ann = map[string]string{}
                }
                ann["carbonkube.io/placement-hint"] = zone
                ann["carbonkube.io/carbonPriorityScore"] = fmt.Sprintf("%.2f", score)
                p.Annotations = ann
                _, _ = c.Client.CoreV1().Pods(p.Namespace).Update(ctx, p, metav1.UpdateOptions{})
            }
        }
    }
}

func requiresGPU(pod *v1.Pod) bool {
    for _, c := range pod.Spec.Containers {
        if c.Resources.Requests != nil {
            if q, ok := c.Resources.Requests["nvidia.com/gpu"]; ok {
                v, _ := strconv.Atoi(q.String())
                if v > 0 {
                    return true
                }
            }
            for r := range c.Resources.Requests {
                if len(r.String()) >= 12 && r.String()[:12] == "nvidia.com/" {
                    return true
                }
            }
        }
    }
    return false
}

func (c *GPUPlacementController) selectZone(ctx context.Context) string {
    cm, err := c.Client.CoreV1().ConfigMaps("default").Get(ctx, "carbon-intensity-data", metav1.GetOptions{})
    if err != nil {
        return "us-east-1"
    }
    data := cm.Data["zones"]
    var arr []zonalIntensity
    if err := json.Unmarshal([]byte(data), &arr); err != nil || len(arr) == 0 {
        return "us-east-1"
    }
    best := arr[0]
    for _, z := range arr {
        if z.Intensity < best.Intensity {
            best = z
        }
    }
    return best.Zone
}

func (c *GPUPlacementController) zoneCarbonScore(ctx context.Context, zone string) float64 {
    cm, err := c.Client.CoreV1().ConfigMaps("default").Get(ctx, "carbon-intensity-data", metav1.GetOptions{})
    if err != nil {
        return 50
    }
    data := cm.Data["zones"]
    var arr []zonalIntensity
    if err := json.Unmarshal([]byte(data), &arr); err != nil {
        return 50
    }
    var intensity float64
    for _, z := range arr {
        if z.Zone == zone {
            intensity = z.Intensity
            break
        }
    }
    if intensity == 0 {
        return 50
    }
    s := 100 - (intensity / 10)
    if s < 0 {
        s = 0
    }
    if s > 100 {
        s = 100
    }
    return s
}
