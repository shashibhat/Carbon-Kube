package controllers

import (
    "context"
    "encoding/json"
    "fmt"
    "strconv"
    "math"
    "net/http"
    "net/url"
    "os"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

type GPUPlacementController struct {
    Client kubernetes.Interface
    HTTP *http.Client
}

func NewGPUPlacementController(cfg *rest.Config) (*GPUPlacementController, error) {
    client, err := kubernetes.NewForConfig(cfg)
    if err != nil {
        return nil, err
    }
    return &GPUPlacementController{Client: client, HTTP: &http.Client{}}, nil
}

type zonalIntensity struct {
	Zone      string  `json:"zone"`
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
                if zone == "" {
                    continue
                }
                cscore := c.zoneCarbonScore(ctx, zone)
                perf := c.nodePerfPerWatt(p)
                gpuScore := perf * 0.4
                carbonScore := (100.0 - cscore) * 0.6
                final := 0.6*carbonScore + 0.4*gpuScore
                ann := p.Annotations
                if ann == nil {
                    ann = map[string]string{}
                }
                ann["carbonkube.io/placement-hint"] = zone
                ann["carbonkube.io/carbonPriorityScore"] = fmt.Sprintf("%.2f", cscore)
                ann["carbonkube.io/placement-score"] = fmt.Sprintf("%.2f", final)
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
		return ""
	}
	data := cm.Data["zones"]
	var arr []zonalIntensity
	if err := json.Unmarshal([]byte(data), &arr); err != nil || len(arr) == 0 {
		return ""
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
    base := os.Getenv("CARBONKUBE_PROMETHEUS_URL")
    u, err := url.Parse(base)
    if err != nil { return 50 }
    u.Path = "/api/v1/query"
    q := url.Values{}
    q.Set("query", fmt.Sprintf("carbon_intensity_gco2_per_kwh{zone=\"%s\"}", zone))
    u.RawQuery = q.Encode()
    resp, err := c.HTTP.Get(u.String())
    if err != nil { return 50 }
    defer resp.Body.Close()
    var body struct{ Data struct{ Result []struct{ Value []interface{} `json:"value"` } `json:"result"` } `json:"data"` }
    if err := json.NewDecoder(resp.Body).Decode(&body); err != nil { return 50 }
    if len(body.Data.Result) == 0 { return 50 }
    valStr := body.Data.Result[0].Value[1].(string)
    f, _ := strconv.ParseFloat(valStr, 64)
    return f
}

func (c *GPUPlacementController) nodePerfPerWatt(p *v1.Pod) float64 {
    nodes, _ := c.Client.CoreV1().Nodes().List(context.Background(), metav1.ListOptions{})
    mm := 0.0
    for _, n := range nodes.Items {
        perf := 0.0
        if v, ok := n.Labels["carbonkube.io/gpu-perf-watt"]; ok {
            f, _ := strconv.ParseFloat(v, 64)
            perf = f
        }
        mm = math.Max(mm, perf)
    }
    node := p.Spec.NodeName
    if node == "" { return 0 }
    nd, err := c.Client.CoreV1().Nodes().Get(context.Background(), node, metav1.GetOptions{})
    if err != nil { return 0 }
    perf := 0.0
    if v, ok := nd.Labels["carbonkube.io/gpu-perf-watt"]; ok { f, _ := strconv.ParseFloat(v, 64); perf = f }
    if mm <= 0 { return 0 }
    return math.Min(perf/mm*100.0, 100.0)
}
