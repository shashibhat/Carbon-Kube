package emission

import (
    "context"
    "fmt"

    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
    "k8s.io/apimachinery/pkg/runtime/schema"
    "k8s.io/client-go/dynamic"
    "k8s.io/client-go/rest"
    "k8s.io/client-go/tools/clientcmd"
)

// kubeCarbonScoreClient reads CarbonScore CRD from the cluster.
type kubeCarbonScoreClient struct {
    dyn       dynamic.Interface
    namespace string
}

var carbonscoresGVR = schema.GroupVersionResource{
    Group:    "emission.carbon-kube.io",
    Version:  "v1alpha1",
    Resource: "carbonscores",
}

// NewKubeCarbonScoreClient constructs a CRD-backed client.
func NewKubeCarbonScoreClient(namespace string) (CarbonScoreClient, error) {
    cfg, err := rest.InClusterConfig()
    if err != nil {
        // fallback to local kubeconfig for dev environments
        cfg, err = clientcmd.BuildConfigFromFlags("", clientcmd.RecommendedHomeFile)
    }
    if err != nil {
        return nil, fmt.Errorf("kube config: %w", err)
    }
    dyn, err := dynamic.NewForConfig(cfg)
    if err != nil {
        return nil, fmt.Errorf("dynamic client: %w", err)
    }
    return &kubeCarbonScoreClient{dyn: dyn, namespace: namespace}, nil
}

func (k *kubeCarbonScoreClient) GetScores(ctx context.Context) ([]CarbonScore, error) {
    obj, err := k.dyn.Resource(carbonscoresGVR).Namespace(k.namespace).Get(ctx, "global", metav1.GetOptions{})
    if err != nil {
        return nil, fmt.Errorf("get carbonscore: %w", err)
    }
    return parseScores(obj)
}

func parseScores(u *unstructured.Unstructured) ([]CarbonScore, error) {
    spec, ok := u.Object["spec"].(map[string]interface{})
    if !ok {
        return nil, fmt.Errorf("missing spec")
    }
    raw, ok := spec["scores"].([]interface{})
    if !ok {
        return nil, fmt.Errorf("missing scores")
    }
    out := make([]CarbonScore, 0, len(raw))
    for _, it := range raw {
        m, _ := it.(map[string]interface{})
        zone, _ := m["zone"].(string)
        intensity, _ := m["intensity_g_per_kwh"].(float64)
        cpuMult, _ := m["cpu_multiplier"].(float64)
        out = append(out, CarbonScore{
            Zone:             zone,
            IntensityGPerKwh: float32(intensity),
            CpuMultiplier:    float32(cpuMult),
        })
    }
    return out, nil
}