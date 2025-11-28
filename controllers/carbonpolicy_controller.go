package controllers

import (
    "context"
    "fmt"
    apiv1 "github.com/shashibhat/Carbon-Kube/api/v1"
    corev1 "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    unstructured "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
    "k8s.io/apimachinery/pkg/runtime/schema"
    "k8s.io/apimachinery/pkg/watch"
    "k8s.io/client-go/dynamic"
    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/rest"
)

type CarbonPolicyController struct {
    Dyn dynamic.Interface
    Client kubernetes.Interface
}

var carbonPolicyGVR = schema.GroupVersionResource{Group: "carbonkube.io", Version: "v1", Resource: "carbonpolicies"}

func NewCarbonPolicyController(cfg *rest.Config) (*CarbonPolicyController, error) {
    dyn, err := dynamic.NewForConfig(cfg)
    if err != nil {
        return nil, err
    }
    client, err := kubernetes.NewForConfig(cfg)
    if err != nil {
        return nil, err
    }
    return &CarbonPolicyController{Dyn: dyn, Client: client}, nil
}

func normalizeWeights(obj apiv1.CarbonPolicyObjectives) map[string]float64 {
    sum := obj.CarbonWeight + obj.CostWeight + obj.SLARiskWeight + obj.DataGravityWeight
    out := map[string]float64{}
    if sum == 0 {
        return out
    }
    out["carbon"] = obj.CarbonWeight / sum
    out["cost"] = obj.CostWeight / sum
    out["sla"] = obj.SLARiskWeight / sum
    out["dataGravity"] = obj.DataGravityWeight / sum
    return out
}

func (c *CarbonPolicyController) Start(ctx context.Context, namespace string) error {
    w, err := c.Dyn.Resource(carbonPolicyGVR).Namespace(namespace).Watch(ctx, metav1.ListOptions{})
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
                spec, ok := u.UnstructuredContent()["spec"].(map[string]interface{})
                if !ok {
                    continue
                }
                objectives := spec["objectives"].(map[string]interface{})
                ow := apiv1.CarbonPolicyObjectives{
                    CarbonWeight: toFloat(objectives["carbonWeight"]),
                    CostWeight: toFloat(objectives["costWeight"]),
                    SLARiskWeight: toFloat(objectives["slaRiskWeight"]),
                    DataGravityWeight: toFloat(objectives["dataGravityWeight"]),
                }
                nw := normalizeWeights(ow)
                cm := map[string]string{}
                for k, v := range nw {
                    cm[k] = formatFloat(v)
                }
                meta := u.UnstructuredContent()["metadata"].(map[string]interface{})
                name := meta["name"].(string)
                ns := meta["namespace"].(string)
                _, _ = c.Client.CoreV1().ConfigMaps(ns).Update(ctx, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: name + "-policy"}, Data: cm}, metav1.UpdateOptions{})
                _, _ = c.Client.CoreV1().ConfigMaps(ns).Create(ctx, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: name + "-policy"}, Data: cm}, metav1.CreateOptions{})
            }
        }
    }
}

func toFloat(v interface{}) float64 {
    switch t := v.(type) {
    case float64:
        return t
    case int64:
        return float64(t)
    case int:
        return float64(t)
    default:
        return 0
    }
}

func formatFloat(f float64) string {
    return fmt.Sprintf("%.4f", f)
}
