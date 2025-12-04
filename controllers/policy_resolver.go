package controllers

import (
    "context"
    "fmt"
    "k8s.io/apimachinery/pkg/apis/meta/v1"
    unstructured "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
    "k8s.io/apimachinery/pkg/watch"
    "k8s.io/client-go/dynamic"
    "k8s.io/client-go/kubernetes"
    "k8s.io/client-go/rest"
)

type PolicyResolver struct {
    Dyn    dynamic.Interface
    Client kubernetes.Interface
}

// use shared GVRs from other controllers in the same package

func NewPolicyResolver(cfg *rest.Config) (*PolicyResolver, error) {
    dyn, err := dynamic.NewForConfig(cfg)
    if err != nil {
        return nil, err
    }
    client, err := kubernetes.NewForConfig(cfg)
    if err != nil {
        return nil, err
    }
    return &PolicyResolver{Dyn: dyn, Client: client}, nil
}

func (p *PolicyResolver) Start(ctx context.Context, namespace string) error {
    w, err := p.Dyn.Resource(carbonJobGVR).Namespace(namespace).Watch(ctx, v1.ListOptions{})
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
                meta := u.UnstructuredContent()["metadata"].(map[string]interface{})
                ns := meta["namespace"].(string)
                labels := map[string]string{}
                if m, ok := meta["labels"].(map[string]interface{}); ok {
                    for k, v := range m {
                        if s, ok := v.(string); ok {
                            labels[k] = s
                        }
                    }
                }
                pols, err := p.Dyn.Resource(carbonPolicyGVR).Namespace(ns).List(ctx, v1.ListOptions{})
                if err != nil {
                    continue
                }
                var chosen *unstructured.Unstructured
                for i := range pols.Items {
                    cp := pols.Items[i]
                    spec, _ := cp.UnstructuredContent()["spec"].(map[string]interface{})
                    target, _ := spec["target"].(map[string]interface{})
                    nsSel, _ := target["namespaceSelector"].(map[string]interface{})
                    wlSel, _ := target["workloadSelector"].(map[string]interface{})
                    nsMatch := true
                    if v, ok := nsSel["matchNames"].([]interface{}); ok && len(v) > 0 {
                        nsMatch = false
                        for _, n := range v {
                            if ns == n.(string) {
                                nsMatch = true
                                break
                            }
                        }
                    }
                    labelMatch := true
                    if ml, ok := wlSel["matchLabels"].(map[string]interface{}); ok {
                        labelMatch = containsAll(labels, ml)
                    }
                    if nsMatch && labelMatch {
                        chosen = cp.DeepCopy()
                        break
                    }
                }
                if chosen == nil {
                    continue
                }
                spec := chosen.UnstructuredContent()["spec"].(map[string]interface{})
                budget, _ := spec["budget"].(map[string]interface{})
                tenant := ""
                if v, ok := budget["tenantId"].(string); ok {
                    tenant = v
                }
                criticality := ""
                if v, ok := spec["criticality"].(string); ok {
                    criticality = v
                }
                carbon, _ := spec["carbon"].(map[string]interface{})
                agg := 0.0
                if v, ok := carbon["aggressiveness"].(float64); ok {
                    agg = v
                }
                anns := map[string]string{
                    "carbonkube.io/policy-name": chosen.GetName(),
                    "carbonkube.io/criticality": criticality,
                    "carbonkube.io/tenant": tenant,
                    "carbonkube.io/carbon-aggressiveness": fmt.Sprintf("%.2f", agg),
                }
                if m, ok := u.UnstructuredContent()["metadata"].(map[string]interface{}); ok {
                    if a, ok := m["annotations"].(map[string]interface{}); ok {
                        for k, v := range anns {
                            a[k] = v
                        }
                        m["annotations"] = a
                    } else {
                        na := map[string]interface{}{}
                        for k, v := range anns {
                            na[k] = v
                        }
                        m["annotations"] = na
                    }
                }
                _, _ = p.Dyn.Resource(carbonJobGVR).Namespace(ns).Update(ctx, u, v1.UpdateOptions{})
            }
        }
    }
}

func containsAll(labels map[string]string, required map[string]interface{}) bool {
    for k, v := range required {
        s, ok := v.(string)
        if !ok {
            continue
        }
        if lv, ok := labels[k]; !ok || lv != s {
            return false
        }
    }
    return true
}
