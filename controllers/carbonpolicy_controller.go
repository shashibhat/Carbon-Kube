package controllers

import (
	"context"
	"fmt"

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
	Dyn    dynamic.Interface
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

// publish selected policy knobs for easy consumption

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
				meta := u.UnstructuredContent()["metadata"].(map[string]interface{})
				name := meta["name"].(string)
				ns := meta["namespace"].(string)
				out := map[string]string{}
				sla := map[string]interface{}{}
				if v, ok := spec["sla"].(map[string]interface{}); ok {
					sla = v
				}
				carbon := map[string]interface{}{}
				if v, ok := spec["carbon"].(map[string]interface{}); ok {
					carbon = v
				}
				budget := map[string]interface{}{}
				if v, ok := spec["budget"].(map[string]interface{}); ok {
					budget = v
				}
				fairness := map[string]interface{}{}
				if v, ok := budget["fairness"].(map[string]interface{}); ok {
					fairness = v
				}
				conflict := map[string]interface{}{}
				if v, ok := spec["conflictResolution"].(map[string]interface{}); ok {
					conflict = v
				}
				out["policy."+name+".sla.maxDelaySeconds"] = formatFloat(toFloat(sla["maxDelaySeconds"]))
				out["policy."+name+".sla.maxSlowdownPercent"] = formatFloat(toFloat(sla["maxSlowdownPercent"]))
				out["policy."+name+".sla.deadlineMode"] = fmt.Sprintf("%v", sla["deadlineMode"])
				out["policy."+name+".sla.defaultRelativeDeadlineSeconds"] = formatFloat(toFloat(sla["defaultRelativeDeadlineSeconds"]))
				out["policy."+name+".carbon.aggressiveness"] = formatFloat(toFloat(carbon["aggressiveness"]))
				out["policy."+name+".carbon.maxCarbonIntensity"] = formatFloat(toFloat(carbon["maxCarbonIntensity"]))
				out["policy."+name+".carbon.minRenewableFraction"] = formatFloat(toFloat(carbon["minRenewableFraction"]))
				out["policy."+name+".budget.tenantId"] = fmt.Sprintf("%v", budget["tenantId"])
				out["policy."+name+".budget.monthlyCarbonBudgetKg"] = formatFloat(toFloat(budget["monthlyCarbonBudgetKg"]))
				out["policy."+name+".budget.perJobCarbonBudgetKg"] = formatFloat(toFloat(budget["perJobBudgetKg"]))
				out["policy."+name+".budget.burstAllowancePercent"] = formatFloat(toFloat(budget["burstAllowancePercent"]))
				out["policy."+name+".fairness.minShare"] = formatFloat(toFloat(fairness["minShare"]))
				out["policy."+name+".fairness.maxShare"] = formatFloat(toFloat(fairness["maxShare"]))
				out["policy."+name+".fairness.overagePolicy"] = fmt.Sprintf("%v", fairness["overagePolicy"])
				out["policy."+name+".onBudgetExhaustion"] = fmt.Sprintf("%v", conflict["onBudgetExhaustion"])
				out["policy."+name+".onSlaRisk"] = fmt.Sprintf("%v", conflict["onSlaRisk"])
				idxName := "carbonpolicy-index-" + ns
				_, _ = c.Client.CoreV1().ConfigMaps(ns).Update(ctx, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: idxName}, Data: out}, metav1.UpdateOptions{})
				_, _ = c.Client.CoreV1().ConfigMaps(ns).Create(ctx, &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: idxName}, Data: out}, metav1.CreateOptions{})
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
