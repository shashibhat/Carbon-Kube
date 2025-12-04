package controllers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"time"

	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

type BudgetEnforcer struct {
	Dyn     dynamic.Interface
	Client  kubernetes.Interface
	PromURL string
	HTTP    *http.Client
}

var bePolicyGVR = schema.GroupVersionResource{Group: "carbonkube.io", Version: "v1", Resource: "carbonpolicies"}

func NewBudgetEnforcer(cfg *rest.Config) (*BudgetEnforcer, error) {
	dyn, err := dynamic.NewForConfig(cfg)
	if err != nil {
		return nil, err
	}
	client, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return nil, err
	}
	return &BudgetEnforcer{Dyn: dyn, Client: client, PromURL: os.Getenv("CARBONKUBE_PROMETHEUS_URL"), HTTP: &http.Client{Timeout: 10 * time.Second}}, nil
}

func (b *BudgetEnforcer) Start(ctx context.Context, namespace string) error {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			pols, err := b.Dyn.Resource(bePolicyGVR).Namespace(namespace).List(ctx, v1.ListOptions{})
			if err != nil {
				continue
			}
			data := map[string]string{}
			for i := range pols.Items {
				cp := pols.Items[i]
				spec, _ := cp.UnstructuredContent()["spec"].(map[string]interface{})
				budget, _ := spec["budget"].(map[string]interface{})
				tenant := safeString(budget["tenantId"])
				monthly := safeInt(budget["monthlyCarbonBudgetKg"])
				used := b.queryTenantCO2(ctx, namespace, tenant)
				remaining := monthly - int(used)
				over := "false"
				throttled := "false"
				if remaining < 0 {
					over = "true"
				}
				prefix := tenant
				data[prefix+".usedCarbonMonthlyKg"] = intString(int(used))
				data[prefix+".remainingBudgetKg"] = intString(remaining)
				data[prefix+".overBudget"] = over
				data[prefix+".throttled"] = throttled
			}
			cm := &corev1.ConfigMap{ObjectMeta: v1.ObjectMeta{Name: "carbonkube-tenant-state"}, Data: data}
			_, _ = b.Client.CoreV1().ConfigMaps(namespace).Update(ctx, cm, v1.UpdateOptions{})
			_, _ = b.Client.CoreV1().ConfigMaps(namespace).Create(ctx, cm, v1.CreateOptions{})
		}
	}
}

func safeString(v interface{}) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func safeInt(v interface{}) int {
	switch x := v.(type) {
	case int:
		return x
	case int64:
		return int(x)
	case float64:
		return int(x)
	default:
		return 0
	}
}

func intString(i int) string {
	return fmtInt(i)
}

func fmtInt(i int) string {
	return fmt.Sprintf("%d", i)
}

func (b *BudgetEnforcer) queryTenantCO2(ctx context.Context, namespace string, tenant string) float64 {
	pods, err := b.Client.CoreV1().Pods(namespace).List(ctx, v1.ListOptions{})
	if err != nil {
		return 0
	}
	total := 0.0
	for _, p := range pods.Items {
		if p.Annotations["carbonkube.io/tenant"] != tenant {
			continue
		}
		region := p.Labels["preferredRegion"]
		ci := b.queryCarbonIntensity(region)
		joules := b.queryPodJoules(p.Namespace, p.Name)
		kwh := joules / 3600000.0
		total += kwh * ci / 1000.0
	}
	return total
}

func (b *BudgetEnforcer) queryPodJoules(namespace string, pod string) float64 {
	u, err := url.Parse(b.PromURL)
	if err != nil {
		return 0
	}
	u.Path = "/api/v1/query"
	q := url.Values{}
	q.Set("query", fmt.Sprintf("kepler_container_joules_total{pod_name=\"%s\",namespace=\"%s\"}", pod, namespace))
	u.RawQuery = q.Encode()
	resp, err := b.HTTP.Get(u.String())
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	var body struct {
		Status string `json:"status"`
		Data   struct {
			ResultType string `json:"resultType"`
			Result     []struct {
				Value []interface{} `json:"value"`
			} `json:"result"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return 0
	}
	if len(body.Data.Result) == 0 {
		return 0
	}
	valStr := body.Data.Result[0].Value[1].(string)
	f, err := parseNum(valStr)
	if err != nil {
		return 0
	}
	return f
}

func (b *BudgetEnforcer) queryCarbonIntensity(region string) float64 {
	u, err := url.Parse(b.PromURL)
	if err != nil {
		return 0
	}
	u.Path = "/api/v1/query"
	q := url.Values{}
	q.Set("query", fmt.Sprintf("carbon_intensity_gco2_per_kwh{zone=\"%s\"}", region))
	u.RawQuery = q.Encode()
	resp, err := b.HTTP.Get(u.String())
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	var body struct {
		Status string `json:"status"`
		Data   struct {
			ResultType string `json:"resultType"`
			Result     []struct {
				Value []interface{} `json:"value"`
			} `json:"result"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return 0
	}
	if len(body.Data.Result) == 0 {
		return 0
	}
	valStr := body.Data.Result[0].Value[1].(string)
	f, err := parseNum(valStr)
	if err != nil {
		return 0
	}
	return f
}
func parseNum(s string) (float64, error) {
	var f float64
	err := json.Unmarshal([]byte(s), &f)
	if err != nil {
		return 0, err
	}
	return f, nil
}
