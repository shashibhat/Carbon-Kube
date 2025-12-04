package controllers

import (
	"context"
	"strings"

	"github.com/shashibhat/Carbon-Kube/pkg/scoring"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	unstructured "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

type CarbonJobController struct {
	Dyn    dynamic.Interface
	Client kubernetes.Interface
}

var carbonJobGVR = schema.GroupVersionResource{Group: "carbonkube.io", Version: "v1", Resource: "carbonjobs"}

func NewCarbonJobController(cfg *rest.Config) (*CarbonJobController, error) {
	dyn, err := dynamic.NewForConfig(cfg)
	if err != nil {
		return nil, err
	}
	client, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return nil, err
	}
	return &CarbonJobController{Dyn: dyn, Client: client}, nil
}

func computeMobilityScore(level string) float64 {
	switch level {
	case "pinned":
		return 10
	case "constrained":
		return 50
	case "highly-mobile":
		return 90
	default:
		return 50
	}
}

func (c *CarbonJobController) Start(ctx context.Context, namespace string) error {
	w, err := c.Dyn.Resource(carbonJobGVR).Namespace(namespace).Watch(ctx, metav1.ListOptions{})
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
				name := meta["name"].(string)
				ns := meta["namespace"].(string)
				_ = u.UnstructuredContent()["spec"].(map[string]interface{})
				hint := ""
				wts := scoring.PolicyWeights{CarbonWeight: 0.4, CostWeight: 0.2, SLARiskWeight: 0.2, DataGravityWeight: 0.2}
				cons := scoring.PolicyConstraints{HighCarbonLimit: 400, ExtremeCarbonLimit: 600, MaxSLAIncreasePercent: 10}
				score := scoring.ComputeScore(scoring.CarbonScores{CarbonScore: 80, CostScore: 60, SLARisk: 20, DataGravityPenalty: 10}, wts, cons, 350, 5)
				ann := map[string]string{"carbonkube.io/placement-hint": hint, "carbonkube.io/carbonPriorityScore": formatFloat(score)}
				if m, ok := u.UnstructuredContent()["metadata"].(map[string]interface{}); ok {
					if a, ok := m["annotations"].(map[string]interface{}); ok {
						for k, v := range ann {
							a[k] = v
						}
						m["annotations"] = a
					} else {
						na := map[string]interface{}{}
						for k, v := range ann {
							na[k] = v
						}
						m["annotations"] = na
					}
				}
				_, _ = c.Dyn.Resource(carbonJobGVR).Namespace(ns).Update(ctx, u, metav1.UpdateOptions{})
				if strings.Contains(name, "") {
				}
			}
		}
	}
}
