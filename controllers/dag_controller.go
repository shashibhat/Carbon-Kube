package controllers

import (
	"context"
	"math"
	"sort"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	unstructured "k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
)

type DAGController struct {
	Dyn dynamic.Interface
}

var dagJobGVR = schema.GroupVersionResource{Group: "carbonkube.io", Version: "v1", Resource: "carbonjobs"}

func NewDAGController(cfg *rest.Config) (*DAGController, error) {
	dyn, err := dynamic.NewForConfig(cfg)
	if err != nil {
		return nil, err
	}
	return &DAGController{Dyn: dyn}, nil
}

func (d *DAGController) Start(ctx context.Context, namespace string) error {
	w, err := d.Dyn.Resource(dagJobGVR).Namespace(namespace).Watch(ctx, v1.ListOptions{})
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
				dagId, _ := spec["dagId"].(string)
				if dagId == "" {
					continue
				}
				ns := u.GetNamespace()
				list, err := d.Dyn.Resource(dagJobGVR).Namespace(ns).List(ctx, v1.ListOptions{})
				if err != nil {
					continue
				}
				nodes := map[string]map[string]interface{}{}
				edges := map[string][]string{}
				for i := range list.Items {
					it := list.Items[i]
					s, _ := it.UnstructuredContent()["spec"].(map[string]interface{})
					if s["dagId"] != dagId {
						continue
					}
					sid, _ := s["stageId"].(string)
					nodes[sid] = s
					ups := []string{}
					if v, ok := s["upstreamStages"].([]interface{}); ok {
						for _, x := range v {
							ups = append(ups, x.(string))
						}
					}
					edges[sid] = ups
				}
				invalid := hasCycle(edges)
				depth := topoDepth(edges)
				dist := longestPath(edges)
				maxDist := 0
				for _, v := range dist {
					if v > maxDist {
						maxDist = v
					}
				}
				for sid := range nodes {
					imp := 0.0
					if maxDist > 0 {
						imp = float64(dist[sid]) / float64(maxDist)
					}
					rt := 0.0
					if v, ok := nodes[sid]["estimatedRuntimeSeconds"].(int64); ok {
						rt = float64(v)
					} else if v, ok := nodes[sid]["estimatedRuntimeSeconds"].(float64); ok {
						rt = v
					}
					imp = imp * math.Log(rt+1)
					status := map[string]interface{}{
						"dag": map[string]interface{}{
							"isCriticalPath":       dist[sid] == maxDist,
							"topoDepth":            depth[sid],
							"normalizedImportance": imp,
						},
					}
					obj := list.Items[findIndex(list.Items, sid)].DeepCopy()
					if m, ok := obj.UnstructuredContent()["status"].(map[string]interface{}); ok {
						for k, v := range status {
							m[k] = v
						}
						obj.Object["status"] = m
					} else {
						obj.Object["status"] = status
					}
					anns := map[string]string{
						"carbonkube.io/dag-id":         dagId,
						"carbonkube.io/stage-id":       sid,
						"carbonkube.io/dag-importance": formatFloat(imp),
						"carbonkube.io/dag-critical":   boolString(dist[sid] == maxDist),
						"carbonkube.io/dag-valid":      boolString(!invalid),
					}
					if md, ok := obj.UnstructuredContent()["metadata"].(map[string]interface{}); ok {
						if a, ok := md["annotations"].(map[string]interface{}); ok {
							for k, v := range anns {
								a[k] = v
							}
							md["annotations"] = a
						} else {
							na := map[string]interface{}{}
							for k, v := range anns {
								na[k] = v
							}
							md["annotations"] = na
						}
					}
					_, _ = d.Dyn.Resource(dagJobGVR).Namespace(ns).Update(ctx, obj, v1.UpdateOptions{})
				}
			}
		}
	}
}

func findIndex(items []unstructured.Unstructured, stageId string) int {
	idx := 0
	for i := range items {
		s, _ := items[i].UnstructuredContent()["spec"].(map[string]interface{})
		if v, _ := s["stageId"].(string); v == stageId {
			idx = i
			break
		}
	}
	return idx
}

func topoDepth(edges map[string][]string) map[string]int {
	depth := map[string]int{}
	order := topologicalOrder(edges)
	for _, n := range order {
		d := 0
		for _, up := range edges[n] {
			if depth[up]+1 > d {
				d = depth[up] + 1
			}
		}
		depth[n] = d
	}
	return depth
}

func longestPath(edges map[string][]string) map[string]int {
	dist := map[string]int{}
	order := topologicalOrder(edges)
	for _, n := range order {
		d := 0
		for _, up := range edges[n] {
			if dist[up]+1 > d {
				d = dist[up] + 1
			}
		}
		dist[n] = d
	}
	return dist
}

func topologicalOrder(edges map[string][]string) []string {
	in := map[string]int{}
	for n := range edges {
		in[n] = len(edges[n])
	}
	q := []string{}
	for n, v := range in {
		if v == 0 {
			q = append(q, n)
		}
	}
	order := []string{}
	for len(q) > 0 {
		sort.Strings(q)
		u := q[0]
		q = q[1:]
		order = append(order, u)
		for v := range edges {
			for _, up := range edges[v] {
				if up == u {
					in[v]--
					if in[v] == 0 {
						q = append(q, v)
					}
					break
				}
			}
		}
	}
	return order
}

func hasCycle(edges map[string][]string) bool {
	visit := map[string]int{}
	var dfs func(string) bool
	dfs = func(u string) bool {
		visit[u] = 1
		for _, v := range edges[u] {
			if visit[v] == 1 {
				return true
			}
			if visit[v] == 0 && dfs(v) {
				return true
			}
		}
		visit[u] = 2
		return false
	}
	for n := range edges {
		if visit[n] == 0 && dfs(n) {
			return true
		}
	}
	return false
}

func boolString(b bool) string {
	if b {
		return "true"
	}
	return "false"
}
