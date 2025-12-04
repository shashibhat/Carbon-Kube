package controllers

import (
	"encoding/json"
	"io"
	"net/http"
	"time"
)

type admissionReview struct {
	ApiVersion string `json:"apiVersion"`
	Kind       string `json:"kind"`
	Request    struct {
		UID    string `json:"uid"`
		Object struct {
			Metadata struct {
				Namespace   string            `json:"namespace"`
				Annotations map[string]string `json:"annotations"`
				Labels      map[string]string `json:"labels"`
			} `json:"metadata"`
		} `json:"object"`
	} `json:"request"`
}

type admissionResponse struct {
	ApiVersion string `json:"apiVersion"`
	Kind       string `json:"kind"`
	Response   struct {
		UID       string `json:"uid"`
		Allowed   bool   `json:"allowed"`
		PatchType string `json:"patchType"`
		Patch     string `json:"patch"`
	} `json:"response"`
}

func jsonPatchAdd(path string, value interface{}) map[string]interface{} {
	return map[string]interface{}{"op": "add", "path": path, "value": value}
}

func HandleMutate(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	var ar admissionReview
	_ = json.Unmarshal(body, &ar)
	uid := ar.Request.UID
	ann := ar.Request.Object.Metadata.Annotations
	labels := ar.Request.Object.Metadata.Labels
	scheduled := ann["carbonkube.io/scheduled-at"]
	deferred := false
	if scheduled != "" {
		if t, err := time.Parse(time.RFC3339, scheduled); err == nil {
			if t.After(time.Now().UTC()) {
				deferred = true
			}
		}
	}
	patch := []map[string]interface{}{}
	if deferred {
		if ann == nil {
			patch = append(patch, jsonPatchAdd("/metadata/annotations", map[string]string{"carbonkube.io/deferred": "true"}))
		} else {
			patch = append(patch, jsonPatchAdd("/metadata/annotations/carbonkube.io~1deferred", "true"))
		}
	} else {
		preferredRegion := ann["carbonkube.io/placement-hint"]
		carbonScore := ann["carbonkube.io/carbonPriorityScore"]
		policyName := ann["carbonkube.io/policy"]
		tenant := ann["carbonkube.io/tenant"]
		if preferredRegion != "" {
			if labels == nil {
				patch = append(patch, jsonPatchAdd("/metadata/labels", map[string]string{"preferredRegion": preferredRegion}))
			} else {
				patch = append(patch, jsonPatchAdd("/metadata/labels/preferredRegion", preferredRegion))
			}
		}
		if carbonScore != "" {
			if ann == nil {
				patch = append(patch, jsonPatchAdd("/metadata/annotations", map[string]string{"carbonPriorityScore": carbonScore}))
			} else {
				patch = append(patch, jsonPatchAdd("/metadata/annotations/carbonPriorityScore", carbonScore))
			}
		}
		if policyName != "" {
			if ann == nil {
				patch = append(patch, jsonPatchAdd("/metadata/annotations", map[string]string{"carbonkube.io/policy": policyName}))
			} else {
				patch = append(patch, jsonPatchAdd("/metadata/annotations/carbonkube.io~1policy", policyName))
			}
		}
		if tenant != "" {
			if ann == nil {
				patch = append(patch, jsonPatchAdd("/metadata/annotations", map[string]string{"carbonkube.io/tenant": tenant}))
			} else {
				patch = append(patch, jsonPatchAdd("/metadata/annotations/carbonkube.io~1tenant", tenant))
			}
		}
	}
	if ann["carbonkube.io/dag-critical"] == "true" {
		if ann == nil {
			patch = append(patch, jsonPatchAdd("/metadata/annotations", map[string]string{"carbonkube.io/sla-weight": "high"}))
		} else {
			patch = append(patch, jsonPatchAdd("/metadata/annotations/carbonkube.io~1sla-weight", "high"))
		}
	}
	resp := admissionResponse{ApiVersion: "admission.k8s.io/v1", Kind: "AdmissionReview"}
	resp.Response.UID = uid
	resp.Response.Allowed = true
	if len(patch) > 0 {
		data, _ := json.Marshal(patch)
		resp.Response.PatchType = "JSONPatch"
		resp.Response.Patch = string(data)
	}
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	_ = enc.Encode(resp)
}
