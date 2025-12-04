package v1

import (
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
)

type CarbonDataSource struct {
    Type string `json:"type"`
    Resource string `json:"resource"`
    Region string `json:"region"`
    AvgIngressGBPerJob float64 `json:"avgIngressGBPerJob,omitempty"`
    AvgReadGBPerJob float64 `json:"avgReadGBPerJob,omitempty"`
}

type CarbonJobSpecSpec struct {
    DagId string `json:"dagId"`
    StageId string `json:"stageId"`
    UpstreamStages []string `json:"upstreamStages,omitempty"`
    EstimatedRuntimeSeconds int `json:"estimatedRuntimeSeconds"`
    EstimatedCpuSeconds int `json:"estimatedCpuSeconds,omitempty"`
    Deadline *metav1.Time `json:"deadline,omitempty"`
    DataSources []CarbonDataSource `json:"dataSources,omitempty"`
    PolicyRef string `json:"policyRef,omitempty"`
}

type CarbonJobDAGStatus struct {
    IsCriticalPath bool `json:"isCriticalPath,omitempty"`
    TopoDepth int `json:"topoDepth,omitempty"`
    NormalizedImportance float64 `json:"normalizedImportance,omitempty"`
}

type CarbonJobSpecStatus struct {
    DAG CarbonJobDAGStatus `json:"dag,omitempty"`
    LastUpdated metav1.Time `json:"lastUpdated,omitempty"`
}

type CarbonJobSpec struct {
    metav1.TypeMeta `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`
    Spec CarbonJobSpecSpec `json:"spec,omitempty"`
    Status CarbonJobSpecStatus `json:"status,omitempty"`
}

type CarbonJobSpecList struct {
    metav1.TypeMeta `json:",inline"`
    metav1.ListMeta `json:"metadata,omitempty"`
    Items []CarbonJobSpec `json:"items"`
}

func (in *CarbonJobSpec) DeepCopyObject() runtime.Object {
    if in == nil {
        return nil
    }
    out := new(CarbonJobSpec)
    *out = *in
    if in.Spec.UpstreamStages != nil {
        out.Spec.UpstreamStages = append([]string{}, in.Spec.UpstreamStages...)
    }
    if in.Spec.DataSources != nil {
        out.Spec.DataSources = make([]CarbonDataSource, len(in.Spec.DataSources))
        copy(out.Spec.DataSources, in.Spec.DataSources)
    }
    return out
}

func (in *CarbonJobSpecList) DeepCopyObject() runtime.Object {
    if in == nil {
        return nil
    }
    out := new(CarbonJobSpecList)
    *out = *in
    if in.Items != nil {
        out.Items = make([]CarbonJobSpec, len(in.Items))
        for i := range in.Items {
            out.Items[i] = *in.Items[i].DeepCopyObject().(*CarbonJobSpec)
        }
    }
    return out
}
