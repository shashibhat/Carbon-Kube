package v1

import (
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
)

type CarbonPolicyObjectives struct {
    CarbonWeight float64 `json:"carbonWeight"`
    CostWeight float64 `json:"costWeight"`
    SLARiskWeight float64 `json:"slaRiskWeight"`
    DataGravityWeight float64 `json:"dataGravityWeight"`
}

type CarbonPolicyConstraints struct {
    MaxSLAIncreasePercent int `json:"maxSLAIncreasePercent"`
    MaxExtraLatencyMs int `json:"maxExtraLatencyMs"`
    MaxCarbonPerDay int `json:"maxCarbonPerDay"`
    MaxCarbonPerJob int `json:"maxCarbonPerJob"`
}

type CarbonPolicyShifting struct {
    AllowTemporalShifting bool `json:"allowTemporalShifting"`
    MaxDelay string `json:"maxDelay"`
    MinGreenWindow string `json:"minGreenWindow"`
}

type CarbonThresholds struct {
    HighCarbonLimit int `json:"highCarbonLimit"`
    ExtremeCarbonLimit int `json:"extremeCarbonLimit"`
}

type CarbonPolicySpatial struct {
    AllowedRegions []string `json:"allowedRegions"`
    AvoidRegions []string `json:"avoidRegions"`
    CarbonThresholds CarbonThresholds `json:"carbonThresholds"`
}

type CarbonPolicyAdaptive struct {
    EnableAutoTune bool `json:"enableAutoTune"`
    LearningRate float64 `json:"learningRate"`
    ExplorationRate float64 `json:"explorationRate"`
    TargetCarbonReductionPercent int `json:"targetCarbonReductionPercent"`
}

type CarbonPolicySpec struct {
    Objectives CarbonPolicyObjectives `json:"objectives"`
    Constraints CarbonPolicyConstraints `json:"constraints"`
    Shifting CarbonPolicyShifting `json:"shifting"`
    Spatial CarbonPolicySpatial `json:"spatial"`
    Adaptive CarbonPolicyAdaptive `json:"adaptive"`
}

type CarbonPolicyStatus struct {
    NormalizedWeights map[string]float64 `json:"normalizedWeights,omitempty"`
    LastUpdated metav1.Time `json:"lastUpdated,omitempty"`
}

type CarbonPolicy struct {
    metav1.TypeMeta `json:",inline"`
    metav1.ObjectMeta `json:"metadata,omitempty"`
    Spec CarbonPolicySpec `json:"spec,omitempty"`
    Status CarbonPolicyStatus `json:"status,omitempty"`
}

type CarbonPolicyList struct {
    metav1.TypeMeta `json:",inline"`
    metav1.ListMeta `json:"metadata,omitempty"`
    Items []CarbonPolicy `json:"items"`
}

func (in *CarbonPolicy) DeepCopyObject() runtime.Object {
    if in == nil {
        return nil
    }
    out := new(CarbonPolicy)
    *out = *in
    if in.Spec.Spatial.AllowedRegions != nil {
        out.Spec.Spatial.AllowedRegions = append([]string{}, in.Spec.Spatial.AllowedRegions...)
    }
    if in.Spec.Spatial.AvoidRegions != nil {
        out.Spec.Spatial.AvoidRegions = append([]string{}, in.Spec.Spatial.AvoidRegions...)
    }
    if in.Status.NormalizedWeights != nil {
        out.Status.NormalizedWeights = map[string]float64{}
        for k, v := range in.Status.NormalizedWeights {
            out.Status.NormalizedWeights[k] = v
        }
    }
    return out
}

func (in *CarbonPolicyList) DeepCopyObject() runtime.Object {
    if in == nil {
        return nil
    }
    out := new(CarbonPolicyList)
    *out = *in
    if in.Items != nil {
        out.Items = make([]CarbonPolicy, len(in.Items))
        for i := range in.Items {
            out.Items[i] = *in.Items[i].DeepCopyObject().(*CarbonPolicy)
        }
    }
    return out
}

