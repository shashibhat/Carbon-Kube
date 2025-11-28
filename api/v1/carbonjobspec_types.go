package v1

import (
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
)

type CarbonJobResources struct {
    CPU string `json:"cpu"`
    Memory string `json:"memory"`
    GPU string `json:"gpu"`
}

type CarbonJobDataAffinity struct {
    PrimaryRegion string `json:"primaryRegion"`
    AllowedRegions []string `json:"allowedRegions"`
    KafkaTopics []string `json:"kafkaTopics"`
    KafkaRegion string `json:"kafkaRegion"`
    ObjectStores []string `json:"objectStores"`
    DBRegion string `json:"dbRegion"`
    MaxExtraLatencyMs int `json:"maxExtraLatencyMs"`
}

type CarbonJobMobility struct {
    Level string `json:"level"`
    Reasons []string `json:"reasons"`
}

type CarbonJobTemporal struct {
    AllowDelay bool `json:"allowDelay"`
    MaxDelay string `json:"maxDelay"`
    RequireGreenWindow string `json:"requireGreenWindow"`
}

type CarbonJobBudgets struct {
    MaxCarbonForThisRun int `json:"maxCarbonForThisRun"`
    DesiredCarbonReduction float64 `json:"desiredCarbonReduction"`
}

type CarbonJobSpecSpec struct {
    JobType string `json:"jobType"`
    CarbonPolicyRef string `json:"carbonPolicyRef"`
    Resources CarbonJobResources `json:"resources"`
    DataAffinity CarbonJobDataAffinity `json:"dataAffinity"`
    Mobility CarbonJobMobility `json:"mobility"`
    Temporal CarbonJobTemporal `json:"temporal"`
    Budgets CarbonJobBudgets `json:"budgets"`
}

type PlacementHint struct {
    Region string `json:"region"`
    CarbonPriorityScore float64 `json:"carbonPriorityScore"`
}

type CarbonJobSpecStatus struct {
    DataGravityScore float64 `json:"dataGravityScore,omitempty"`
    MobilityScore float64 `json:"mobilityScore,omitempty"`
    SLARiskScore float64 `json:"slaRiskScore,omitempty"`
    PlacementHint PlacementHint `json:"placementHint,omitempty"`
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
    if in.Spec.DataAffinity.AllowedRegions != nil {
        out.Spec.DataAffinity.AllowedRegions = append([]string{}, in.Spec.DataAffinity.AllowedRegions...)
    }
    if in.Spec.DataAffinity.KafkaTopics != nil {
        out.Spec.DataAffinity.KafkaTopics = append([]string{}, in.Spec.DataAffinity.KafkaTopics...)
    }
    if in.Spec.DataAffinity.ObjectStores != nil {
        out.Spec.DataAffinity.ObjectStores = append([]string{}, in.Spec.DataAffinity.ObjectStores...)
    }
    if in.Spec.Mobility.Reasons != nil {
        out.Spec.Mobility.Reasons = append([]string{}, in.Spec.Mobility.Reasons...)
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

