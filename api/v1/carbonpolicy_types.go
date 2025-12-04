package v1

import (
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
)

type NamespaceSelector struct {
    MatchNames []string `json:"matchNames,omitempty"`
}

type WorkloadSelector struct {
    MatchLabels map[string]string `json:"matchLabels,omitempty"`
}

type PolicyTarget struct {
    NamespaceSelector NamespaceSelector `json:"namespaceSelector,omitempty"`
    WorkloadSelector WorkloadSelector `json:"workloadSelector,omitempty"`
}

type SLAEnvelope struct {
    MaxDelaySeconds int `json:"maxDelaySeconds"`
    MaxSlowdownPercent int `json:"maxSlowdownPercent"`
    DeadlineMode string `json:"deadlineMode"`
    DefaultRelativeDeadlineSeconds int `json:"defaultRelativeDeadlineSeconds"`
}

type CarbonKnobs struct {
    Aggressiveness float64 `json:"aggressiveness"`
    MaxCarbonIntensity int `json:"maxCarbonIntensity,omitempty"`
    MinRenewableFraction float64 `json:"minRenewableFraction,omitempty"`
}

type Fairness struct {
    MinShare float64 `json:"minShare,omitempty"`
    MaxShare float64 `json:"maxShare,omitempty"`
    OveragePolicy string `json:"overagePolicy,omitempty"`
}

type Budget struct {
    Enabled bool `json:"enabled"`
    TenantId string `json:"tenantId"`
    MonthlyCarbonBudgetKg int `json:"monthlyCarbonBudgetKg"`
    PerJobBudgetKg int `json:"perJobBudgetKg"`
    BurstAllowancePercent int `json:"burstAllowancePercent"`
    Fairness Fairness `json:"fairness,omitempty"`
}

type ConflictResolution struct {
    OnBudgetExhaustion string `json:"onBudgetExhaustion,omitempty"`
    OnSlaRisk string `json:"onSlaRisk,omitempty"`
    AllowOverrideAnnotations bool `json:"allowOverrideAnnotations,omitempty"`
}

type CarbonPolicySpec struct {
    Target PolicyTarget `json:"target"`
    Criticality string `json:"criticality"`
    SLA SLAEnvelope `json:"sla"`
    Carbon CarbonKnobs `json:"carbon"`
    Budget Budget `json:"budget,omitempty"`
    ConflictResolution ConflictResolution `json:"conflictResolution,omitempty"`
}

type CarbonPolicyStatus struct {
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
    if in.Spec.Target.NamespaceSelector.MatchNames != nil {
        out.Spec.Target.NamespaceSelector.MatchNames = append([]string{}, in.Spec.Target.NamespaceSelector.MatchNames...)
    }
    if in.Spec.Target.WorkloadSelector.MatchLabels != nil {
        out.Spec.Target.WorkloadSelector.MatchLabels = map[string]string{}
        for k, v := range in.Spec.Target.WorkloadSelector.MatchLabels {
            out.Spec.Target.WorkloadSelector.MatchLabels[k] = v
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
