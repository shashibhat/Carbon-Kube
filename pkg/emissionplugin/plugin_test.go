package emissionplugin

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestEmissionPlugin_Name(t *testing.T) {
	plugin := &EmissionPlugin{}
	assert.Equal(t, PluginName, plugin.Name())
}

func TestEmissionPlugin_Execute_WithValidData(t *testing.T) {
	// Setup
	ctx := context.Background()
	
	// Create fake Kubernetes client with ConfigMap
	intensityData := map[string]ZonalIntensity{
		"us-west-2a": {
			Zone:      "us-west-2a",
			Intensity: 100.0, // Low carbon intensity (green)
			Timestamp: time.Now().Unix(),
			Forecast:  []float64{95.0, 105.0, 110.0},
		},
		"us-east-1a": {
			Zone:      "us-east-1a",
			Intensity: 300.0, // High carbon intensity (dirty)
			Timestamp: time.Now().Unix(),
			Forecast:  []float64{295.0, 305.0, 310.0},
		},
	}
	
	intensityJSON, _ := json.Marshal(intensityData)
	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ConfigMapName,
			Namespace: ConfigMapNamespace,
		},
		Data: map[string]string{
			"zones":     string(intensityJSON),
			"threshold": "200.0",
		},
	}
	
	fakeClient := fake.NewSimpleClientset(configMap)
	
	plugin := &EmissionPlugin{
		kubeClient: fakeClient,
		metrics:    &Metrics{}, // Mock metrics for testing
	}
	
	// Create test pod with resource requests
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-pod",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "test-container",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("1000m"), // 1 CPU
							v1.ResourceMemory: resource.MustParse("1Gi"),
						},
					},
				},
			},
		},
	}
	
	// Create test nodes
	nodes := []*v1.Node{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "green-node",
				Labels: map[string]string{
					ZoneLabel: "us-west-2a",
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "dirty-node",
				Labels: map[string]string{
					ZoneLabel: "us-east-1a",
				},
			},
		},
	}
	
	// Execute plugin
	scores, success := plugin.Execute(ctx, pod, nodes)
	
	// Assertions
	assert.True(t, success)
	assert.Len(t, scores, 2)
	
	// Green node should have lower score (better for scheduling)
	greenScore := scores["green-node"]
	dirtyScore := scores["dirty-node"]
	
	assert.True(t, greenScore < dirtyScore, "Green node should have lower emission score")
	
	// Verify score calculations
	expectedGreenScore := 100.0 * (1000.0 / 1000.0 / 1000.0) // 100 gCO2/kWh * 0.001 kW
	expectedDirtyScore := 300.0 * (1000.0 / 1000.0 / 1000.0) // 300 gCO2/kWh * 0.001 kW
	
	assert.InDelta(t, expectedGreenScore, greenScore, 0.001)
	assert.InDelta(t, expectedDirtyScore, dirtyScore, 0.001)
}

func TestEmissionPlugin_Execute_WithMissingConfigMap(t *testing.T) {
	// Setup with empty fake client (no ConfigMap)
	ctx := context.Background()
	fakeClient := fake.NewSimpleClientset()
	
	plugin := &EmissionPlugin{
		kubeClient: fakeClient,
		metrics:    &Metrics{},
	}
	
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod"},
	}
	
	nodes := []*v1.Node{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-node",
				Labels: map[string]string{
					ZoneLabel: "us-west-2a",
				},
			},
		},
	}
	
	// Execute plugin
	scores, success := plugin.Execute(ctx, pod, nodes)
	
	// Should fallback gracefully with zero scores
	assert.True(t, success)
	assert.Len(t, scores, 1)
	assert.Equal(t, 0.0, scores["test-node"])
}

func TestEmissionPlugin_Execute_WithStaleData(t *testing.T) {
	// Setup with stale data (older than 10 minutes)
	ctx := context.Background()
	
	staleTimestamp := time.Now().Unix() - 700 // 11+ minutes ago
	intensityData := map[string]ZonalIntensity{
		"us-west-2a": {
			Zone:      "us-west-2a",
			Intensity: 100.0,
			Timestamp: staleTimestamp, // Stale data
			Forecast:  []float64{95.0, 105.0},
		},
	}
	
	intensityJSON, _ := json.Marshal(intensityData)
	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ConfigMapName,
			Namespace: ConfigMapNamespace,
		},
		Data: map[string]string{
			"zones": string(intensityJSON),
		},
	}
	
	fakeClient := fake.NewSimpleClientset(configMap)
	plugin := &EmissionPlugin{
		kubeClient: fakeClient,
		metrics:    &Metrics{},
	}
	
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("1000m"),
						},
					},
				},
			},
		},
	}
	
	nodes := []*v1.Node{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-node",
				Labels: map[string]string{
					ZoneLabel: "us-west-2a",
				},
			},
		},
	}
	
	scores, success := plugin.Execute(ctx, pod, nodes)
	
	// Should use default intensity for stale data
	assert.True(t, success)
	assert.Len(t, scores, 1)
	
	expectedScore := DefaultIntensity * (1000.0 / 1000.0 / 1000.0)
	assert.InDelta(t, expectedScore, scores["test-node"], 0.001)
}

func TestExtractZoneFromNode(t *testing.T) {
	plugin := &EmissionPlugin{}
	
	tests := []struct {
		name     string
		node     *v1.Node
		expected string
	}{
		{
			name: "Node with zone label",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						ZoneLabel: "us-west-2b",
					},
				},
			},
			expected: "us-west-2b",
		},
		{
			name: "Node with us-west in name",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "ip-10-0-1-100.us-west-2.compute.internal",
				},
			},
			expected: "us-west-2a",
		},
		{
			name: "Node with us-east in name",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "ip-10-0-1-100.us-east-1.compute.internal",
				},
			},
			expected: "us-east-1a",
		},
		{
			name: "Unknown node",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "unknown-node",
				},
			},
			expected: "unknown",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := plugin.extractZoneFromNode(tt.node)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestGetPodResourceRequests(t *testing.T) {
	plugin := &EmissionPlugin{}
	
	pod := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("500m"),
							v1.ResourceMemory: resource.MustParse("1Gi"),
						},
					},
				},
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("1500m"),
							v1.ResourceMemory: resource.MustParse("2Gi"),
						},
					},
				},
			},
		},
	}
	
	requests := plugin.getPodResourceRequests(pod)
	
	assert.Equal(t, int64(2000), requests.CPU)    // 500m + 1500m = 2000m
	assert.Equal(t, int64(3221225472), requests.Memory) // 1Gi + 2Gi = 3Gi in bytes
}

func TestGetThreshold(t *testing.T) {
	ctx := context.Background()
	
	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ConfigMapName,
			Namespace: ConfigMapNamespace,
		},
		Data: map[string]string{
			"threshold": "250.5",
		},
	}
	
	fakeClient := fake.NewSimpleClientset(configMap)
	plugin := &EmissionPlugin{
		kubeClient: fakeClient,
	}
	
	threshold, err := plugin.GetThreshold(ctx)
	
	assert.NoError(t, err)
	assert.Equal(t, 250.5, threshold)
}

func TestGetThreshold_MissingConfigMap(t *testing.T) {
	ctx := context.Background()
	fakeClient := fake.NewSimpleClientset()
	
	plugin := &EmissionPlugin{
		kubeClient: fakeClient,
	}
	
	threshold, err := plugin.GetThreshold(ctx)
	
	assert.Error(t, err)
	assert.Equal(t, 200.0, threshold) // Default threshold
}

func TestGetThreshold_MissingThresholdKey(t *testing.T) {
	ctx := context.Background()
	
	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ConfigMapName,
			Namespace: ConfigMapNamespace,
		},
		Data: map[string]string{
			"zones": "{}",
		},
	}
	
	fakeClient := fake.NewSimpleClientset(configMap)
	plugin := &EmissionPlugin{
		kubeClient: fakeClient,
	}
	
	threshold, err := plugin.GetThreshold(ctx)
	
	assert.NoError(t, err)
	assert.Equal(t, 200.0, threshold) // Default threshold
}