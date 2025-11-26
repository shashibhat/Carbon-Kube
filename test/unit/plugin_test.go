package emissionplugin

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

func TestCarbonEmissionPlugin_Name(t *testing.T) {
	plugin := &CarbonEmissionPlugin{}
	assert.Equal(t, "CarbonEmissionPlugin", plugin.Name())
}

func TestCarbonEmissionPlugin_Score(t *testing.T) {
	tests := []struct {
		name           string
		carbonData     map[string]interface{}
		nodeName       string
		expectedScore  int64
		expectedError  bool
	}{
		{
			name: "valid carbon data",
			carbonData: map[string]interface{}{
				"US-CA": map[string]interface{}{
					"carbon_intensity": 200.0,
					"timestamp":        float64(time.Now().Unix()),
				},
				"US-TX": map[string]interface{}{
					"carbon_intensity": 400.0,
					"timestamp":        float64(time.Now().Unix()),
				},
			},
			nodeName:      "node-us-ca-1",
			expectedScore: 80, // (1000 - 200) * 100 / 1000 = 80
			expectedError: false,
		},
		{
			name: "high carbon intensity",
			carbonData: map[string]interface{}{
				"US-CA": map[string]interface{}{
					"carbon_intensity": 800.0,
					"timestamp":        float64(time.Now().Unix()),
				},
			},
			nodeName:      "node-us-ca-1",
			expectedScore: 20, // (1000 - 800) * 100 / 1000 = 20
			expectedError: false,
		},
		{
			name: "zero carbon intensity",
			carbonData: map[string]interface{}{
				"US-CA": map[string]interface{}{
					"carbon_intensity": 0.0,
					"timestamp":        float64(time.Now().Unix()),
				},
			},
			nodeName:      "node-us-ca-1",
			expectedScore: 100, // (1000 - 0) * 100 / 1000 = 100
			expectedError: false,
		},
		{
			name:          "missing carbon data",
			carbonData:    map[string]interface{}{},
			nodeName:      "node-us-ca-1",
			expectedScore: 50, // default score
			expectedError: false,
		},
		{
			name: "unknown zone",
			carbonData: map[string]interface{}{
				"US-TX": map[string]interface{}{
					"carbon_intensity": 300.0,
					"timestamp":        float64(time.Now().Unix()),
				},
			},
			nodeName:      "node-us-ca-1",
			expectedScore: 50, // default score
			expectedError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create fake Kubernetes client
			client := fake.NewSimpleClientset()

			// Create ConfigMap with carbon data
			configMap := &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "carbon-intensity-data",
					Namespace: "carbon-kube",
				},
				Data: map[string]string{
					"carbon-intensity": marshalCarbonData(tt.carbonData),
				},
			}
			_, err := client.CoreV1().ConfigMaps("carbon-kube").Create(
				context.TODO(), configMap, metav1.CreateOptions{})
			require.NoError(t, err)

			// Create plugin
			plugin := &CarbonEmissionPlugin{
				handle: &mockFrameworkHandle{client: client},
			}

			// Create test node
			node := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: tt.nodeName,
					Labels: map[string]string{
						"topology.kubernetes.io/zone": extractZoneFromNodeName(tt.nodeName),
					},
				},
			}

			// Create test pod
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "test-container",
							Image: "nginx",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: resource.MustParse("128Mi"),
								},
							},
						},
					},
				},
			}

			// Execute Score
			score, status := plugin.Score(context.TODO(), nil, pod, tt.nodeName)

			if tt.expectedError {
				assert.False(t, status.IsSuccess())
			} else {
				assert.True(t, status.IsSuccess())
				assert.Equal(t, tt.expectedScore, score)
			}
		})
	}
}

func TestCarbonEmissionPlugin_Filter(t *testing.T) {
	tests := []struct {
		name          string
		carbonData    map[string]interface{}
		threshold     float64
		nodeName      string
		expectedPass  bool
	}{
		{
			name: "carbon intensity below threshold",
			carbonData: map[string]interface{}{
				"US-CA": map[string]interface{}{
					"carbon_intensity": 200.0,
					"timestamp":        float64(time.Now().Unix()),
				},
			},
			threshold:    300.0,
			nodeName:     "node-us-ca-1",
			expectedPass: true,
		},
		{
			name: "carbon intensity above threshold",
			carbonData: map[string]interface{}{
				"US-CA": map[string]interface{}{
					"carbon_intensity": 400.0,
					"timestamp":        float64(time.Now().Unix()),
				},
			},
			threshold:    300.0,
			nodeName:     "node-us-ca-1",
			expectedPass: false,
		},
		{
			name: "carbon intensity equals threshold",
			carbonData: map[string]interface{}{
				"US-CA": map[string]interface{}{
					"carbon_intensity": 300.0,
					"timestamp":        float64(time.Now().Unix()),
				},
			},
			threshold:    300.0,
			nodeName:     "node-us-ca-1",
			expectedPass: true,
		},
		{
			name:         "missing carbon data allows scheduling",
			carbonData:   map[string]interface{}{},
			threshold:    300.0,
			nodeName:     "node-us-ca-1",
			expectedPass: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create fake Kubernetes client
			client := fake.NewSimpleClientset()

			// Create ConfigMaps
			carbonConfigMap := &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "carbon-intensity-data",
					Namespace: "carbon-kube",
				},
				Data: map[string]string{
					"carbon-intensity": marshalCarbonData(tt.carbonData),
				},
			}
			_, err := client.CoreV1().ConfigMaps("carbon-kube").Create(
				context.TODO(), carbonConfigMap, metav1.CreateOptions{})
			require.NoError(t, err)

			thresholdConfigMap := &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "carbon-kube-config",
					Namespace: "carbon-kube",
				},
				Data: map[string]string{
					"threshold": fmt.Sprintf("%.1f", tt.threshold),
				},
			}
			_, err = client.CoreV1().ConfigMaps("carbon-kube").Create(
				context.TODO(), thresholdConfigMap, metav1.CreateOptions{})
			require.NoError(t, err)

			// Create plugin
			plugin := &CarbonEmissionPlugin{
				handle: &mockFrameworkHandle{client: client},
			}

			// Create test node
			nodeInfo := &framework.NodeInfo{}
			node := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: tt.nodeName,
					Labels: map[string]string{
						"topology.kubernetes.io/zone": extractZoneFromNodeName(tt.nodeName),
					},
				},
			}
			nodeInfo.SetNode(node)

			// Create test pod
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
				},
			}

			// Execute Filter
			status := plugin.Filter(context.TODO(), nil, pod, nodeInfo)

			if tt.expectedPass {
				assert.True(t, status.IsSuccess())
			} else {
				assert.False(t, status.IsSuccess())
				assert.Contains(t, status.Message(), "carbon intensity too high")
			}
		})
	}
}

func TestExtractZoneFromNode(t *testing.T) {
	tests := []struct {
		name         string
		nodeLabels   map[string]string
		expectedZone string
	}{
		{
			name: "zone from topology label",
			nodeLabels: map[string]string{
				"topology.kubernetes.io/zone": "us-west-2a",
			},
			expectedZone: "us-west-2a",
		},
		{
			name: "zone from failure domain label",
			nodeLabels: map[string]string{
				"failure-domain.beta.kubernetes.io/zone": "us-east-1b",
			},
			expectedZone: "us-east-1b",
		},
		{
			name: "zone from node name",
			nodeLabels: map[string]string{
				"kubernetes.io/hostname": "node-us-ca-1",
			},
			expectedZone: "US-CA",
		},
		{
			name:         "no zone information",
			nodeLabels:   map[string]string{},
			expectedZone: "unknown",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Labels: tt.nodeLabels,
				},
			}

			plugin := &CarbonEmissionPlugin{}
			zone := plugin.extractZoneFromNode(node)
			assert.Equal(t, tt.expectedZone, zone)
		})
	}
}

func TestGetPodResourceRequests(t *testing.T) {
	tests := []struct {
		name         string
		pod          *v1.Pod
		expectedCPU  float64
		expectedMem  float64
	}{
		{
			name: "pod with resource requests",
			pod: &v1.Pod{
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
									v1.ResourceCPU:    resource.MustParse("200m"),
									v1.ResourceMemory: resource.MustParse("512Mi"),
								},
							},
						},
					},
				},
			},
			expectedCPU: 0.7, // 500m + 200m = 700m = 0.7 cores
			expectedMem: 1.5, // 1Gi + 512Mi = 1536Mi = 1.5Gi
		},
		{
			name: "pod without resource requests",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{},
						},
					},
				},
			},
			expectedCPU: 0.0,
			expectedMem: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plugin := &CarbonEmissionPlugin{}
			cpu, mem := plugin.getPodResourceRequests(tt.pod)
			
			assert.InDelta(t, tt.expectedCPU, cpu, 0.01)
			assert.InDelta(t, tt.expectedMem, mem, 0.01)
		})
	}
}

func TestGetThreshold(t *testing.T) {
	tests := []struct {
		name              string
		configMapData     map[string]string
		expectedThreshold float64
		expectedError     bool
	}{
		{
			name: "valid threshold",
			configMapData: map[string]string{
				"threshold": "250.5",
			},
			expectedThreshold: 250.5,
			expectedError:     false,
		},
		{
			name: "integer threshold",
			configMapData: map[string]string{
				"threshold": "300",
			},
			expectedThreshold: 300.0,
			expectedError:     false,
		},
		{
			name:              "missing threshold key",
			configMapData:     map[string]string{},
			expectedThreshold: 250.0, // default
			expectedError:     false,
		},
		{
			name: "invalid threshold format",
			configMapData: map[string]string{
				"threshold": "invalid",
			},
			expectedThreshold: 250.0, // default
			expectedError:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create fake Kubernetes client
			client := fake.NewSimpleClientset()

			// Create ConfigMap
			configMap := &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "carbon-kube-config",
					Namespace: "carbon-kube",
				},
				Data: tt.configMapData,
			}
			_, err := client.CoreV1().ConfigMaps("carbon-kube").Create(
				context.TODO(), configMap, metav1.CreateOptions{})
			require.NoError(t, err)

			// Create plugin
			plugin := &CarbonEmissionPlugin{
				handle: &mockFrameworkHandle{client: client},
			}

			// Execute GetThreshold
			threshold := plugin.GetThreshold()
			assert.Equal(t, tt.expectedThreshold, threshold)
		})
	}
}

// Helper functions and mocks

func marshalCarbonData(data map[string]interface{}) string {
	if len(data) == 0 {
		return "{}"
	}
	
	// Simple JSON marshaling for test data
	result := "{"
	first := true
	for zone, zoneData := range data {
		if !first {
			result += ","
		}
		first = false
		
		result += fmt.Sprintf(`"%s":`, zone)
		if zoneMap, ok := zoneData.(map[string]interface{}); ok {
			result += "{"
			firstInner := true
			for key, value := range zoneMap {
				if !firstInner {
					result += ","
				}
				firstInner = false
				
				if key == "carbon_intensity" || key == "timestamp" {
					result += fmt.Sprintf(`"%s":%v`, key, value)
				} else {
					result += fmt.Sprintf(`"%s":"%v"`, key, value)
				}
			}
			result += "}"
		}
	}
	result += "}"
	return result
}

func extractZoneFromNodeName(nodeName string) string {
	// Simple zone extraction for testing
	if strings.Contains(nodeName, "us-ca") {
		return "US-CA"
	} else if strings.Contains(nodeName, "us-tx") {
		return "US-TX"
	}
	return "unknown"
}

// Mock framework handle
type mockFrameworkHandle struct {
	client kubernetes.Interface
}

func (m *mockFrameworkHandle) ClientSet() kubernetes.Interface {
	return m.client
}

func (m *mockFrameworkHandle) KubeConfig() *rest.Config {
	return nil
}

func (m *mockFrameworkHandle) EventRecorder() events.EventRecorder {
	return nil
}

func (m *mockFrameworkHandle) SharedInformerFactory() informers.SharedInformerFactory {
	return nil
}

func (m *mockFrameworkHandle) SnapshotSharedLister() framework.SharedLister {
	return nil
}

func (m *mockFrameworkHandle) IterateOverWaitingPods(callback func(framework.WaitingPod)) {
}

func (m *mockFrameworkHandle) GetWaitingPod(uid types.UID) framework.WaitingPod {
	return nil
}

func (m *mockFrameworkHandle) RejectWaitingPod(uid types.UID) bool {
	return false
}

func (m *mockFrameworkHandle) RunFilterPluginsWithNominatedPods(ctx context.Context, state *framework.CycleState, pod *v1.Pod, info *framework.NodeInfo) *framework.Status {
	return nil
}

func (m *mockFrameworkHandle) Extenders() []framework.Extender {
	return nil
}

func (m *mockFrameworkHandle) Parallelizer() parallelize.Parallelizer {
	return nil
}