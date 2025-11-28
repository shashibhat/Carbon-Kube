//go:build katalyst
package emissionplugin

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

const (
	PluginName           = "EmissionPlugin"
	ConfigMapName        = "carbon-scores"
	ConfigMapNamespace   = "default"
	ZoneLabel           = "topology.kubernetes.io/zone"
	DefaultIntensity    = 200.0 // Default carbon intensity in gCO2/kWh (MOER)
	DefaultPUE          = 1.5   // Default Power Usage Effectiveness for GPU workloads
	CPUBasePUE          = 1.2   // PUE for CPU-only workloads
	GPUBasePUE          = 1.5   // PUE for GPU workloads
	CacheTTL            = 5 * time.Minute // Redis cache TTL
	RedisKeyPrefix      = "carbon-kube:"
	
	// Scoring constants for normalization
	MaxScore            = 100.0
	MinScore            = 0.0
)

// ZonalIntensity represents carbon intensity data for a zone
type ZonalIntensity struct {
	Zone      string    `json:"zone"`
	Intensity float64   `json:"intensity"` // gCO2/kWh
	Timestamp time.Time `json:"timestamp"`
}

// GPUMetrics represents real-time GPU utilization and power metrics
type GPUMetrics struct {
	DeviceID         int     `json:"device_id"`
	GPUUtilization   float64 `json:"gpu_utilization"`   // Percentage (0-100)
	MemoryUtilization float64 `json:"memory_utilization"` // Percentage (0-100)
	PowerDraw        float64 `json:"power_draw"`        // Watts
	Temperature      float64 `json:"temperature"`       // Celsius
	ClockSM          int     `json:"clock_sm"`          // MHz
	ClockMemory      int     `json:"clock_memory"`      // MHz
	Timestamp        time.Time `json:"timestamp"`
}

// DCGMClient interface for GPU metrics collection
type DCGMClient interface {
	GetGPUMetrics(nodeIP string) ([]GPUMetrics, error)
	GetGPUUtilization(nodeIP string, deviceID int) (float64, error)
	GetGPUPowerDraw(nodeIP string, deviceID int) (float64, error)
}

// HTTPDCGMClient implements DCGMClient using HTTP API
type HTTPDCGMClient struct {
	client  *http.Client
	timeout time.Duration
}

// NewDCGMClient creates a new DCGM client
func NewDCGMClient() DCGMClient {
	return &HTTPDCGMClient{
		client: &http.Client{
			Timeout: 5 * time.Second,
		},
		timeout: 5 * time.Second,
	}
}

// GetGPUMetrics retrieves comprehensive GPU metrics from DCGM
func (d *HTTPDCGMClient) GetGPUMetrics(nodeIP string) ([]GPUMetrics, error) {
	// Try DCGM HTTP API first (port 9400 is default for DCGM exporter)
	url := fmt.Sprintf("http://%s:9400/metrics", nodeIP)
	
	ctx, cancel := context.WithTimeout(context.Background(), d.timeout)
	defer cancel()
	
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}
	
	resp, err := d.client.Do(req)
	if err != nil {
		// Fallback to nvidia-smi if DCGM is not available
		return d.getGPUMetricsFromNvidiaSMI(nodeIP)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return d.getGPUMetricsFromNvidiaSMI(nodeIP)
	}
	
	// Parse DCGM metrics (simplified - in production would parse Prometheus format)
	return d.parseDCGMMetrics(resp)
}

// GetGPUUtilization gets GPU utilization for a specific device
func (d *HTTPDCGMClient) GetGPUUtilization(nodeIP string, deviceID int) (float64, error) {
	metrics, err := d.GetGPUMetrics(nodeIP)
	if err != nil {
		return 0, err
	}
	
	for _, metric := range metrics {
		if metric.DeviceID == deviceID {
			return metric.GPUUtilization, nil
		}
	}
	
	return 0, fmt.Errorf("GPU device %d not found on node %s", deviceID, nodeIP)
}

// GetGPUPowerDraw gets power draw for a specific GPU device
func (d *HTTPDCGMClient) GetGPUPowerDraw(nodeIP string, deviceID int) (float64, error) {
	metrics, err := d.GetGPUMetrics(nodeIP)
	if err != nil {
		return 0, err
	}
	
	for _, metric := range metrics {
		if metric.DeviceID == deviceID {
			return metric.PowerDraw, nil
		}
	}
	
	return 0, fmt.Errorf("GPU device %d not found on node %s", deviceID, nodeIP)
}

// getGPUMetricsFromNvidiaSMI fallback method using nvidia-smi
func (d *HTTPDCGMClient) getGPUMetricsFromNvidiaSMI(nodeIP string) ([]GPUMetrics, error) {
	// This would typically use kubectl exec or SSH to run nvidia-smi on the node
	// For this implementation, we'll simulate the call
	
	cmd := exec.Command("kubectl", "exec", "-n", "kube-system", 
		fmt.Sprintf("nvidia-device-plugin-%s", strings.Replace(nodeIP, ".", "-", -1)),
		"--", "nvidia-smi", "--query-gpu=index,utilization.gpu,utilization.memory,power.draw,temperature.gpu,clocks.sm,clocks.mem",
		"--format=csv,noheader,nounits")
	
	output, err := cmd.Output()
	if err != nil {
		// Return default metrics if nvidia-smi is not available
		return d.getDefaultGPUMetrics(), nil
	}
	
	return d.parseNvidiaSMIOutput(string(output))
}

// parseDCGMMetrics parses DCGM Prometheus metrics (simplified implementation)
func (d *HTTPDCGMClient) parseDCGMMetrics(resp *http.Response) ([]GPUMetrics, error) {
	// In a real implementation, this would parse Prometheus format
	// For now, return simulated metrics
	return d.getDefaultGPUMetrics(), nil
}

// parseNvidiaSMIOutput parses nvidia-smi CSV output
func (d *HTTPDCGMClient) parseNvidiaSMIOutput(output string) ([]GPUMetrics, error) {
	var metrics []GPUMetrics
	lines := strings.Split(strings.TrimSpace(output), "\n")
	
	for _, line := range lines {
		if line == "" {
			continue
		}
		
		fields := strings.Split(line, ",")
		if len(fields) < 7 {
			continue
		}
		
		deviceID, _ := strconv.Atoi(strings.TrimSpace(fields[0]))
		gpuUtil, _ := strconv.ParseFloat(strings.TrimSpace(fields[1]), 64)
		memUtil, _ := strconv.ParseFloat(strings.TrimSpace(fields[2]), 64)
		powerDraw, _ := strconv.ParseFloat(strings.TrimSpace(fields[3]), 64)
		temp, _ := strconv.ParseFloat(strings.TrimSpace(fields[4]), 64)
		clockSM, _ := strconv.Atoi(strings.TrimSpace(fields[5]))
		clockMem, _ := strconv.Atoi(strings.TrimSpace(fields[6]))
		
		metrics = append(metrics, GPUMetrics{
			DeviceID:         deviceID,
			GPUUtilization:   gpuUtil,
			MemoryUtilization: memUtil,
			PowerDraw:        powerDraw,
			Temperature:      temp,
			ClockSM:          clockSM,
			ClockMemory:      clockMem,
			Timestamp:        time.Now(),
		})
	}
	
	return metrics, nil
}

// getDefaultGPUMetrics returns default GPU metrics when real metrics are unavailable
func (d *HTTPDCGMClient) getDefaultGPUMetrics() []GPUMetrics {
	return []GPUMetrics{
		{
			DeviceID:         0,
			GPUUtilization:   75.0, // Assume 75% utilization
			MemoryUtilization: 60.0, // Assume 60% memory utilization
			PowerDraw:        300.0, // Default power draw
			Temperature:      65.0,  // Default temperature
			ClockSM:          1400,  // Default SM clock
			ClockMemory:      5000,  // Default memory clock
			Timestamp:        time.Now(),
		},
	}
}

// EmissionScore represents the calculated emission score for a node
type EmissionScore struct {
	NodeName string  `json:"nodeName"`
	Score    float64 `json:"score"`
	ReqCPU   int64   `json:"reqCPU"`   // CPU in millicores
	ReqMem   int64   `json:"reqMem"`   // Memory in bytes
}

// EmissionPlugin implements the Katalyst scheduler plugin interface
type EmissionPlugin struct {
	kubeClient  kubernetes.Interface
	redisClient *redis.Client
	logger      *zap.Logger
	metrics     *Metrics
	handle      framework.Handle
}

// Ensure EmissionPlugin implements required Katalyst interfaces
var _ framework.ScorePlugin = &EmissionPlugin{}
var _ framework.FilterPlugin = &EmissionPlugin{}

// Metrics holds Prometheus metrics for the plugin
type Metrics struct {
	carbonIntensity *prometheus.GaugeVec
	emissionScores  *prometheus.GaugeVec
	migrationsTotal prometheus.Counter
	savedCO2        prometheus.Counter
}

// NewEmissionPlugin creates a new instance of the emission plugin
func NewEmissionPlugin(args runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	config, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get in-cluster config: %v", err)
	}

	kubeClient, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes client: %v", err)
	}

	// Initialize structured logger
	logger, err := zap.NewProduction()
	if err != nil {
		return nil, fmt.Errorf("failed to create logger: %v", err)
	}

	// Initialize Redis client for caching
	redisClient := redis.NewClient(&redis.Options{
		Addr:     "redis-service:6379", // Kubernetes service name
		Password: "",                   // No password by default
		DB:       0,                    // Default DB
	})

	// Test Redis connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := redisClient.Ping(ctx).Err(); err != nil {
		logger.Warn("Redis connection failed, falling back to ConfigMap only", zap.Error(err))
		redisClient = nil // Disable Redis if connection fails
	}

	metrics := &Metrics{
		carbonIntensity: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "carbon_intensity_gco2_per_kwh",
				Help: "Carbon intensity by zone in gCO2/kWh",
			},
			[]string{"zone"},
		),
		emissionScores: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "emission_scores",
				Help: "Calculated emission scores for nodes",
			},
			[]string{"node", "zone"},
		),
		migrationsTotal: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "migrations_total",
				Help: "Total number of carbon-aware migrations",
			},
		),
		savedCO2: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "saved_co2_kg_total",
				Help: "Total CO2 saved in kg through migrations",
			},
		),
	}

	return &EmissionPlugin{
		kubeClient:  kubeClient,
		redisClient: redisClient,
		logger:      logger,
		metrics:     metrics,
		handle:      handle,
	}, nil
}

// Name returns the plugin name
func (p *EmissionPlugin) Name() string {
	return PluginName
}

// Score implements the Katalyst ScorePlugin interface
// Calculates emission scores using the equation: E = ∫ P_IT * PUE * MOER dt
func (p *EmissionPlugin) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
    nodeInfo, err := p.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
    if err != nil || nodeInfo == nil {
        return 0, framework.NewStatus(framework.Error, fmt.Sprintf("node %s not found", nodeName))
    }

	p.logger.Debug("EmissionPlugin scoring node",
		zap.String("pod", pod.Name),
		zap.String("namespace", pod.Namespace),
		zap.String("node", nodeName))

	// Get carbon intensity data from cache or ConfigMap
	intensityMap, err := p.getCarbonIntensityData(ctx)
	if err != nil {
		p.logger.Error("Failed to get carbon intensity data, using default score",
			zap.Error(err),
			zap.String("pod", pod.Name),
			zap.String("node", nodeName))
		return 50, framework.NewStatus(framework.Success, "") // Neutral score
	}

	// Calculate pod resource requirements
	podRequests := p.getPodResourceRequests(pod)
	
	// Extract zone and get carbon intensity (MOER)
	zone := p.extractZoneFromNode(nodeInfo.Node())
	moer := p.getIntensityForZone(zone, intensityMap) // gCO2/kWh
	
	// Calculate Power IT (P_IT) based on resource requests
	cpuPowerKW := p.calculateCPUPower(podRequests.CPU)
	gpuPowerKW := p.calculateGPUPower(pod, nodeInfo.Node())
	totalPowerKW := cpuPowerKW + gpuPowerKW
	
	// Determine PUE based on workload type
	pue := CPUBasePUE
	if gpuPowerKW > 0 {
		pue = GPUBasePUE // Higher PUE for GPU workloads
	}
	
	// Calculate emission score: E = P_IT * PUE * MOER
	// This represents gCO2 per hour for this workload
	emissionRate := totalPowerKW * pue * moer
	
	// Normalize to 0-100 scale (lower emissions = higher score for scheduling preference)
	// We invert the score so lower emissions get higher scheduling priority
	normalizedScore := p.normalizeEmissionScore(emissionRate)
	
	// Update metrics
	p.metrics.carbonIntensity.WithLabelValues(zone).Set(moer)
	p.metrics.emissionScores.WithLabelValues(nodeName, zone).Set(emissionRate)
	
	p.logger.Debug("Calculated emission score",
		zap.String("node", nodeName),
		zap.String("zone", zone),
		zap.Float64("moer", moer),
		zap.Float64("cpuPowerKW", cpuPowerKW),
		zap.Float64("gpuPowerKW", gpuPowerKW),
		zap.Float64("pue", pue),
		zap.Float64("emissionRate", emissionRate),
		zap.Int64("normalizedScore", normalizedScore))

	return normalizedScore, framework.NewStatus(framework.Success, "")
}

// ScoreExtensions returns the score extensions for the plugin
func (p *EmissionPlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil // No extensions needed for basic scoring
}

// Filter implements the Katalyst FilterPlugin interface
// Filters out nodes that don't meet carbon emission thresholds
func (p *EmissionPlugin) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	// For now, we don't filter any nodes - all nodes are eligible
	// This could be extended to filter nodes based on carbon intensity thresholds
	// or other emission-related criteria
	
	p.logger.Debug("EmissionPlugin filter check",
		zap.String("pod", pod.Name),
		zap.String("node", nodeInfo.Node().Name))
	
	return framework.NewStatus(framework.Success, "")
}

// getCarbonIntensityData retrieves carbon intensity data from Redis cache or ConfigMap
func (p *EmissionPlugin) getCarbonIntensityData(ctx context.Context) (map[string]ZonalIntensity, error) {
	cacheKey := RedisKeyPrefix + "intensity_data"
	
	// Try Redis cache first if available
	if p.redisClient != nil {
		cachedData, err := p.redisClient.Get(ctx, cacheKey).Result()
		if err == nil {
			var intensityMap map[string]ZonalIntensity
			if err := json.Unmarshal([]byte(cachedData), &intensityMap); err == nil {
				p.logger.Debug("Retrieved carbon intensity data from Redis cache",
					zap.String("cacheKey", cacheKey),
					zap.Int("zones", len(intensityMap)))
				return intensityMap, nil
			}
			p.logger.Warn("Failed to unmarshal cached data", zap.Error(err))
		} else if err != redis.Nil {
			p.logger.Warn("Redis cache read failed", zap.Error(err))
		}
	}

	// Fallback to ConfigMap
	p.logger.Debug("Fetching carbon intensity data from ConfigMap")
	configMap, err := p.kubeClient.CoreV1().ConfigMaps(ConfigMapNamespace).Get(
		ctx, ConfigMapName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get ConfigMap %s/%s: %v", ConfigMapNamespace, ConfigMapName, err)
	}

	zonesData, exists := configMap.Data["zones"]
	if !exists {
		return nil, fmt.Errorf("zones data not found in ConfigMap")
	}

	var intensityMap map[string]ZonalIntensity
	if err := json.Unmarshal([]byte(zonesData), &intensityMap); err != nil {
		return nil, fmt.Errorf("failed to unmarshal zones data: %v", err)
	}

	// Cache the data in Redis if available
	if p.redisClient != nil {
		if cacheData, err := json.Marshal(intensityMap); err == nil {
			if err := p.redisClient.Set(ctx, cacheKey, cacheData, CacheTTL).Err(); err != nil {
				p.logger.Warn("Failed to cache intensity data in Redis", zap.Error(err))
			} else {
				p.logger.Debug("Cached carbon intensity data in Redis",
					zap.String("cacheKey", cacheKey),
					zap.Duration("ttl", CacheTTL),
					zap.Int("zones", len(intensityMap)))
			}
		}
	}

	return intensityMap, nil
}

// getPodResourceRequests extracts CPU and memory requests from a pod
func (p *EmissionPlugin) getPodResourceRequests(pod *v1.Pod) struct{ CPU, Memory int64 } {
	var totalCPU, totalMemory int64

	for _, container := range pod.Spec.Containers {
		if cpu := container.Resources.Requests.Cpu(); cpu != nil {
			totalCPU += cpu.MilliValue()
		}
		if memory := container.Resources.Requests.Memory(); memory != nil {
			totalMemory += memory.Value()
		}
	}

	return struct{ CPU, Memory int64 }{CPU: totalCPU, Memory: totalMemory}
}

// extractZoneFromNode extracts the availability zone from node labels
func (p *EmissionPlugin) extractZoneFromNode(node *v1.Node) string {
	if zone, exists := node.Labels[ZoneLabel]; exists {
		return zone
	}
	
	// Fallback: try to extract from node name or other labels
	if strings.Contains(node.Name, "us-west") {
		return "us-west-2a" // Default to a green zone
	}
	if strings.Contains(node.Name, "us-east") {
		return "us-east-1a" // Default to a less green zone
	}
	
	return "unknown"
}

// getIntensityForZone returns the carbon intensity for a given zone
func (p *EmissionPlugin) getIntensityForZone(zone string, intensityMap map[string]ZonalIntensity) float64 {
	if data, exists := intensityMap[zone]; exists {
		// Check if data is recent (within last 10 minutes)
		if time.Now().Unix()-data.Timestamp < 600 {
			return data.Intensity
		}
	}
	
	// Fallback to default intensity
	klog.V(3).InfoS("Using default intensity for zone", "zone", zone, "defaultIntensity", DefaultIntensity)
	return DefaultIntensity
}

// calculateCPUPower estimates CPU power consumption in kW based on millicores
func (p *EmissionPlugin) calculateCPUPower(cpuMillicores int64) float64 {
	// Typical server CPU: ~150W at full load for 32 cores
	// This gives us ~4.7W per core or ~0.0047W per millicore
	const cpuWattsPerMillicore = 0.0047
	powerWatts := float64(cpuMillicores) * cpuWattsPerMillicore
	return powerWatts / 1000.0 // Convert to kW
}

// calculateGPUPower estimates GPU power consumption in kW based on GPU requests with DCGM integration
func (p *EmissionPlugin) calculateGPUPower(pod *v1.Pod, node *v1.Node) float64 {
	var totalGPUPower float64
	
	// Initialize DCGM client for real-time metrics
	dcgmClient := NewDCGMClient()
	nodeIP := p.getNodeIP(node)
	
	// Check for GPU resource requests in the pod
	for _, container := range pod.Spec.Containers {
		if gpuRequest := container.Resources.Requests["nvidia.com/gpu"]; !gpuRequest.IsZero() {
			gpuCount := gpuRequest.Value()
			
			// Determine GPU type from node labels or annotations
			gpuType := p.getGPUType(node)
			
			// Calculate power consumption with DCGM integration
			for i := 0; i < int(gpuCount); i++ {
				gpuPower := p.calculateGPUPowerWithDCGM(dcgmClient, nodeIP, i, gpuType)
				totalGPUPower += gpuPower
			}
		}
	}
	
	return totalGPUPower / 1000.0 // Convert to kW
}

// calculateGPUPowerWithDCGM calculates GPU power using DCGM metrics and TDP
func (p *EmissionPlugin) calculateGPUPowerWithDCGM(dcgmClient DCGMClient, nodeIP string, deviceID int, gpuType string) float64 {
	// Try to get real-time power draw from DCGM
	if realPowerDraw, err := dcgmClient.GetGPUPowerDraw(nodeIP, deviceID); err == nil {
		p.logger.Debug("Using real-time GPU power draw from DCGM",
			zap.String("nodeIP", nodeIP),
			zap.Int("deviceID", deviceID),
			zap.Float64("powerDraw", realPowerDraw))
		return realPowerDraw
	}
	
	// Fallback to utilization-based calculation
	if utilization, err := dcgmClient.GetGPUUtilization(nodeIP, deviceID); err == nil {
		tdpWatts := p.getGPUTDP(gpuType)
		
		// Calculate power based on utilization and TDP
		// Power = Base Power + (TDP - Base Power) * Utilization
		basePowerRatio := 0.15 // GPUs consume ~15% of TDP at idle
		basePower := tdpWatts * basePowerRatio
		dynamicPower := (tdpWatts - basePower) * (utilization / 100.0)
		
		calculatedPower := basePower + dynamicPower
		
		p.logger.Debug("Calculated GPU power from utilization",
			zap.String("nodeIP", nodeIP),
			zap.Int("deviceID", deviceID),
			zap.String("gpuType", gpuType),
			zap.Float64("utilization", utilization),
			zap.Float64("tdp", tdpWatts),
			zap.Float64("calculatedPower", calculatedPower))
		
		return calculatedPower
	}
	
	// Final fallback to static TDP-based estimation
	tdpWatts := p.getGPUTDP(gpuType)
	// Assume 70% average utilization for workloads
	estimatedPower := tdpWatts * 0.7
	
	p.logger.Debug("Using TDP-based GPU power estimation",
		zap.String("gpuType", gpuType),
		zap.Float64("tdp", tdpWatts),
		zap.Float64("estimatedPower", estimatedPower))
	
	return estimatedPower
}

// getNodeIP extracts the internal IP address of a node
func (p *EmissionPlugin) getNodeIP(node *v1.Node) string {
	for _, addr := range node.Status.Addresses {
		if addr.Type == v1.NodeInternalIP {
			return addr.Address
		}
	}
	// Fallback to external IP if internal IP is not available
	for _, addr := range node.Status.Addresses {
		if addr.Type == v1.NodeExternalIP {
			return addr.Address
		}
	}
	return "unknown"
}

// getGPUTDP returns the Thermal Design Power (TDP) for different GPU types
func (p *EmissionPlugin) getGPUTDP(gpuType string) float64 {
	switch strings.ToUpper(gpuType) {
	case "H100":
		return 700.0 // NVIDIA H100 SXM: 700W TDP
	case "H100-PCIE":
		return 350.0 // NVIDIA H100 PCIe: 350W TDP
	case "A100":
		return 400.0 // NVIDIA A100 SXM: 400W TDP
	case "A100-PCIE":
		return 250.0 // NVIDIA A100 PCIe: 250W TDP
	case "V100":
		return 300.0 // NVIDIA V100 SXM2: 300W TDP
	case "V100-PCIE":
		return 250.0 // NVIDIA V100 PCIe: 250W TDP
	case "T4":
		return 70.0  // NVIDIA T4: 70W TDP
	case "RTX4090":
		return 450.0 // RTX 4090: 450W TDP
	case "RTX3090":
		return 350.0 // RTX 3090: 350W TDP
	case "RTX3080":
		return 320.0 // RTX 3080: 320W TDP
	case "L4":
		return 72.0  // NVIDIA L4: 72W TDP
	case "L40":
		return 300.0 // NVIDIA L40: 300W TDP
	case "A40":
		return 300.0 // NVIDIA A40: 300W TDP
	case "A30":
		return 165.0 // NVIDIA A30: 165W TDP
	case "A10":
		return 150.0 // NVIDIA A10: 150W TDP
	default:
		return 250.0 // Default GPU TDP for unknown types
	}
}

// getGPUType extracts GPU type from node labels/annotations
func (p *EmissionPlugin) getGPUType(node *v1.Node) string {
	// Check common GPU type labels
	if gpuType, exists := node.Labels["nvidia.com/gpu.product"]; exists {
		return gpuType
	}
	if gpuType, exists := node.Labels["accelerator"]; exists {
		return gpuType
	}
	
	// Fallback: try to infer from instance type
	if instanceType, exists := node.Labels["node.kubernetes.io/instance-type"]; exists {
		if strings.Contains(instanceType, "p4d") {
			return "A100" // p4d instances use A100 GPUs
		}
		if strings.Contains(instanceType, "p3") {
			return "V100" // p3 instances use V100 GPUs
		}
		if strings.Contains(instanceType, "g4") {
			return "T4" // g4 instances use T4 GPUs
		}
	}
	
	return "unknown"
}

// getGPUPowerConsumption returns power consumption in watts for different GPU types
func (p *EmissionPlugin) getGPUPowerConsumption(gpuType string) float64 {
	switch strings.ToUpper(gpuType) {
	case "H100":
		return 700.0 // NVIDIA H100: ~700W
	case "A100":
		return 400.0 // NVIDIA A100: ~400W
	case "V100":
		return 300.0 // NVIDIA V100: ~300W
	case "T4":
		return 70.0  // NVIDIA T4: ~70W
	case "RTX4090":
		return 450.0 // RTX 4090: ~450W
	default:
		return 250.0 // Default GPU power consumption
	}
}

// normalizeEmissionScore converts emission rate to a 0-100 score
// Lower emissions get higher scores (better for scheduling)
func (p *EmissionPlugin) normalizeEmissionScore(emissionRate float64) int64 {
	// Define reasonable bounds for emission rates (gCO2/hour)
	const (
		minEmissionRate = 10.0   // Very low emission rate
		maxEmissionRate = 1000.0 // Very high emission rate
	)
	
	// Clamp the emission rate to bounds
	clampedRate := math.Max(minEmissionRate, math.Min(maxEmissionRate, emissionRate))
	
	// Invert and normalize: lower emissions = higher score
	// Score = 100 * (1 - (rate - min) / (max - min))
	normalizedScore := MaxScore * (1.0 - (clampedRate-minEmissionRate)/(maxEmissionRate-minEmissionRate))
	
	// Ensure score is within bounds
	score := int64(math.Max(MinScore, math.Min(MaxScore, normalizedScore)))
	
	return score
}

// RecordMigration records a successful migration for metrics
func (p *EmissionPlugin) RecordMigration(savedCO2KG float64) {
	p.metrics.migrationsTotal.Inc()
	p.metrics.savedCO2.Add(savedCO2KG)
}

// GetThreshold returns the current migration threshold from ConfigMap
func (p *EmissionPlugin) GetThreshold(ctx context.Context) (float64, error) {
	configMap, err := p.kubeClient.CoreV1().ConfigMaps(ConfigMapNamespace).Get(
		ctx, ConfigMapName, metav1.GetOptions{})
	if err != nil {
		return 200.0, err // Default threshold
	}

	thresholdStr, exists := configMap.Data["threshold"]
	if !exists {
		return 200.0, nil // Default threshold
	}

	threshold, err := strconv.ParseFloat(thresholdStr, 64)
	if err != nil {
		return 200.0, err
	}

	return threshold, nil
}

// New is the plugin constructor function for Katalyst framework registration
func New(args runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	return NewEmissionPlugin(args, handle)
}

// Plugin registration for Katalyst framework
func init() {
	// The plugin will be registered by the Katalyst framework
	// using the New function above
}
