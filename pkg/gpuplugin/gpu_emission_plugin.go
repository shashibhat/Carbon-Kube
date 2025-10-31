package gpuplugin

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
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
)

const (
	PluginName              = "GPUEmissionPlugin"
	ConfigMapName           = "gpu-carbon-scores"
	ConfigMapNamespace      = "default"
	ZoneLabel              = "topology.kubernetes.io/zone"
	GPUProductLabel        = "nvidia.com/gpu.product"
	GPUMemoryLabel         = "nvidia.com/gpu.memory"
	GPUCountLabel          = "nvidia.com/gpu.count"
	MIGLabel               = "nvidia.com/mig.config"
	DefaultIntensity       = 200.0 // Default carbon intensity in gCO2/kWh
	DefaultPUE             = 1.4   // Default Power Usage Effectiveness
	CacheTTL               = 5 * time.Minute
	RedisKeyPrefix         = "carbon-kube:gpu:"
	MigrationCooldown      = 10 * time.Minute
	CheckpointThreshold    = 0.8   // Checkpoint if migration saves >80% carbon
)

// GPUZonalIntensity represents carbon intensity data for GPU workloads
type GPUZonalIntensity struct {
	Zone         string    `json:"zone"`
	Intensity    float64   `json:"intensity"`    // gCO2/kWh
	Timestamp    int64     `json:"timestamp"`
	Forecast     []float64 `json:"forecast"`     // 1-4 hour forecast
	PUE          float64   `json:"pue"`          // Power Usage Effectiveness
	GPUBasePower float64   `json:"gpu_base_power"` // Base GPU power in watts
}

// GPUEmissionScore represents the calculated emission score for GPU nodes
type GPUEmissionScore struct {
	NodeName        string  `json:"nodeName"`
	Score           float64 `json:"score"`
	GPUCount        int     `json:"gpuCount"`
	GPUProduct      string  `json:"gpuProduct"`
	EstimatedPower  float64 `json:"estimatedPower"`  // Watts
	CarbonRate      float64 `json:"carbonRate"`      // gCO2/hour
	MIGEnabled      bool    `json:"migEnabled"`
	LastMigration   int64   `json:"lastMigration"`   // Unix timestamp
	SLACritical     bool    `json:"slaCritical"`
}

// GPUWorkloadProfile defines power characteristics for different GPU workloads
type GPUWorkloadProfile struct {
	Name           string  `json:"name"`
	BasePowerRatio float64 `json:"basePowerRatio"` // Ratio of TDP
	MemoryRatio    float64 `json:"memoryRatio"`    // Memory utilization factor
	ComputeRatio   float64 `json:"computeRatio"`   // Compute utilization factor
}

// GPUEmissionPlugin implements carbon-aware GPU scheduling
type GPUEmissionPlugin struct {
	kubeClient      kubernetes.Interface
	redisClient     *redis.Client
	logger          *zap.Logger
	metrics         *GPUMetrics
	workloadProfiles map[string]GPUWorkloadProfile
}

// GPUMetrics holds Prometheus metrics for GPU carbon monitoring
type GPUMetrics struct {
	gpuCarbonIntensity    *prometheus.GaugeVec
	gpuEmissionScores     *prometheus.GaugeVec
	gpuPowerConsumption   *prometheus.GaugeVec
	gpuMigrationsTotal    prometheus.Counter
	gpuSavedCO2           prometheus.Counter
	gpuCheckpointLatency  prometheus.Histogram
	gpuMIGUtilization     *prometheus.GaugeVec
	gpuSLAViolations      prometheus.Counter
}

// NewGPUEmissionPlugin creates a new GPU-aware emission plugin
func NewGPUEmissionPlugin() (*GPUEmissionPlugin, error) {
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
		Addr:     "redis-service:6379",
		Password: "",
		DB:       1, // Use DB 1 for GPU data
	})

	// Test Redis connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := redisClient.Ping(ctx).Err(); err != nil {
		logger.Warn("Redis connection failed, falling back to ConfigMap only", zap.Error(err))
		redisClient = nil
	}

	// Initialize GPU workload profiles
	workloadProfiles := map[string]GPUWorkloadProfile{
		"training": {
			Name:           "ML Training",
			BasePowerRatio: 0.85, // 85% of TDP for training
			MemoryRatio:    0.90, // High memory usage
			ComputeRatio:   0.95, // High compute usage
		},
		"inference": {
			Name:           "ML Inference", 
			BasePowerRatio: 0.60, // 60% of TDP for inference
			MemoryRatio:    0.70, // Moderate memory usage
			ComputeRatio:   0.75, // Moderate compute usage
		},
		"batch": {
			Name:           "Batch Processing",
			BasePowerRatio: 0.75, // 75% of TDP for batch
			MemoryRatio:    0.80, // High memory usage
			ComputeRatio:   0.85, // High compute usage
		},
		"interactive": {
			Name:           "Interactive/Jupyter",
			BasePowerRatio: 0.40, // 40% of TDP for interactive
			MemoryRatio:    0.50, // Low memory usage
			ComputeRatio:   0.45, // Low compute usage
		},
	}

	metrics := &GPUMetrics{
		gpuCarbonIntensity: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "gpu_carbon_intensity_gco2_per_kwh",
				Help: "GPU carbon intensity by zone in gCO2/kWh",
			},
			[]string{"zone", "gpu_product"},
		),
		gpuEmissionScores: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "gpu_emission_scores",
				Help: "Calculated GPU emission scores for nodes",
			},
			[]string{"node", "zone", "gpu_product", "mig_enabled"},
		),
		gpuPowerConsumption: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "gpu_estimated_power_watts",
				Help: "Estimated GPU power consumption in watts",
			},
			[]string{"node", "gpu_product", "workload_type"},
		),
		gpuMigrationsTotal: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "gpu_migrations_total",
				Help: "Total number of carbon-aware GPU migrations",
			},
		),
		gpuSavedCO2: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "gpu_saved_co2_kg_total",
				Help: "Total CO2 saved in kg through GPU migrations",
			},
		),
		gpuCheckpointLatency: promauto.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "gpu_checkpoint_latency_seconds",
				Help:    "GPU workload checkpoint latency in seconds",
				Buckets: prometheus.ExponentialBuckets(1, 2, 10), // 1s to 512s
			},
		),
		gpuMIGUtilization: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "gpu_mig_utilization_ratio",
				Help: "MIG instance utilization ratio",
			},
			[]string{"node", "mig_profile"},
		),
		gpuSLAViolations: promauto.NewCounter(
			prometheus.CounterOpts{
				Name: "gpu_sla_violations_total",
				Help: "Total number of GPU SLA violations",
			},
		),
	}

	return &GPUEmissionPlugin{
		kubeClient:       kubeClient,
		redisClient:      redisClient,
		logger:           logger,
		metrics:          metrics,
		workloadProfiles: workloadProfiles,
	}, nil
}

// Name returns the plugin name
func (p *GPUEmissionPlugin) Name() string {
	return PluginName
}

// Execute implements the main GPU scheduling logic
func (p *GPUEmissionPlugin) Execute(ctx context.Context, pod *v1.Pod, nodes []*v1.Node) (map[string]float64, bool) {
	p.logger.Info("GPUEmissionPlugin executing",
		zap.String("pod", pod.Name),
		zap.String("namespace", pod.Namespace),
		zap.Int("nodeCount", len(nodes)))

	// Check if pod requires GPU resources
	if !p.requiresGPU(pod) {
		p.logger.Debug("Pod does not require GPU resources, skipping",
			zap.String("pod", pod.Name))
		return map[string]float64{}, false
	}

	scores := make(map[string]float64)
	
	// Get GPU carbon intensity data
	intensityMap, err := p.getGPUCarbonIntensityData(ctx)
	if err != nil {
		p.logger.Error("Failed to get GPU carbon intensity data",
			zap.Error(err),
			zap.String("pod", pod.Name))
		// Fallback: assign neutral scores
		for _, node := range nodes {
			scores[node.Name] = 0.0
		}
		return scores, true
	}

	// Determine workload profile
	workloadType := p.getWorkloadType(pod)
	profile := p.workloadProfiles[workloadType]
	
	// Check SLA criticality
	slaCritical := p.isSLACritical(pod)
	
	p.logger.Info("GPU workload analysis",
		zap.String("pod", pod.Name),
		zap.String("workloadType", workloadType),
		zap.Bool("slaCritical", slaCritical))

	// Score each GPU-enabled node
	for _, node := range nodes {
		if !p.hasGPU(node) {
			continue // Skip non-GPU nodes
		}

		zone := p.extractZoneFromNode(node)
		gpuProduct := p.getGPUProduct(node)
		gpuCount := p.getGPUCount(node)
		migEnabled := p.isMIGEnabled(node)
		
		// Get carbon intensity for this zone
		intensity, exists := intensityMap[zone]
		if !exists {
			intensity = GPUZonalIntensity{
				Zone:         zone,
				Intensity:    DefaultIntensity,
				PUE:          DefaultPUE,
				GPUBasePower: p.getGPUBasePower(gpuProduct),
			}
		}

		// Calculate migration cooldown penalty
		migrationPenalty := p.getMigrationPenalty(ctx, node.Name)
		
		// Calculate carbon emission score
		emissionScore := p.calculateGPUEmissionScore(
			intensity, profile, gpuCount, slaCritical, migrationPenalty)
		
		scores[node.Name] = emissionScore
		
		// Update metrics
		p.updateMetrics(node, zone, gpuProduct, migEnabled, intensity, profile, emissionScore)
		
		p.logger.Debug("Calculated GPU emission score",
			zap.String("node", node.Name),
			zap.String("zone", zone),
			zap.String("gpuProduct", gpuProduct),
			zap.Int("gpuCount", gpuCount),
			zap.Bool("migEnabled", migEnabled),
			zap.Float64("intensity", intensity.Intensity),
			zap.Float64("score", emissionScore),
			zap.Float64("migrationPenalty", migrationPenalty))
	}

	p.logger.Info("GPUEmissionPlugin completed",
		zap.String("pod", pod.Name),
		zap.Int("gpuNodesScored", len(scores)))
	
	return scores, len(scores) > 0
}

// requiresGPU checks if pod requires GPU resources
func (p *GPUEmissionPlugin) requiresGPU(pod *v1.Pod) bool {
	for _, container := range pod.Spec.Containers {
		if requests := container.Resources.Requests; requests != nil {
			if _, hasGPU := requests["nvidia.com/gpu"]; hasGPU {
				return true
			}
			// Check for MIG resources
			for resource := range requests {
				if strings.Contains(string(resource), "nvidia.com/mig-") {
					return true
				}
			}
		}
	}
	return false
}

// getWorkloadType determines the GPU workload type from pod annotations
func (p *GPUEmissionPlugin) getWorkloadType(pod *v1.Pod) string {
	if workloadType, exists := pod.Annotations["carbon-kube.io/gpu-workload-type"]; exists {
		return workloadType
	}
	
	// Infer from pod labels and image
	if _, exists := pod.Labels["app.kubernetes.io/name"]; exists {
		name := pod.Labels["app.kubernetes.io/name"]
		if strings.Contains(name, "jupyter") || strings.Contains(name, "notebook") {
			return "interactive"
		}
		if strings.Contains(name, "training") || strings.Contains(name, "train") {
			return "training"
		}
		if strings.Contains(name, "inference") || strings.Contains(name, "serve") {
			return "inference"
		}
	}
	
	// Default to batch processing
	return "batch"
}

// isSLACritical checks if pod has SLA criticality annotation
func (p *GPUEmissionPlugin) isSLACritical(pod *v1.Pod) bool {
	if critical, exists := pod.Annotations["carbon-kube.io/sla-critical"]; exists {
		return critical == "true"
	}
	
	// Check for priority class
	if pod.Spec.PriorityClassName != "" {
		return strings.Contains(pod.Spec.PriorityClassName, "critical") ||
			   strings.Contains(pod.Spec.PriorityClassName, "high")
	}
	
	return false
}

// hasGPU checks if node has GPU resources
func (p *GPUEmissionPlugin) hasGPU(node *v1.Node) bool {
	if capacity := node.Status.Capacity; capacity != nil {
		if _, hasGPU := capacity["nvidia.com/gpu"]; hasGPU {
			return true
		}
	}
	return false
}

// getGPUProduct extracts GPU product from node labels
func (p *GPUEmissionPlugin) getGPUProduct(node *v1.Node) string {
	if product, exists := node.Labels[GPUProductLabel]; exists {
		return product
	}
	return "unknown"
}

// getGPUCount gets the number of GPUs on the node
func (p *GPUEmissionPlugin) getGPUCount(node *v1.Node) int {
	if capacity := node.Status.Capacity; capacity != nil {
		if gpuQuantity, exists := capacity["nvidia.com/gpu"]; exists {
			if count, err := strconv.Atoi(gpuQuantity.String()); err == nil {
				return count
			}
		}
	}
	return 0
}

// isMIGEnabled checks if MIG is enabled on the node
func (p *GPUEmissionPlugin) isMIGEnabled(node *v1.Node) bool {
	if migConfig, exists := node.Labels[MIGLabel]; exists {
		return migConfig != "all-disabled"
	}
	return false
}

// extractZoneFromNode extracts availability zone from node labels
func (p *GPUEmissionPlugin) extractZoneFromNode(node *v1.Node) string {
	if zone, exists := node.Labels[ZoneLabel]; exists {
		return zone
	}
	return "unknown"
}

// getGPUBasePower returns base power consumption for GPU product
func (p *GPUEmissionPlugin) getGPUBasePower(gpuProduct string) float64 {
	// GPU TDP values in watts
	gpuPowerMap := map[string]float64{
		"NVIDIA-A100-SXM4-40GB":  400.0,
		"NVIDIA-A100-SXM4-80GB":  400.0,
		"NVIDIA-H100-SXM5-80GB":  700.0,
		"NVIDIA-V100-SXM2-16GB":  300.0,
		"NVIDIA-V100-SXM2-32GB":  300.0,
		"NVIDIA-T4":              70.0,
		"NVIDIA-RTX-A6000":       300.0,
		"NVIDIA-RTX-A5000":       230.0,
	}
	
	if power, exists := gpuPowerMap[gpuProduct]; exists {
		return power
	}
	return 250.0 // Default GPU power
}

// calculateGPUEmissionScore computes the carbon emission score for GPU scheduling
func (p *GPUEmissionPlugin) calculateGPUEmissionScore(
	intensity GPUZonalIntensity,
	profile GPUWorkloadProfile,
	gpuCount int,
	slaCritical bool,
	migrationPenalty float64) float64 {
	
	// Base power consumption calculation
	basePower := intensity.GPUBasePower * profile.BasePowerRatio * float64(gpuCount)
	
	// Apply PUE (Power Usage Effectiveness)
	totalPower := basePower * intensity.PUE
	
	// Convert to carbon emission rate (gCO2/hour)
	carbonRate := totalPower * intensity.Intensity / 1000.0 // Convert W to kW
	
	// Apply forecast if available (use 1-hour forecast)
	if len(intensity.Forecast) > 0 {
		forecastIntensity := intensity.Forecast[0]
		forecastCarbonRate := totalPower * forecastIntensity / 1000.0
		// Weight current vs forecast (70% current, 30% forecast)
		carbonRate = 0.7*carbonRate + 0.3*forecastCarbonRate
	}
	
	// SLA penalty: critical workloads get lower scores (higher priority)
	slaMultiplier := 1.0
	if slaCritical {
		slaMultiplier = 0.5 // Prefer critical workloads
	}
	
	// Migration penalty: recently migrated nodes get higher scores
	migrationMultiplier := 1.0 + migrationPenalty
	
	// Final emission score (higher = worse for scheduling)
	score := carbonRate * slaMultiplier * migrationMultiplier
	
	return score
}

// getMigrationPenalty calculates penalty for recent migrations
func (p *GPUEmissionPlugin) getMigrationPenalty(ctx context.Context, nodeName string) float64 {
	if p.redisClient == nil {
		return 0.0
	}
	
	key := RedisKeyPrefix + "migration:" + nodeName
	lastMigrationStr, err := p.redisClient.Get(ctx, key).Result()
	if err != nil {
		return 0.0 // No recent migration
	}
	
	lastMigration, err := strconv.ParseInt(lastMigrationStr, 10, 64)
	if err != nil {
		return 0.0
	}
	
	timeSinceMigration := time.Since(time.Unix(lastMigration, 0))
	if timeSinceMigration < MigrationCooldown {
		// Linear penalty that decreases over time
		penalty := 1.0 - (timeSinceMigration.Seconds() / MigrationCooldown.Seconds())
		return penalty * 0.5 // Max 50% penalty
	}
	
	return 0.0
}

// updateMetrics updates Prometheus metrics
func (p *GPUEmissionPlugin) updateMetrics(
	node *v1.Node,
	zone, gpuProduct string,
	migEnabled bool,
	intensity GPUZonalIntensity,
	profile GPUWorkloadProfile,
	emissionScore float64) {
	
	p.metrics.gpuCarbonIntensity.WithLabelValues(zone, gpuProduct).Set(intensity.Intensity)
	p.metrics.gpuEmissionScores.WithLabelValues(
		node.Name, zone, gpuProduct, fmt.Sprintf("%t", migEnabled)).Set(emissionScore)
	
	estimatedPower := intensity.GPUBasePower * profile.BasePowerRatio * intensity.PUE
	p.metrics.gpuPowerConsumption.WithLabelValues(
		node.Name, gpuProduct, profile.Name).Set(estimatedPower)
}

// getGPUCarbonIntensityData retrieves GPU-specific carbon intensity data
func (p *GPUEmissionPlugin) getGPUCarbonIntensityData(ctx context.Context) (map[string]GPUZonalIntensity, error) {
	cacheKey := RedisKeyPrefix + "gpu_intensity_data"
	
	// Try Redis cache first
	if p.redisClient != nil {
		cachedData, err := p.redisClient.Get(ctx, cacheKey).Result()
		if err == nil {
			var intensityMap map[string]GPUZonalIntensity
			if err := json.Unmarshal([]byte(cachedData), &intensityMap); err == nil {
				p.logger.Debug("Retrieved GPU carbon intensity data from Redis cache",
					zap.String("cacheKey", cacheKey),
					zap.Int("zones", len(intensityMap)))
				return intensityMap, nil
			}
		}
	}
	
	// Fallback to ConfigMap
	configMap, err := p.kubeClient.CoreV1().ConfigMaps(ConfigMapNamespace).Get(
		ctx, ConfigMapName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get GPU ConfigMap %s/%s: %v", 
			ConfigMapNamespace, ConfigMapName, err)
	}
	
	zonesData, exists := configMap.Data["gpu_zones"]
	if !exists {
		return nil, fmt.Errorf("gpu_zones data not found in ConfigMap")
	}
	
	var intensityMap map[string]GPUZonalIntensity
	if err := json.Unmarshal([]byte(zonesData), &intensityMap); err != nil {
		return nil, fmt.Errorf("failed to unmarshal GPU zones data: %v", err)
	}
	
	// Cache the data
	if p.redisClient != nil {
		if cacheData, err := json.Marshal(intensityMap); err == nil {
			p.redisClient.Set(ctx, cacheKey, cacheData, CacheTTL)
		}
	}
	
	return intensityMap, nil
}

// RecordGPUMigration records a GPU migration event
func (p *GPUEmissionPlugin) RecordGPUMigration(nodeName string, savedCO2KG float64) {
	p.metrics.gpuMigrationsTotal.Inc()
	p.metrics.gpuSavedCO2.Add(savedCO2KG)
	
	// Record migration timestamp in Redis
	if p.redisClient != nil {
		ctx := context.Background()
		key := RedisKeyPrefix + "migration:" + nodeName
		timestamp := time.Now().Unix()
		p.redisClient.Set(ctx, key, timestamp, MigrationCooldown*2)
	}
	
	p.logger.Info("GPU migration recorded",
		zap.String("node", nodeName),
		zap.Float64("savedCO2KG", savedCO2KG))
}

// RecordCheckpointLatency records checkpoint operation latency
func (p *GPUEmissionPlugin) RecordCheckpointLatency(latencySeconds float64) {
	p.metrics.gpuCheckpointLatency.Observe(latencySeconds)
}

// RecordSLAViolation records an SLA violation
func (p *GPUEmissionPlugin) RecordSLAViolation() {
	p.metrics.gpuSLAViolations.Inc()
}

var GPUPlugin GPUEmissionPlugin

func init() {
	plugin, err := NewGPUEmissionPlugin()
	if err != nil {
		panic(fmt.Sprintf("Failed to initialize GPU emission plugin: %v", err))
	}
	GPUPlugin = *plugin
}