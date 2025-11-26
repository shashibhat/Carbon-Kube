package scheduler

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/go-redis/redis/v8"
	"go.uber.org/zap"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

const (
	// Plugin name
	Name = "CarbonAwareGPUScheduler"
	
	// Configuration keys
	ConfigMapName      = "gpu-carbon-config"
	ConfigMapNamespace = "kube-system"
	
	// Scoring weights
	CarbonWeight       = 40 // 40% weight for carbon efficiency
	PerformanceWeight  = 30 // 30% weight for performance
	UtilizationWeight  = 20 // 20% weight for resource utilization
	SLAWeight         = 10 // 10% weight for SLA compliance
	
	// Thresholds
	MaxCarbonIntensity = 800.0  // gCO2/kWh
	MinGPUUtilization  = 0.1    // 10% minimum utilization
	MaxGPUUtilization  = 0.9    // 90% maximum utilization
	SLALatencyThreshold = 100   // milliseconds
	
	// Migration parameters
	MigrationCooldown     = 10 * time.Minute
	CarbonSavingsThreshold = 0.2 // 20% carbon savings required for migration
	CheckpointOverhead    = 30   // seconds
)

// CarbonAwareGPUScheduler implements carbon-aware GPU scheduling
type CarbonAwareGPUScheduler struct {
	handle      framework.Handle
	kubeClient  kubernetes.Interface
	redisClient *redis.Client
	logger      *zap.Logger
	config      *SchedulerConfig
}

// SchedulerConfig holds configuration for the GPU scheduler
type SchedulerConfig struct {
	CarbonWeight       float64 `json:"carbonWeight"`
	PerformanceWeight  float64 `json:"performanceWeight"`
	UtilizationWeight  float64 `json:"utilizationWeight"`
	SLAWeight         float64 `json:"slaWeight"`
	EnableMigration   bool    `json:"enableMigration"`
	EnableMIG         bool    `json:"enableMIG"`
	EnableCheckpoint  bool    `json:"enableCheckpoint"`
	ForecastHorizon   int     `json:"forecastHorizon"` // hours
}

// GPUNodeInfo contains GPU-specific node information
type GPUNodeInfo struct {
	NodeName        string             `json:"nodeName"`
	Zone           string             `json:"zone"`
	GPUCount       int                `json:"gpuCount"`
	GPUProduct     string             `json:"gpuProduct"`
	GPUMemoryTotal int64              `json:"gpuMemoryTotal"`
	GPUMemoryFree  int64              `json:"gpuMemoryFree"`
	MIGEnabled     bool               `json:"migEnabled"`
	MIGProfiles    []string           `json:"migProfiles"`
	CarbonData     CarbonIntensityData `json:"carbonData"`
	Utilization    GPUUtilization     `json:"utilization"`
	SLAMetrics     SLAMetrics         `json:"slaMetrics"`
}

// CarbonIntensityData holds carbon-related metrics
type CarbonIntensityData struct {
	Intensity    float64   `json:"intensity"`    // gCO2/kWh
	Forecast     []float64 `json:"forecast"`     // 1-4 hour forecast
	PUE          float64   `json:"pue"`          // Power Usage Effectiveness
	LastUpdated  time.Time `json:"lastUpdated"`
	Source       string    `json:"source"`       // electricity-maps, noaa, etc.
}

// GPUUtilization holds GPU utilization metrics
type GPUUtilization struct {
	GPUUtil     float64 `json:"gpuUtil"`     // 0-100%
	MemoryUtil  float64 `json:"memoryUtil"`  // 0-100%
	PowerUsage  float64 `json:"powerUsage"`  // Watts
	Temperature float64 `json:"temperature"` // Celsius
}

// SLAMetrics holds SLA-related metrics
type SLAMetrics struct {
	AvgLatency    float64 `json:"avgLatency"`    // milliseconds
	P99Latency    float64 `json:"p99Latency"`    // milliseconds
	ErrorRate     float64 `json:"errorRate"`     // 0-1
	Availability  float64 `json:"availability"`  // 0-1
	LastViolation time.Time `json:"lastViolation"`
}

// GPUWorkloadRequirements defines GPU resource requirements
type GPUWorkloadRequirements struct {
	GPUCount       int     `json:"gpuCount"`
	GPUMemory      int64   `json:"gpuMemory"`      // MB
	GPUType        string  `json:"gpuType"`        // preferred GPU type
	MIGProfile     string  `json:"migProfile"`     // MIG profile if needed
	WorkloadType   string  `json:"workloadType"`   // training, inference, batch
	SLACritical    bool    `json:"slaCritical"`
	MaxLatency     float64 `json:"maxLatency"`     // milliseconds
	MinThroughput  float64 `json:"minThroughput"`  // requests/second
	Checkpointable bool    `json:"checkpointable"`
}

// New creates a new CarbonAwareGPUScheduler plugin
func New(obj runtime.Object, h framework.Handle) (framework.Plugin, error) {
	logger, err := zap.NewProduction()
	if err != nil {
		return nil, fmt.Errorf("failed to create logger: %v", err)
	}

	kubeClient, err := kubernetes.NewForConfig(h.ClientSet().RESTConfig())
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes client: %v", err)
	}

	// Initialize Redis client
	redisClient := redis.NewClient(&redis.Options{
		Addr:     "redis-service:6379",
		Password: "",
		DB:       3, // Use DB 3 for scheduler data
	})

	// Test Redis connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := redisClient.Ping(ctx).Err(); err != nil {
		logger.Warn("Redis connection failed, some features will be disabled", zap.Error(err))
		redisClient = nil
	}

	// Load configuration
	config, err := loadSchedulerConfig(kubeClient, logger)
	if err != nil {
		logger.Warn("Failed to load scheduler config, using defaults", zap.Error(err))
		config = getDefaultConfig()
	}

	return &CarbonAwareGPUScheduler{
		handle:      h,
		kubeClient:  kubeClient,
		redisClient: redisClient,
		logger:      logger,
		config:      config,
	}, nil
}

// Name returns the plugin name
func (s *CarbonAwareGPUScheduler) Name() string {
	return Name
}

// Filter implements the Filter extension point
func (s *CarbonAwareGPUScheduler) Filter(ctx context.Context, state *framework.CycleState, 
	pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	
	s.logger.Debug("Filtering node for GPU workload",
		zap.String("pod", pod.Name),
		zap.String("node", nodeInfo.Node().Name))

	// Check if pod requires GPU resources
	gpuReqs := s.extractGPURequirements(pod)
	if gpuReqs.GPUCount == 0 {
		// Not a GPU workload, allow scheduling
		return nil
	}

	node := nodeInfo.Node()
	
	// Check basic GPU availability
	if !s.hasGPUResources(node) {
		return framework.NewStatus(framework.UnschedulableAndUnresolvable, 
			"Node does not have GPU resources")
	}

	// Get detailed GPU node information
	gpuNodeInfo, err := s.getGPUNodeInfo(ctx, node)
	if err != nil {
		s.logger.Error("Failed to get GPU node info", 
			zap.String("node", node.Name), zap.Error(err))
		return framework.NewStatus(framework.Error, 
			fmt.Sprintf("Failed to get GPU node info: %v", err))
	}

	// Check GPU resource availability
	if !s.hasAvailableGPUResources(gpuNodeInfo, gpuReqs) {
		return framework.NewStatus(framework.Unschedulable, 
			"Insufficient GPU resources available")
	}

	// Check MIG compatibility
	if gpuReqs.MIGProfile != "" && !s.supportsMIGProfile(gpuNodeInfo, gpuReqs.MIGProfile) {
		return framework.NewStatus(framework.UnschedulableAndUnresolvable, 
			fmt.Sprintf("Node does not support MIG profile: %s", gpuReqs.MIGProfile))
	}

	// Check carbon intensity constraints
	if s.exceedsCarbonThreshold(gpuNodeInfo.CarbonData) {
		if gpuReqs.SLACritical {
			s.logger.Warn("High carbon intensity but SLA critical workload",
				zap.String("node", node.Name),
				zap.Float64("intensity", gpuNodeInfo.CarbonData.Intensity))
		} else {
			return framework.NewStatus(framework.Unschedulable, 
				"Carbon intensity too high for non-critical workload")
		}
	}

	// Check SLA constraints
	if gpuReqs.SLACritical && !s.meetsSLARequirements(gpuNodeInfo.SLAMetrics, gpuReqs) {
		return framework.NewStatus(framework.Unschedulable, 
			"Node does not meet SLA requirements")
	}

	return nil
}

// Score implements the Score extension point
func (s *CarbonAwareGPUScheduler) Score(ctx context.Context, state *framework.CycleState, 
	pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	
	s.logger.Debug("Scoring node for GPU workload",
		zap.String("pod", pod.Name),
		zap.String("node", nodeName))

	// Get node information
	nodeInfo, err := s.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, 
			fmt.Sprintf("Failed to get node info: %v", err))
	}

	node := nodeInfo.Node()
	gpuReqs := s.extractGPURequirements(pod)
	
	// If not a GPU workload, return neutral score
	if gpuReqs.GPUCount == 0 {
		return 50, nil // Neutral score
	}

	// Get detailed GPU node information
	gpuNodeInfo, err := s.getGPUNodeInfo(ctx, node)
	if err != nil {
		s.logger.Error("Failed to get GPU node info for scoring", 
			zap.String("node", nodeName), zap.Error(err))
		return 0, framework.NewStatus(framework.Error, 
			fmt.Sprintf("Failed to get GPU node info: %v", err))
	}

	// Calculate component scores
	carbonScore := s.calculateCarbonScore(gpuNodeInfo.CarbonData, gpuReqs)
	performanceScore := s.calculatePerformanceScore(gpuNodeInfo, gpuReqs)
	utilizationScore := s.calculateUtilizationScore(gpuNodeInfo, gpuReqs)
	slaScore := s.calculateSLAScore(gpuNodeInfo.SLAMetrics, gpuReqs)

	// Calculate weighted final score
	finalScore := (carbonScore*s.config.CarbonWeight +
		performanceScore*s.config.PerformanceWeight +
		utilizationScore*s.config.UtilizationWeight +
		slaScore*s.config.SLAWeight) / 100.0

	// Apply migration bonus if applicable
	if s.config.EnableMigration {
		migrationBonus := s.calculateMigrationBonus(ctx, nodeName, gpuReqs)
		finalScore += migrationBonus
	}

	// Ensure score is in valid range [0, 100]
	finalScore = math.Max(0, math.Min(100, finalScore))

	s.logger.Debug("GPU node scoring completed",
		zap.String("node", nodeName),
		zap.Float64("carbonScore", carbonScore),
		zap.Float64("performanceScore", performanceScore),
		zap.Float64("utilizationScore", utilizationScore),
		zap.Float64("slaScore", slaScore),
		zap.Float64("finalScore", finalScore))

	return int64(finalScore), nil
}

// ScoreExtensions returns score extensions
func (s *CarbonAwareGPUScheduler) ScoreExtensions() framework.ScoreExtensions {
	return s
}

// NormalizeScore normalizes scores across all nodes
func (s *CarbonAwareGPUScheduler) NormalizeScore(ctx context.Context, state *framework.CycleState, 
	pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	
	if len(scores) == 0 {
		return nil
	}

	// Find min and max scores
	var minScore, maxScore int64 = scores[0].Score, scores[0].Score
	for _, score := range scores {
		if score.Score < minScore {
			minScore = score.Score
		}
		if score.Score > maxScore {
			maxScore = score.Score
		}
	}

	// Normalize scores to [0, 100] range
	scoreRange := maxScore - minScore
	if scoreRange == 0 {
		// All scores are the same, set them all to 50
		for i := range scores {
			scores[i].Score = 50
		}
	} else {
		for i := range scores {
			normalizedScore := (scores[i].Score - minScore) * 100 / scoreRange
			scores[i].Score = normalizedScore
		}
	}

	return nil
}

// extractGPURequirements extracts GPU requirements from pod spec
func (s *CarbonAwareGPUScheduler) extractGPURequirements(pod *v1.Pod) GPUWorkloadRequirements {
	reqs := GPUWorkloadRequirements{}

	// Extract from resource requests
	for _, container := range pod.Spec.Containers {
		if requests := container.Resources.Requests; requests != nil {
			if gpuQuantity, exists := requests["nvidia.com/gpu"]; exists {
				if count, err := strconv.Atoi(gpuQuantity.String()); err == nil {
					reqs.GPUCount += count
				}
			}

			// Check for MIG resources
			for resource, quantity := range requests {
				if strings.Contains(string(resource), "nvidia.com/mig-") {
					reqs.MIGProfile = strings.TrimPrefix(string(resource), "nvidia.com/mig-")
					if count, err := strconv.Atoi(quantity.String()); err == nil {
						reqs.GPUCount += count
					}
				}
			}
		}
	}

	// Extract from annotations
	if annotations := pod.Annotations; annotations != nil {
		if workloadType, exists := annotations["carbon-kube.io/gpu-workload-type"]; exists {
			reqs.WorkloadType = workloadType
		}
		if slaCritical, exists := annotations["carbon-kube.io/sla-critical"]; exists {
			reqs.SLACritical = slaCritical == "true"
		}
		if maxLatency, exists := annotations["carbon-kube.io/max-latency-ms"]; exists {
			if latency, err := strconv.ParseFloat(maxLatency, 64); err == nil {
				reqs.MaxLatency = latency
			}
		}
		if checkpointable, exists := annotations["carbon-kube.io/checkpointable"]; exists {
			reqs.Checkpointable = checkpointable == "true"
		}
		if gpuType, exists := annotations["carbon-kube.io/gpu-type"]; exists {
			reqs.GPUType = gpuType
		}
	}

	// Set defaults
	if reqs.WorkloadType == "" {
		reqs.WorkloadType = "batch"
	}

	return reqs
}

// hasGPUResources checks if node has GPU resources
func (s *CarbonAwareGPUScheduler) hasGPUResources(node *v1.Node) bool {
	if capacity := node.Status.Capacity; capacity != nil {
		if _, hasGPU := capacity["nvidia.com/gpu"]; hasGPU {
			return true
		}
	}
	return false
}

// getGPUNodeInfo retrieves detailed GPU information for a node
func (s *CarbonAwareGPUScheduler) getGPUNodeInfo(ctx context.Context, node *v1.Node) (*GPUNodeInfo, error) {
	info := &GPUNodeInfo{
		NodeName: node.Name,
		Zone:     s.extractZone(node),
	}

	// Get GPU hardware info from node labels
	if labels := node.Labels; labels != nil {
		if product, exists := labels["nvidia.com/gpu.product"]; exists {
			info.GPUProduct = product
		}
		if count, exists := labels["nvidia.com/gpu.count"]; exists {
			if gpuCount, err := strconv.Atoi(count); err == nil {
				info.GPUCount = gpuCount
			}
		}
		if migConfig, exists := labels["nvidia.com/mig.config"]; exists {
			info.MIGEnabled = migConfig != "all-disabled"
		}
	}

	// Get carbon intensity data
	carbonData, err := s.getCarbonIntensityData(ctx, info.Zone)
	if err != nil {
		s.logger.Warn("Failed to get carbon intensity data", 
			zap.String("zone", info.Zone), zap.Error(err))
		// Use default values
		carbonData = CarbonIntensityData{
			Intensity: 400.0, // Default intensity
			PUE:       1.4,   // Default PUE
		}
	}
	info.CarbonData = carbonData

	// Get GPU utilization metrics from DCGM/Prometheus
	utilization, err := s.getGPUUtilization(ctx, node.Name)
	if err != nil {
		s.logger.Warn("Failed to get GPU utilization", 
			zap.String("node", node.Name), zap.Error(err))
		utilization = GPUUtilization{} // Default empty utilization
	}
	info.Utilization = utilization

	// Get SLA metrics
	slaMetrics, err := s.getSLAMetrics(ctx, node.Name)
	if err != nil {
		s.logger.Warn("Failed to get SLA metrics", 
			zap.String("node", node.Name), zap.Error(err))
		slaMetrics = SLAMetrics{
			Availability: 1.0, // Default high availability
		}
	}
	info.SLAMetrics = slaMetrics

	return info, nil
}

// calculateCarbonScore calculates carbon efficiency score (0-100, higher is better)
func (s *CarbonAwareGPUScheduler) calculateCarbonScore(carbonData CarbonIntensityData, 
	reqs GPUWorkloadRequirements) float64 {
	
	// Base score inversely proportional to carbon intensity
	baseScore := math.Max(0, 100-carbonData.Intensity/MaxCarbonIntensity*100)
	
	// Apply PUE penalty
	pueScore := math.Max(0, 100-(carbonData.PUE-1.0)*50) // Penalty for PUE > 1.0
	
	// Forecast bonus: prefer nodes with decreasing carbon intensity
	forecastBonus := 0.0
	if len(carbonData.Forecast) > 0 {
		futureIntensity := carbonData.Forecast[0]
		if futureIntensity < carbonData.Intensity {
			// Carbon intensity is decreasing, give bonus
			improvement := (carbonData.Intensity - futureIntensity) / carbonData.Intensity
			forecastBonus = improvement * 20 // Up to 20 point bonus
		}
	}
	
	// SLA critical workloads get reduced carbon weight
	if reqs.SLACritical {
		baseScore = baseScore*0.7 + 30 // Reduce carbon importance for critical workloads
	}
	
	return math.Min(100, baseScore*0.7+pueScore*0.3+forecastBonus)
}

// calculatePerformanceScore calculates performance score based on GPU capabilities
func (s *CarbonAwareGPUScheduler) calculatePerformanceScore(info *GPUNodeInfo, 
	reqs GPUWorkloadRequirements) float64 {
	
	score := 50.0 // Base score
	
	// GPU type matching bonus
	if reqs.GPUType != "" && strings.Contains(info.GPUProduct, reqs.GPUType) {
		score += 20
	}
	
	// GPU count adequacy
	if info.GPUCount >= reqs.GPUCount {
		if info.GPUCount == reqs.GPUCount {
			score += 15 // Perfect match
		} else {
			score += 10 // Adequate but not perfect
		}
	} else {
		score -= 30 // Insufficient GPUs
	}
	
	// MIG support bonus
	if reqs.MIGProfile != "" && info.MIGEnabled {
		score += 10
	}
	
	// Workload type optimization
	switch reqs.WorkloadType {
	case "training":
		// Prefer newer, high-memory GPUs for training
		if strings.Contains(info.GPUProduct, "A100") || strings.Contains(info.GPUProduct, "H100") {
			score += 15
		}
	case "inference":
		// Prefer efficient GPUs for inference
		if strings.Contains(info.GPUProduct, "T4") || info.MIGEnabled {
			score += 15
		}
	}
	
	return math.Max(0, math.Min(100, score))
}

// calculateUtilizationScore calculates resource utilization score
func (s *CarbonAwareGPUScheduler) calculateUtilizationScore(info *GPUNodeInfo, 
	reqs GPUWorkloadRequirements) float64 {
	
	// Prefer nodes with moderate utilization (not too high, not too low)
	targetUtilization := 0.7 // 70% target utilization
	
	currentUtil := info.Utilization.GPUUtil / 100.0
	utilizationDiff := math.Abs(currentUtil - targetUtilization)
	
	// Score decreases as we move away from target utilization
	utilizationScore := math.Max(0, 100-utilizationDiff*200)
	
	// Memory utilization consideration
	memoryUtil := info.Utilization.MemoryUtil / 100.0
	memoryScore := math.Max(0, 100-memoryUtil*100) // Prefer nodes with available memory
	
	// Temperature penalty
	tempScore := 100.0
	if info.Utilization.Temperature > 80 {
		tempScore = math.Max(0, 100-(info.Utilization.Temperature-80)*5)
	}
	
	// Power efficiency bonus
	powerEfficiency := info.Utilization.GPUUtil / info.Utilization.PowerUsage
	powerScore := math.Min(100, powerEfficiency*50) // Normalize power efficiency
	
	return (utilizationScore*0.4 + memoryScore*0.3 + tempScore*0.2 + powerScore*0.1)
}

// calculateSLAScore calculates SLA compliance score
func (s *CarbonAwareGPUScheduler) calculateSLAScore(slaMetrics SLAMetrics, 
	reqs GPUWorkloadRequirements) float64 {
	
	score := 100.0
	
	// Latency score
	if reqs.MaxLatency > 0 {
		if slaMetrics.AvgLatency > reqs.MaxLatency {
			latencyPenalty := (slaMetrics.AvgLatency - reqs.MaxLatency) / reqs.MaxLatency * 50
			score -= latencyPenalty
		}
		
		if slaMetrics.P99Latency > reqs.MaxLatency*2 {
			score -= 20 // Additional penalty for high P99 latency
		}
	}
	
	// Error rate penalty
	if slaMetrics.ErrorRate > 0.01 { // 1% error rate threshold
		score -= slaMetrics.ErrorRate * 1000 // Heavy penalty for errors
	}
	
	// Availability bonus/penalty
	if slaMetrics.Availability < 0.99 { // 99% availability threshold
		availabilityPenalty := (0.99 - slaMetrics.Availability) * 500
		score -= availabilityPenalty
	}
	
	// Recent violation penalty
	if time.Since(slaMetrics.LastViolation) < time.Hour {
		score -= 30 // Penalty for recent SLA violations
	}
	
	return math.Max(0, math.Min(100, score))
}

// calculateMigrationBonus calculates bonus for migration-friendly nodes
func (s *CarbonAwareGPUScheduler) calculateMigrationBonus(ctx context.Context, 
	nodeName string, reqs GPUWorkloadRequirements) float64 {