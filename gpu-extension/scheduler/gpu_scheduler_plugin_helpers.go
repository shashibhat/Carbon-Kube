package scheduler

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"go.uber.org/zap"
)

// Helper functions for the GPU scheduler plugin

// loadSchedulerConfig loads scheduler configuration from ConfigMap
func loadSchedulerConfig(kubeClient kubernetes.Interface, logger *zap.Logger) (*SchedulerConfig, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	configMap, err := kubeClient.CoreV1().ConfigMaps(ConfigMapNamespace).Get(ctx, ConfigMapName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get config map: %v", err)
	}

	configData, exists := configMap.Data["scheduler-config.json"]
	if !exists {
		return nil, fmt.Errorf("scheduler-config.json not found in config map")
	}

	var config SchedulerConfig
	if err := json.Unmarshal([]byte(configData), &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %v", err)
	}

	logger.Info("Loaded scheduler configuration",
		zap.Float64("carbonWeight", config.CarbonWeight),
		zap.Float64("performanceWeight", config.PerformanceWeight),
		zap.Bool("enableMigration", config.EnableMigration),
		zap.Bool("enableMIG", config.EnableMIG))

	return &config, nil
}

// getDefaultConfig returns default scheduler configuration
func getDefaultConfig() *SchedulerConfig {
	return &SchedulerConfig{
		CarbonWeight:      CarbonWeight,
		PerformanceWeight: PerformanceWeight,
		UtilizationWeight: UtilizationWeight,
		SLAWeight:        SLAWeight,
		EnableMigration:  true,
		EnableMIG:        true,
		EnableCheckpoint: true,
		ForecastHorizon:  4, // 4 hours
	}
}

// extractZone extracts zone information from node labels
func (s *CarbonAwareGPUScheduler) extractZone(node *v1.Node) string {
	if labels := node.Labels; labels != nil {
		// Try standard zone labels
		if zone, exists := labels["topology.kubernetes.io/zone"]; exists {
			return zone
		}
		if zone, exists := labels["failure-domain.beta.kubernetes.io/zone"]; exists {
			return zone
		}
		// Try cloud provider specific labels
		if zone, exists := labels["cloud.google.com/gce-zone"]; exists {
			return zone
		}
		if zone, exists := labels["topology.ebs.csi.aws.com/zone"]; exists {
			return zone
		}
	}
	return "unknown"
}

// hasAvailableGPUResources checks if node has sufficient GPU resources
func (s *CarbonAwareGPUScheduler) hasAvailableGPUResources(info *GPUNodeInfo, reqs GPUWorkloadRequirements) bool {
	// Check basic GPU count
	if info.GPUCount < reqs.GPUCount {
		return false
	}

	// Check GPU memory if specified
	if reqs.GPUMemory > 0 && info.GPUMemoryFree < reqs.GPUMemory {
		return false
	}

	// Check MIG profile availability
	if reqs.MIGProfile != "" {
		return s.supportsMIGProfile(info, reqs.MIGProfile)
	}

	return true
}

// supportsMIGProfile checks if node supports the requested MIG profile
func (s *CarbonAwareGPUScheduler) supportsMIGProfile(info *GPUNodeInfo, profile string) bool {
	if !info.MIGEnabled {
		return false
	}

	for _, supportedProfile := range info.MIGProfiles {
		if supportedProfile == profile {
			return true
		}
	}
	return false
}

// exceedsCarbonThreshold checks if carbon intensity exceeds threshold
func (s *CarbonAwareGPUScheduler) exceedsCarbonThreshold(carbonData CarbonIntensityData) bool {
	return carbonData.Intensity > MaxCarbonIntensity
}

// meetsSLARequirements checks if node meets SLA requirements
func (s *CarbonAwareGPUScheduler) meetsSLARequirements(slaMetrics SLAMetrics, reqs GPUWorkloadRequirements) bool {
	// Check latency requirements
	if reqs.MaxLatency > 0 && slaMetrics.AvgLatency > reqs.MaxLatency {
		return false
	}

	// Check availability requirements
	if slaMetrics.Availability < 0.99 { // 99% availability requirement
		return false
	}

	// Check error rate
	if slaMetrics.ErrorRate > 0.01 { // 1% error rate threshold
		return false
	}

	return true
}

// getCarbonIntensityData retrieves carbon intensity data for a zone
func (s *CarbonAwareGPUScheduler) getCarbonIntensityData(ctx context.Context, zone string) (CarbonIntensityData, error) {
	var carbonData CarbonIntensityData

	// Try to get from Redis cache first
	if s.redisClient != nil {
		cacheKey := fmt.Sprintf("carbon:intensity:%s", zone)
		cached, err := s.redisClient.Get(ctx, cacheKey).Result()
		if err == nil {
			if err := json.Unmarshal([]byte(cached), &carbonData); err == nil {
				// Check if data is still fresh (within 15 minutes)
				if time.Since(carbonData.LastUpdated) < 15*time.Minute {
					return carbonData, nil
				}
			}
		}
	}

	// Fallback to ConfigMap
	configMap, err := s.kubeClient.CoreV1().ConfigMaps(ConfigMapNamespace).Get(ctx, "carbon-intensity-data", metav1.GetOptions{})
	if err != nil {
		return carbonData, fmt.Errorf("failed to get carbon intensity config map: %v", err)
	}

	zoneData, exists := configMap.Data[zone]
	if !exists {
		// Use default zone data
		zoneData = configMap.Data["default"]
		if zoneData == "" {
			return carbonData, fmt.Errorf("no carbon intensity data for zone %s", zone)
		}
	}

	if err := json.Unmarshal([]byte(zoneData), &carbonData); err != nil {
		return carbonData, fmt.Errorf("failed to unmarshal carbon data: %v", err)
	}

	// Cache the data in Redis
	if s.redisClient != nil {
		carbonData.LastUpdated = time.Now()
		if data, err := json.Marshal(carbonData); err == nil {
			s.redisClient.Set(ctx, fmt.Sprintf("carbon:intensity:%s", zone), data, 15*time.Minute)
		}
	}

	return carbonData, nil
}

// getGPUUtilization retrieves GPU utilization metrics from DCGM/Prometheus
func (s *CarbonAwareGPUScheduler) getGPUUtilization(ctx context.Context, nodeName string) (GPUUtilization, error) {
	var utilization GPUUtilization

	// Try to get from Redis cache first
	if s.redisClient != nil {
		cacheKey := fmt.Sprintf("gpu:utilization:%s", nodeName)
		cached, err := s.redisClient.Get(ctx, cacheKey).Result()
		if err == nil {
			if err := json.Unmarshal([]byte(cached), &utilization); err == nil {
				return utilization, nil
			}
		}
	}

	// In a real implementation, this would query Prometheus/DCGM
	// For now, we'll simulate with some default values
	utilization = GPUUtilization{
		GPUUtil:     50.0, // 50% utilization
		MemoryUtil:  60.0, // 60% memory utilization
		PowerUsage:  200.0, // 200W power usage
		Temperature: 65.0,  // 65°C
	}

	// Cache the data
	if s.redisClient != nil {
		if data, err := json.Marshal(utilization); err == nil {
			s.redisClient.Set(ctx, fmt.Sprintf("gpu:utilization:%s", nodeName), data, 1*time.Minute)
		}
	}

	return utilization, nil
}

// getSLAMetrics retrieves SLA metrics for a node
func (s *CarbonAwareGPUScheduler) getSLAMetrics(ctx context.Context, nodeName string) (SLAMetrics, error) {
	var slaMetrics SLAMetrics

	// Try to get from Redis cache first
	if s.redisClient != nil {
		cacheKey := fmt.Sprintf("sla:metrics:%s", nodeName)
		cached, err := s.redisClient.Get(ctx, cacheKey).Result()
		if err == nil {
			if err := json.Unmarshal([]byte(cached), &slaMetrics); err == nil {
				return slaMetrics, nil
			}
		}
	}

	// In a real implementation, this would query monitoring systems
	// For now, we'll simulate with some default values
	slaMetrics = SLAMetrics{
		AvgLatency:    50.0, // 50ms average latency
		P99Latency:    150.0, // 150ms P99 latency
		ErrorRate:     0.001, // 0.1% error rate
		Availability:  0.999, // 99.9% availability
		LastViolation: time.Now().Add(-2 * time.Hour), // Last violation 2 hours ago
	}

	// Cache the data
	if s.redisClient != nil {
		if data, err := json.Marshal(slaMetrics); err == nil {
			s.redisClient.Set(ctx, fmt.Sprintf("sla:metrics:%s", nodeName), data, 5*time.Minute)
		}
	}

	return slaMetrics, nil
}

// calculateMigrationBonus calculates bonus for migration-friendly nodes
func (s *CarbonAwareGPUScheduler) calculateMigrationBonus(ctx context.Context, nodeName string, reqs GPUWorkloadRequirements) float64 {
	if !reqs.Checkpointable {
		return 0.0 // No migration bonus for non-checkpointable workloads
	}

	// Check if there are workloads that could benefit from migration
	migrationKey := fmt.Sprintf("migration:candidates:%s", nodeName)
	if s.redisClient != nil {
		candidates, err := s.redisClient.SMembers(ctx, migrationKey).Result()
		if err == nil && len(candidates) > 0 {
			// There are migration candidates, give bonus for accepting new workloads
			return 10.0 // 10 point bonus
		}
	}

	// Check migration history
	historyKey := fmt.Sprintf("migration:history:%s", nodeName)
	if s.redisClient != nil {
		recentMigrations, err := s.redisClient.ZCount(ctx, historyKey, 
			fmt.Sprintf("%d", time.Now().Add(-MigrationCooldown).Unix()), "+inf").Result()
		if err == nil && recentMigrations > 0 {
			// Recent migrations, reduce bonus
			return -5.0 // 5 point penalty
		}
	}

	return 0.0
}

// Additional helper functions for workload management

// WorkloadProfile represents different types of GPU workloads
type WorkloadProfile struct {
	Name                string        `json:"name"`
	GPUUtilizationMin   float64       `json:"gpuUtilizationMin"`
	GPUUtilizationMax   float64       `json:"gpuUtilizationMax"`
	MemoryUtilizationMin float64      `json:"memoryUtilizationMin"`
	MemoryUtilizationMax float64      `json:"memoryUtilizationMax"`
	PowerConsumptionMin float64       `json:"powerConsumptionMin"`
	PowerConsumptionMax float64       `json:"powerConsumptionMax"`
	TypicalDuration     time.Duration `json:"typicalDuration"`
	Checkpointable      bool          `json:"checkpointable"`
	SLASensitive        bool          `json:"slaSensitive"`
}

// getWorkloadProfiles returns predefined workload profiles
func getWorkloadProfiles() map[string]WorkloadProfile {
	return map[string]WorkloadProfile{
		"training": {
			Name:                "training",
			GPUUtilizationMin:   80.0,
			GPUUtilizationMax:   100.0,
			MemoryUtilizationMin: 70.0,
			MemoryUtilizationMax: 95.0,
			PowerConsumptionMin: 200.0,
			PowerConsumptionMax: 400.0,
			TypicalDuration:     4 * time.Hour,
			Checkpointable:      true,
			SLASensitive:        false,
		},
		"inference": {
			Name:                "inference",
			GPUUtilizationMin:   20.0,
			GPUUtilizationMax:   60.0,
			MemoryUtilizationMin: 30.0,
			MemoryUtilizationMax: 70.0,
			PowerConsumptionMin: 50.0,
			PowerConsumptionMax: 200.0,
			TypicalDuration:     24 * time.Hour,
			Checkpointable:      false,
			SLASensitive:        true,
		},
		"batch": {
			Name:                "batch",
			GPUUtilizationMin:   60.0,
			GPUUtilizationMax:   90.0,
			MemoryUtilizationMin: 40.0,
			MemoryUtilizationMax: 80.0,
			PowerConsumptionMin: 150.0,
			PowerConsumptionMax: 300.0,
			TypicalDuration:     2 * time.Hour,
			Checkpointable:      true,
			SLASensitive:        false,
		},
		"interactive": {
			Name:                "interactive",
			GPUUtilizationMin:   10.0,
			GPUUtilizationMax:   40.0,
			MemoryUtilizationMin: 20.0,
			MemoryUtilizationMax: 50.0,
			PowerConsumptionMin: 30.0,
			PowerConsumptionMax: 150.0,
			TypicalDuration:     1 * time.Hour,
			Checkpointable:      false,
			SLASensitive:        true,
		},
	}
}

// predictWorkloadProfile predicts workload profile based on resource requirements and annotations
func (s *CarbonAwareGPUScheduler) predictWorkloadProfile(reqs GPUWorkloadRequirements) WorkloadProfile {
	profiles := getWorkloadProfiles()
	
	// If workload type is explicitly specified, use it
	if profile, exists := profiles[reqs.WorkloadType]; exists {
		return profile
	}
	
	// Predict based on characteristics
	if reqs.SLACritical {
		if reqs.MaxLatency > 0 && reqs.MaxLatency < 100 {
			return profiles["inference"]
		}
		return profiles["interactive"]
	}
	
	if reqs.Checkpointable {
		if reqs.GPUCount > 1 {
			return profiles["training"]
		}
		return profiles["batch"]
	}
	
	// Default to batch processing
	return profiles["batch"]
}

// MIGProfileInfo contains information about MIG profiles
type MIGProfileInfo struct {
	Name        string `json:"name"`
	GPUSlice    int    `json:"gpuSlice"`    // GPU compute slice (1g, 2g, 3g, 7g)
	MemorySlice int    `json:"memorySlice"` // Memory slice in GB (5gb, 10gb, 20gb, 40gb, 80gb)
	MaxInstances int   `json:"maxInstances"` // Maximum instances per GPU
	PowerLimit  int    `json:"powerLimit"`   // Power limit in watts
}

// getMIGProfiles returns available MIG profiles
func getMIGProfiles() map[string]MIGProfileInfo {
	return map[string]MIGProfileInfo{
		"1g.5gb": {
			Name:        "1g.5gb",
			GPUSlice:    1,
			MemorySlice: 5,
			MaxInstances: 7,
			PowerLimit:  50,
		},
		"1g.10gb": {
			Name:        "1g.10gb",
			GPUSlice:    1,
			MemorySlice: 10,
			MaxInstances: 4,
			PowerLimit:  60,
		},
		"2g.20gb": {
			Name:        "2g.20gb",
			GPUSlice:    2,
			MemorySlice: 20,
			MaxInstances: 3,
			PowerLimit:  120,
		},
		"3g.40gb": {
			Name:        "3g.40gb",
			GPUSlice:    3,
			MemorySlice: 40,
			MaxInstances: 2,
			PowerLimit:  180,
		},
		"7g.80gb": {
			Name:        "7g.80gb",
			GPUSlice:    7,
			MemorySlice: 80,
			MaxInstances: 1,
			PowerLimit:  350,
		},
	}
}

// recommendMIGProfile recommends optimal MIG profile for workload
func (s *CarbonAwareGPUScheduler) recommendMIGProfile(reqs GPUWorkloadRequirements, 
	workloadProfile WorkloadProfile) string {
	
	profiles := getMIGProfiles()
	
	// For inference workloads, prefer smaller profiles for better utilization
	if workloadProfile.Name == "inference" {
		if reqs.GPUMemory <= 5*1024 { // 5GB
			return "1g.5gb"
		} else if reqs.GPUMemory <= 10*1024 { // 10GB
			return "1g.10gb"
		}
		return "2g.20gb"
	}
	
	// For training workloads, prefer larger profiles
	if workloadProfile.Name == "training" {
		if reqs.GPUMemory > 40*1024 { // > 40GB
			return "7g.80gb"
		} else if reqs.GPUMemory > 20*1024 { // > 20GB
			return "3g.40gb"
		}
		return "2g.20gb"
	}
	
	// For batch workloads, balance between resource needs and efficiency
	if workloadProfile.Name == "batch" {
		if reqs.GPUMemory <= 10*1024 {
			return "1g.10gb"
		} else if reqs.GPUMemory <= 20*1024 {
			return "2g.20gb"
		}
		return "3g.40gb"
	}
	
	// Default recommendation
	return "2g.20gb"
}

// CarbonForecast represents carbon intensity forecast
type CarbonForecast struct {
	Timestamp time.Time `json:"timestamp"`
	Intensity float64   `json:"intensity"`
	Source    string    `json:"source"`
}

// getCarbonForecast retrieves carbon intensity forecast for a zone
func (s *CarbonAwareGPUScheduler) getCarbonForecast(ctx context.Context, zone string, 
	hours int) ([]CarbonForecast, error) {
	
	var forecast []CarbonForecast
	
	// Try to get from Redis cache first
	if s.redisClient != nil {
		cacheKey := fmt.Sprintf("carbon:forecast:%s:%d", zone, hours)
		cached, err := s.redisClient.Get(ctx, cacheKey).Result()
		if err == nil {
			if err := json.Unmarshal([]byte(cached), &forecast); err == nil {
				return forecast, nil
			}
		}
	}
	
	// In a real implementation, this would query external APIs
	// For now, simulate forecast data
	baseIntensity := 400.0
	for i := 0; i < hours; i++ {
		forecast = append(forecast, CarbonForecast{
			Timestamp: time.Now().Add(time.Duration(i) * time.Hour),
			Intensity: baseIntensity + float64(i*10), // Simulate increasing intensity
			Source:    "simulated",
		})
	}
	
	// Cache the forecast
	if s.redisClient != nil {
		if data, err := json.Marshal(forecast); err == nil {
			s.redisClient.Set(ctx, fmt.Sprintf("carbon:forecast:%s:%d", zone, hours), 
				data, 30*time.Minute)
		}
	}
	
	return forecast, nil
}