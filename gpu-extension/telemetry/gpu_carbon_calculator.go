package telemetry

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/go-redis/redis/v8"
	"go.uber.org/zap"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// GPUCarbonCalculator calculates carbon emissions for GPU workloads
type GPUCarbonCalculator struct {
	kubeClient  kubernetes.Interface
	redisClient *redis.Client
	logger      *zap.Logger
	config      *CarbonConfig
}

// CarbonConfig holds carbon calculation configuration
type CarbonConfig struct {
	DefaultPUE              float64           `json:"defaultPUE"`              // Power Usage Effectiveness
	CarbonIntensitySource   string            `json:"carbonIntensitySource"`   // electricity-maps, noaa, static
	UpdateInterval          time.Duration     `json:"updateInterval"`          // How often to update carbon data
	GPUPowerModels          map[string]GPUPowerModel `json:"gpuPowerModels"`
	DatacenterEfficiency    map[string]float64 `json:"datacenterEfficiency"`   // Zone-specific PUE values
	CarbonAccountingMethod  string            `json:"carbonAccountingMethod"`  // operational, embodied, lifecycle
	EmissionFactors         EmissionFactors   `json:"emissionFactors"`
}

// GPUPowerModel defines power consumption model for different GPU types
type GPUPowerModel struct {
	GPUType           string  `json:"gpuType"`           // e.g., "A100", "V100", "T4"
	TDPWatts          float64 `json:"tdpWatts"`          // Thermal Design Power
	IdlePowerWatts    float64 `json:"idlePowerWatts"`    // Idle power consumption
	MaxPowerWatts     float64 `json:"maxPowerWatts"`     // Maximum power consumption
	MemoryPowerWatts  float64 `json:"memoryPowerWatts"`  // Memory power per GB
	PowerEfficiency   float64 `json:"powerEfficiency"`   // Performance per watt
	CoolingOverhead   float64 `json:"coolingOverhead"`   // Additional cooling power ratio
}

// EmissionFactors holds various emission factors for carbon calculations
type EmissionFactors struct {
	ManufacturingGCO2   float64 `json:"manufacturingGCO2"`   // gCO2 for manufacturing per GPU
	TransportGCO2       float64 `json:"transportGCO2"`       // gCO2 for transport per GPU
	EndOfLifeGCO2       float64 `json:"endOfLifeGCO2"`       // gCO2 for end-of-life per GPU
	LifespanYears       float64 `json:"lifespanYears"`       // Expected lifespan in years
	RecyclingEfficiency float64 `json:"recyclingEfficiency"` // Recycling efficiency (0-1)
}

// GPUCarbonMetrics holds carbon-related metrics for a GPU workload
type GPUCarbonMetrics struct {
	WorkloadID          string    `json:"workloadId"`
	NodeName            string    `json:"nodeName"`
	Zone                string    `json:"zone"`
	Timestamp           time.Time `json:"timestamp"`
	
	// Power metrics
	GPUPowerWatts       float64   `json:"gpuPowerWatts"`       // Current GPU power consumption
	SystemPowerWatts    float64   `json:"systemPowerWatts"`    // Total system power
	CoolingPowerWatts   float64   `json:"coolingPowerWatts"`   // Cooling power overhead
	TotalPowerWatts     float64   `json:"totalPowerWatts"`     // Total power including PUE
	
	// Carbon metrics
	CarbonIntensity     float64   `json:"carbonIntensity"`     // gCO2/kWh from grid
	OperationalGCO2     float64   `json:"operationalGCO2"`     // Operational emissions gCO2/hour
	EmbodiedGCO2        float64   `json:"embodiedGCO2"`        // Embodied emissions gCO2/hour
	TotalGCO2           float64   `json:"totalGCO2"`           // Total emissions gCO2/hour
	
	// Efficiency metrics
	PUE                 float64   `json:"pue"`                 // Power Usage Effectiveness
	CarbonEfficiency    float64   `json:"carbonEfficiency"`    // gCO2 per FLOP
	PowerEfficiency     float64   `json:"powerEfficiency"`     // FLOPS per watt
	
	// Utilization metrics
	GPUUtilization      float64   `json:"gpuUtilization"`      // GPU utilization %
	MemoryUtilization   float64   `json:"memoryUtilization"`   // Memory utilization %
	ComputeUtilization  float64   `json:"computeUtilization"`  // Compute utilization %
	
	// Forecast data
	CarbonForecast      []float64 `json:"carbonForecast"`      // 1-4 hour carbon intensity forecast
	OptimalWindow       *TimeWindow `json:"optimalWindow"`     // Optimal execution window
}

// TimeWindow represents an optimal time window for workload execution
type TimeWindow struct {
	StartTime       time.Time `json:"startTime"`
	EndTime         time.Time `json:"endTime"`
	AvgIntensity    float64   `json:"avgIntensity"`    // Average carbon intensity in window
	CarbonSavings   float64   `json:"carbonSavings"`   // Potential carbon savings %
	Confidence      float64   `json:"confidence"`      // Forecast confidence (0-1)
}

// GPUWorkloadProfile defines characteristics of different GPU workload types
type GPUWorkloadProfile struct {
	WorkloadType        string        `json:"workloadType"`        // training, inference, batch, interactive
	TypicalDuration     time.Duration `json:"typicalDuration"`     // Typical execution duration
	PowerProfile        string        `json:"powerProfile"`        // constant, variable, bursty
	CarbonSensitivity   float64       `json:"carbonSensitivity"`   // How sensitive to carbon (0-1)
	DelayTolerance      time.Duration `json:"delayTolerance"`      // How long can be delayed
	Checkpointable      bool          `json:"checkpointable"`      // Can be checkpointed/migrated
	SLACritical         bool          `json:"slaCritical"`         // Has strict SLA requirements
	ResourceIntensity   string        `json:"resourceIntensity"`   // low, medium, high
}

// NewGPUCarbonCalculator creates a new GPU carbon calculator
func NewGPUCarbonCalculator(kubeClient kubernetes.Interface, redisClient *redis.Client, 
	logger *zap.Logger) (*GPUCarbonCalculator, error) {
	
	config := &CarbonConfig{
		DefaultPUE:             1.4,
		CarbonIntensitySource:  "electricity-maps",
		UpdateInterval:         5 * time.Minute,
		CarbonAccountingMethod: "lifecycle", // Include operational + embodied
		GPUPowerModels:         getDefaultGPUPowerModels(),
		DatacenterEfficiency:   getDefaultDatacenterEfficiency(),
		EmissionFactors:        getDefaultEmissionFactors(),
	}

	return &GPUCarbonCalculator{
		kubeClient:  kubeClient,
		redisClient: redisClient,
		logger:      logger,
		config:      config,
	}, nil
}

// CalculateGPUCarbon calculates carbon emissions for a GPU workload
func (gcc *GPUCarbonCalculator) CalculateGPUCarbon(ctx context.Context, 
	workloadID, nodeName string, gpuMetrics map[string]interface{}) (*GPUCarbonMetrics, error) {
	
	gcc.logger.Debug("Calculating GPU carbon emissions",
		zap.String("workloadId", workloadID),
		zap.String("node", nodeName))

	metrics := &GPUCarbonMetrics{
		WorkloadID: workloadID,
		NodeName:   nodeName,
		Timestamp:  time.Now(),
	}

	// Get node information
	node, err := gcc.kubeClient.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get node info: %v", err)
	}
	
	metrics.Zone = gcc.extractZone(node)

	// Extract GPU metrics from DCGM data
	if err := gcc.extractGPUMetrics(gpuMetrics, metrics); err != nil {
		return nil, fmt.Errorf("failed to extract GPU metrics: %v", err)
	}

	// Get carbon intensity data
	carbonIntensity, err := gcc.getCarbonIntensity(ctx, metrics.Zone)
	if err != nil {
		gcc.logger.Warn("Failed to get carbon intensity, using default", zap.Error(err))
		carbonIntensity = 400.0 // Default carbon intensity
	}
	metrics.CarbonIntensity = carbonIntensity

	// Get PUE for the zone/datacenter
	pue := gcc.getPUE(metrics.Zone)
	metrics.PUE = pue

	// Calculate power consumption
	if err := gcc.calculatePowerConsumption(ctx, node, metrics); err != nil {
		return nil, fmt.Errorf("failed to calculate power consumption: %v", err)
	}

	// Calculate carbon emissions
	gcc.calculateCarbonEmissions(metrics)

	// Calculate efficiency metrics
	gcc.calculateEfficiencyMetrics(metrics)

	// Get carbon forecast and optimal window
	if err := gcc.calculateOptimalWindow(ctx, metrics); err != nil {
		gcc.logger.Warn("Failed to calculate optimal window", zap.Error(err))
	}

	// Store metrics in Redis for caching
	if err := gcc.storeMetrics(ctx, metrics); err != nil {
		gcc.logger.Warn("Failed to store carbon metrics", zap.Error(err))
	}

	return metrics, nil
}

// extractGPUMetrics extracts GPU utilization metrics from DCGM data
func (gcc *GPUCarbonCalculator) extractGPUMetrics(gpuMetrics map[string]interface{}, 
	metrics *GPUCarbonMetrics) error {
	
	// Extract GPU utilization
	if util, ok := gpuMetrics["gpu_utilization"].(float64); ok {
		metrics.GPUUtilization = util
	}

	// Extract memory utilization
	if memUtil, ok := gpuMetrics["memory_utilization"].(float64); ok {
		metrics.MemoryUtilization = memUtil
	}

	// Extract power consumption
	if power, ok := gpuMetrics["power_draw"].(float64); ok {
		metrics.GPUPowerWatts = power
	}

	// Extract compute utilization (if available)
	if compUtil, ok := gpuMetrics["compute_utilization"].(float64); ok {
		metrics.ComputeUtilization = compUtil
	} else {
		// Estimate compute utilization from GPU utilization
		metrics.ComputeUtilization = metrics.GPUUtilization * 0.8
	}

	return nil
}

// calculatePowerConsumption calculates total power consumption including PUE
func (gcc *GPUCarbonCalculator) calculatePowerConsumption(ctx context.Context, 
	node *v1.Node, metrics *GPUCarbonMetrics) error {
	
	// Get GPU type from node labels
	gpuType := gcc.getGPUType(node)
	powerModel, exists := gcc.config.GPUPowerModels[gpuType]
	if !exists {
		// Use default power model
		powerModel = gcc.config.GPUPowerModels["default"]
	}

	// If we don't have actual power measurement, estimate it
	if metrics.GPUPowerWatts == 0 {
		// Estimate power based on utilization
		utilizationFactor := metrics.GPUUtilization / 100.0
		estimatedPower := powerModel.IdlePowerWatts + 
			(powerModel.MaxPowerWatts-powerModel.IdlePowerWatts)*utilizationFactor
		metrics.GPUPowerWatts = estimatedPower
	}

	// Calculate cooling overhead
	metrics.CoolingPowerWatts = metrics.GPUPowerWatts * powerModel.CoolingOverhead

	// Calculate system power (GPU + CPU + memory + storage)
	systemOverhead := metrics.GPUPowerWatts * 0.3 // Estimate 30% overhead for other components
	metrics.SystemPowerWatts = metrics.GPUPowerWatts + systemOverhead

	// Apply PUE to get total power consumption
	metrics.TotalPowerWatts = metrics.SystemPowerWatts * metrics.PUE

	return nil
}

// calculateCarbonEmissions calculates operational and embodied carbon emissions
func (gcc *GPUCarbonCalculator) calculateCarbonEmissions(metrics *GPUCarbonMetrics) {
	// Operational emissions (gCO2/hour)
	powerKW := metrics.TotalPowerWatts / 1000.0
	metrics.OperationalGCO2 = powerKW * metrics.CarbonIntensity

	// Embodied emissions (gCO2/hour) - amortized over lifespan
	if gcc.config.CarbonAccountingMethod == "lifecycle" || gcc.config.CarbonAccountingMethod == "embodied" {
		totalEmbodiedGCO2 := gcc.config.EmissionFactors.ManufacturingGCO2 + 
			gcc.config.EmissionFactors.TransportGCO2 + 
			gcc.config.EmissionFactors.EndOfLifeGCO2

		// Amortize over lifespan (assuming 24/7 usage)
		hoursInLifespan := gcc.config.EmissionFactors.LifespanYears * 365 * 24
		metrics.EmbodiedGCO2 = totalEmbodiedGCO2 / hoursInLifespan
	}

	// Total emissions
	if gcc.config.CarbonAccountingMethod == "operational" {
		metrics.TotalGCO2 = metrics.OperationalGCO2
	} else {
		metrics.TotalGCO2 = metrics.OperationalGCO2 + metrics.EmbodiedGCO2
	}
}

// calculateEfficiencyMetrics calculates various efficiency metrics
func (gcc *GPUCarbonCalculator) calculateEfficiencyMetrics(metrics *GPUCarbonMetrics) {
	// Power efficiency (FLOPS per watt) - estimated based on GPU utilization
	if metrics.GPUPowerWatts > 0 {
		// This is a simplified calculation - in practice, you'd measure actual FLOPS
		estimatedFLOPS := metrics.GPUUtilization * 1e12 // Assume 1 TFLOPS at 100% utilization
		metrics.PowerEfficiency = estimatedFLOPS / metrics.GPUPowerWatts
	}

	// Carbon efficiency (gCO2 per FLOP)
	if metrics.PowerEfficiency > 0 {
		metrics.CarbonEfficiency = metrics.TotalGCO2 / (metrics.PowerEfficiency * metrics.GPUPowerWatts)
	}
}

// calculateOptimalWindow calculates optimal execution window based on carbon forecast
func (gcc *GPUCarbonCalculator) calculateOptimalWindow(ctx context.Context, 
	metrics *GPUCarbonMetrics) error {
	
	// Get carbon intensity forecast
	forecast, err := gcc.getCarbonForecast(ctx, metrics.Zone, 4) // 4-hour forecast
	if err != nil {
		return fmt.Errorf("failed to get carbon forecast: %v", err)
	}

	metrics.CarbonForecast = forecast

	// Find optimal window (lowest average carbon intensity)
	if len(forecast) >= 2 {
		minAvgIntensity := math.Inf(1)
		var optimalWindow *TimeWindow

		// Check different window sizes (1-4 hours)
		for windowSize := 1; windowSize <= len(forecast); windowSize++ {
			for start := 0; start <= len(forecast)-windowSize; start++ {
				// Calculate average intensity for this window
				sum := 0.0
				for i := start; i < start+windowSize; i++ {
					sum += forecast[i]
				}
				avgIntensity := sum / float64(windowSize)

				if avgIntensity < minAvgIntensity {
					minAvgIntensity = avgIntensity
					startTime := time.Now().Add(time.Duration(start) * time.Hour)
					endTime := startTime.Add(time.Duration(windowSize) * time.Hour)
					
					carbonSavings := (metrics.CarbonIntensity - avgIntensity) / metrics.CarbonIntensity * 100
					
					optimalWindow = &TimeWindow{
						StartTime:     startTime,
						EndTime:       endTime,
						AvgIntensity:  avgIntensity,
						CarbonSavings: math.Max(0, carbonSavings),
						Confidence:    0.8, // Simplified confidence calculation
					}
				}
			}
		}

		metrics.OptimalWindow = optimalWindow
	}

	return nil
}

// GetWorkloadProfile returns the workload profile for a given workload type
func (gcc *GPUCarbonCalculator) GetWorkloadProfile(workloadType string) GPUWorkloadProfile {
	profiles := map[string]GPUWorkloadProfile{
		"training": {
			WorkloadType:      "training",
			TypicalDuration:   4 * time.Hour,
			PowerProfile:      "constant",
			CarbonSensitivity: 0.8,
			DelayTolerance:    2 * time.Hour,
			Checkpointable:    true,
			SLACritical:       false,
			ResourceIntensity: "high",
		},
		"inference": {
			WorkloadType:      "inference",
			TypicalDuration:   24 * time.Hour,
			PowerProfile:      "variable",
			CarbonSensitivity: 0.6,
			DelayTolerance:    5 * time.Minute,
			Checkpointable:    false,
			SLACritical:       true,
			ResourceIntensity: "medium",
		},
		"batch": {
			WorkloadType:      "batch",
			TypicalDuration:   2 * time.Hour,
			PowerProfile:      "bursty",
			CarbonSensitivity: 0.9,
			DelayTolerance:    4 * time.Hour,
			Checkpointable:    true,
			SLACritical:       false,
			ResourceIntensity: "medium",
		},
		"interactive": {
			WorkloadType:      "interactive",
			TypicalDuration:   1 * time.Hour,
			PowerProfile:      "variable",
			CarbonSensitivity: 0.3,
			DelayTolerance:    30 * time.Second,
			Checkpointable:    false,
			SLACritical:       true,
			ResourceIntensity: "low",
		},
	}

	if profile, exists := profiles[workloadType]; exists {
		return profile
	}
	return profiles["batch"] // Default to batch
}

// Helper functions

func (gcc *GPUCarbonCalculator) extractZone(node *v1.Node) string {
	if labels := node.Labels; labels != nil {
		if zone, exists := labels["topology.kubernetes.io/zone"]; exists {
			return zone
		}
		if zone, exists := labels["failure-domain.beta.kubernetes.io/zone"]; exists {
			return zone
		}
	}
	return "unknown"
}

func (gcc *GPUCarbonCalculator) getGPUType(node *v1.Node) string {
	if labels := node.Labels; labels != nil {
		if gpuType, exists := labels["nvidia.com/gpu.product"]; exists {
			return gpuType
		}
	}
	return "default"
}

func (gcc *GPUCarbonCalculator) getCarbonIntensity(ctx context.Context, zone string) (float64, error) {
	// Try to get from Redis cache first
	if gcc.redisClient != nil {
		cacheKey := fmt.Sprintf("carbon:intensity:%s", zone)
		cached, err := gcc.redisClient.Get(ctx, cacheKey).Result()
		if err == nil {
			var intensity float64
			if err := json.Unmarshal([]byte(cached), &intensity); err == nil {
				return intensity, nil
			}
		}
	}

	// In a real implementation, this would query external APIs
	// For now, simulate zone-specific carbon intensity
	zoneIntensities := map[string]float64{
		"us-west1-a": 350.0,  // California - cleaner grid
		"us-west1-b": 360.0,
		"us-east1-a": 450.0,  // Virginia - dirtier grid
		"us-east1-b": 440.0,
		"europe-west1-a": 300.0, // Belgium - clean grid
		"asia-east1-a": 600.0,   // Taiwan - coal-heavy
		"unknown": 400.0,        // Default
	}

	intensity, exists := zoneIntensities[zone]
	if !exists {
		intensity = zoneIntensities["unknown"]
	}

	// Cache the result
	if gcc.redisClient != nil {
		data, _ := json.Marshal(intensity)
		gcc.redisClient.Set(ctx, fmt.Sprintf("carbon:intensity:%s", zone), data, 15*time.Minute)
	}

	return intensity, nil
}

func (gcc *GPUCarbonCalculator) getPUE(zone string) float64 {
	if pue, exists := gcc.config.DatacenterEfficiency[zone]; exists {
		return pue
	}
	return gcc.config.DefaultPUE
}

func (gcc *GPUCarbonCalculator) getCarbonForecast(ctx context.Context, zone string, 
	hours int) ([]float64, error) {
	
	// Try to get from Redis cache first
	if gcc.redisClient != nil {
		cacheKey := fmt.Sprintf("carbon:forecast:%s:%d", zone, hours)
		cached, err := gcc.redisClient.Get(ctx, cacheKey).Result()
		if err == nil {
			var forecast []float64
			if err := json.Unmarshal([]byte(cached), &forecast); err == nil {
				return forecast, nil
			}
		}
	}

	// Simulate forecast data (in practice, this would query external APIs)
	baseIntensity, _ := gcc.getCarbonIntensity(ctx, zone)
	forecast := make([]float64, hours)
	
	for i := 0; i < hours; i++ {
		// Simulate daily pattern (lower at night, higher during day)
		hour := (time.Now().Hour() + i) % 24
		dailyFactor := 0.8 + 0.4*math.Sin(float64(hour-6)*math.Pi/12) // Peak at 6 PM
		forecast[i] = baseIntensity * dailyFactor
	}

	// Cache the forecast
	if gcc.redisClient != nil {
		data, _ := json.Marshal(forecast)
		gcc.redisClient.Set(ctx, fmt.Sprintf("carbon:forecast:%s:%d", zone, hours), 
			data, 30*time.Minute)
	}

	return forecast, nil
}

func (gcc *GPUCarbonCalculator) storeMetrics(ctx context.Context, metrics *GPUCarbonMetrics) error {
	if gcc.redisClient == nil {
		return nil
	}

	key := fmt.Sprintf("gpu:carbon:metrics:%s:%d", metrics.WorkloadID, metrics.Timestamp.Unix())
	data, err := json.Marshal(metrics)
	if err != nil {
		return err
	}

	return gcc.redisClient.Set(ctx, key, data, 24*time.Hour).Err()
}

// Default configuration functions

func getDefaultGPUPowerModels() map[string]GPUPowerModel {
	return map[string]GPUPowerModel{
		"A100": {
			GPUType:          "A100",
			TDPWatts:         400.0,
			IdlePowerWatts:   50.0,
			MaxPowerWatts:    400.0,
			MemoryPowerWatts: 2.0,
			PowerEfficiency:  2.5e9, // FLOPS per watt
			CoolingOverhead:  0.3,   // 30% cooling overhead
		},
		"V100": {
			GPUType:          "V100",
			TDPWatts:         300.0,
			IdlePowerWatts:   40.0,
			MaxPowerWatts:    300.0,
			MemoryPowerWatts: 1.5,
			PowerEfficiency:  2.0e9,
			CoolingOverhead:  0.3,
		},
		"T4": {
			GPUType:          "T4",
			TDPWatts:         70.0,
			IdlePowerWatts:   15.0,
			MaxPowerWatts:    70.0,
			MemoryPowerWatts: 1.0,
			PowerEfficiency:  1.8e9,
			CoolingOverhead:  0.25,
		},
		"H100": {
			GPUType:          "H100",
			TDPWatts:         700.0,
			IdlePowerWatts:   80.0,
			MaxPowerWatts:    700.0,
			MemoryPowerWatts: 3.0,
			PowerEfficiency:  3.0e9,
			CoolingOverhead:  0.35,
		},
		"default": {
			GPUType:          "default",
			TDPWatts:         250.0,
			IdlePowerWatts:   30.0,
			MaxPowerWatts:    250.0,
			MemoryPowerWatts: 1.5,
			PowerEfficiency:  2.0e9,
			CoolingOverhead:  0.3,
		},
	}
}

func getDefaultDatacenterEfficiency() map[string]float64 {
	return map[string]float64{
		"us-west1-a":     1.2, // Google's efficient datacenter
		"us-west1-b":     1.2,
		"us-east1-a":     1.4, // Older datacenter
		"us-east1-b":     1.4,
		"europe-west1-a": 1.1, // Very efficient
		"asia-east1-a":   1.5, // Less efficient
		"unknown":        1.4, // Default
	}
}

func getDefaultEmissionFactors() EmissionFactors {
	return EmissionFactors{
		ManufacturingGCO2:   150000.0, // 150 kg CO2 for manufacturing
		TransportGCO2:       5000.0,   // 5 kg CO2 for transport
		EndOfLifeGCO2:       2000.0,   // 2 kg CO2 for end-of-life
		LifespanYears:       4.0,      // 4 years expected lifespan
		RecyclingEfficiency: 0.8,      // 80% recycling efficiency
	}
}