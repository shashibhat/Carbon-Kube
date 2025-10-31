package telemetry

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/prometheus/client_golang/api"
	v1 "github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/common/model"
	"go.uber.org/zap"
)

// DCGMCollector collects GPU metrics from DCGM Exporter
type DCGMCollector struct {
	prometheusClient v1.API
	redisClient      *redis.Client
	logger           *zap.Logger
	config           *DCGMConfig
}

// DCGMConfig holds DCGM collector configuration
type DCGMConfig struct {
	PrometheusURL      string        `json:"prometheusURL"`      // Prometheus server URL
	CollectionInterval time.Duration `json:"collectionInterval"` // How often to collect metrics
	MetricRetention    time.Duration `json:"metricRetention"`    // How long to retain metrics
	GPUMetrics         []string      `json:"gpuMetrics"`         // List of GPU metrics to collect
	CarbonMetrics      []string      `json:"carbonMetrics"`      // List of carbon-specific metrics
	AlertThresholds    map[string]float64 `json:"alertThresholds"` // Alert thresholds for metrics
}

// GPUMetrics represents comprehensive GPU telemetry data
type GPUMetrics struct {
	NodeName        string                 `json:"nodeName"`
	Zone            string                 `json:"zone"`
	Timestamp       time.Time              `json:"timestamp"`
	GPUs            map[string]*GPUDevice  `json:"gpus"`            // GPU UUID -> device data
	NodeMetrics     *NodeMetrics           `json:"nodeMetrics"`     // Node-level metrics
	CarbonMetrics   *CarbonMetrics         `json:"carbonMetrics"`   // Carbon-specific metrics
	WorkloadMetrics map[string]*WorkloadMetrics `json:"workloadMetrics"` // Pod -> workload metrics
}

// GPUDevice represents individual GPU device metrics
type GPUDevice struct {
	UUID                string             `json:"uuid"`
	Name                string             `json:"name"`
	Index               int                `json:"index"`
	PowerDraw           float64            `json:"powerDraw"`           // Watts
	PowerLimit          float64            `json:"powerLimit"`          // Watts
	Temperature         float64            `json:"temperature"`         // Celsius
	MemoryUsed          uint64             `json:"memoryUsed"`          // Bytes
	MemoryTotal         uint64             `json:"memoryTotal"`         // Bytes
	UtilizationGPU      float64            `json:"utilizationGPU"`      // Percentage
	UtilizationMemory   float64            `json:"utilizationMemory"`   // Percentage
	ClockSM             float64            `json:"clockSM"`             // MHz
	ClockMemory         float64            `json:"clockMemory"`         // MHz
	ClockVideo          float64            `json:"clockVideo"`          // MHz
	FanSpeed            float64            `json:"fanSpeed"`            // Percentage
	PerformanceState    string             `json:"performanceState"`    // P0-P12
	ThrottleReasons     []string           `json:"throttleReasons"`     // Throttling reasons
	PCIeGeneration      int                `json:"pcieGeneration"`      // PCIe generation
	PCIeWidth           int                `json:"pcieWidth"`           // PCIe width
	NVLinkBandwidth     float64            `json:"nvlinkBandwidth"`     // GB/s
	EnergyConsumption   float64            `json:"energyConsumption"`   // Joules
	ProcessCount        int                `json:"processCount"`        // Number of processes
	MIGMode             bool               `json:"migMode"`             // MIG enabled
	MIGDevices          map[string]*MIGDevice `json:"migDevices"`       // MIG instance UUID -> device
	XIDErrors           int                `json:"xidErrors"`           // XID error count
	ECCErrors           *ECCErrors         `json:"eccErrors"`           // ECC error counts
	RetiredPages        *RetiredPages      `json:"retiredPages"`        // Retired page counts
	RemappedRows        int                `json:"remappedRows"`        // Remapped row count
	ViolationTime       float64            `json:"violationTime"`       // Power/thermal violation time
}

// MIGDevice represents MIG instance metrics
type MIGDevice struct {
	UUID              string  `json:"uuid"`
	InstanceID        int     `json:"instanceID"`
	ProfileName       string  `json:"profileName"`
	MemoryUsed        uint64  `json:"memoryUsed"`        // Bytes
	MemoryTotal       uint64  `json:"memoryTotal"`       // Bytes
	UtilizationGPU    float64 `json:"utilizationGPU"`    // Percentage
	UtilizationMemory float64 `json:"utilizationMemory"` // Percentage
	ProcessCount      int     `json:"processCount"`      // Number of processes
}

// ECCErrors represents ECC error counts
type ECCErrors struct {
	SingleBit struct {
		Volatile    int `json:"volatile"`
		Aggregate   int `json:"aggregate"`
	} `json:"singleBit"`
	DoubleBit struct {
		Volatile    int `json:"volatile"`
		Aggregate   int `json:"aggregate"`
	} `json:"doubleBit"`
}

// RetiredPages represents retired page counts
type RetiredPages struct {
	SingleBitECC int `json:"singleBitECC"`
	DoubleBitECC int `json:"doubleBitECC"`
	Pending      int `json:"pending"`
}

// NodeMetrics represents node-level metrics
type NodeMetrics struct {
	CPUUsage       float64 `json:"cpuUsage"`       // Percentage
	MemoryUsage    float64 `json:"memoryUsage"`    // Percentage
	DiskUsage      float64 `json:"diskUsage"`      // Percentage
	NetworkRxBytes uint64  `json:"networkRxBytes"` // Bytes
	NetworkTxBytes uint64  `json:"networkTxBytes"` // Bytes
	PowerDraw      float64 `json:"powerDraw"`      // Watts (if available)
	Temperature    float64 `json:"temperature"`    // Celsius (if available)
}

// CarbonMetrics represents carbon-specific metrics
type CarbonMetrics struct {
	CarbonIntensity     float64 `json:"carbonIntensity"`     // gCO2/kWh
	CarbonRate          float64 `json:"carbonRate"`          // gCO2/hour
	EnergyEfficiency    float64 `json:"energyEfficiency"`    // Performance/Watt
	CarbonEfficiency    float64 `json:"carbonEfficiency"`    // Performance/gCO2
	OptimalWindow       bool    `json:"optimalWindow"`       // Is this an optimal carbon window?
	CarbonBudgetUsed    float64 `json:"carbonBudgetUsed"`    // Percentage of carbon budget used
	CarbonSavings       float64 `json:"carbonSavings"`       // gCO2 saved vs baseline
	RenewablePercent    float64 `json:"renewablePercent"`    // Percentage renewable energy
	PUE                 float64 `json:"pue"`                 // Power Usage Effectiveness
	CarbonForecast      []float64 `json:"carbonForecast"`    // Next 24h carbon intensity forecast
}

// WorkloadMetrics represents workload-specific metrics
type WorkloadMetrics struct {
	PodName           string            `json:"podName"`
	Namespace         string            `json:"namespace"`
	WorkloadType      string            `json:"workloadType"`      // training, inference, batch
	GPUAllocation     map[string]float64 `json:"gpuAllocation"`    // GPU UUID -> allocation percentage
	PowerConsumption  float64           `json:"powerConsumption"`  // Watts
	CarbonEmissions   float64           `json:"carbonEmissions"`   // gCO2/hour
	Performance       float64           `json:"performance"`       // Task-specific metric
	Efficiency        float64           `json:"efficiency"`        // Performance/Watt
	SLACompliance     float64           `json:"slaCompliance"`     // Percentage
	CheckpointReady   bool              `json:"checkpointReady"`   // Can be checkpointed
	MigrationScore    float64           `json:"migrationScore"`    // Migration feasibility score
}

// NewDCGMCollector creates a new DCGM collector
func NewDCGMCollector(prometheusURL string, redisClient *redis.Client, 
	logger *zap.Logger) (*DCGMCollector, error) {
	
	// Create Prometheus client
	client, err := api.NewClient(api.Config{
		Address: prometheusURL,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Prometheus client: %w", err)
	}

	config := &DCGMConfig{
		PrometheusURL:      prometheusURL,
		CollectionInterval: 30 * time.Second,
		MetricRetention:    24 * time.Hour,
		GPUMetrics:         getDefaultGPUMetrics(),
		CarbonMetrics:      getDefaultCarbonMetrics(),
		AlertThresholds:    getDefaultAlertThresholds(),
	}

	return &DCGMCollector{
		prometheusClient: v1.NewAPI(client),
		redisClient:      redisClient,
		logger:           logger,
		config:           config,
	}, nil
}

// CollectMetrics collects GPU metrics from DCGM Exporter
func (d *DCGMCollector) CollectMetrics(ctx context.Context) (*GPUMetrics, error) {
	d.logger.Debug("Collecting GPU metrics from DCGM")

	metrics := &GPUMetrics{
		Timestamp:       time.Now(),
		GPUs:            make(map[string]*GPUDevice),
		WorkloadMetrics: make(map[string]*WorkloadMetrics),
	}

	// Collect GPU device metrics
	if err := d.collectGPUDeviceMetrics(ctx, metrics); err != nil {
		d.logger.Error("Failed to collect GPU device metrics", zap.Error(err))
		return nil, err
	}

	// Collect node metrics
	if err := d.collectNodeMetrics(ctx, metrics); err != nil {
		d.logger.Warn("Failed to collect node metrics", zap.Error(err))
		// Continue without node metrics
	}

	// Collect carbon metrics
	if err := d.collectCarbonMetrics(ctx, metrics); err != nil {
		d.logger.Warn("Failed to collect carbon metrics", zap.Error(err))
		// Continue without carbon metrics
	}

	// Collect workload metrics
	if err := d.collectWorkloadMetrics(ctx, metrics); err != nil {
		d.logger.Warn("Failed to collect workload metrics", zap.Error(err))
		// Continue without workload metrics
	}

	// Cache metrics
	if err := d.cacheMetrics(ctx, metrics); err != nil {
		d.logger.Warn("Failed to cache metrics", zap.Error(err))
	}

	return metrics, nil
}

// collectGPUDeviceMetrics collects individual GPU device metrics
func (d *DCGMCollector) collectGPUDeviceMetrics(ctx context.Context, metrics *GPUMetrics) error {
	// Query GPU power metrics
	powerQuery := `DCGM_FI_DEV_POWER_USAGE`
	powerResult, _, err := d.prometheusClient.Query(ctx, powerQuery, time.Now())
	if err != nil {
		return fmt.Errorf("failed to query GPU power: %w", err)
	}

	// Process power metrics
	if vector, ok := powerResult.(model.Vector); ok {
		for _, sample := range vector {
			gpu := d.getOrCreateGPU(metrics, sample.Metric)
			gpu.PowerDraw = float64(sample.Value)
		}
	}

	// Query GPU temperature
	tempQuery := `DCGM_FI_DEV_GPU_TEMP`
	tempResult, _, err := d.prometheusClient.Query(ctx, tempQuery, time.Now())
	if err == nil {
		if vector, ok := tempResult.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				gpu.Temperature = float64(sample.Value)
			}
		}
	}

	// Query GPU utilization
	utilizationQuery := `DCGM_FI_DEV_GPU_UTIL`
	utilizationResult, _, err := d.prometheusClient.Query(ctx, utilizationQuery, time.Now())
	if err == nil {
		if vector, ok := utilizationResult.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				gpu.UtilizationGPU = float64(sample.Value)
			}
		}
	}

	// Query memory metrics
	memUsedQuery := `DCGM_FI_DEV_FB_USED`
	memUsedResult, _, err := d.prometheusClient.Query(ctx, memUsedQuery, time.Now())
	if err == nil {
		if vector, ok := memUsedResult.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				gpu.MemoryUsed = uint64(sample.Value) * 1024 * 1024 // Convert MB to bytes
			}
		}
	}

	memTotalQuery := `DCGM_FI_DEV_FB_TOTAL`
	memTotalResult, _, err := d.prometheusClient.Query(ctx, memTotalQuery, time.Now())
	if err == nil {
		if vector, ok := memTotalResult.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				gpu.MemoryTotal = uint64(sample.Value) * 1024 * 1024 // Convert MB to bytes
			}
		}
	}

	// Query additional metrics
	d.collectAdditionalGPUMetrics(ctx, metrics)

	return nil
}

// collectAdditionalGPUMetrics collects additional GPU metrics
func (d *DCGMCollector) collectAdditionalGPUMetrics(ctx context.Context, metrics *GPUMetrics) {
	// Clock frequencies
	smClockQuery := `DCGM_FI_DEV_SM_CLOCK`
	if result, _, err := d.prometheusClient.Query(ctx, smClockQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				gpu.ClockSM = float64(sample.Value)
			}
		}
	}

	memClockQuery := `DCGM_FI_DEV_MEM_CLOCK`
	if result, _, err := d.prometheusClient.Query(ctx, memClockQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				gpu.ClockMemory = float64(sample.Value)
			}
		}
	}

	// Fan speed
	fanQuery := `DCGM_FI_DEV_FAN_SPEED`
	if result, _, err := d.prometheusClient.Query(ctx, fanQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				gpu.FanSpeed = float64(sample.Value)
			}
		}
	}

	// Performance state
	perfStateQuery := `DCGM_FI_DEV_PSTATE`
	if result, _, err := d.prometheusClient.Query(ctx, perfStateQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				gpu.PerformanceState = fmt.Sprintf("P%d", int(sample.Value))
			}
		}
	}

	// Energy consumption
	energyQuery := `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION`
	if result, _, err := d.prometheusClient.Query(ctx, energyQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				gpu.EnergyConsumption = float64(sample.Value)
			}
		}
	}

	// Process count
	processQuery := `DCGM_FI_DEV_COMPUTE_PIDS`
	if result, _, err := d.prometheusClient.Query(ctx, processQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				gpu.ProcessCount = int(sample.Value)
			}
		}
	}

	// XID errors
	xidQuery := `DCGM_FI_DEV_XID_ERRORS`
	if result, _, err := d.prometheusClient.Query(ctx, xidQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				gpu.XIDErrors = int(sample.Value)
			}
		}
	}

	// Collect ECC errors
	d.collectECCErrors(ctx, metrics)

	// Collect MIG metrics if available
	d.collectMIGMetrics(ctx, metrics)
}

// collectECCErrors collects ECC error metrics
func (d *DCGMCollector) collectECCErrors(ctx context.Context, metrics *GPUMetrics) {
	// Single-bit ECC errors
	sbeVolQuery := `DCGM_FI_DEV_ECC_SBE_VOL_TOTAL`
	if result, _, err := d.prometheusClient.Query(ctx, sbeVolQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				if gpu.ECCErrors == nil {
					gpu.ECCErrors = &ECCErrors{}
				}
				gpu.ECCErrors.SingleBit.Volatile = int(sample.Value)
			}
		}
	}

	sbeAggQuery := `DCGM_FI_DEV_ECC_SBE_AGG_TOTAL`
	if result, _, err := d.prometheusClient.Query(ctx, sbeAggQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				if gpu.ECCErrors == nil {
					gpu.ECCErrors = &ECCErrors{}
				}
				gpu.ECCErrors.SingleBit.Aggregate = int(sample.Value)
			}
		}
	}

	// Double-bit ECC errors
	dbeVolQuery := `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL`
	if result, _, err := d.prometheusClient.Query(ctx, dbeVolQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				if gpu.ECCErrors == nil {
					gpu.ECCErrors = &ECCErrors{}
				}
				gpu.ECCErrors.DoubleBit.Volatile = int(sample.Value)
			}
		}
	}

	dbeAggQuery := `DCGM_FI_DEV_ECC_DBE_AGG_TOTAL`
	if result, _, err := d.prometheusClient.Query(ctx, dbeAggQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				if gpu.ECCErrors == nil {
					gpu.ECCErrors = &ECCErrors{}
				}
				gpu.ECCErrors.DoubleBit.Aggregate = int(sample.Value)
			}
		}
	}
}

// collectMIGMetrics collects MIG-specific metrics
func (d *DCGMCollector) collectMIGMetrics(ctx context.Context, metrics *GPUMetrics) {
	// MIG instance utilization
	migUtilQuery := `DCGM_FI_DEV_GPU_UTIL{mig_instance!=""}`
	if result, _, err := d.prometheusClient.Query(ctx, migUtilQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				migUUID := string(sample.Metric["mig_instance"])
				
				if gpu.MIGDevices == nil {
					gpu.MIGDevices = make(map[string]*MIGDevice)
				}
				
				if _, exists := gpu.MIGDevices[migUUID]; !exists {
					gpu.MIGDevices[migUUID] = &MIGDevice{
						UUID: migUUID,
					}
				}
				
				gpu.MIGDevices[migUUID].UtilizationGPU = float64(sample.Value)
				gpu.MIGMode = true
			}
		}
	}

	// MIG memory usage
	migMemQuery := `DCGM_FI_DEV_FB_USED{mig_instance!=""}`
	if result, _, err := d.prometheusClient.Query(ctx, migMemQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			for _, sample := range vector {
				gpu := d.getOrCreateGPU(metrics, sample.Metric)
				migUUID := string(sample.Metric["mig_instance"])
				
				if gpu.MIGDevices == nil {
					gpu.MIGDevices = make(map[string]*MIGDevice)
				}
				
				if _, exists := gpu.MIGDevices[migUUID]; !exists {
					gpu.MIGDevices[migUUID] = &MIGDevice{
						UUID: migUUID,
					}
				}
				
				gpu.MIGDevices[migUUID].MemoryUsed = uint64(sample.Value) * 1024 * 1024
			}
		}
	}
}

// collectNodeMetrics collects node-level metrics
func (d *DCGMCollector) collectNodeMetrics(ctx context.Context, metrics *GPUMetrics) error {
	nodeMetrics := &NodeMetrics{}

	// CPU usage
	cpuQuery := `100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)`
	if result, _, err := d.prometheusClient.Query(ctx, cpuQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			if len(vector) > 0 {
				nodeMetrics.CPUUsage = float64(vector[0].Value)
			}
		}
	}

	// Memory usage
	memQuery := `(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100`
	if result, _, err := d.prometheusClient.Query(ctx, memQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			if len(vector) > 0 {
				nodeMetrics.MemoryUsage = float64(vector[0].Value)
			}
		}
	}

	// Disk usage
	diskQuery := `(1 - (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"})) * 100`
	if result, _, err := d.prometheusClient.Query(ctx, diskQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			if len(vector) > 0 {
				nodeMetrics.DiskUsage = float64(vector[0].Value)
			}
		}
	}

	// Network metrics
	netRxQuery := `rate(node_network_receive_bytes_total[5m])`
	if result, _, err := d.prometheusClient.Query(ctx, netRxQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			var total float64
			for _, sample := range vector {
				total += float64(sample.Value)
			}
			nodeMetrics.NetworkRxBytes = uint64(total)
		}
	}

	netTxQuery := `rate(node_network_transmit_bytes_total[5m])`
	if result, _, err := d.prometheusClient.Query(ctx, netTxQuery, time.Now()); err == nil {
		if vector, ok := result.(model.Vector); ok {
			var total float64
			for _, sample := range vector {
				total += float64(sample.Value)
			}
			nodeMetrics.NetworkTxBytes = uint64(total)
		}
	}

	metrics.NodeMetrics = nodeMetrics
	return nil
}

// collectCarbonMetrics collects carbon-specific metrics
func (d *DCGMCollector) collectCarbonMetrics(ctx context.Context, metrics *GPUMetrics) error {
	carbonMetrics := &CarbonMetrics{}

	// Get zone from first GPU (assuming all GPUs on same node)
	var zone string
	for _, gpu := range metrics.GPUs {
		if nodeName := gpu.Name; nodeName != "" {
			// Extract zone from node name or labels
			zone = d.extractZoneFromNode(nodeName)
			break
		}
	}

	if zone == "" {
		zone = "default"
	}

	// Get carbon intensity from cache or external source
	if intensity, err := d.getCarbonIntensity(ctx, zone); err == nil {
		carbonMetrics.CarbonIntensity = intensity
	} else {
		carbonMetrics.CarbonIntensity = 400.0 // Default intensity
	}

	// Calculate total power consumption
	var totalPower float64
	for _, gpu := range metrics.GPUs {
		totalPower += gpu.PowerDraw
	}

	// Calculate carbon rate (gCO2/hour)
	carbonMetrics.CarbonRate = (totalPower / 1000.0) * carbonMetrics.CarbonIntensity

	// Get PUE
	if pue, err := d.getPUE(ctx, zone); err == nil {
		carbonMetrics.PUE = pue
	} else {
		carbonMetrics.PUE = 1.4 // Default PUE
	}

	// Adjust carbon rate with PUE
	carbonMetrics.CarbonRate *= carbonMetrics.PUE

	// Calculate efficiency metrics
	var totalUtilization float64
	gpuCount := len(metrics.GPUs)
	if gpuCount > 0 {
		for _, gpu := range metrics.GPUs {
			totalUtilization += gpu.UtilizationGPU
		}
		avgUtilization := totalUtilization / float64(gpuCount)
		
		// Energy efficiency (utilization per watt)
		if totalPower > 0 {
			carbonMetrics.EnergyEfficiency = avgUtilization / totalPower
		}
		
		// Carbon efficiency (utilization per gCO2/hour)
		if carbonMetrics.CarbonRate > 0 {
			carbonMetrics.CarbonEfficiency = avgUtilization / carbonMetrics.CarbonRate
		}
	}

	// Get renewable percentage
	if renewable, err := d.getRenewablePercent(ctx, zone); err == nil {
		carbonMetrics.RenewablePercent = renewable
	} else {
		carbonMetrics.RenewablePercent = 30.0 // Default
	}

	// Check if in optimal carbon window
	carbonMetrics.OptimalWindow = d.isOptimalCarbonWindow(ctx, zone, carbonMetrics.CarbonIntensity)

	// Get carbon forecast
	if forecast, err := d.getCarbonForecast(ctx, zone, 24); err == nil {
		carbonMetrics.CarbonForecast = forecast
	}

	metrics.CarbonMetrics = carbonMetrics
	return nil
}

// collectWorkloadMetrics collects workload-specific metrics
func (d *DCGMCollector) collectWorkloadMetrics(ctx context.Context, metrics *GPUMetrics) error {
	// Query pod GPU usage
	podGPUQuery := `DCGM_FI_DEV_GPU_UTIL * on(gpu, UUID) group_left(pod, namespace) kube_pod_info`
	result, _, err := d.prometheusClient.Query(ctx, podGPUQuery, time.Now())
	if err != nil {
		// Fallback to simulated workload metrics
		return d.simulateWorkloadMetrics(ctx, metrics)
	}

	if vector, ok := result.(model.Vector); ok {
		for _, sample := range vector {
			podName := string(sample.Metric["pod"])
			namespace := string(sample.Metric["namespace"])
			gpuUUID := string(sample.Metric["UUID"])
			
			workloadKey := fmt.Sprintf("%s/%s", namespace, podName)
			
			if _, exists := metrics.WorkloadMetrics[workloadKey]; !exists {
				metrics.WorkloadMetrics[workloadKey] = &WorkloadMetrics{
					PodName:       podName,
					Namespace:     namespace,
					GPUAllocation: make(map[string]float64),
				}
			}
			
			workload := metrics.WorkloadMetrics[workloadKey]
			workload.GPUAllocation[gpuUUID] = float64(sample.Value)
			
			// Calculate power consumption for this workload
			if gpu, exists := metrics.GPUs[gpuUUID]; exists {
				utilizationRatio := float64(sample.Value) / 100.0
				workload.PowerConsumption += gpu.PowerDraw * utilizationRatio
			}
		}
	}

	// Enhance workload metrics with additional data
	for _, workload := range metrics.WorkloadMetrics {
		d.enhanceWorkloadMetrics(ctx, workload, metrics)
	}

	return nil
}

// simulateWorkloadMetrics creates simulated workload metrics for testing
func (d *DCGMCollector) simulateWorkloadMetrics(ctx context.Context, metrics *GPUMetrics) error {
	// Create sample workloads
	workloads := []struct {
		name      string
		namespace string
		wlType    string
	}{
		{"training-job-1", "ml-training", "training"},
		{"inference-server", "ml-inference", "inference"},
		{"batch-job-2", "ml-batch", "batch"},
	}

	for i, wl := range workloads {
		workloadKey := fmt.Sprintf("%s/%s", wl.namespace, wl.name)
		
		workload := &WorkloadMetrics{
			PodName:       wl.name,
			Namespace:     wl.namespace,
			WorkloadType:  wl.wlType,
			GPUAllocation: make(map[string]float64),
		}

		// Assign GPUs to workloads
		gpuIndex := 0
		for uuid, gpu := range metrics.GPUs {
			if gpuIndex%len(workloads) == i {
				allocation := 50.0 + float64(i*20) // Vary allocation
				workload.GPUAllocation[uuid] = allocation
				workload.PowerConsumption += gpu.PowerDraw * (allocation / 100.0)
			}
			gpuIndex++
		}

		d.enhanceWorkloadMetrics(ctx, workload, metrics)
		metrics.WorkloadMetrics[workloadKey] = workload
	}

	return nil
}

// enhanceWorkloadMetrics adds calculated metrics to workload data
func (d *DCGMCollector) enhanceWorkloadMetrics(ctx context.Context, workload *WorkloadMetrics, 
	metrics *GPUMetrics) {
	
	// Calculate carbon emissions
	if metrics.CarbonMetrics != nil {
		powerKW := workload.PowerConsumption / 1000.0
		workload.CarbonEmissions = powerKW * metrics.CarbonMetrics.CarbonIntensity * metrics.CarbonMetrics.PUE
	}

	// Calculate performance (simplified)
	var totalUtilization float64
	gpuCount := len(workload.GPUAllocation)
	if gpuCount > 0 {
		for _, allocation := range workload.GPUAllocation {
			totalUtilization += allocation
		}
		workload.Performance = totalUtilization / float64(gpuCount)
	}

	// Calculate efficiency
	if workload.PowerConsumption > 0 {
		workload.Efficiency = workload.Performance / workload.PowerConsumption
	}

	// Set workload type if not set
	if workload.WorkloadType == "" {
		workload.WorkloadType = d.inferWorkloadType(workload.PodName)
	}

	// SLA compliance (simplified)
	workload.SLACompliance = 95.0 + (workload.Performance-50.0)*0.1

	// Checkpoint readiness
	workload.CheckpointReady = d.isCheckpointReady(workload.WorkloadType, workload.Performance)

	// Migration score
	workload.MigrationScore = d.calculateMigrationScore(workload)
}

// Helper functions

func (d *DCGMCollector) getOrCreateGPU(metrics *GPUMetrics, labels model.Metric) *GPUDevice {
	uuid := string(labels["UUID"])
	if uuid == "" {
		uuid = string(labels["gpu"]) // Fallback
	}

	if _, exists := metrics.GPUs[uuid]; !exists {
		gpu := &GPUDevice{
			UUID:  uuid,
			Name:  string(labels["device"]),
			Index: d.parseGPUIndex(labels),
		}
		
		// Extract node name and zone
		if nodeName := string(labels["instance"]); nodeName != "" {
			metrics.NodeName = nodeName
			metrics.Zone = d.extractZoneFromNode(nodeName)
		}
		
		metrics.GPUs[uuid] = gpu
	}

	return metrics.GPUs[uuid]
}

func (d *DCGMCollector) parseGPUIndex(labels model.Metric) int {
	if indexStr := string(labels["gpu"]); indexStr != "" {
		if index, err := strconv.Atoi(indexStr); err == nil {
			return index
		}
	}
	return 0
}

func (d *DCGMCollector) extractZoneFromNode(nodeName string) string {
	// Extract zone from node name (e.g., gke-cluster-pool-1-abc123 -> us-west1-a)
	// This is a simplified implementation
	if strings.Contains(nodeName, "west1") {
		return "us-west1-a"
	} else if strings.Contains(nodeName, "east1") {
		return "us-east1-a"
	} else if strings.Contains(nodeName, "europe") {
		return "europe-west1-a"
	}
	return "default"
}

func (d *DCGMCollector) getCarbonIntensity(ctx context.Context, zone string) (float64, error) {
	if d.redisClient != nil {
		key := fmt.Sprintf("carbon:intensity:%s", zone)
		val, err := d.redisClient.Get(ctx, key).Result()
		if err == nil {
			if intensity, err := strconv.ParseFloat(val, 64); err == nil {
				return intensity, nil
			}
		}
	}
	
	// Default intensities by zone
	intensities := map[string]float64{
		"us-west1-a":     250.0, // California - high renewable
		"us-east1-a":     450.0, // Virginia - mixed grid
		"europe-west1-a": 200.0, // Belgium - high renewable
		"default":        400.0,
	}
	
	if intensity, exists := intensities[zone]; exists {
		return intensity, nil
	}
	return intensities["default"], nil
}

func (d *DCGMCollector) getPUE(ctx context.Context, zone string) (float64, error) {
	if d.redisClient != nil {
		key := fmt.Sprintf("pue:current:%s", zone)
		val, err := d.redisClient.Get(ctx, key).Result()
		if err == nil {
			var measurement PUEMeasurement
			if err := json.Unmarshal([]byte(val), &measurement); err == nil {
				return measurement.PUE, nil
			}
		}
	}
	
	// Default PUE by zone
	pueValues := map[string]float64{
		"us-west1-a":     1.2, // Efficient datacenter
		"us-east1-a":     1.4, // Standard datacenter
		"europe-west1-a": 1.1, // Very efficient
		"default":        1.4,
	}
	
	if pue, exists := pueValues[zone]; exists {
		return pue, nil
	}
	return pueValues["default"], nil
}

func (d *DCGMCollector) getRenewablePercent(ctx context.Context, zone string) (float64, error) {
	// Default renewable percentages by zone
	renewable := map[string]float64{
		"us-west1-a":     85.0, // California - high renewable
		"us-east1-a":     45.0, // Virginia - mixed
		"europe-west1-a": 95.0, // Belgium - very high renewable
		"default":        30.0,
	}
	
	if pct, exists := renewable[zone]; exists {
		return pct, nil
	}
	return renewable["default"], nil
}

func (d *DCGMCollector) isOptimalCarbonWindow(ctx context.Context, zone string, 
	currentIntensity float64) bool {
	
	// Get forecast to determine if current time is optimal
	if forecast, err := d.getCarbonForecast(ctx, zone, 6); err == nil {
		// Check if current intensity is in lowest 25% of next 6 hours
		minIntensity := currentIntensity
		for _, intensity := range forecast {
			if intensity < minIntensity {
				minIntensity = intensity
			}
		}
		
		threshold := minIntensity * 1.1 // Within 10% of minimum
		return currentIntensity <= threshold
	}
	
	// Fallback: consider low-carbon hours (typically night/early morning)
	hour := time.Now().Hour()
	return hour >= 2 && hour <= 6 // 2 AM - 6 AM typically lowest demand
}

func (d *DCGMCollector) getCarbonForecast(ctx context.Context, zone string, 
	hours int) ([]float64, error) {
	
	if d.redisClient != nil {
		key := fmt.Sprintf("carbon:forecast:%s:%d", zone, hours)
		val, err := d.redisClient.Get(ctx, key).Result()
		if err == nil {
			var forecast []float64
			if err := json.Unmarshal([]byte(val), &forecast); err == nil {
				return forecast, nil
			}
		}
	}
	
	// Generate simulated forecast
	baseIntensity, _ := d.getCarbonIntensity(ctx, zone)
	forecast := make([]float64, hours)
	
	for i := 0; i < hours; i++ {
		// Simulate daily pattern
		hour := (time.Now().Hour() + i) % 24
		dailyFactor := 0.8 + 0.4*math.Sin(float64(hour-6)*math.Pi/12)
		forecast[i] = baseIntensity * dailyFactor
	}
	
	return forecast, nil
}

func (d *DCGMCollector) inferWorkloadType(podName string) string {
	if strings.Contains(podName, "training") {
		return "training"
	} else if strings.Contains(podName, "inference") || strings.Contains(podName, "serve") {
		return "inference"
	} else if strings.Contains(podName, "batch") {
		return "batch"
	}
	return "unknown"
}

func (d *DCGMCollector) isCheckpointReady(workloadType string, performance float64) bool {
	// Training workloads are typically checkpointable
	if workloadType == "training" {
		return performance > 30.0 // Only if actively training
	}
	
	// Batch workloads may be checkpointable
	if workloadType == "batch" {
		return performance > 20.0
	}
	
	// Inference workloads typically not checkpointable
	return false
}

func (d *DCGMCollector) calculateMigrationScore(workload *WorkloadMetrics) float64 {
	score := 0.0
	
	// Base score from checkpoint readiness
	if workload.CheckpointReady {
		score += 50.0
	}
	
	// Adjust based on performance (lower performance = easier to migrate)
	score += (100.0 - workload.Performance) * 0.3
	
	// Adjust based on SLA compliance (higher compliance = harder to migrate)
	score += (100.0 - workload.SLACompliance) * 0.2
	
	// Workload type factor
	switch workload.WorkloadType {
	case "batch":
		score += 30.0 // Batch jobs are easier to migrate
	case "training":
		score += 20.0 // Training can be checkpointed
	case "inference":
		score -= 20.0 // Inference is harder to migrate
	}
	
	return math.Max(0.0, math.Min(100.0, score))
}

func (d *DCGMCollector) cacheMetrics(ctx context.Context, metrics *GPUMetrics) error {
	if d.redisClient == nil {
		return nil
	}
	
	data, err := json.Marshal(metrics)
	if err != nil {
		return err
	}
	
	// Cache current metrics
	key := fmt.Sprintf("gpu:metrics:%s:%d", metrics.Zone, metrics.Timestamp.Unix())
	return d.redisClient.Set(ctx, key, data, d.config.MetricRetention).Err()
}

// Default configuration functions

func getDefaultGPUMetrics() []string {
	return []string{
		"DCGM_FI_DEV_POWER_USAGE",
		"DCGM_FI_DEV_GPU_TEMP",
		"DCGM_FI_DEV_GPU_UTIL",
		"DCGM_FI_DEV_FB_USED",
		"DCGM_FI_DEV_FB_TOTAL",
		"DCGM_FI_DEV_SM_CLOCK",
		"DCGM_FI_DEV_MEM_CLOCK",
		"DCGM_FI_DEV_FAN_SPEED",
		"DCGM_FI_DEV_PSTATE",
		"DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
		"DCGM_FI_DEV_COMPUTE_PIDS",
		"DCGM_FI_DEV_XID_ERRORS",
		"DCGM_FI_DEV_ECC_SBE_VOL_TOTAL",
		"DCGM_FI_DEV_ECC_DBE_VOL_TOTAL",
	}
}

func getDefaultCarbonMetrics() []string {
	return []string{
		"gpu_carbon_rate_gco2_per_hour",
		"gpu_energy_efficiency_perf_per_watt",
		"gpu_carbon_efficiency_perf_per_gco2",
		"gpu_carbon_intensity_gco2_per_kwh",
		"gpu_pue_ratio",
		"gpu_renewable_percent",
	}
}

func getDefaultAlertThresholds() map[string]float64 {
	return map[string]float64{
		"power_usage_watts":     300.0, // Alert if GPU power > 300W
		"temperature_celsius":   85.0,  // Alert if GPU temp > 85°C
		"memory_usage_percent":  90.0,  // Alert if GPU memory > 90%
		"carbon_rate_gco2_hour": 500.0, // Alert if carbon rate > 500 gCO2/hour
		"utilization_percent":   95.0,  // Alert if GPU utilization > 95%
		"xid_errors":           5.0,    // Alert if XID errors > 5
		"ecc_errors":           10.0,   // Alert if ECC errors > 10
	}
}