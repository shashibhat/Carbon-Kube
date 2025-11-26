package workloads

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/go-redis/redis/v8"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"go.uber.org/zap"
)

// PyTorchWorkloadManager manages PyTorch training and inference workloads
type PyTorchWorkloadManager struct {
	kubeClient  kubernetes.Interface
	redisClient *redis.Client
	logger      *zap.Logger
	config      *PyTorchConfig
}

// PyTorchConfig holds PyTorch workload configuration
type PyTorchConfig struct {
	DefaultImage           string            `json:"defaultImage"`           // Default PyTorch image
	SupportedVersions      []string          `json:"supportedVersions"`      // Supported PyTorch versions
	DefaultResources       ResourceLimits    `json:"defaultResources"`       // Default resource limits
	CarbonOptimization     bool              `json:"carbonOptimization"`     // Enable carbon optimization
	CheckpointInterval     time.Duration     `json:"checkpointInterval"`     // Checkpoint interval
	ModelRegistry          string            `json:"modelRegistry"`          // Model registry URL
	DatasetRegistry        string            `json:"datasetRegistry"`        // Dataset registry URL
	DistributedTraining    DistributedConfig `json:"distributedTraining"`    // Distributed training config
	MixedPrecision         bool              `json:"mixedPrecision"`         // Enable mixed precision
	GradientCompression    bool              `json:"gradientCompression"`    // Enable gradient compression
	ProfilerConfig         ProfilerConfig    `json:"profilerConfig"`         // Profiler configuration
}

// ResourceLimits defines resource limits for workloads
type ResourceLimits struct {
	CPURequest    string `json:"cpuRequest"`    // CPU request
	CPULimit      string `json:"cpuLimit"`      // CPU limit
	MemoryRequest string `json:"memoryRequest"` // Memory request
	MemoryLimit   string `json:"memoryLimit"`   // Memory limit
	GPURequest    int    `json:"gpuRequest"`    // GPU request
	GPUMemory     string `json:"gpuMemory"`     // GPU memory requirement
}

// DistributedConfig defines distributed training configuration
type DistributedConfig struct {
	Backend           string `json:"backend"`           // nccl, gloo, mpi
	MaxNodes          int    `json:"maxNodes"`          // Maximum nodes
	MaxGPUsPerNode    int    `json:"maxGPUsPerNode"`    // Maximum GPUs per node
	CommunicationOpt  bool   `json:"communicationOpt"`  // Enable communication optimization
	GradientAllReduce bool   `json:"gradientAllReduce"` // Enable gradient all-reduce
}

// ProfilerConfig defines profiler configuration
type ProfilerConfig struct {
	Enabled         bool          `json:"enabled"`         // Enable profiler
	ProfileInterval time.Duration `json:"profileInterval"` // Profile interval
	TraceMemory     bool          `json:"traceMemory"`     // Trace memory usage
	TraceCompute    bool          `json:"traceCompute"`    // Trace compute usage
	ExportFormat    string        `json:"exportFormat"`    // Export format (tensorboard, chrome)
}

// PyTorchWorkload represents a PyTorch workload
type PyTorchWorkload struct {
	Name                string                 `json:"name"`
	Namespace           string                 `json:"namespace"`
	WorkloadType        string                 `json:"workloadType"`        // training, inference, fine-tuning
	ModelName           string                 `json:"modelName"`           // Model name
	ModelVersion        string                 `json:"modelVersion"`        // Model version
	DatasetName         string                 `json:"datasetName"`         // Dataset name
	DatasetVersion      string                 `json:"datasetVersion"`      // Dataset version
	PyTorchVersion      string                 `json:"pytorchVersion"`      // PyTorch version
	CUDAVersion         string                 `json:"cudaVersion"`         // CUDA version
	Resources           ResourceLimits         `json:"resources"`           // Resource requirements
	DistributedConfig   *DistributedConfig     `json:"distributedConfig"`   // Distributed training config
	Hyperparameters     map[string]interface{} `json:"hyperparameters"`     // Hyperparameters
	Environment         map[string]string      `json:"environment"`         // Environment variables
	CarbonConstraints   *CarbonConstraints     `json:"carbonConstraints"`   // Carbon constraints
	SLARequirements     *SLARequirements       `json:"slaRequirements"`     // SLA requirements
	CheckpointConfig    *CheckpointConfig      `json:"checkpointConfig"`    // Checkpoint configuration
	MonitoringConfig    *MonitoringConfig      `json:"monitoringConfig"`    // Monitoring configuration
	Status              WorkloadStatus         `json:"status"`              // Current status
	Metrics             *WorkloadMetrics       `json:"metrics"`             // Runtime metrics
	CreatedAt           time.Time              `json:"createdAt"`           // Creation time
	UpdatedAt           time.Time              `json:"updatedAt"`           // Last update time
}

// CarbonConstraints defines carbon-aware constraints
type CarbonConstraints struct {
	MaxCarbonRate       float64   `json:"maxCarbonRate"`       // Max gCO2/hour
	CarbonBudget        float64   `json:"carbonBudget"`        // Total carbon budget (gCO2)
	PreferLowCarbon     bool      `json:"preferLowCarbon"`     // Prefer low-carbon zones
	OptimalWindows      []string  `json:"optimalWindows"`      // Preferred time windows
	CarbonIntensityMax  float64   `json:"carbonIntensityMax"`  // Max carbon intensity (gCO2/kWh)
	RenewableMinPercent float64   `json:"renewableMinPercent"` // Min renewable energy %
	PUEMax              float64   `json:"pueMax"`              // Max PUE
}

// SLARequirements defines SLA requirements
type SLARequirements struct {
	MaxLatency          time.Duration `json:"maxLatency"`          // Max response latency
	MinThroughput       float64       `json:"minThroughput"`       // Min throughput (req/s)
	MaxErrorRate        float64       `json:"maxErrorRate"`        // Max error rate (%)
	MinAvailability     float64       `json:"minAvailability"`     // Min availability (%)
	MaxTrainingTime     time.Duration `json:"maxTrainingTime"`     // Max training time
	MinAccuracy         float64       `json:"minAccuracy"`         // Min model accuracy
	CostBudget          float64       `json:"costBudget"`          // Cost budget ($)
}

// CheckpointConfig defines checkpoint configuration
type CheckpointConfig struct {
	Enabled           bool          `json:"enabled"`           // Enable checkpointing
	Interval          time.Duration `json:"interval"`          // Checkpoint interval
	StorageClass      string        `json:"storageClass"`      // Storage class for checkpoints
	RetentionPolicy   string        `json:"retentionPolicy"`   // Retention policy
	CompressionLevel  int           `json:"compressionLevel"`  // Compression level (0-9)
	AsyncSave         bool          `json:"asyncSave"`         // Async checkpoint saving
	VerifyCheckpoints bool          `json:"verifyCheckpoints"` // Verify checkpoint integrity
}

// MonitoringConfig defines monitoring configuration
type MonitoringConfig struct {
	MetricsEnabled    bool          `json:"metricsEnabled"`    // Enable metrics collection
	LogLevel          string        `json:"logLevel"`          // Log level
	ProfilerEnabled   bool          `json:"profilerEnabled"`   // Enable profiler
	TracingEnabled    bool          `json:"tracingEnabled"`    // Enable distributed tracing
	AlertsEnabled     bool          `json:"alertsEnabled"`     // Enable alerts
	DashboardEnabled  bool          `json:"dashboardEnabled"`  // Enable dashboard
	MetricsInterval   time.Duration `json:"metricsInterval"`   // Metrics collection interval
}

// WorkloadStatus represents workload status
type WorkloadStatus struct {
	Phase             string            `json:"phase"`             // pending, running, completed, failed
	Message           string            `json:"message"`           // Status message
	Reason            string            `json:"reason"`            // Status reason
	StartTime         *time.Time        `json:"startTime"`         // Start time
	CompletionTime    *time.Time        `json:"completionTime"`    // Completion time
	Progress          float64           `json:"progress"`          // Progress percentage
	CurrentEpoch      int               `json:"currentEpoch"`      // Current training epoch
	TotalEpochs       int               `json:"totalEpochs"`       // Total training epochs
	LastCheckpoint    *time.Time        `json:"lastCheckpoint"`    // Last checkpoint time
	Conditions        []WorkloadCondition `json:"conditions"`      // Status conditions
}

// WorkloadCondition represents a workload condition
type WorkloadCondition struct {
	Type               string    `json:"type"`               // condition type
	Status             string    `json:"status"`             // True, False, Unknown
	LastTransitionTime time.Time `json:"lastTransitionTime"` // last transition time
	Reason             string    `json:"reason"`             // reason for condition
	Message            string    `json:"message"`            // human readable message
}

// WorkloadMetrics represents runtime metrics
type WorkloadMetrics struct {
	PowerConsumption    float64           `json:"powerConsumption"`    // Current power (W)
	CarbonEmissions     float64           `json:"carbonEmissions"`     // Current carbon rate (gCO2/h)
	GPUUtilization      map[string]float64 `json:"gpuUtilization"`     // GPU utilization by device
	GPUMemoryUsage      map[string]float64 `json:"gpuMemoryUsage"`     // GPU memory usage by device
	CPUUtilization      float64           `json:"cpuUtilization"`      // CPU utilization %
	MemoryUsage         float64           `json:"memoryUsage"`         // Memory usage %
	NetworkIO           NetworkMetrics    `json:"networkIO"`           // Network I/O metrics
	DiskIO              DiskMetrics       `json:"diskIO"`              // Disk I/O metrics
	TrainingMetrics     *TrainingMetrics  `json:"trainingMetrics"`     // Training-specific metrics
	InferenceMetrics    *InferenceMetrics `json:"inferenceMetrics"`    // Inference-specific metrics
	CarbonEfficiency    float64           `json:"carbonEfficiency"`    // Performance/gCO2
	EnergyEfficiency    float64           `json:"energyEfficiency"`    // Performance/Watt
	CostEfficiency      float64           `json:"costEfficiency"`      // Performance/$
}

// NetworkMetrics represents network I/O metrics
type NetworkMetrics struct {
	RxBytes   uint64  `json:"rxBytes"`   // Received bytes
	TxBytes   uint64  `json:"txBytes"`   // Transmitted bytes
	RxPackets uint64  `json:"rxPackets"` // Received packets
	TxPackets uint64  `json:"txPackets"` // Transmitted packets
	Bandwidth float64 `json:"bandwidth"` // Current bandwidth usage (Mbps)
}

// DiskMetrics represents disk I/O metrics
type DiskMetrics struct {
	ReadBytes  uint64  `json:"readBytes"`  // Read bytes
	WriteBytes uint64  `json:"writeBytes"` // Write bytes
	ReadOps    uint64  `json:"readOps"`    // Read operations
	WriteOps   uint64  `json:"writeOps"`   // Write operations
	IOPS       float64 `json:"iops"`       // Current IOPS
}

// TrainingMetrics represents training-specific metrics
type TrainingMetrics struct {
	Loss              float64   `json:"loss"`              // Current loss
	Accuracy          float64   `json:"accuracy"`          // Current accuracy
	LearningRate      float64   `json:"learningRate"`      // Current learning rate
	BatchSize         int       `json:"batchSize"`         // Batch size
	SamplesPerSecond  float64   `json:"samplesPerSecond"`  // Training throughput
	TimePerEpoch      float64   `json:"timePerEpoch"`      // Time per epoch (seconds)
	EstimatedTimeLeft float64   `json:"estimatedTimeLeft"` // Estimated time left (seconds)
	GradientNorm      float64   `json:"gradientNorm"`      // Gradient norm
	ValidationLoss    float64   `json:"validationLoss"`    // Validation loss
	ValidationAcc     float64   `json:"validationAcc"`     // Validation accuracy
}

// InferenceMetrics represents inference-specific metrics
type InferenceMetrics struct {
	RequestsPerSecond float64 `json:"requestsPerSecond"` // Requests per second
	LatencyP50        float64 `json:"latencyP50"`        // 50th percentile latency (ms)
	LatencyP95        float64 `json:"latencyP95"`        // 95th percentile latency (ms)
	LatencyP99        float64 `json:"latencyP99"`        // 99th percentile latency (ms)
	ErrorRate         float64 `json:"errorRate"`         // Error rate (%)
	BatchSize         int     `json:"batchSize"`         // Inference batch size
	ModelSize         uint64  `json:"modelSize"`         // Model size in bytes
	CacheHitRate      float64 `json:"cacheHitRate"`      // Cache hit rate (%)
}

// NewPyTorchWorkloadManager creates a new PyTorch workload manager
func NewPyTorchWorkloadManager(kubeClient kubernetes.Interface, redisClient *redis.Client, 
	logger *zap.Logger) *PyTorchWorkloadManager {
	
	config := &PyTorchConfig{
		DefaultImage:      "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel",
		SupportedVersions: []string{"1.13", "2.0", "2.1", "2.2"},
		DefaultResources: ResourceLimits{
			CPURequest:    "2",
			CPULimit:      "8",
			MemoryRequest: "8Gi",
			MemoryLimit:   "32Gi",
			GPURequest:    1,
			GPUMemory:     "16Gi",
		},
		CarbonOptimization: true,
		CheckpointInterval: 10 * time.Minute,
		ModelRegistry:      "registry.carbon-kube.io/models",
		DatasetRegistry:    "registry.carbon-kube.io/datasets",
		DistributedTraining: DistributedConfig{
			Backend:           "nccl",
			MaxNodes:          8,
			MaxGPUsPerNode:    8,
			CommunicationOpt:  true,
			GradientAllReduce: true,
		},
		MixedPrecision:      true,
		GradientCompression: true,
		ProfilerConfig: ProfilerConfig{
			Enabled:         true,
			ProfileInterval: 5 * time.Minute,
			TraceMemory:     true,
			TraceCompute:    true,
			ExportFormat:    "tensorboard",
		},
	}

	return &PyTorchWorkloadManager{
		kubeClient:  kubeClient,
		redisClient: redisClient,
		logger:      logger,
		config:      config,
	}
}

// CreateTrainingWorkload creates a new PyTorch training workload
func (p *PyTorchWorkloadManager) CreateTrainingWorkload(ctx context.Context, 
	spec *PyTorchWorkloadSpec) (*PyTorchWorkload, error) {
	
	p.logger.Info("Creating PyTorch training workload", 
		zap.String("name", spec.Name), 
		zap.String("model", spec.ModelName))

	// Validate specification
	if err := p.validateWorkloadSpec(spec); err != nil {
		return nil, fmt.Errorf("invalid workload spec: %w", err)
	}

	// Create workload object
	workload := &PyTorchWorkload{
		Name:              spec.Name,
		Namespace:         spec.Namespace,
		WorkloadType:      "training",
		ModelName:         spec.ModelName,
		ModelVersion:      spec.ModelVersion,
		DatasetName:       spec.DatasetName,
		DatasetVersion:    spec.DatasetVersion,
		PyTorchVersion:    spec.PyTorchVersion,
		CUDAVersion:       spec.CUDAVersion,
		Resources:         spec.Resources,
		DistributedConfig: spec.DistributedConfig,
		Hyperparameters:   spec.Hyperparameters,
		Environment:       spec.Environment,
		CarbonConstraints: spec.CarbonConstraints,
		SLARequirements:   spec.SLARequirements,
		CheckpointConfig:  spec.CheckpointConfig,
		MonitoringConfig:  spec.MonitoringConfig,
		Status: WorkloadStatus{
			Phase:   "pending",
			Message: "Workload created, waiting for scheduling",
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	// Apply defaults
	p.applyDefaults(workload)

	// Optimize for carbon efficiency if enabled
	if p.config.CarbonOptimization {
		if err := p.optimizeForCarbon(ctx, workload); err != nil {
			p.logger.Warn("Failed to optimize for carbon", zap.Error(err))
		}
	}

	// Create Kubernetes resources
	if err := p.createKubernetesResources(ctx, workload); err != nil {
		return nil, fmt.Errorf("failed to create Kubernetes resources: %w", err)
	}

	// Store workload metadata
	if err := p.storeWorkload(ctx, workload); err != nil {
		p.logger.Warn("Failed to store workload metadata", zap.Error(err))
	}

	// Start monitoring
	go p.monitorWorkload(ctx, workload)

	return workload, nil
}

// CreateInferenceWorkload creates a new PyTorch inference workload
func (p *PyTorchWorkloadManager) CreateInferenceWorkload(ctx context.Context, 
	spec *PyTorchInferenceSpec) (*PyTorchWorkload, error) {
	
	p.logger.Info("Creating PyTorch inference workload", 
		zap.String("name", spec.Name), 
		zap.String("model", spec.ModelName))

	workload := &PyTorchWorkload{
		Name:             spec.Name,
		Namespace:        spec.Namespace,
		WorkloadType:     "inference",
		ModelName:        spec.ModelName,
		ModelVersion:     spec.ModelVersion,
		PyTorchVersion:   spec.PyTorchVersion,
		CUDAVersion:      spec.CUDAVersion,
		Resources:        spec.Resources,
		Environment:      spec.Environment,
		CarbonConstraints: spec.CarbonConstraints,
		SLARequirements:  spec.SLARequirements,
		MonitoringConfig: spec.MonitoringConfig,
		Status: WorkloadStatus{
			Phase:   "pending",
			Message: "Inference workload created, waiting for deployment",
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	p.applyDefaults(workload)

	// Optimize for inference performance and carbon efficiency
	if err := p.optimizeForInference(ctx, workload); err != nil {
		p.logger.Warn("Failed to optimize for inference", zap.Error(err))
	}

	// Create Kubernetes resources
	if err := p.createInferenceResources(ctx, workload); err != nil {
		return nil, fmt.Errorf("failed to create inference resources: %w", err)
	}

	// Store workload metadata
	if err := p.storeWorkload(ctx, workload); err != nil {
		p.logger.Warn("Failed to store workload metadata", zap.Error(err))
	}

	// Start monitoring
	go p.monitorWorkload(ctx, workload)

	return workload, nil
}

// GetWorkload retrieves a workload by name and namespace
func (p *PyTorchWorkloadManager) GetWorkload(ctx context.Context, namespace, 
	name string) (*PyTorchWorkload, error) {
	
	if p.redisClient != nil {
		key := fmt.Sprintf("pytorch:workload:%s:%s", namespace, name)
		data, err := p.redisClient.Get(ctx, key).Result()
		if err == nil {
			var workload PyTorchWorkload
			if err := json.Unmarshal([]byte(data), &workload); err == nil {
				return &workload, nil
			}
		}
	}

	return nil, fmt.Errorf("workload not found: %s/%s", namespace, name)
}

// ListWorkloads lists all workloads in a namespace
func (p *PyTorchWorkloadManager) ListWorkloads(ctx context.Context, 
	namespace string) ([]*PyTorchWorkload, error) {
	
	var workloads []*PyTorchWorkload

	if p.redisClient != nil {
		pattern := fmt.Sprintf("pytorch:workload:%s:*", namespace)
		keys, err := p.redisClient.Keys(ctx, pattern).Result()
		if err != nil {
			return nil, err
		}

		for _, key := range keys {
			data, err := p.redisClient.Get(ctx, key).Result()
			if err != nil {
				continue
			}

			var workload PyTorchWorkload
			if err := json.Unmarshal([]byte(data), &workload); err == nil {
				workloads = append(workloads, &workload)
			}
		}
	}

	return workloads, nil
}

// UpdateWorkload updates a workload
func (p *PyTorchWorkloadManager) UpdateWorkload(ctx context.Context, 
	workload *PyTorchWorkload) error {
	
	workload.UpdatedAt = time.Now()
	return p.storeWorkload(ctx, workload)
}

// DeleteWorkload deletes a workload
func (p *PyTorchWorkloadManager) DeleteWorkload(ctx context.Context, namespace, 
	name string) error {
	
	p.logger.Info("Deleting PyTorch workload", 
		zap.String("namespace", namespace), 
		zap.String("name", name))

	// Delete Kubernetes resources
	if err := p.deleteKubernetesResources(ctx, namespace, name); err != nil {
		p.logger.Warn("Failed to delete Kubernetes resources", zap.Error(err))
	}

	// Delete from cache
	if p.redisClient != nil {
		key := fmt.Sprintf("pytorch:workload:%s:%s", namespace, name)
		p.redisClient.Del(ctx, key)
	}

	return nil
}

// Helper functions

func (p *PyTorchWorkloadManager) validateWorkloadSpec(spec *PyTorchWorkloadSpec) error {
	if spec.Name == "" {
		return fmt.Errorf("workload name is required")
	}
	if spec.Namespace == "" {
		return fmt.Errorf("workload namespace is required")
	}
	if spec.ModelName == "" {
		return fmt.Errorf("model name is required")
	}
	
	// Validate PyTorch version
	if spec.PyTorchVersion != "" {
		supported := false
		for _, version := range p.config.SupportedVersions {
			if strings.HasPrefix(spec.PyTorchVersion, version) {
				supported = true
				break
			}
		}
		if !supported {
			return fmt.Errorf("unsupported PyTorch version: %s", spec.PyTorchVersion)
		}
	}

	return nil
}

func (p *PyTorchWorkloadManager) applyDefaults(workload *PyTorchWorkload) {
	// Apply default PyTorch version
	if workload.PyTorchVersion == "" {
		workload.PyTorchVersion = p.config.SupportedVersions[len(p.config.SupportedVersions)-1]
	}

	// Apply default CUDA version
	if workload.CUDAVersion == "" {
		workload.CUDAVersion = "12.1"
	}

	// Apply default resources
	if workload.Resources.CPURequest == "" {
		workload.Resources = p.config.DefaultResources
	}

	// Apply default checkpoint config for training
	if workload.WorkloadType == "training" && workload.CheckpointConfig == nil {
		workload.CheckpointConfig = &CheckpointConfig{
			Enabled:           true,
			Interval:          p.config.CheckpointInterval,
			StorageClass:      "fast-ssd",
			RetentionPolicy:   "keep-last-5",
			CompressionLevel:  6,
			AsyncSave:         true,
			VerifyCheckpoints: true,
		}
	}

	// Apply default monitoring config
	if workload.MonitoringConfig == nil {
		workload.MonitoringConfig = &MonitoringConfig{
			MetricsEnabled:   true,
			LogLevel:         "INFO",
			ProfilerEnabled:  p.config.ProfilerConfig.Enabled,
			TracingEnabled:   true,
			AlertsEnabled:    true,
			DashboardEnabled: true,
			MetricsInterval:  30 * time.Second,
		}
	}

	// Apply default environment variables
	if workload.Environment == nil {
		workload.Environment = make(map[string]string)
	}
	
	// Set PyTorch-specific environment variables
	workload.Environment["PYTORCH_VERSION"] = workload.PyTorchVersion
	workload.Environment["CUDA_VERSION"] = workload.CUDAVersion
	workload.Environment["NCCL_DEBUG"] = "INFO"
	workload.Environment["NCCL_TREE_THRESHOLD"] = "0"
	
	if p.config.MixedPrecision {
		workload.Environment["PYTORCH_MIXED_PRECISION"] = "1"
	}
	
	if p.config.GradientCompression {
		workload.Environment["PYTORCH_GRADIENT_COMPRESSION"] = "1"
	}
}

func (p *PyTorchWorkloadManager) optimizeForCarbon(ctx context.Context, 
	workload *PyTorchWorkload) error {
	
	// Get current carbon intensity
	intensity, err := p.getCurrentCarbonIntensity(ctx)
	if err != nil {
		return err
	}

	// Check if carbon constraints are met
	if workload.CarbonConstraints != nil {
		if intensity > workload.CarbonConstraints.CarbonIntensityMax {
			// Defer workload to optimal window
			return p.scheduleForOptimalWindow(ctx, workload)
		}
	}

	// Optimize resource allocation for carbon efficiency
	p.optimizeResourcesForCarbon(workload, intensity)

	return nil
}

func (p *PyTorchWorkloadManager) optimizeForInference(ctx context.Context, 
	workload *PyTorchWorkload) error {
	
	// Enable inference optimizations
	if workload.Environment == nil {
		workload.Environment = make(map[string]string)
	}

	// Set inference-specific optimizations
	workload.Environment["PYTORCH_JIT"] = "1"
	workload.Environment["PYTORCH_TENSORRT"] = "1"
	workload.Environment["PYTORCH_INFERENCE_MODE"] = "1"

	// Optimize batch size for throughput
	if workload.Hyperparameters == nil {
		workload.Hyperparameters = make(map[string]interface{})
	}

	// Set optimal batch size based on GPU memory
	gpuMemoryGB := p.parseMemoryString(workload.Resources.GPUMemory)
	optimalBatchSize := p.calculateOptimalBatchSize(gpuMemoryGB, workload.ModelName)
	workload.Hyperparameters["batch_size"] = optimalBatchSize

	return nil
}

func (p *PyTorchWorkloadManager) createKubernetesResources(ctx context.Context, 
	workload *PyTorchWorkload) error {
	
	// Create ConfigMap for workload configuration
	configMap := p.createWorkloadConfigMap(workload)
	_, err := p.kubeClient.CoreV1().ConfigMaps(workload.Namespace).Create(ctx, configMap, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("failed to create ConfigMap: %w", err)
	}

	// Create Job for training workload
	if workload.WorkloadType == "training" {
		job := p.createTrainingJob(workload)
		_, err = p.kubeClient.BatchV1().Jobs(workload.Namespace).Create(ctx, job, metav1.CreateOptions{})
		if err != nil {
			return fmt.Errorf("failed to create Job: %w", err)
		}
	}

	return nil
}

func (p *PyTorchWorkloadManager) createInferenceResources(ctx context.Context, 
	workload *PyTorchWorkload) error {
	
	// Create Deployment for inference workload
	deployment := p.createInferenceDeployment(workload)
	_, err := p.kubeClient.AppsV1().Deployments(workload.Namespace).Create(ctx, deployment, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("failed to create Deployment: %w", err)
	}

	// Create Service for inference endpoint
	service := p.createInferenceService(workload)
	_, err = p.kubeClient.CoreV1().Services(workload.Namespace).Create(ctx, service, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("failed to create Service: %w", err)
	}

	return nil
}

func (p *PyTorchWorkloadManager) storeWorkload(ctx context.Context, 
	workload *PyTorchWorkload) error {
	
	if p.redisClient == nil {
		return nil
	}

	data, err := json.Marshal(workload)
	if err != nil {
		return err
	}

	key := fmt.Sprintf("pytorch:workload:%s:%s", workload.Namespace, workload.Name)
	return p.redisClient.Set(ctx, key, data, 24*time.Hour).Err()
}

func (p *PyTorchWorkloadManager) monitorWorkload(ctx context.Context, 
	workload *PyTorchWorkload) {
	
	ticker := time.NewTicker(workload.MonitoringConfig.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := p.updateWorkloadMetrics(ctx, workload); err != nil {
				p.logger.Warn("Failed to update workload metrics", 
					zap.String("workload", workload.Name), 
					zap.Error(err))
			}
		}
	}
}

func (p *PyTorchWorkloadManager) updateWorkloadMetrics(ctx context.Context, 
	workload *PyTorchWorkload) error {
	
	// Get current metrics from Prometheus/DCGM
	metrics, err := p.collectWorkloadMetrics(ctx, workload)
	if err != nil {
		return err
	}

	workload.Metrics = metrics
	workload.UpdatedAt = time.Now()

	// Update status based on metrics
	p.updateWorkloadStatus(workload, metrics)

	// Store updated workload
	return p.storeWorkload(ctx, workload)
}

// Additional helper functions would be implemented here...
// This includes functions for:
// - createWorkloadConfigMap
// - createTrainingJob  
// - createInferenceDeployment
// - createInferenceService
// - deleteKubernetesResources
// - getCurrentCarbonIntensity
// - scheduleForOptimalWindow
// - optimizeResourcesForCarbon
// - parseMemoryString
// - calculateOptimalBatchSize
// - collectWorkloadMetrics
// - updateWorkloadStatus

// PyTorchWorkloadSpec defines the specification for creating a PyTorch workload
type PyTorchWorkloadSpec struct {
	Name              string                 `json:"name"`
	Namespace         string                 `json:"namespace"`
	ModelName         string                 `json:"modelName"`
	ModelVersion      string                 `json:"modelVersion"`
	DatasetName       string                 `json:"datasetName"`
	DatasetVersion    string                 `json:"datasetVersion"`
	PyTorchVersion    string                 `json:"pytorchVersion"`
	CUDAVersion       string                 `json:"cudaVersion"`
	Resources         ResourceLimits         `json:"resources"`
	DistributedConfig *DistributedConfig     `json:"distributedConfig"`
	Hyperparameters   map[string]interface{} `json:"hyperparameters"`
	Environment       map[string]string      `json:"environment"`
	CarbonConstraints *CarbonConstraints     `json:"carbonConstraints"`
	SLARequirements   *SLARequirements       `json:"slaRequirements"`
	CheckpointConfig  *CheckpointConfig      `json:"checkpointConfig"`
	MonitoringConfig  *MonitoringConfig      `json:"monitoringConfig"`
}

// PyTorchInferenceSpec defines the specification for creating a PyTorch inference workload
type PyTorchInferenceSpec struct {
	Name              string             `json:"name"`
	Namespace         string             `json:"namespace"`
	ModelName         string             `json:"modelName"`
	ModelVersion      string             `json:"modelVersion"`
	PyTorchVersion    string             `json:"pytorchVersion"`
	CUDAVersion       string             `json:"cudaVersion"`
	Resources         ResourceLimits     `json:"resources"`
	Environment       map[string]string  `json:"environment"`
	CarbonConstraints *CarbonConstraints `json:"carbonConstraints"`
	SLARequirements   *SLARequirements   `json:"slaRequirements"`
	MonitoringConfig  *MonitoringConfig  `json:"monitoringConfig"`
	Replicas          int32              `json:"replicas"`
	AutoScaling       *AutoScalingConfig `json:"autoScaling"`
}

// AutoScalingConfig defines auto-scaling configuration
type AutoScalingConfig struct {
	Enabled                bool    `json:"enabled"`
	MinReplicas            int32   `json:"minReplicas"`
	MaxReplicas            int32   `json:"maxReplicas"`
	TargetCPUUtilization   int32   `json:"targetCPUUtilization"`
	TargetGPUUtilization   int32   `json:"targetGPUUtilization"`
	TargetLatency          float64 `json:"targetLatency"`          // Target latency in ms
	ScaleUpCooldown        int32   `json:"scaleUpCooldown"`        // Scale up cooldown in seconds
	ScaleDownCooldown      int32   `json:"scaleDownCooldown"`      // Scale down cooldown in seconds
}