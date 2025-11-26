package evaluation

import (
	"context"
	"fmt"
	"log"
	"time"
)

// EvaluationFramework orchestrates scientific evaluations
type EvaluationFramework struct {
	config           *FrameworkConfig
	baselineManager  *BaselineManager
	statisticalAnalyzer *StatisticalAnalyzer
	ablationManager  *AblationStudyManager
	reproManager     *ReproducibilityManager
	validator        *ReproducibilityValidator
	artifactStore    ArtifactStore
}

// FrameworkConfig holds evaluation framework configuration
type FrameworkConfig struct {
	BaselineStorage     string                 `json:"baselineStorage"`
	ExperimentStorage   string                 `json:"experimentStorage"`
	ResultsStorage      string                 `json:"resultsStorage"`
	StatisticalConfig   StatisticalConfig      `json:"statisticalConfig"`
	ValidationConfig    ValidationConfig       `json:"validationConfig"`
	ReproducibilityConfig ReproducibilityConfig `json:"reproducibilityConfig"`
	MetricsConfig       MetricsConfig          `json:"metricsConfig"`
	ReportingConfig     ReportingConfig        `json:"reportingConfig"`
}

// StatisticalConfig defines statistical analysis parameters
type StatisticalConfig struct {
	ConfidenceLevel     float64 `json:"confidenceLevel"`
	SignificanceLevel   float64 `json:"significanceLevel"`
	MinSampleSize       int     `json:"minSampleSize"`
	MaxSampleSize       int     `json:"maxSampleSize"`
	PowerAnalysis       bool    `json:"powerAnalysis"`
	EffectSizeThreshold float64 `json:"effectSizeThreshold"`
	MultipleComparisons string  `json:"multipleComparisons"`
	OutlierDetection    bool    `json:"outlierDetection"`
	NormalityTesting    bool    `json:"normalityTesting"`
}

// MetricsConfig defines metrics configuration
type MetricsConfig struct {
	PrimaryMetrics      []string          `json:"primaryMetrics"`
	SecondaryMetrics    []string          `json:"secondaryMetrics"`
	CustomMetrics       []CustomMetric    `json:"customMetrics"`
	AggregationMethods  []string          `json:"aggregationMethods"`
	NormalizationMethod string            `json:"normalizationMethod"`
	WeightingScheme     map[string]float64 `json:"weightingScheme"`
	ThresholdValues     map[string]float64 `json:"thresholdValues"`
	ComparisonMethods   []string          `json:"comparisonMethods"`
}

// CustomMetric defines a custom metric
type CustomMetric struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Formula     string                 `json:"formula"`
	Unit        string                 `json:"unit"`
	Type        string                 `json:"type"`
	Direction   string                 `json:"direction"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// ReportingConfig defines reporting configuration
type ReportingConfig struct {
	ReportFormats       []string `json:"reportFormats"`
	IncludeCharts       bool     `json:"includeCharts"`
	IncludeStatistics   bool     `json:"includeStatistics"`
	IncludeBaselines    bool     `json:"includeBaselines"`
	IncludeAblations    bool     `json:"includeAblations"`
	DetailLevel         string   `json:"detailLevel"`
	AutoGeneration      bool     `json:"autoGeneration"`
	TemplateCustomization bool   `json:"templateCustomization"`
}

// EvaluationRequest represents a request for evaluation
type EvaluationRequest struct {
	ID                  string                 `json:"id"`
	Name                string                 `json:"name"`
	Description         string                 `json:"description"`
	ExperimentConfig    ExperimentConfig       `json:"experimentConfig"`
	BaselineConfigs     []BaselineConfig       `json:"baselineConfigs"`
	AblationConfig      *AblationConfig        `json:"ablationConfig"`
	StatisticalTests    []string               `json:"statisticalTests"`
	ReproducibilityReqs ReproducibilityRequirements `json:"reproducibilityReqs"`
	Metadata            map[string]interface{} `json:"metadata"`
	CreatedAt           time.Time              `json:"createdAt"`
}

// ExperimentConfig defines experiment configuration
type ExperimentConfig struct {
	DataSources         []DataSource           `json:"dataSources"`
	ModelConfig         ModelConfig            `json:"modelConfig"`
	Metrics             []MetricConfig         `json:"metrics"`
	ResourceRequirements ResourceRequirements  `json:"resourceRequirements"`
	RetryPolicy         RetryPolicy            `json:"retryPolicy"`
	CheckpointConfig    CheckpointConfig       `json:"checkpointConfig"`
	MonitoringConfig    MonitoringConfig       `json:"monitoringConfig"`
	Parameters          map[string]interface{} `json:"parameters"`
}

// DataSource defines a data source
type DataSource struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Location    string                 `json:"location"`
	Format      string                 `json:"format"`
	Schema      map[string]interface{} `json:"schema"`
	Validation  DataValidationConfig   `json:"validation"`
	Preprocessing []PreprocessingStep  `json:"preprocessing"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// ModelConfig defines model configuration
type ModelConfig struct {
	Type            string                 `json:"type"`
	Architecture    string                 `json:"architecture"`
	Parameters      map[string]interface{} `json:"parameters"`
	TrainingConfig  TrainingConfig         `json:"trainingConfig"`
	InferenceConfig InferenceConfig        `json:"inferenceConfig"`
	Optimization    OptimizationConfig     `json:"optimization"`
	Regularization  RegularizationConfig   `json:"regularization"`
}

// MetricConfig defines metric configuration
type MetricConfig struct {
	Name        string                 `json:"name"`
	Type        string                 `json:"type"`
	Parameters  map[string]interface{} `json:"parameters"`
	Aggregation string                 `json:"aggregation"`
	Weight      float64                `json:"weight"`
	Threshold   *float64               `json:"threshold"`
	Direction   string                 `json:"direction"`
}

// ResourceRequirements defines resource requirements
type ResourceRequirements struct {
	CPU     int               `json:"cpu"`
	Memory  string            `json:"memory"`
	GPU     int               `json:"gpu"`
	Storage string            `json:"storage"`
	Network string            `json:"network"`
	CustomResources map[string]string `json:"customResources"`
}

// RetryPolicy defines retry policy
type RetryPolicy struct {
	MaxRetries    int           `json:"maxRetries"`
	BackoffFactor float64       `json:"backoffFactor"`
	MaxDelay      time.Duration `json:"maxDelay"`
	RetryableErrors []string    `json:"retryableErrors"`
}

// CheckpointConfig defines checkpoint configuration
type CheckpointConfig struct {
	Enabled   bool          `json:"enabled"`
	Frequency time.Duration `json:"frequency"`
	Location  string        `json:"location"`
	Retention int           `json:"retention"`
}

// MonitoringConfig defines monitoring configuration
type MonitoringConfig struct {
	Enabled     bool          `json:"enabled"`
	Interval    time.Duration `json:"interval"`
	Metrics     []string      `json:"metrics"`
	Alerts      []AlertConfig `json:"alerts"`
	Notifications []NotificationConfig `json:"notifications"`
}

// AlertConfig defines alert configuration
type AlertConfig struct {
	Name      string                 `json:"name"`
	Condition string                 `json:"condition"`
	Threshold float64                `json:"threshold"`
	Severity  string                 `json:"severity"`
	Actions   []string               `json:"actions"`
	Parameters map[string]interface{} `json:"parameters"`
}

// NotificationConfig defines notification configuration
type NotificationConfig struct {
	Type       string                 `json:"type"`
	Target     string                 `json:"target"`
	Template   string                 `json:"template"`
	Conditions []string               `json:"conditions"`
	Parameters map[string]interface{} `json:"parameters"`
}

// DataValidationConfig defines data validation configuration
type DataValidationConfig struct {
	Enabled       bool     `json:"enabled"`
	SchemaCheck   bool     `json:"schemaCheck"`
	QualityChecks []string `json:"qualityChecks"`
	Constraints   []string `json:"constraints"`
	SampleSize    int      `json:"sampleSize"`
}

// PreprocessingStep defines a preprocessing step
type PreprocessingStep struct {
	Name       string                 `json:"name"`
	Type       string                 `json:"type"`
	Parameters map[string]interface{} `json:"parameters"`
	Order      int                    `json:"order"`
	Enabled    bool                   `json:"enabled"`
}

// TrainingConfig defines training configuration
type TrainingConfig struct {
	Epochs          int                    `json:"epochs"`
	BatchSize       int                    `json:"batchSize"`
	LearningRate    float64                `json:"learningRate"`
	Optimizer       string                 `json:"optimizer"`
	LossFunction    string                 `json:"lossFunction"`
	ValidationSplit float64                `json:"validationSplit"`
	EarlyStopping   bool                   `json:"earlyStopping"`
	Parameters      map[string]interface{} `json:"parameters"`
}

// InferenceConfig defines inference configuration
type InferenceConfig struct {
	BatchSize   int                    `json:"batchSize"`
	Precision   string                 `json:"precision"`
	Optimization string                `json:"optimization"`
	Caching     bool                   `json:"caching"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// OptimizationConfig defines optimization configuration
type OptimizationConfig struct {
	Method     string                 `json:"method"`
	Goals      []OptimizationGoal     `json:"goals"`
	Constraints []string              `json:"constraints"`
	Budget     int                    `json:"budget"`
	Parameters map[string]interface{} `json:"parameters"`
}

// OptimizationGoal defines an optimization goal
type OptimizationGoal struct {
	Metric    string  `json:"metric"`
	Direction string  `json:"direction"`
	Weight    float64 `json:"weight"`
	Target    *float64 `json:"target"`
}

// RegularizationConfig defines regularization configuration
type RegularizationConfig struct {
	L1Lambda    float64                `json:"l1Lambda"`
	L2Lambda    float64                `json:"l2Lambda"`
	Dropout     float64                `json:"dropout"`
	BatchNorm   bool                   `json:"batchNorm"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// ReproducibilityRequirements defines reproducibility requirements
type ReproducibilityRequirements struct {
	SeedManagement     bool   `json:"seedManagement"`
	EnvironmentCapture bool   `json:"environmentCapture"`
	DependencyTracking bool   `json:"dependencyTracking"`
	DataVersioning     bool   `json:"dataVersioning"`
	CodeVersioning     bool   `json:"codeVersioning"`
	ConfigVersioning   bool   `json:"configVersioning"`
	ResultsArchiving   bool   `json:"resultsArchiving"`
	ArtifactGeneration bool   `json:"artifactGeneration"`
	DocumentationLevel string `json:"documentationLevel"`
}

// EvaluationResult represents the result of an evaluation
type EvaluationResult struct {
	ID                  string                 `json:"id"`
	RequestID           string                 `json:"requestId"`
	Status              string                 `json:"status"`
	Summary             EvaluationSummary      `json:"summary"`
	BaselineResults     []BaselineResult       `json:"baselineResults"`
	StatisticalAnalysis StatisticalAnalysis    `json:"statisticalAnalysis"`
	AblationResults     *AblationResults       `json:"ablationResults"`
	ReproducibilityReport ReproducibilityReport `json:"reproducibilityReport"`
	Artifacts           []string               `json:"artifacts"`
	Recommendations     []string               `json:"recommendations"`
	Metadata            map[string]interface{} `json:"metadata"`
	CreatedAt           time.Time              `json:"createdAt"`
	CompletedAt         *time.Time             `json:"completedAt"`
}

// EvaluationSummary provides a summary of evaluation results
type EvaluationSummary struct {
	TotalExperiments    int                    `json:"totalExperiments"`
	SuccessfulRuns      int                    `json:"successfulRuns"`
	FailedRuns          int                    `json:"failedRuns"`
	TotalDuration       time.Duration          `json:"totalDuration"`
	BestPerformance     map[string]float64     `json:"bestPerformance"`
	AveragePerformance  map[string]float64     `json:"averagePerformance"`
	StatisticalSignificance bool               `json:"statisticalSignificance"`
	EffectSize          float64                `json:"effectSize"`
	ConfidenceLevel     float64                `json:"confidenceLevel"`
	QualityScore        float64                `json:"qualityScore"`
	ReproducibilityScore float64               `json:"reproducibilityScore"`
	KeyFindings         []string               `json:"keyFindings"`
	Limitations         []string               `json:"limitations"`
}

// ReproducibilityReport provides reproducibility information
type ReproducibilityReport struct {
	EnvironmentSnapshot EnvironmentSnapshot    `json:"environmentSnapshot"`
	ArtifactManifest    []ArtifactInfo         `json:"artifactManifest"`
	ValidationResults   ValidationResult       `json:"validationResults"`
	ReproducibilityScore float64               `json:"reproducibilityScore"`
	Issues              []string               `json:"issues"`
	Recommendations     []string               `json:"recommendations"`
	Metadata            map[string]interface{} `json:"metadata"`
}

// NewEvaluationFramework creates a new evaluation framework
func NewEvaluationFramework(config *FrameworkConfig) (*EvaluationFramework, error) {
	if config == nil {
		config = &FrameworkConfig{
			BaselineStorage:   "./baselines",
			ExperimentStorage: "./experiments",
			ResultsStorage:    "./results",
			StatisticalConfig: StatisticalConfig{
				ConfidenceLevel:     0.95,
				SignificanceLevel:   0.05,
				MinSampleSize:       10,
				MaxSampleSize:       1000,
				PowerAnalysis:       true,
				EffectSizeThreshold: 0.2,
				MultipleComparisons: "bonferroni",
				OutlierDetection:    true,
				NormalityTesting:    true,
			},
			ValidationConfig: ValidationConfig{
				StrictMode:          true,
				ChecksumValidation:  true,
				EnvironmentValidation: true,
				DependencyValidation: true,
				SeedValidation:      true,
				DataValidation:      true,
				CodeValidation:      true,
				ResultValidation:    true,
			},
			ReproducibilityConfig: ReproducibilityConfig{
				SeedManagement:     true,
				EnvironmentCapture: true,
				DependencyTracking: true,
				DataVersioning:     true,
				CodeVersioning:     true,
				ConfigVersioning:   true,
				ResultsArchiving:   true,
				ArtifactGeneration: true,
				DocumentationLevel: "comprehensive",
			},
		}
	}

	// Initialize components
	baselineManager := NewBaselineManager(&BaselineConfig{
		StoragePath:     config.BaselineStorage,
		CacheSize:       100,
		CompressionEnabled: true,
		EncryptionEnabled:  false,
	})

	statisticalAnalyzer := NewStatisticalAnalyzer(&StatisticalAnalyzerConfig{
		ConfidenceLevel:   config.StatisticalConfig.ConfidenceLevel,
		SignificanceLevel: config.StatisticalConfig.SignificanceLevel,
		MinSampleSize:     config.StatisticalConfig.MinSampleSize,
		PowerAnalysis:     config.StatisticalConfig.PowerAnalysis,
		OutlierDetection:  config.StatisticalConfig.OutlierDetection,
		NormalityTesting:  config.StatisticalConfig.NormalityTesting,
	})

	ablationManager := NewAblationStudyManager(&AblationConfig{
		MaxComponents:     50,
		MaxInteractions:   100,
		ParallelExecution: true,
		CacheResults:      true,
		ValidationStrategy: ValidationStrategy{
			CrossValidation: CrossValidationConfig{
				Enabled: true,
				KFolds:  5,
			},
		},
	})

	reproManager := NewReproducibilityManager(&ReproducibilityConfig{
		ArtifactStorage: ArtifactStorageConfig{
			BasePath:    config.ResultsStorage + "/artifacts",
			Compression: true,
			Encryption:  false,
		},
		EnvironmentCapture: EnvironmentCaptureConfig{
			SystemInfo:    true,
			HardwareInfo:  true,
			SoftwareInfo:  true,
			NetworkInfo:   true,
			ContainerInfo: true,
		},
		SeedManagement: SeedManagementConfig{
			AutoGeneration: true,
			Validation:     true,
			Storage:        true,
		},
	})

	validator := NewReproducibilityValidator(&ValidationConfig{
		StrictMode:          config.ValidationConfig.StrictMode,
		ChecksumValidation:  config.ValidationConfig.ChecksumValidation,
		EnvironmentValidation: config.ValidationConfig.EnvironmentValidation,
		DependencyValidation: config.ValidationConfig.DependencyValidation,
		SeedValidation:      config.ValidationConfig.SeedValidation,
		DataValidation:      config.ValidationConfig.DataValidation,
		CodeValidation:      config.ValidationConfig.CodeValidation,
		ResultValidation:    config.ValidationConfig.ResultValidation,
	})

	artifactStore, err := NewFileSystemArtifactStore(&FileSystemStoreConfig{
		BasePath:    config.ResultsStorage + "/artifacts",
		Permissions: 0755,
		SyncWrites:  true,
		BufferSize:  8192,
		Compression: true,
		Encryption:  false,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create artifact store: %w", err)
	}

	return &EvaluationFramework{
		config:              config,
		baselineManager:     baselineManager,
		statisticalAnalyzer: statisticalAnalyzer,
		ablationManager:     ablationManager,
		reproManager:        reproManager,
		validator:           validator,
		artifactStore:       artifactStore,
	}, nil
}

// RunEvaluation executes a complete scientific evaluation
func (f *EvaluationFramework) RunEvaluation(ctx context.Context, request *EvaluationRequest) (*EvaluationResult, error) {
	log.Printf("Starting evaluation: %s", request.ID)
	
	result := &EvaluationResult{
		ID:        fmt.Sprintf("eval_%d", time.Now().Unix()),
		RequestID: request.ID,
		Status:    "running",
		CreatedAt: time.Now(),
	}

	// Step 1: Create reproducibility snapshot
	snapshot, err := f.reproManager.CaptureEnvironment(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to capture environment: %w", err)
	}

	// Step 2: Run baseline evaluations
	var baselineResults []BaselineResult
	for _, baselineConfig := range request.BaselineConfigs {
		baselineResult, err := f.baselineManager.EvaluateBaseline(ctx, &baselineConfig)
		if err != nil {
			log.Printf("Baseline evaluation failed: %v", err)
			continue
		}
		baselineResults = append(baselineResults, *baselineResult)
	}

	// Step 3: Perform statistical analysis
	statisticalAnalysis, err := f.statisticalAnalyzer.AnalyzeExperiment(ctx, &Experiment{
		ID:          request.ID,
		Name:        request.Name,
		Description: request.Description,
		Baselines:   baselineResults,
	})
	if err != nil {
		return nil, fmt.Errorf("statistical analysis failed: %w", err)
	}

	// Step 4: Run ablation studies if configured
	var ablationResults *AblationResults
	if request.AblationConfig != nil {
		study, err := f.ablationManager.CreateAblationStudy(ctx, request.AblationConfig)
		if err != nil {
			log.Printf("Failed to create ablation study: %v", err)
		} else {
			ablationResults, err = f.ablationManager.RunAblationStudy(ctx, study.ID)
			if err != nil {
				log.Printf("Ablation study failed: %v", err)
			}
		}
	}

	// Step 5: Create and store artifacts
	artifact := &Artifact{
		ID:          fmt.Sprintf("artifact_%d", time.Now().Unix()),
		Type:        "evaluation_results",
		Name:        fmt.Sprintf("evaluation_%s", request.ID),
		Description: "Complete evaluation results and reproducibility artifacts",
		Data:        result,
		Metadata: map[string]interface{}{
			"evaluation_id": request.ID,
			"timestamp":     time.Now(),
			"version":       "1.0",
		},
		CreatedAt: time.Now(),
	}

	artifactID, err := f.artifactStore.Store(ctx, artifact)
	if err != nil {
		log.Printf("Failed to store artifact: %v", err)
	}

	// Step 6: Validate reproducibility
	validationResult, err := f.validator.ValidateArtifact(ctx, artifactID)
	if err != nil {
		log.Printf("Validation failed: %v", err)
	}

	// Step 7: Generate summary and recommendations
	summary := f.generateSummary(baselineResults, statisticalAnalysis, ablationResults)
	recommendations := f.generateRecommendations(statisticalAnalysis, ablationResults, validationResult)

	// Complete the result
	completedAt := time.Now()
	result.Status = "completed"
	result.Summary = summary
	result.BaselineResults = baselineResults
	result.StatisticalAnalysis = *statisticalAnalysis
	result.AblationResults = ablationResults
	result.ReproducibilityReport = ReproducibilityReport{
		EnvironmentSnapshot:  *snapshot,
		ValidationResults:    *validationResult,
		ReproducibilityScore: validationResult.Score,
	}
	result.Artifacts = []string{artifactID}
	result.Recommendations = recommendations
	result.CompletedAt = &completedAt

	log.Printf("Evaluation completed: %s", request.ID)
	return result, nil
}

// generateSummary creates an evaluation summary
func (f *EvaluationFramework) generateSummary(baselines []BaselineResult, stats *StatisticalAnalysis, ablations *AblationResults) EvaluationSummary {
	summary := EvaluationSummary{
		TotalExperiments:     len(baselines),
		SuccessfulRuns:       0,
		FailedRuns:          0,
		BestPerformance:     make(map[string]float64),
		AveragePerformance:  make(map[string]float64),
		ConfidenceLevel:     stats.ConfidenceLevel,
		KeyFindings:         []string{},
		Limitations:         []string{},
	}

	// Calculate success/failure rates
	for _, baseline := range baselines {
		if baseline.Status == "completed" {
			summary.SuccessfulRuns++
		} else {
			summary.FailedRuns++
		}
	}

	// Extract key findings from statistical analysis
	if stats.HypothesisTests != nil && len(stats.HypothesisTests) > 0 {
		for _, test := range stats.HypothesisTests {
			if test.Significant {
				summary.KeyFindings = append(summary.KeyFindings, 
					fmt.Sprintf("Significant %s result (p=%.4f)", test.TestName, test.PValue))
			}
		}
	}

	// Add ablation findings if available
	if ablations != nil && len(ablations.ComponentAnalyses) > 0 {
		summary.KeyFindings = append(summary.KeyFindings, 
			fmt.Sprintf("Ablation study identified %d critical components", len(ablations.ComponentAnalyses)))
	}

	return summary
}

// generateRecommendations creates recommendations based on results
func (f *EvaluationFramework) generateRecommendations(stats *StatisticalAnalysis, ablations *AblationResults, validation *ValidationResult) []string {
	var recommendations []string

	// Statistical recommendations
	if stats.EffectSizes != nil && len(stats.EffectSizes) > 0 {
		for _, effect := range stats.EffectSizes {
			if effect.Magnitude == "small" {
				recommendations = append(recommendations, 
					fmt.Sprintf("Consider increasing sample size for %s to detect small effects", effect.Comparison))
			}
		}
	}

	// Ablation recommendations
	if ablations != nil && len(ablations.ComponentAnalyses) > 0 {
		recommendations = append(recommendations, 
			"Focus optimization efforts on components identified as critical in ablation study")
	}

	// Reproducibility recommendations
	if validation.Score < 0.8 {
		recommendations = append(recommendations, 
			"Improve reproducibility by addressing validation issues")
	}

	// Default recommendations
	if len(recommendations) == 0 {
		recommendations = append(recommendations, 
			"Results appear robust. Consider expanding evaluation to additional scenarios.")
	}

	return recommendations
}