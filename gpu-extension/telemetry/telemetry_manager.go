package telemetry

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	"go.uber.org/zap"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// TelemetryManager orchestrates GPU carbon telemetry collection and reporting
type TelemetryManager struct {
	kubeClient       kubernetes.Interface
	redisClient      *redis.Client
	logger           *zap.Logger
	calculator       *GPUCarbonCalculator
	exporter         *PrometheusExporter
	
	// Configuration
	config           *TelemetryConfig
	
	// State management
	activeWorkloads  map[string]*WorkloadTracker
	zoneMetrics      map[string]*ZoneMetrics
	mutex            sync.RWMutex
	
	// Channels for coordination
	stopCh           chan struct{}
	workloadCh       chan *WorkloadEvent
}

// TelemetryConfig holds telemetry collection configuration
type TelemetryConfig struct {
	CollectionInterval    time.Duration `json:"collectionInterval"`    // How often to collect metrics
	AggregationInterval   time.Duration `json:"aggregationInterval"`   // How often to aggregate zone metrics
	RetentionPeriod       time.Duration `json:"retentionPeriod"`       // How long to keep metrics
	DCGMEndpoint          string        `json:"dcgmEndpoint"`          // DCGM exporter endpoint
	PrometheusPort        int           `json:"prometheusPort"`        // Prometheus metrics port
	EnableForecast        bool          `json:"enableForecast"`        // Enable carbon forecasting
	EnableOptimization    bool          `json:"enableOptimization"`    // Enable workload optimization
	AlertThresholds       AlertThresholds `json:"alertThresholds"`     // Alert thresholds
}

// AlertThresholds defines thresholds for carbon and power alerts
type AlertThresholds struct {
	CarbonRateThreshold   float64 `json:"carbonRateThreshold"`   // gCO2/hour threshold
	PowerThreshold        float64 `json:"powerThreshold"`        // Watts threshold
	EfficiencyThreshold   float64 `json:"efficiencyThreshold"`   // gCO2/FLOP threshold
	UtilizationThreshold  float64 `json:"utilizationThreshold"`  // GPU utilization % threshold
	SLAViolationThreshold int     `json:"slaViolationThreshold"` // Number of violations threshold
}

// WorkloadTracker tracks metrics for an active GPU workload
type WorkloadTracker struct {
	WorkloadID       string                 `json:"workloadId"`
	NodeName         string                 `json:"nodeName"`
	Zone             string                 `json:"zone"`
	GPUType          string                 `json:"gpuType"`
	WorkloadType     string                 `json:"workloadType"`
	StartTime        time.Time              `json:"startTime"`
	LastUpdate       time.Time              `json:"lastUpdate"`
	TotalCarbon      float64                `json:"totalCarbon"`      // Cumulative gCO2
	TotalEnergy      float64                `json:"totalEnergy"`      // Cumulative kWh
	CurrentMetrics   *GPUCarbonMetrics      `json:"currentMetrics"`
	MetricsHistory   []*GPUCarbonMetrics    `json:"metricsHistory"`
	SLARequirements  *SLARequirements       `json:"slaRequirements"`
	Alerts           []Alert                `json:"alerts"`
}

// ZoneMetrics aggregates metrics for a zone
type ZoneMetrics struct {
	Zone                string                    `json:"zone"`
	LastUpdate          time.Time                 `json:"lastUpdate"`
	TotalCarbonRate     float64                   `json:"totalCarbonRate"`     // gCO2/hour
	TotalPowerConsumption float64                 `json:"totalPowerConsumption"` // Watts
	ActiveWorkloads     int                       `json:"activeWorkloads"`
	GPUCounts           map[string]int            `json:"gpuCounts"`           // GPU type -> count
	AverageEfficiency   float64                   `json:"averageEfficiency"`   // gCO2/FLOP
	CarbonIntensity     float64                   `json:"carbonIntensity"`     // gCO2/kWh
	PUE                 float64                   `json:"pue"`
	Forecast            []float64                 `json:"forecast"`            // 4-hour forecast
}

// WorkloadEvent represents workload lifecycle events
type WorkloadEvent struct {
	Type        string    `json:"type"`        // start, update, stop
	WorkloadID  string    `json:"workloadId"`
	NodeName    string    `json:"nodeName"`
	GPUMetrics  map[string]interface{} `json:"gpuMetrics"`
	Timestamp   time.Time `json:"timestamp"`
}

// SLARequirements defines SLA requirements for a workload
type SLARequirements struct {
	MaxCarbonBudget    float64       `json:"maxCarbonBudget"`    // gCO2 budget
	MaxPowerBudget     float64       `json:"maxPowerBudget"`     // Watts budget
	MaxLatency         time.Duration `json:"maxLatency"`         // Response time SLA
	MinThroughput      float64       `json:"minThroughput"`      // Minimum throughput
	MaxCostBudget      float64       `json:"maxCostBudget"`      // Cost budget
}

// Alert represents a carbon/power alert
type Alert struct {
	ID          string    `json:"id"`
	Type        string    `json:"type"`        // carbon, power, efficiency, sla
	Severity    string    `json:"severity"`    // info, warning, critical
	Message     string    `json:"message"`
	Threshold   float64   `json:"threshold"`
	ActualValue float64   `json:"actualValue"`
	Timestamp   time.Time `json:"timestamp"`
	Resolved    bool      `json:"resolved"`
}

// NewTelemetryManager creates a new telemetry manager
func NewTelemetryManager(kubeClient kubernetes.Interface, redisClient *redis.Client, 
	logger *zap.Logger) (*TelemetryManager, error) {
	
	calculator, err := NewGPUCarbonCalculator(kubeClient, redisClient, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create carbon calculator: %v", err)
	}
	
	exporter := NewPrometheusExporter(logger, calculator)
	
	config := &TelemetryConfig{
		CollectionInterval:  30 * time.Second,
		AggregationInterval: 5 * time.Minute,
		RetentionPeriod:     24 * time.Hour,
		DCGMEndpoint:        "http://dcgm-exporter:9400/metrics",
		PrometheusPort:      9090,
		EnableForecast:      true,
		EnableOptimization:  true,
		AlertThresholds: AlertThresholds{
			CarbonRateThreshold:   1000.0, // 1000 gCO2/hour
			PowerThreshold:        500.0,  // 500 Watts
			EfficiencyThreshold:   0.1,    // 0.1 gCO2/FLOP
			UtilizationThreshold:  90.0,   // 90% utilization
			SLAViolationThreshold: 3,      // 3 violations
		},
	}
	
	return &TelemetryManager{
		kubeClient:      kubeClient,
		redisClient:     redisClient,
		logger:          logger,
		calculator:      calculator,
		exporter:        exporter,
		config:          config,
		activeWorkloads: make(map[string]*WorkloadTracker),
		zoneMetrics:     make(map[string]*ZoneMetrics),
		stopCh:          make(chan struct{}),
		workloadCh:      make(chan *WorkloadEvent, 100),
	}, nil
}

// Start starts the telemetry manager
func (tm *TelemetryManager) Start(ctx context.Context) error {
	tm.logger.Info("Starting GPU carbon telemetry manager")
	
	// Start Prometheus metrics server
	go func() {
		if err := tm.exporter.StartMetricsServer(ctx, tm.config.PrometheusPort); err != nil {
			tm.logger.Error("Failed to start Prometheus server", zap.Error(err))
		}
	}()
	
	// Start workload event processor
	go tm.processWorkloadEvents(ctx)
	
	// Start metrics collection loop
	go tm.collectMetrics(ctx)
	
	// Start zone aggregation loop
	go tm.aggregateZoneMetrics(ctx)
	
	// Start alert monitoring
	go tm.monitorAlerts(ctx)
	
	// Wait for context cancellation
	<-ctx.Done()
	close(tm.stopCh)
	
	tm.logger.Info("Stopping GPU carbon telemetry manager")
	return nil
}

// TrackWorkload starts tracking a new GPU workload
func (tm *TelemetryManager) TrackWorkload(workloadID, nodeName, gpuType, workloadType string, 
	slaRequirements *SLARequirements) error {
	
	tm.mutex.Lock()
	defer tm.mutex.Unlock()
	
	// Get node information
	node, err := tm.kubeClient.CoreV1().Nodes().Get(context.Background(), nodeName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get node info: %v", err)
	}
	
	zone := tm.extractZone(node)
	
	tracker := &WorkloadTracker{
		WorkloadID:      workloadID,
		NodeName:        nodeName,
		Zone:            zone,
		GPUType:         gpuType,
		WorkloadType:    workloadType,
		StartTime:       time.Now(),
		LastUpdate:      time.Now(),
		TotalCarbon:     0.0,
		TotalEnergy:     0.0,
		MetricsHistory:  make([]*GPUCarbonMetrics, 0),
		SLARequirements: slaRequirements,
		Alerts:          make([]Alert, 0),
	}
	
	tm.activeWorkloads[workloadID] = tracker
	
	// Send start event
	tm.workloadCh <- &WorkloadEvent{
		Type:       "start",
		WorkloadID: workloadID,
		NodeName:   nodeName,
		Timestamp:  time.Now(),
	}
	
	tm.logger.Info("Started tracking GPU workload",
		zap.String("workloadId", workloadID),
		zap.String("node", nodeName),
		zap.String("zone", zone),
		zap.String("gpuType", gpuType))
	
	return nil
}

// UpdateWorkload updates metrics for an active workload
func (tm *TelemetryManager) UpdateWorkload(workloadID string, gpuMetrics map[string]interface{}) error {
	tm.mutex.Lock()
	defer tm.mutex.Unlock()
	
	tracker, exists := tm.activeWorkloads[workloadID]
	if !exists {
		return fmt.Errorf("workload %s not found", workloadID)
	}
	
	// Send update event
	tm.workloadCh <- &WorkloadEvent{
		Type:       "update",
		WorkloadID: workloadID,
		NodeName:   tracker.NodeName,
		GPUMetrics: gpuMetrics,
		Timestamp:  time.Now(),
	}
	
	return nil
}

// StopWorkload stops tracking a workload
func (tm *TelemetryManager) StopWorkload(workloadID string) error {
	tm.mutex.Lock()
	defer tm.mutex.Unlock()
	
	tracker, exists := tm.activeWorkloads[workloadID]
	if !exists {
		return fmt.Errorf("workload %s not found", workloadID)
	}
	
	// Calculate final metrics
	duration := time.Since(tracker.StartTime)
	
	// Update workload completion metrics
	tm.exporter.UpdateWorkloadMetrics(
		tracker.WorkloadType,
		tracker.NodeName,
		tracker.Zone,
		tracker.GPUType,
		duration,
		tracker.TotalCarbon,
		tracker.TotalEnergy,
	)
	
	// Send stop event
	tm.workloadCh <- &WorkloadEvent{
		Type:       "stop",
		WorkloadID: workloadID,
		NodeName:   tracker.NodeName,
		Timestamp:  time.Now(),
	}
	
	// Archive workload data
	if err := tm.archiveWorkload(tracker); err != nil {
		tm.logger.Warn("Failed to archive workload", zap.Error(err))
	}
	
	delete(tm.activeWorkloads, workloadID)
	
	tm.logger.Info("Stopped tracking GPU workload",
		zap.String("workloadId", workloadID),
		zap.Duration("duration", duration),
		zap.Float64("totalCarbon", tracker.TotalCarbon),
		zap.Float64("totalEnergy", tracker.TotalEnergy))
	
	return nil
}

// processWorkloadEvents processes workload lifecycle events
func (tm *TelemetryManager) processWorkloadEvents(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case event := <-tm.workloadCh:
			tm.handleWorkloadEvent(event)
		}
	}
}

// handleWorkloadEvent handles a single workload event
func (tm *TelemetryManager) handleWorkloadEvent(event *WorkloadEvent) {
	switch event.Type {
	case "start":
		tm.logger.Debug("Processing workload start event", zap.String("workloadId", event.WorkloadID))
		
	case "update":
		tm.mutex.Lock()
		tracker, exists := tm.activeWorkloads[event.WorkloadID]
		tm.mutex.Unlock()
		
		if !exists {
			tm.logger.Warn("Received update for unknown workload", zap.String("workloadId", event.WorkloadID))
			return
		}
		
		// Calculate carbon metrics
		metrics, err := tm.calculator.CalculateGPUCarbon(context.Background(), 
			event.WorkloadID, event.NodeName, event.GPUMetrics)
		if err != nil {
			tm.logger.Error("Failed to calculate carbon metrics", zap.Error(err))
			return
		}
		
		// Update tracker
		tm.mutex.Lock()
		tracker.CurrentMetrics = metrics
		tracker.LastUpdate = time.Now()
		tracker.MetricsHistory = append(tracker.MetricsHistory, metrics)
		
		// Update cumulative metrics
		timeDelta := time.Since(tracker.LastUpdate).Hours()
		tracker.TotalCarbon += metrics.TotalGCO2 * timeDelta
		tracker.TotalEnergy += metrics.TotalPowerWatts * timeDelta / 1000.0 // Convert to kWh
		
		// Check for alerts
		alerts := tm.checkAlerts(tracker, metrics)
		tracker.Alerts = append(tracker.Alerts, alerts...)
		tm.mutex.Unlock()
		
		// Update Prometheus metrics
		tm.exporter.UpdateMetrics(metrics, tracker.GPUType, tracker.WorkloadType)
		
		// Check SLA compliance
		if tracker.SLARequirements != nil {
			compliance, violations := tm.checkSLACompliance(tracker, metrics)
			remainingBudget := tracker.SLARequirements.MaxCarbonBudget - tracker.TotalCarbon
			
			violationType := ""
			if len(violations) > 0 {
				violationType = violations[0] // Report first violation
			}
			
			tm.exporter.UpdateSLAMetrics(event.WorkloadID, event.NodeName, 
				tracker.Zone, violationType, compliance, remainingBudget)
		}
		
	case "stop":
		tm.logger.Debug("Processing workload stop event", zap.String("workloadId", event.WorkloadID))
	}
}

// collectMetrics periodically collects GPU metrics from DCGM
func (tm *TelemetryManager) collectMetrics(ctx context.Context) {
	ticker := time.NewTicker(tm.config.CollectionInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			tm.collectDCGMMetrics()
		}
	}
}

// collectDCGMMetrics collects metrics from DCGM exporter
func (tm *TelemetryManager) collectDCGMMetrics() {
	// In a real implementation, this would query the DCGM exporter HTTP endpoint
	// For now, we'll simulate the collection process
	
	tm.mutex.RLock()
	workloads := make([]*WorkloadTracker, 0, len(tm.activeWorkloads))
	for _, tracker := range tm.activeWorkloads {
		workloads = append(workloads, tracker)
	}
	tm.mutex.RUnlock()
	
	for _, tracker := range workloads {
		// Simulate DCGM metrics collection
		simulatedMetrics := tm.simulateDCGMMetrics(tracker)
		
		// Update workload with new metrics
		if err := tm.UpdateWorkload(tracker.WorkloadID, simulatedMetrics); err != nil {
			tm.logger.Error("Failed to update workload metrics", 
				zap.String("workloadId", tracker.WorkloadID), zap.Error(err))
		}
	}
}

// simulateDCGMMetrics simulates DCGM metrics for testing
func (tm *TelemetryManager) simulateDCGMMetrics(tracker *WorkloadTracker) map[string]interface{} {
	// Simulate realistic GPU metrics based on workload type
	baseUtilization := 70.0
	if tracker.WorkloadType == "training" {
		baseUtilization = 85.0
	} else if tracker.WorkloadType == "inference" {
		baseUtilization = 60.0
	}
	
	// Add some variation
	variation := (time.Now().Unix() % 20) - 10 // -10 to +10
	utilization := baseUtilization + float64(variation)
	
	return map[string]interface{}{
		"gpu_utilization":     utilization,
		"memory_utilization":  utilization * 0.9,
		"compute_utilization": utilization * 0.8,
		"power_draw":          250.0 + utilization*2.0, // Simulate power based on utilization
		"temperature":         65.0 + utilization*0.3,
		"memory_used":         utilization * 0.16, // 16GB * utilization
	}
}

// aggregateZoneMetrics aggregates metrics by zone
func (tm *TelemetryManager) aggregateZoneMetrics(ctx context.Context) {
	ticker := time.NewTicker(tm.config.AggregationInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			tm.calculateZoneMetrics()
		}
	}
}

// calculateZoneMetrics calculates aggregated metrics for each zone
func (tm *TelemetryManager) calculateZoneMetrics() {
	tm.mutex.Lock()
	defer tm.mutex.Unlock()
	
	zoneData := make(map[string]*ZoneMetrics)
	
	// Aggregate metrics by zone
	for _, tracker := range tm.activeWorkloads {
		if tracker.CurrentMetrics == nil {
			continue
		}
		
		zone := tracker.Zone
		if zoneData[zone] == nil {
			zoneData[zone] = &ZoneMetrics{
				Zone:       zone,
				LastUpdate: time.Now(),
				GPUCounts:  make(map[string]int),
			}
		}
		
		zoneMetrics := zoneData[zone]
		zoneMetrics.TotalCarbonRate += tracker.CurrentMetrics.TotalGCO2
		zoneMetrics.TotalPowerConsumption += tracker.CurrentMetrics.TotalPowerWatts
		zoneMetrics.ActiveWorkloads++
		zoneMetrics.GPUCounts[tracker.GPUType]++
		
		// Update carbon intensity and PUE (use latest values)
		zoneMetrics.CarbonIntensity = tracker.CurrentMetrics.CarbonIntensity
		zoneMetrics.PUE = tracker.CurrentMetrics.PUE
	}
	
	// Calculate average efficiency and update zone metrics
	for zone, metrics := range zoneData {
		if metrics.ActiveWorkloads > 0 {
			// Calculate average efficiency (simplified)
			totalEfficiency := 0.0
			count := 0
			for _, tracker := range tm.activeWorkloads {
				if tracker.Zone == zone && tracker.CurrentMetrics != nil {
					totalEfficiency += tracker.CurrentMetrics.CarbonEfficiency
					count++
				}
			}
			if count > 0 {
				metrics.AverageEfficiency = totalEfficiency / float64(count)
			}
		}
		
		// Get carbon forecast for the zone
		if tm.config.EnableForecast {
			forecast, err := tm.calculator.getCarbonForecast(context.Background(), zone, 4)
			if err == nil {
				metrics.Forecast = forecast
			}
		}
		
		tm.zoneMetrics[zone] = metrics
		
		// Update Prometheus zone metrics
		tm.exporter.UpdateZoneMetrics(zone, metrics.TotalCarbonRate, 
			metrics.TotalPowerConsumption, metrics.GPUCounts, metrics.AverageEfficiency)
	}
}

// monitorAlerts monitors for carbon and power alerts
func (tm *TelemetryManager) monitorAlerts(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			tm.processAlerts()
		}
	}
}

// processAlerts processes and resolves alerts
func (tm *TelemetryManager) processAlerts() {
	tm.mutex.RLock()
	defer tm.mutex.RUnlock()
	
	for _, tracker := range tm.activeWorkloads {
		if tracker.CurrentMetrics == nil {
			continue
		}
		
		// Check for new alerts
		newAlerts := tm.checkAlerts(tracker, tracker.CurrentMetrics)
		
		// Log critical alerts
		for _, alert := range newAlerts {
			if alert.Severity == "critical" {
				tm.logger.Warn("Critical carbon/power alert",
					zap.String("workloadId", tracker.WorkloadID),
					zap.String("type", alert.Type),
					zap.String("message", alert.Message),
					zap.Float64("threshold", alert.Threshold),
					zap.Float64("actual", alert.ActualValue))
			}
		}
	}
}

// checkAlerts checks for carbon and power alerts
func (tm *TelemetryManager) checkAlerts(tracker *WorkloadTracker, metrics *GPUCarbonMetrics) []Alert {
	var alerts []Alert
	
	// Carbon rate alert
	if metrics.TotalGCO2 > tm.config.AlertThresholds.CarbonRateThreshold {
		alerts = append(alerts, Alert{
			ID:          fmt.Sprintf("%s-carbon-%d", tracker.WorkloadID, time.Now().Unix()),
			Type:        "carbon",
			Severity:    "warning",
			Message:     "High carbon emission rate detected",
			Threshold:   tm.config.AlertThresholds.CarbonRateThreshold,
			ActualValue: metrics.TotalGCO2,
			Timestamp:   time.Now(),
		})
	}
	
	// Power alert
	if metrics.TotalPowerWatts > tm.config.AlertThresholds.PowerThreshold {
		alerts = append(alerts, Alert{
			ID:          fmt.Sprintf("%s-power-%d", tracker.WorkloadID, time.Now().Unix()),
			Type:        "power",
			Severity:    "warning",
			Message:     "High power consumption detected",
			Threshold:   tm.config.AlertThresholds.PowerThreshold,
			ActualValue: metrics.TotalPowerWatts,
			Timestamp:   time.Now(),
		})
	}
	
	// Efficiency alert
	if metrics.CarbonEfficiency > tm.config.AlertThresholds.EfficiencyThreshold {
		alerts = append(alerts, Alert{
			ID:          fmt.Sprintf("%s-efficiency-%d", tracker.WorkloadID, time.Now().Unix()),
			Type:        "efficiency",
			Severity:    "info",
			Message:     "Low carbon efficiency detected",
			Threshold:   tm.config.AlertThresholds.EfficiencyThreshold,
			ActualValue: metrics.CarbonEfficiency,
			Timestamp:   time.Now(),
		})
	}
	
	return alerts
}

// checkSLACompliance checks SLA compliance for a workload
func (tm *TelemetryManager) checkSLACompliance(tracker *WorkloadTracker, 
	metrics *GPUCarbonMetrics) (float64, []string) {
	
	if tracker.SLARequirements == nil {
		return 100.0, nil
	}
	
	var violations []string
	compliance := 100.0
	
	// Check carbon budget
	if tracker.TotalCarbon > tracker.SLARequirements.MaxCarbonBudget {
		violations = append(violations, "carbon_budget")
		compliance -= 25.0
	}
	
	// Check power budget
	if metrics.TotalPowerWatts > tracker.SLARequirements.MaxPowerBudget {
		violations = append(violations, "power_budget")
		compliance -= 25.0
	}
	
	// Simulate latency and throughput checks
	// In practice, these would be measured from the actual workload
	
	return compliance, violations
}

// archiveWorkload archives completed workload data
func (tm *TelemetryManager) archiveWorkload(tracker *WorkloadTracker) error {
	if tm.redisClient == nil {
		return nil
	}
	
	key := fmt.Sprintf("workload:archive:%s", tracker.WorkloadID)
	data, err := json.Marshal(tracker)
	if err != nil {
		return err
	}
	
	return tm.redisClient.Set(context.Background(), key, data, tm.config.RetentionPeriod).Err()
}

// GetWorkloadMetrics returns current metrics for a workload
func (tm *TelemetryManager) GetWorkloadMetrics(workloadID string) (*WorkloadTracker, error) {
	tm.mutex.RLock()
	defer tm.mutex.RUnlock()
	
	tracker, exists := tm.activeWorkloads[workloadID]
	if !exists {
		return nil, fmt.Errorf("workload %s not found", workloadID)
	}
	
	return tracker, nil
}

// GetZoneMetrics returns aggregated metrics for a zone
func (tm *TelemetryManager) GetZoneMetrics(zone string) (*ZoneMetrics, error) {
	tm.mutex.RLock()
	defer tm.mutex.RUnlock()
	
	metrics, exists := tm.zoneMetrics[zone]
	if !exists {
		return nil, fmt.Errorf("zone %s not found", zone)
	}
	
	return metrics, nil
}

// GetActiveWorkloads returns all active workloads
func (tm *TelemetryManager) GetActiveWorkloads() map[string]*WorkloadTracker {
	tm.mutex.RLock()
	defer tm.mutex.RUnlock()
	
	// Return a copy to avoid race conditions
	result := make(map[string]*WorkloadTracker)
	for k, v := range tm.activeWorkloads {
		result[k] = v
	}
	
	return result
}

// Helper functions

func (tm *TelemetryManager) extractZone(node *v1.Node) string {
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