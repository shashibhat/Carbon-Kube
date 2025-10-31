package telemetry

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
)

// PrometheusExporter exports GPU carbon metrics to Prometheus
type PrometheusExporter struct {
	logger     *zap.Logger
	calculator *GPUCarbonCalculator
	
	// Carbon metrics
	gpuCarbonRate         *prometheus.GaugeVec
	gpuOperationalCarbon  *prometheus.GaugeVec
	gpuEmbodiedCarbon     *prometheus.GaugeVec
	gpuCarbonIntensity    *prometheus.GaugeVec
	gpuCarbonEfficiency   *prometheus.GaugeVec
	
	// Power metrics
	gpuPowerConsumption   *prometheus.GaugeVec
	gpuSystemPower        *prometheus.GaugeVec
	gpuCoolingPower       *prometheus.GaugeVec
	gpuTotalPower         *prometheus.GaugeVec
	gpuPUE                *prometheus.GaugeVec
	
	// Efficiency metrics
	gpuPowerEfficiency    *prometheus.GaugeVec
	gpuUtilizationRate    *prometheus.GaugeVec
	gpuMemoryUtilization  *prometheus.GaugeVec
	gpuComputeUtilization *prometheus.GaugeVec
	
	// Forecast metrics
	gpuCarbonForecast     *prometheus.GaugeVec
	gpuOptimalWindow      *prometheus.GaugeVec
	gpuCarbonSavings      *prometheus.GaugeVec
	
	// Workload metrics
	gpuWorkloadDuration   *prometheus.HistogramVec
	gpuWorkloadCarbon     *prometheus.CounterVec
	gpuWorkloadEnergy     *prometheus.CounterVec
	
	// SLA metrics
	gpuSLAViolations      *prometheus.CounterVec
	gpuSLACompliance      *prometheus.GaugeVec
	gpuCarbonBudget       *prometheus.GaugeVec
	
	// Zone aggregation metrics
	zoneCarbonRate        *prometheus.GaugeVec
	zonePowerConsumption  *prometheus.GaugeVec
	zoneGPUCount          *prometheus.GaugeVec
	zoneEfficiency        *prometheus.GaugeVec
}

// NewPrometheusExporter creates a new Prometheus exporter for GPU carbon metrics
func NewPrometheusExporter(logger *zap.Logger, calculator *GPUCarbonCalculator) *PrometheusExporter {
	exporter := &PrometheusExporter{
		logger:     logger,
		calculator: calculator,
	}
	
	exporter.initializeMetrics()
	exporter.registerMetrics()
	
	return exporter
}

// initializeMetrics initializes all Prometheus metrics
func (pe *PrometheusExporter) initializeMetrics() {
	// Carbon metrics
	pe.gpuCarbonRate = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_carbon_rate_gco2_per_hour",
			Help: "GPU carbon emission rate in gCO2 per hour",
		},
		[]string{"workload_id", "node", "zone", "gpu_type", "workload_type"},
	)
	
	pe.gpuOperationalCarbon = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_operational_carbon_gco2_per_hour",
			Help: "GPU operational carbon emissions in gCO2 per hour",
		},
		[]string{"workload_id", "node", "zone", "gpu_type"},
	)
	
	pe.gpuEmbodiedCarbon = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_embodied_carbon_gco2_per_hour",
			Help: "GPU embodied carbon emissions in gCO2 per hour",
		},
		[]string{"workload_id", "node", "zone", "gpu_type"},
	)
	
	pe.gpuCarbonIntensity = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_carbon_intensity_gco2_per_kwh",
			Help: "Grid carbon intensity in gCO2 per kWh",
		},
		[]string{"zone"},
	)
	
	pe.gpuCarbonEfficiency = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_carbon_efficiency_gco2_per_flop",
			Help: "GPU carbon efficiency in gCO2 per FLOP",
		},
		[]string{"workload_id", "node", "zone", "gpu_type"},
	)
	
	// Power metrics
	pe.gpuPowerConsumption = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_power_consumption_watts",
			Help: "GPU power consumption in watts",
		},
		[]string{"workload_id", "node", "zone", "gpu_type"},
	)
	
	pe.gpuSystemPower = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_system_power_watts",
			Help: "Total system power consumption in watts",
		},
		[]string{"workload_id", "node", "zone"},
	)
	
	pe.gpuCoolingPower = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_cooling_power_watts",
			Help: "Cooling power overhead in watts",
		},
		[]string{"workload_id", "node", "zone"},
	)
	
	pe.gpuTotalPower = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_total_power_watts",
			Help: "Total power consumption including PUE in watts",
		},
		[]string{"workload_id", "node", "zone"},
	)
	
	pe.gpuPUE = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_pue_ratio",
			Help: "Power Usage Effectiveness ratio",
		},
		[]string{"zone"},
	)
	
	// Efficiency metrics
	pe.gpuPowerEfficiency = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_power_efficiency_flops_per_watt",
			Help: "GPU power efficiency in FLOPS per watt",
		},
		[]string{"workload_id", "node", "zone", "gpu_type"},
	)
	
	pe.gpuUtilizationRate = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_utilization_percent",
			Help: "GPU utilization percentage",
		},
		[]string{"workload_id", "node", "zone", "gpu_type"},
	)
	
	pe.gpuMemoryUtilization = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_memory_utilization_percent",
			Help: "GPU memory utilization percentage",
		},
		[]string{"workload_id", "node", "zone", "gpu_type"},
	)
	
	pe.gpuComputeUtilization = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_compute_utilization_percent",
			Help: "GPU compute utilization percentage",
		},
		[]string{"workload_id", "node", "zone", "gpu_type"},
	)
	
	// Forecast metrics
	pe.gpuCarbonForecast = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_carbon_forecast_gco2_per_kwh",
			Help: "Carbon intensity forecast in gCO2 per kWh",
		},
		[]string{"zone", "hour_offset"},
	)
	
	pe.gpuOptimalWindow = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_optimal_window_start_timestamp",
			Help: "Optimal execution window start timestamp",
		},
		[]string{"zone", "workload_type"},
	)
	
	pe.gpuCarbonSavings = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_carbon_savings_percent",
			Help: "Potential carbon savings percentage",
		},
		[]string{"zone", "workload_type"},
	)
	
	// Workload metrics
	pe.gpuWorkloadDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "gpu_workload_duration_seconds",
			Help:    "GPU workload execution duration in seconds",
			Buckets: []float64{60, 300, 900, 1800, 3600, 7200, 14400, 28800, 86400},
		},
		[]string{"workload_type", "node", "zone", "gpu_type"},
	)
	
	pe.gpuWorkloadCarbon = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gpu_workload_carbon_total_gco2",
			Help: "Total carbon emissions for completed workloads in gCO2",
		},
		[]string{"workload_type", "node", "zone", "gpu_type"},
	)
	
	pe.gpuWorkloadEnergy = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gpu_workload_energy_total_kwh",
			Help: "Total energy consumption for completed workloads in kWh",
		},
		[]string{"workload_type", "node", "zone", "gpu_type"},
	)
	
	// SLA metrics
	pe.gpuSLAViolations = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gpu_sla_violations_total",
			Help: "Total number of SLA violations",
		},
		[]string{"workload_id", "node", "zone", "violation_type"},
	)
	
	pe.gpuSLACompliance = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_sla_compliance_percent",
			Help: "SLA compliance percentage",
		},
		[]string{"workload_id", "node", "zone"},
	)
	
	pe.gpuCarbonBudget = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "gpu_carbon_budget_remaining_gco2",
			Help: "Remaining carbon budget in gCO2",
		},
		[]string{"workload_id", "node", "zone"},
	)
	
	// Zone aggregation metrics
	pe.zoneCarbonRate = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "zone_carbon_rate_gco2_per_hour",
			Help: "Total carbon emission rate per zone in gCO2 per hour",
		},
		[]string{"zone"},
	)
	
	pe.zonePowerConsumption = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "zone_power_consumption_watts",
			Help: "Total power consumption per zone in watts",
		},
		[]string{"zone"},
	)
	
	pe.zoneGPUCount = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "zone_gpu_count",
			Help: "Number of active GPUs per zone",
		},
		[]string{"zone", "gpu_type"},
	)
	
	pe.zoneEfficiency = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "zone_carbon_efficiency_gco2_per_flop",
			Help: "Average carbon efficiency per zone in gCO2 per FLOP",
		},
		[]string{"zone"},
	)
}

// registerMetrics registers all metrics with Prometheus
func (pe *PrometheusExporter) registerMetrics() {
	prometheus.MustRegister(
		// Carbon metrics
		pe.gpuCarbonRate,
		pe.gpuOperationalCarbon,
		pe.gpuEmbodiedCarbon,
		pe.gpuCarbonIntensity,
		pe.gpuCarbonEfficiency,
		
		// Power metrics
		pe.gpuPowerConsumption,
		pe.gpuSystemPower,
		pe.gpuCoolingPower,
		pe.gpuTotalPower,
		pe.gpuPUE,
		
		// Efficiency metrics
		pe.gpuPowerEfficiency,
		pe.gpuUtilizationRate,
		pe.gpuMemoryUtilization,
		pe.gpuComputeUtilization,
		
		// Forecast metrics
		pe.gpuCarbonForecast,
		pe.gpuOptimalWindow,
		pe.gpuCarbonSavings,
		
		// Workload metrics
		pe.gpuWorkloadDuration,
		pe.gpuWorkloadCarbon,
		pe.gpuWorkloadEnergy,
		
		// SLA metrics
		pe.gpuSLAViolations,
		pe.gpuSLACompliance,
		pe.gpuCarbonBudget,
		
		// Zone aggregation metrics
		pe.zoneCarbonRate,
		pe.zonePowerConsumption,
		pe.zoneGPUCount,
		pe.zoneEfficiency,
	)
}

// UpdateMetrics updates Prometheus metrics with GPU carbon data
func (pe *PrometheusExporter) UpdateMetrics(metrics *GPUCarbonMetrics, gpuType, workloadType string) {
	labels := prometheus.Labels{
		"workload_id":   metrics.WorkloadID,
		"node":          metrics.NodeName,
		"zone":          metrics.Zone,
		"gpu_type":      gpuType,
		"workload_type": workloadType,
	}
	
	// Update carbon metrics
	pe.gpuCarbonRate.With(labels).Set(metrics.TotalGCO2)
	pe.gpuOperationalCarbon.With(prometheus.Labels{
		"workload_id": metrics.WorkloadID,
		"node":        metrics.NodeName,
		"zone":        metrics.Zone,
		"gpu_type":    gpuType,
	}).Set(metrics.OperationalGCO2)
	pe.gpuEmbodiedCarbon.With(prometheus.Labels{
		"workload_id": metrics.WorkloadID,
		"node":        metrics.NodeName,
		"zone":        metrics.Zone,
		"gpu_type":    gpuType,
	}).Set(metrics.EmbodiedGCO2)
	pe.gpuCarbonIntensity.With(prometheus.Labels{"zone": metrics.Zone}).Set(metrics.CarbonIntensity)
	pe.gpuCarbonEfficiency.With(prometheus.Labels{
		"workload_id": metrics.WorkloadID,
		"node":        metrics.NodeName,
		"zone":        metrics.Zone,
		"gpu_type":    gpuType,
	}).Set(metrics.CarbonEfficiency)
	
	// Update power metrics
	pe.gpuPowerConsumption.With(prometheus.Labels{
		"workload_id": metrics.WorkloadID,
		"node":        metrics.NodeName,
		"zone":        metrics.Zone,
		"gpu_type":    gpuType,
	}).Set(metrics.GPUPowerWatts)
	pe.gpuSystemPower.With(prometheus.Labels{
		"workload_id": metrics.WorkloadID,
		"node":        metrics.NodeName,
		"zone":        metrics.Zone,
	}).Set(metrics.SystemPowerWatts)
	pe.gpuCoolingPower.With(prometheus.Labels{
		"workload_id": metrics.WorkloadID,
		"node":        metrics.NodeName,
		"zone":        metrics.Zone,
	}).Set(metrics.CoolingPowerWatts)
	pe.gpuTotalPower.With(prometheus.Labels{
		"workload_id": metrics.WorkloadID,
		"node":        metrics.NodeName,
		"zone":        metrics.Zone,
	}).Set(metrics.TotalPowerWatts)
	pe.gpuPUE.With(prometheus.Labels{"zone": metrics.Zone}).Set(metrics.PUE)
	
	// Update efficiency metrics
	pe.gpuPowerEfficiency.With(prometheus.Labels{
		"workload_id": metrics.WorkloadID,
		"node":        metrics.NodeName,
		"zone":        metrics.Zone,
		"gpu_type":    gpuType,
	}).Set(metrics.PowerEfficiency)
	pe.gpuUtilizationRate.With(prometheus.Labels{
		"workload_id": metrics.WorkloadID,
		"node":        metrics.NodeName,
		"zone":        metrics.Zone,
		"gpu_type":    gpuType,
	}).Set(metrics.GPUUtilization)
	pe.gpuMemoryUtilization.With(prometheus.Labels{
		"workload_id": metrics.WorkloadID,
		"node":        metrics.NodeName,
		"zone":        metrics.Zone,
		"gpu_type":    gpuType,
	}).Set(metrics.MemoryUtilization)
	pe.gpuComputeUtilization.With(prometheus.Labels{
		"workload_id": metrics.WorkloadID,
		"node":        metrics.NodeName,
		"zone":        metrics.Zone,
		"gpu_type":    gpuType,
	}).Set(metrics.ComputeUtilization)
	
	// Update forecast metrics
	if len(metrics.CarbonForecast) > 0 {
		for i, forecast := range metrics.CarbonForecast {
			pe.gpuCarbonForecast.With(prometheus.Labels{
				"zone":        metrics.Zone,
				"hour_offset": fmt.Sprintf("%d", i),
			}).Set(forecast)
		}
	}
	
	if metrics.OptimalWindow != nil {
		pe.gpuOptimalWindow.With(prometheus.Labels{
			"zone":          metrics.Zone,
			"workload_type": workloadType,
		}).Set(float64(metrics.OptimalWindow.StartTime.Unix()))
		pe.gpuCarbonSavings.With(prometheus.Labels{
			"zone":          metrics.Zone,
			"workload_type": workloadType,
		}).Set(metrics.OptimalWindow.CarbonSavings)
	}
}

// UpdateWorkloadMetrics updates workload completion metrics
func (pe *PrometheusExporter) UpdateWorkloadMetrics(workloadType, node, zone, gpuType string, 
	duration time.Duration, totalCarbon, totalEnergy float64) {
	
	labels := prometheus.Labels{
		"workload_type": workloadType,
		"node":          node,
		"zone":          zone,
		"gpu_type":      gpuType,
	}
	
	pe.gpuWorkloadDuration.With(labels).Observe(duration.Seconds())
	pe.gpuWorkloadCarbon.With(labels).Add(totalCarbon)
	pe.gpuWorkloadEnergy.With(labels).Add(totalEnergy)
}

// UpdateSLAMetrics updates SLA-related metrics
func (pe *PrometheusExporter) UpdateSLAMetrics(workloadID, node, zone, violationType string, 
	compliance float64, remainingBudget float64) {
	
	if violationType != "" {
		pe.gpuSLAViolations.With(prometheus.Labels{
			"workload_id":    workloadID,
			"node":           node,
			"zone":           zone,
			"violation_type": violationType,
		}).Inc()
	}
	
	pe.gpuSLACompliance.With(prometheus.Labels{
		"workload_id": workloadID,
		"node":        node,
		"zone":        zone,
	}).Set(compliance)
	
	pe.gpuCarbonBudget.With(prometheus.Labels{
		"workload_id": workloadID,
		"node":        node,
		"zone":        zone,
	}).Set(remainingBudget)
}

// UpdateZoneMetrics updates zone-level aggregated metrics
func (pe *PrometheusExporter) UpdateZoneMetrics(zone string, totalCarbonRate, totalPower float64, 
	gpuCounts map[string]int, avgEfficiency float64) {
	
	pe.zoneCarbonRate.With(prometheus.Labels{"zone": zone}).Set(totalCarbonRate)
	pe.zonePowerConsumption.With(prometheus.Labels{"zone": zone}).Set(totalPower)
	pe.zoneEfficiency.With(prometheus.Labels{"zone": zone}).Set(avgEfficiency)
	
	for gpuType, count := range gpuCounts {
		pe.zoneGPUCount.With(prometheus.Labels{
			"zone":     zone,
			"gpu_type": gpuType,
		}).Set(float64(count))
	}
}

// StartMetricsServer starts the Prometheus metrics HTTP server
func (pe *PrometheusExporter) StartMetricsServer(ctx context.Context, port int) error {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	
	// Add health check endpoint
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})
	
	// Add carbon metrics summary endpoint
	mux.HandleFunc("/carbon-summary", pe.handleCarbonSummary)
	
	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", port),
		Handler: mux,
	}
	
	pe.logger.Info("Starting Prometheus metrics server", zap.Int("port", port))
	
	go func() {
		<-ctx.Done()
		pe.logger.Info("Shutting down Prometheus metrics server")
		server.Shutdown(context.Background())
	}()
	
	return server.ListenAndServe()
}

// handleCarbonSummary provides a JSON summary of carbon metrics
func (pe *PrometheusExporter) handleCarbonSummary(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	// This is a simplified summary - in practice, you'd aggregate from Prometheus
	summary := map[string]interface{}{
		"timestamp": time.Now().Unix(),
		"summary": map[string]interface{}{
			"total_carbon_rate_gco2_per_hour": 1500.0,
			"total_power_consumption_watts":   5000.0,
			"average_carbon_efficiency":       0.05,
			"active_workloads":               12,
			"zones": []string{"us-west1-a", "us-east1-a"},
		},
	}
	
	w.WriteHeader(http.StatusOK)
	if err := json.NewEncoder(w).Encode(summary); err != nil {
		pe.logger.Error("Failed to encode carbon summary", zap.Error(err))
	}
}

// GetMetricsSnapshot returns a snapshot of current metrics for debugging
func (pe *PrometheusExporter) GetMetricsSnapshot() map[string]interface{} {
	return map[string]interface{}{
		"carbon_metrics": map[string]interface{}{
			"total_workloads": "tracked via gpu_carbon_rate_gco2_per_hour",
			"zones":          "tracked via zone labels",
		},
		"power_metrics": map[string]interface{}{
			"gpu_power":    "tracked via gpu_power_consumption_watts",
			"system_power": "tracked via gpu_system_power_watts",
			"total_power":  "tracked via gpu_total_power_watts",
		},
		"efficiency_metrics": map[string]interface{}{
			"power_efficiency": "tracked via gpu_power_efficiency_flops_per_watt",
			"carbon_efficiency": "tracked via gpu_carbon_efficiency_gco2_per_flop",
		},
		"forecast_metrics": map[string]interface{}{
			"carbon_forecast": "tracked via gpu_carbon_forecast_gco2_per_kwh",
			"optimal_windows": "tracked via gpu_optimal_window_start_timestamp",
		},
	}
}