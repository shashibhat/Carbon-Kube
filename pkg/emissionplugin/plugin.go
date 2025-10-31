package emissionplugin

import (
	"context"
	"encoding/json"
	"fmt"
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
)

const (
	PluginName           = "EmissionPlugin"
	ConfigMapName        = "carbon-scores"
	ConfigMapNamespace   = "default"
	ZoneLabel           = "topology.kubernetes.io/zone"
	DefaultIntensity    = 200.0 // Default carbon intensity in gCO2/kWh
	CacheTTL            = 5 * time.Minute // Redis cache TTL
	RedisKeyPrefix      = "carbon-kube:"
)

// ZonalIntensity represents carbon intensity data for a zone
type ZonalIntensity struct {
	Zone      string    `json:"zone"`
	Intensity float64   `json:"intensity"` // gCO2/kWh
	Timestamp int64     `json:"timestamp"`
	Forecast  []float64 `json:"forecast"` // 24h forecast array
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
}

// Metrics holds Prometheus metrics for the plugin
type Metrics struct {
	carbonIntensity *prometheus.GaugeVec
	emissionScores  *prometheus.GaugeVec
	migrationsTotal prometheus.Counter
	savedCO2        prometheus.Counter
}

// NewEmissionPlugin creates a new instance of the emission plugin
func NewEmissionPlugin() (*EmissionPlugin, error) {
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
	}, nil
}

// Name returns the plugin name
func (p *EmissionPlugin) Name() string {
	return PluginName
}

// Execute implements the main plugin logic for scoring nodes based on carbon emissions
func (p *EmissionPlugin) Execute(ctx context.Context, pod *v1.Pod, nodes []*v1.Node) (map[string]float64, bool) {
	p.logger.Info("EmissionPlugin executing",
		zap.String("pod", pod.Name),
		zap.String("namespace", pod.Namespace),
		zap.Int("nodeCount", len(nodes)))

	scores := make(map[string]float64)
	
	// Get carbon intensity data from cache or ConfigMap
	intensityMap, err := p.getCarbonIntensityData(ctx)
	if err != nil {
		p.logger.Error("Failed to get carbon intensity data, using default scores",
			zap.Error(err),
			zap.String("pod", pod.Name))
		// Fallback: assign zero scores (no preference)
		for _, node := range nodes {
			scores[node.Name] = 0.0
		}
		return scores, true
	}

	// Calculate pod resource requirements
	podRequests := p.getPodResourceRequests(pod)
	
	p.logger.Debug("Pod resource requirements calculated",
		zap.String("pod", pod.Name),
		zap.Int64("cpuMillicores", podRequests.CPU),
		zap.Int64("memoryBytes", podRequests.Memory))
	
	// Score each node based on carbon intensity and resource requirements
	for _, node := range nodes {
		zone := p.extractZoneFromNode(node)
		intensity := p.getIntensityForZone(zone, intensityMap)
		
		// Calculate emission score: intensity * normalized CPU requirement
		// Higher scores indicate higher emissions (worse for scheduling)
		cpuKW := float64(podRequests.CPU) / 1000.0 / 1000.0 // Convert millicores to kW equivalent
		emissionScore := intensity * cpuKW
		
		scores[node.Name] = emissionScore
		
		// Update metrics
		p.metrics.carbonIntensity.WithLabelValues(zone).Set(intensity)
		p.metrics.emissionScores.WithLabelValues(node.Name, zone).Set(emissionScore)
		
		p.logger.Debug("Calculated emission score",
			zap.String("node", node.Name),
			zap.String("zone", zone),
			zap.Float64("intensity", intensity),
			zap.Float64("cpuKW", cpuKW),
			zap.Float64("score", emissionScore))
	}

	p.logger.Info("EmissionPlugin completed",
		zap.String("pod", pod.Name),
		zap.Int("scoresCalculated", len(scores)))
	return scores, true
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

// Plugin interface for dynamic loading
var Plugin EmissionPlugin

func init() {
	plugin, err := NewEmissionPlugin()
	if err != nil {
		klog.ErrorS(err, "Failed to initialize EmissionPlugin")
		return
	}
	Plugin = *plugin
}