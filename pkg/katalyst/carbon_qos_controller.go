package katalyst

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
)

// QoSProfile defines carbon-aware Quality of Service profiles
type QoSProfile struct {
	Name                    string  `json:"name"`
	CarbonThreshold        float64 `json:"carbonThreshold"`        // gCO2/kWh
	ResourceGuarantee      float64 `json:"resourceGuarantee"`      // Percentage (0-100)
	Priority               int32   `json:"priority"`               // Kubernetes priority
	TolerateHighCarbon     bool    `json:"tolerateHighCarbon"`     // Can run in high-carbon zones
	MigrationPolicy        string  `json:"migrationPolicy"`        // "aggressive", "conservative", "disabled"
	EnergyEfficiencyTarget float64 `json:"energyEfficiencyTarget"` // Target PUE
	MaxCarbonBudget        float64 `json:"maxCarbonBudget"`        // Daily CO2 budget in kg
}

// CarbonNodeTopology represents carbon and topology information for a node
type CarbonNodeTopology struct {
	NodeName         string                `json:"nodeName"`
	Zone             string                `json:"zone"`
	CarbonIntensity  float64               `json:"carbonIntensity"`
	EnergySource     EnergySourceBreakdown `json:"energySource"`
	QoSClass         string                `json:"qosClass"`
	ResourceProfile  ResourceProfile       `json:"resourceProfile"`
	TopologyInfo     TopologyInfo          `json:"topologyInfo"`
	LastUpdated      time.Time             `json:"lastUpdated"`
}

// EnergySourceBreakdown shows the energy mix for a region
type EnergySourceBreakdown struct {
	Renewable   float64   `json:"renewable"`   // Percentage
	Nuclear     float64   `json:"nuclear"`     // Percentage
	Gas         float64   `json:"gas"`         // Percentage
	Coal        float64   `json:"coal"`        // Percentage
	LastUpdated time.Time `json:"lastUpdated"`
}

// ResourceProfile contains performance and efficiency metrics
type ResourceProfile struct {
	CPUEfficiency    float64       `json:"cpuEfficiency"`    // Performance per watt
	MemoryEfficiency float64       `json:"memoryEfficiency"` // GB per watt
	NetworkLatency   time.Duration `json:"networkLatency"`   // Average network latency
	StorageIOPS      int64         `json:"storageIOPS"`      // Storage IOPS capacity
	PowerUsage       float64       `json:"powerUsage"`       // Current power usage in watts
}

// TopologyInfo contains NUMA and device topology information
type TopologyInfo struct {
	NUMANodes      []NUMANode      `json:"numaNodes"`
	GPUs           []GPUInfo       `json:"gpus"`
	NetworkDevices []NetworkDevice `json:"networkDevices"`
	PowerZones     []PowerZone     `json:"powerZones"`
}

// NUMANode represents a NUMA topology node
type NUMANode struct {
	ID       int     `json:"id"`
	CPUs     []int   `json:"cpus"`
	Memory   int64   `json:"memory"` // Memory in bytes
	Distance []int   `json:"distance"`
	PowerW   float64 `json:"powerW"` // Power consumption in watts
}

// GPUInfo represents GPU device information
type GPUInfo struct {
	ID           string  `json:"id"`
	Model        string  `json:"model"`
	MemoryMB     int64   `json:"memoryMB"`
	PowerLimitW  float64 `json:"powerLimitW"`
	UtilPercent  float64 `json:"utilPercent"`
	TempCelsius  float64 `json:"tempCelsius"`
	NUMANode     int     `json:"numaNode"`
}

// NetworkDevice represents network device information
type NetworkDevice struct {
	Name         string  `json:"name"`
	BandwidthMbps int64   `json:"bandwidthMbps"`
	LatencyMs    float64 `json:"latencyMs"`
	PowerW       float64 `json:"powerW"`
}

// PowerZone represents a power management zone
type PowerZone struct {
	ID          string  `json:"id"`
	MaxPowerW   float64 `json:"maxPowerW"`
	CurrentW    float64 `json:"currentW"`
	Efficiency  float64 `json:"efficiency"` // PUE (Power Usage Effectiveness)
	CoolingType string  `json:"coolingType"`
}

// CarbonQoSController manages carbon-aware QoS profiles and node topology
type CarbonQoSController struct {
	kubeClient   kubernetes.Interface
	qosProfiles  map[string]*QoSProfile
	nodeTopology map[string]*CarbonNodeTopology
	metrics      *CarbonQoSMetrics
}

// CarbonQoSMetrics holds Prometheus metrics for the controller
type CarbonQoSMetrics struct {
	qosProfilesTotal    *prometheus.GaugeVec
	carbonEfficiency    *prometheus.GaugeVec
	resourceUtilization *prometheus.GaugeVec
	migrationEvents     *prometheus.CounterVec
	energyConsumption   *prometheus.GaugeVec
}

// NewCarbonQoSController creates a new carbon-aware QoS controller
func NewCarbonQoSController(kubeClient kubernetes.Interface) (*CarbonQoSController, error) {
	metrics := &CarbonQoSMetrics{
		qosProfilesTotal: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "katalyst_carbon_qos_profiles_total",
				Help: "Total number of carbon-aware QoS profiles",
			},
			[]string{"profile_name", "carbon_class"},
		),
		carbonEfficiency: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "katalyst_carbon_efficiency_ratio",
				Help: "Carbon efficiency ratio (performance per gCO2)",
			},
			[]string{"node", "zone", "qos_class"},
		),
		resourceUtilization: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "katalyst_carbon_resource_utilization",
				Help: "Resource utilization with carbon awareness",
			},
			[]string{"node", "resource_type", "qos_class"},
		),
		migrationEvents: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "katalyst_carbon_migration_events_total",
				Help: "Total carbon-aware migration events",
			},
			[]string{"from_zone", "to_zone", "reason"},
		),
		energyConsumption: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "katalyst_energy_consumption_watts",
				Help: "Current energy consumption in watts",
			},
			[]string{"node", "component", "power_zone"},
		),
	}

	controller := &CarbonQoSController{
		kubeClient:   kubeClient,
		qosProfiles:  make(map[string]*QoSProfile),
		nodeTopology: make(map[string]*CarbonNodeTopology),
		metrics:      metrics,
	}

	// Initialize default QoS profiles
	controller.initializeDefaultProfiles()

	return controller, nil
}

// initializeDefaultProfiles sets up default carbon-aware QoS profiles
func (c *CarbonQoSController) initializeDefaultProfiles() {
	profiles := []*QoSProfile{
		{
			Name:                    "green-guaranteed",
			CarbonThreshold:        100.0, // Very low carbon
			ResourceGuarantee:      100.0, // Full resource guarantee
			Priority:               1000,  // Highest priority
			TolerateHighCarbon:     false, // Must run on green energy
			MigrationPolicy:        "disabled",
			EnergyEfficiencyTarget: 1.2, // Excellent PUE
			MaxCarbonBudget:        10.0, // 10kg CO2 per day
		},
		{
			Name:                    "mixed-burstable",
			CarbonThreshold:        300.0, // Moderate carbon
			ResourceGuarantee:      80.0,  // 80% resource guarantee
			Priority:               500,   // Medium priority
			TolerateHighCarbon:     true,  // Can run on mixed energy
			MigrationPolicy:        "conservative",
			EnergyEfficiencyTarget: 1.5, // Good PUE
			MaxCarbonBudget:        50.0, // 50kg CO2 per day
		},
		{
			Name:                    "dirty-besteffort",
			CarbonThreshold:        500.0, // High carbon tolerance
			ResourceGuarantee:      50.0,  // 50% resource guarantee
			Priority:               100,   // Low priority
			TolerateHighCarbon:     true,  // Can run anywhere
			MigrationPolicy:        "aggressive",
			EnergyEfficiencyTarget: 2.0, // Acceptable PUE
			MaxCarbonBudget:        100.0, // 100kg CO2 per day
		},
	}

	for _, profile := range profiles {
		c.qosProfiles[profile.Name] = profile
		c.metrics.qosProfilesTotal.WithLabelValues(
			profile.Name,
			c.getCarbonClass(profile.CarbonThreshold),
		).Set(1)
	}

	klog.Infof("Initialized %d default carbon-aware QoS profiles", len(profiles))
}

// getCarbonClass returns the carbon classification for a threshold
func (c *CarbonQoSController) getCarbonClass(threshold float64) string {
	if threshold <= 100 {
		return "green"
	} else if threshold <= 300 {
		return "mixed"
	}
	return "dirty"
}

// ReconcileQoSProfiles updates QoS profiles based on current carbon intensity
func (c *CarbonQoSController) ReconcileQoSProfiles(ctx context.Context) error {
	klog.V(2).Info("Reconciling carbon-aware QoS profiles")

	// Get current carbon intensity data from ConfigMap
	carbonData, err := c.getCarbonIntensityData(ctx)
	if err != nil {
		return fmt.Errorf("failed to get carbon intensity data: %v", err)
	}

	// Update node topology information
	if err := c.updateNodeTopology(ctx, carbonData); err != nil {
		return fmt.Errorf("failed to update node topology: %v", err)
	}

	// Determine optimal QoS profiles for each zone
	for zone, intensity := range carbonData {
		profile := c.determineOptimalQoSProfile(intensity)
		if err := c.updateZoneQoSProfile(ctx, zone, profile); err != nil {
			klog.Errorf("Failed to update QoS profile for zone %s: %v", zone, err)
			continue
		}

		// Update metrics
		c.updateMetrics(zone, intensity, profile)
	}

	return nil
}

// getCarbonIntensityData retrieves carbon intensity from ConfigMap
func (c *CarbonQoSController) getCarbonIntensityData(ctx context.Context) (map[string]float64, error) {
	cm, err := c.kubeClient.CoreV1().ConfigMaps("default").Get(ctx, "carbon-scores", metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	carbonData := make(map[string]float64)
	if data, exists := cm.Data["carbon-intensity"]; exists {
		var intensityMap map[string]interface{}
		if err := json.Unmarshal([]byte(data), &intensityMap); err != nil {
			return nil, err
		}

		for zone, value := range intensityMap {
			if intensity, ok := value.(float64); ok {
				carbonData[zone] = intensity
			}
		}
	}

	return carbonData, nil
}

// determineOptimalQoSProfile selects the best QoS profile for given carbon intensity
func (c *CarbonQoSController) determineOptimalQoSProfile(carbonIntensity float64) *QoSProfile {
	// Find the most restrictive profile that can handle this carbon intensity
	var bestProfile *QoSProfile
	
	for _, profile := range c.qosProfiles {
		if carbonIntensity <= profile.CarbonThreshold {
			if bestProfile == nil || profile.Priority > bestProfile.Priority {
				bestProfile = profile
			}
		}
	}

	// If no profile can handle the intensity, use the most tolerant one
	if bestProfile == nil {
		for _, profile := range c.qosProfiles {
			if bestProfile == nil || profile.CarbonThreshold > bestProfile.CarbonThreshold {
				bestProfile = profile
			}
		}
	}

	return bestProfile
}

// updateNodeTopology updates the carbon and topology information for nodes
func (c *CarbonQoSController) updateNodeTopology(ctx context.Context, carbonData map[string]float64) error {
	nodes, err := c.kubeClient.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return err
	}

	for _, node := range nodes.Items {
		zone := c.extractZoneFromNode(&node)
		if zone == "" {
			continue
		}

		carbonIntensity, exists := carbonData[zone]
		if !exists {
			carbonIntensity = 200.0 // Default value
		}

		topology := &CarbonNodeTopology{
			NodeName:        node.Name,
			Zone:            zone,
			CarbonIntensity: carbonIntensity,
			EnergySource:    c.estimateEnergySource(zone, carbonIntensity),
			QoSClass:        c.determineNodeQoSClass(carbonIntensity),
			ResourceProfile: c.getNodeResourceProfile(&node),
			TopologyInfo:    c.getNodeTopologyInfo(&node),
			LastUpdated:     time.Now(),
		}

		c.nodeTopology[node.Name] = topology

		// Update energy consumption metrics
		c.updateEnergyMetrics(topology)
	}

	return nil
}

// extractZoneFromNode extracts the availability zone from node labels
func (c *CarbonQoSController) extractZoneFromNode(node *v1.Node) string {
	if zone, exists := node.Labels["topology.kubernetes.io/zone"]; exists {
		return zone
	}
	if zone, exists := node.Labels["failure-domain.beta.kubernetes.io/zone"]; exists {
		return zone
	}
	return ""
}

// estimateEnergySource estimates the energy source breakdown for a zone
func (c *CarbonQoSController) estimateEnergySource(zone string, carbonIntensity float64) EnergySourceBreakdown {
	// Simple heuristic based on carbon intensity
	// In practice, this would come from real energy data APIs
	
	renewable := 100.0 - (carbonIntensity / 10.0) // Higher carbon = less renewable
	if renewable < 0 {
		renewable = 0
	}
	if renewable > 100 {
		renewable = 100
	}

	nuclear := 20.0 // Assume 20% nuclear baseline
	remaining := 100.0 - renewable - nuclear
	
	gas := remaining * 0.6  // 60% of remaining is gas
	coal := remaining * 0.4 // 40% of remaining is coal

	return EnergySourceBreakdown{
		Renewable:   renewable,
		Nuclear:     nuclear,
		Gas:         gas,
		Coal:        coal,
		LastUpdated: time.Now(),
	}
}

// determineNodeQoSClass determines the QoS class for a node based on carbon intensity
func (c *CarbonQoSController) determineNodeQoSClass(carbonIntensity float64) string {
	if carbonIntensity <= 100 {
		return "green-guaranteed"
	} else if carbonIntensity <= 300 {
		return "mixed-burstable"
	}
	return "dirty-besteffort"
}

// getNodeResourceProfile gets the resource profile for a node
func (c *CarbonQoSController) getNodeResourceProfile(node *v1.Node) ResourceProfile {
	// In practice, this would query actual performance metrics
	// For now, we'll use estimates based on node capacity
	
	cpu := node.Status.Capacity.Cpu().MilliValue()
	memory := node.Status.Capacity.Memory().Value()

	return ResourceProfile{
		CPUEfficiency:    float64(cpu) / 100.0,    // Estimated performance per watt
		MemoryEfficiency: float64(memory) / 1e9 / 50.0, // Estimated GB per watt
		NetworkLatency:   time.Millisecond * 10,   // Estimated 10ms
		StorageIOPS:      10000,                   // Estimated 10K IOPS
		PowerUsage:       200.0,                   // Estimated 200W
	}
}

// getNodeTopologyInfo gets the topology information for a node
func (c *CarbonQoSController) getNodeTopologyInfo(node *v1.Node) TopologyInfo {
	// In practice, this would query actual topology information
	// For now, we'll create a basic topology structure
	
	return TopologyInfo{
		NUMANodes: []NUMANode{
			{
				ID:       0,
				CPUs:     []int{0, 1, 2, 3},
				Memory:   8 * 1024 * 1024 * 1024, // 8GB
				Distance: []int{10, 20},
				PowerW:   100.0,
			},
		},
		GPUs: []GPUInfo{}, // No GPUs by default
		NetworkDevices: []NetworkDevice{
			{
				Name:         "eth0",
				BandwidthMbps: 1000,
				LatencyMs:    1.0,
				PowerW:       10.0,
			},
		},
		PowerZones: []PowerZone{
			{
				ID:          "zone-1",
				MaxPowerW:   1000.0,
				CurrentW:    200.0,
				Efficiency:  1.3, // PUE
				CoolingType: "air",
			},
		},
	}
}

// updateZoneQoSProfile updates the QoS profile for a specific zone
func (c *CarbonQoSController) updateZoneQoSProfile(ctx context.Context, zone string, profile *QoSProfile) error {
	// Create or update ConfigMap with QoS profile information
	cmName := fmt.Sprintf("carbon-qos-%s", zone)
	
	profileData, err := json.Marshal(profile)
	if err != nil {
		return err
	}

	cm := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      cmName,
			Namespace: "default",
			Labels: map[string]string{
				"app":                    "carbon-kube",
				"component":              "qos-profile",
				"carbon-kube.io/zone":    zone,
				"carbon-kube.io/profile": profile.Name,
			},
		},
		Data: map[string]string{
			"profile.json": string(profileData),
			"zone":         zone,
			"updated":      time.Now().Format(time.RFC3339),
		},
	}

	_, err = c.kubeClient.CoreV1().ConfigMaps("default").Get(ctx, cmName, metav1.GetOptions{})
	if err != nil {
		// ConfigMap doesn't exist, create it
		_, err = c.kubeClient.CoreV1().ConfigMaps("default").Create(ctx, cm, metav1.CreateOptions{})
	} else {
		// ConfigMap exists, update it
		_, err = c.kubeClient.CoreV1().ConfigMaps("default").Update(ctx, cm, metav1.UpdateOptions{})
	}

	if err != nil {
		return fmt.Errorf("failed to update QoS profile ConfigMap for zone %s: %v", zone, err)
	}

	klog.V(2).Infof("Updated QoS profile for zone %s to %s", zone, profile.Name)
	return nil
}

// updateMetrics updates Prometheus metrics
func (c *CarbonQoSController) updateMetrics(zone string, carbonIntensity float64, profile *QoSProfile) {
	carbonClass := c.getCarbonClass(carbonIntensity)
	
	// Update carbon efficiency metric
	efficiency := profile.ResourceGuarantee / carbonIntensity
	c.metrics.carbonEfficiency.WithLabelValues("", zone, profile.Name).Set(efficiency)
}

// updateEnergyMetrics updates energy consumption metrics
func (c *CarbonQoSController) updateEnergyMetrics(topology *CarbonNodeTopology) {
	// Update energy consumption for different components
	c.metrics.energyConsumption.WithLabelValues(
		topology.NodeName, "cpu", topology.TopologyInfo.PowerZones[0].ID,
	).Set(topology.ResourceProfile.PowerUsage * 0.7) // 70% for CPU

	c.metrics.energyConsumption.WithLabelValues(
		topology.NodeName, "memory", topology.TopologyInfo.PowerZones[0].ID,
	).Set(topology.ResourceProfile.PowerUsage * 0.2) // 20% for memory

	c.metrics.energyConsumption.WithLabelValues(
		topology.NodeName, "network", topology.TopologyInfo.PowerZones[0].ID,
	).Set(topology.ResourceProfile.PowerUsage * 0.1) // 10% for network
}

// GetQoSProfile returns the QoS profile for a given name
func (c *CarbonQoSController) GetQoSProfile(name string) (*QoSProfile, bool) {
	profile, exists := c.qosProfiles[name]
	return profile, exists
}

// GetNodeTopology returns the carbon topology for a given node
func (c *CarbonQoSController) GetNodeTopology(nodeName string) (*CarbonNodeTopology, bool) {
	topology, exists := c.nodeTopology[nodeName]
	return topology, exists
}

// ListQoSProfiles returns all available QoS profiles
func (c *CarbonQoSController) ListQoSProfiles() map[string]*QoSProfile {
	profiles := make(map[string]*QoSProfile)
	for name, profile := range c.qosProfiles {
		profiles[name] = profile
	}
	return profiles
}

// Start begins the controller's reconciliation loop
func (c *CarbonQoSController) Start(ctx context.Context) error {
	klog.Info("Starting Carbon QoS Controller")

	ticker := time.NewTicker(30 * time.Second) // Reconcile every 30 seconds
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			klog.Info("Stopping Carbon QoS Controller")
			return ctx.Err()
		case <-ticker.C:
			if err := c.ReconcileQoSProfiles(ctx); err != nil {
				klog.Errorf("Failed to reconcile QoS profiles: %v", err)
			}
		}
	}
}