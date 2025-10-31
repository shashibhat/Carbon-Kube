package katalyst

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

const (
	// Plugin name
	KatalystCarbonPluginName = "KatalystCarbonScheduler"
	
	// Scoring weights
	CarbonIntensityWeight = 40 // 40% weight for carbon intensity
	QoSAffinityWeight     = 30 // 30% weight for QoS affinity
	TopologyWeight        = 20 // 20% weight for topology optimization
	ResourceEffWeight     = 10 // 10% weight for resource efficiency
	
	// Carbon intensity thresholds
	GreenThreshold  = 100.0 // gCO2/kWh
	MixedThreshold  = 300.0 // gCO2/kWh
	DirtyThreshold  = 500.0 // gCO2/kWh
	
	// QoS annotations
	QoSProfileAnnotation     = "carbon-kube.io/qos-profile"
	CarbonBudgetAnnotation   = "carbon-kube.io/carbon-budget"
	EnergyEfficiencyAnnotation = "carbon-kube.io/energy-efficiency"
	TopologyAffinityAnnotation = "carbon-kube.io/topology-affinity"
)

// KatalystCarbonPlugin integrates carbon awareness with Katalyst's QoS and topology management
type KatalystCarbonPlugin struct {
	handle          framework.Handle
	kubeClient      kubernetes.Interface
	qosController   *CarbonQoSController
	metrics         *KatalystCarbonMetrics
	config          *KatalystCarbonConfig
}

// KatalystCarbonConfig holds configuration for the enhanced plugin
type KatalystCarbonConfig struct {
	EnableTopologyAware    bool    `json:"enableTopologyAware"`
	EnableQoSOptimization  bool    `json:"enableQoSOptimization"`
	CarbonWeightMultiplier float64 `json:"carbonWeightMultiplier"`
	MaxMigrationFrequency  int     `json:"maxMigrationFrequency"` // per hour
	EnergyEfficiencyTarget float64 `json:"energyEfficiencyTarget"` // Target PUE
}

// KatalystCarbonMetrics holds enhanced metrics
type KatalystCarbonMetrics struct {
	schedulingDecisions    *prometheus.CounterVec
	carbonOptimization     *prometheus.GaugeVec
	qosViolations         *prometheus.CounterVec
	topologyOptimization  *prometheus.GaugeVec
	energyEfficiency      *prometheus.GaugeVec
	migrationRecommendations *prometheus.CounterVec
}

// NodeScore represents the comprehensive scoring for a node
type NodeScore struct {
	NodeName           string
	CarbonScore        float64
	QoSScore          float64
	TopologyScore     float64
	ResourceEffScore  float64
	TotalScore        float64
	CarbonIntensity   float64
	QoSClass          string
	RecommendMigration bool
}

// PodRequirements represents the carbon and QoS requirements of a pod
type PodRequirements struct {
	QoSProfile         string
	CarbonBudget       float64 // kg CO2 per day
	EnergyEfficiency   float64 // Target PUE
	TopologyAffinity   []string
	ResourceGuarantee  float64 // Percentage
	TolerateHighCarbon bool
}

// New creates a new KatalystCarbonPlugin
func New(obj runtime.Object, h framework.Handle) (framework.Plugin, error) {
	config := &KatalystCarbonConfig{
		EnableTopologyAware:    true,
		EnableQoSOptimization:  true,
		CarbonWeightMultiplier: 1.0,
		MaxMigrationFrequency:  10,
		EnergyEfficiencyTarget: 1.5,
	}

	// Parse configuration if provided
	if obj != nil {
		// Configuration parsing would go here
		klog.V(2).Info("Using default Katalyst Carbon plugin configuration")
	}

	kubeClient, err := kubernetes.NewForConfig(h.ClientSet().RESTConfig())
	if err != nil {
		return nil, fmt.Errorf("failed to create Kubernetes client: %v", err)
	}

	qosController, err := NewCarbonQoSController(kubeClient)
	if err != nil {
		return nil, fmt.Errorf("failed to create QoS controller: %v", err)
	}

	metrics := &KatalystCarbonMetrics{
		schedulingDecisions: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "katalyst_carbon_scheduling_decisions_total",
				Help: "Total scheduling decisions made by Katalyst Carbon plugin",
			},
			[]string{"decision_type", "qos_class", "carbon_class"},
		),
		carbonOptimization: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "katalyst_carbon_optimization_score",
				Help: "Carbon optimization score for scheduled pods",
			},
			[]string{"node", "zone", "qos_class"},
		),
		qosViolations: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "katalyst_qos_violations_total",
				Help: "Total QoS violations detected",
			},
			[]string{"violation_type", "qos_class", "node"},
		),
		topologyOptimization: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "katalyst_topology_optimization_score",
				Help: "Topology optimization score",
			},
			[]string{"node", "numa_node", "optimization_type"},
		),
		energyEfficiency: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "katalyst_energy_efficiency_pue",
				Help: "Power Usage Effectiveness (PUE) metric",
			},
			[]string{"node", "power_zone"},
		),
		migrationRecommendations: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "katalyst_migration_recommendations_total",
				Help: "Total migration recommendations made",
			},
			[]string{"from_node", "to_node", "reason"},
		),
	}

	plugin := &KatalystCarbonPlugin{
		handle:        h,
		kubeClient:    kubeClient,
		qosController: qosController,
		metrics:       metrics,
		config:        config,
	}

	// Start the QoS controller
	go func() {
		ctx := context.Background()
		if err := qosController.Start(ctx); err != nil {
			klog.Errorf("QoS controller failed: %v", err)
		}
	}()

	klog.Info("Katalyst Carbon Plugin initialized successfully")
	return plugin, nil
}

// Name returns the plugin name
func (p *KatalystCarbonPlugin) Name() string {
	return KatalystCarbonPluginName
}

// Filter implements the Filter extension point
func (p *KatalystCarbonPlugin) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	node := nodeInfo.Node()
	if node == nil {
		return framework.NewStatus(framework.Error, "node not found")
	}

	// Extract pod requirements
	podReqs := p.extractPodRequirements(pod)

	// Get node topology and carbon information
	nodeTopology, exists := p.qosController.GetNodeTopology(node.Name)
	if !exists {
		klog.V(4).Infof("No topology information for node %s, allowing scheduling", node.Name)
		return framework.NewStatus(framework.Success, "")
	}

	// Check carbon budget constraints
	if !p.checkCarbonBudget(podReqs, nodeTopology) {
		p.metrics.qosViolations.WithLabelValues("carbon_budget", podReqs.QoSProfile, node.Name).Inc()
		return framework.NewStatus(framework.Unschedulable, 
			fmt.Sprintf("node %s exceeds carbon budget for pod", node.Name))
	}

	// Check QoS profile compatibility
	if !p.checkQoSCompatibility(podReqs, nodeTopology) {
		p.metrics.qosViolations.WithLabelValues("qos_incompatible", podReqs.QoSProfile, node.Name).Inc()
		return framework.NewStatus(framework.Unschedulable, 
			fmt.Sprintf("node %s QoS class incompatible with pod requirements", node.Name))
	}

	// Check topology affinity if enabled
	if p.config.EnableTopologyAware && !p.checkTopologyAffinity(podReqs, nodeTopology) {
		return framework.NewStatus(framework.Unschedulable, 
			fmt.Sprintf("node %s topology doesn't match pod affinity", node.Name))
	}

	// Check energy efficiency requirements
	if !p.checkEnergyEfficiency(podReqs, nodeTopology) {
		p.metrics.qosViolations.WithLabelValues("energy_efficiency", podReqs.QoSProfile, node.Name).Inc()
		return framework.NewStatus(framework.Unschedulable, 
			fmt.Sprintf("node %s doesn't meet energy efficiency requirements", node.Name))
	}

	return framework.NewStatus(framework.Success, "")
}

// Score implements the Score extension point
func (p *KatalystCarbonPlugin) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := p.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}

	node := nodeInfo.Node()
	if node == nil {
		return 0, framework.NewStatus(framework.Error, "node not found")
	}

	// Extract pod requirements
	podReqs := p.extractPodRequirements(pod)

	// Get node topology and carbon information
	nodeTopology, exists := p.qosController.GetNodeTopology(node.Name)
	if !exists {
		klog.V(4).Infof("No topology information for node %s, using default score", node.Name)
		return 50, framework.NewStatus(framework.Success, "")
	}

	// Calculate comprehensive node score
	nodeScore := p.calculateNodeScore(podReqs, nodeTopology, node)

	// Update metrics
	p.updateScoringMetrics(nodeScore, podReqs)

	// Check if migration should be recommended
	if nodeScore.RecommendMigration {
		p.recommendMigration(pod, node, nodeScore)
	}

	// Convert to framework score (0-100)
	score := int64(nodeScore.TotalScore)
	if score > 100 {
		score = 100
	}
	if score < 0 {
		score = 0
	}

	p.metrics.schedulingDecisions.WithLabelValues(
		"score", podReqs.QoSProfile, p.getCarbonClass(nodeScore.CarbonIntensity),
	).Inc()

	return score, framework.NewStatus(framework.Success, "")
}

// extractPodRequirements extracts carbon and QoS requirements from pod annotations
func (p *KatalystCarbonPlugin) extractPodRequirements(pod *v1.Pod) *PodRequirements {
	reqs := &PodRequirements{
		QoSProfile:         "mixed-burstable", // Default
		CarbonBudget:       50.0,              // Default 50kg CO2/day
		EnergyEfficiency:   1.5,               // Default PUE
		TopologyAffinity:   []string{},
		ResourceGuarantee:  80.0,              // Default 80%
		TolerateHighCarbon: true,              // Default tolerant
	}

	if pod.Annotations == nil {
		return reqs
	}

	// Extract QoS profile
	if profile, exists := pod.Annotations[QoSProfileAnnotation]; exists {
		reqs.QoSProfile = profile
	}

	// Extract carbon budget
	if budget, exists := pod.Annotations[CarbonBudgetAnnotation]; exists {
		if val, err := strconv.ParseFloat(budget, 64); err == nil {
			reqs.CarbonBudget = val
		}
	}

	// Extract energy efficiency target
	if efficiency, exists := pod.Annotations[EnergyEfficiencyAnnotation]; exists {
		if val, err := strconv.ParseFloat(efficiency, 64); err == nil {
			reqs.EnergyEfficiency = val
		}
	}

	// Extract topology affinity
	if affinity, exists := pod.Annotations[TopologyAffinityAnnotation]; exists {
		var affinityList []string
		if err := json.Unmarshal([]byte(affinity), &affinityList); err == nil {
			reqs.TopologyAffinity = affinityList
		}
	}

	// Determine tolerance based on QoS profile
	if qosProfile, exists := p.qosController.GetQoSProfile(reqs.QoSProfile); exists {
		reqs.TolerateHighCarbon = qosProfile.TolerateHighCarbon
		reqs.ResourceGuarantee = qosProfile.ResourceGuarantee
	}

	return reqs
}

// checkCarbonBudget checks if the node meets the pod's carbon budget requirements
func (p *KatalystCarbonPlugin) checkCarbonBudget(podReqs *PodRequirements, nodeTopology *CarbonNodeTopology) bool {
	// Estimate daily carbon consumption for this pod on this node
	estimatedConsumption := p.estimatePodCarbonConsumption(podReqs, nodeTopology)
	
	return estimatedConsumption <= podReqs.CarbonBudget
}

// checkQoSCompatibility checks if the node's QoS class is compatible with pod requirements
func (p *KatalystCarbonPlugin) checkQoSCompatibility(podReqs *PodRequirements, nodeTopology *CarbonNodeTopology) bool {
	// Check if pod can tolerate the node's carbon intensity
	if !podReqs.TolerateHighCarbon && nodeTopology.CarbonIntensity > MixedThreshold {
		return false
	}

	// Check if QoS profile matches or is compatible
	if qosProfile, exists := p.qosController.GetQoSProfile(podReqs.QoSProfile); exists {
		return nodeTopology.CarbonIntensity <= qosProfile.CarbonThreshold
	}

	return true
}

// checkTopologyAffinity checks if the node topology matches pod affinity requirements
func (p *KatalystCarbonPlugin) checkTopologyAffinity(podReqs *PodRequirements, nodeTopology *CarbonNodeTopology) bool {
	if len(podReqs.TopologyAffinity) == 0 {
		return true // No specific requirements
	}

	// Check NUMA affinity
	for _, affinity := range podReqs.TopologyAffinity {
		switch affinity {
		case "numa-local":
			// Prefer single NUMA node placement
			if len(nodeTopology.TopologyInfo.NUMANodes) > 1 {
				return true // Multi-NUMA available
			}
		case "gpu-accelerated":
			// Require GPU availability
			if len(nodeTopology.TopologyInfo.GPUs) == 0 {
				return false
			}
		case "high-bandwidth":
			// Require high network bandwidth
			hasHighBandwidth := false
			for _, netDev := range nodeTopology.TopologyInfo.NetworkDevices {
				if netDev.BandwidthMbps >= 1000 {
					hasHighBandwidth = true
					break
				}
			}
			if !hasHighBandwidth {
				return false
			}
		}
	}

	return true
}

// checkEnergyEfficiency checks if the node meets energy efficiency requirements
func (p *KatalystCarbonPlugin) checkEnergyEfficiency(podReqs *PodRequirements, nodeTopology *CarbonNodeTopology) bool {
	// Check PUE (Power Usage Effectiveness)
	for _, powerZone := range nodeTopology.TopologyInfo.PowerZones {
		if powerZone.Efficiency <= podReqs.EnergyEfficiency {
			return true // At least one power zone meets requirements
		}
	}

	return false
}

// calculateNodeScore calculates a comprehensive score for the node
func (p *KatalystCarbonPlugin) calculateNodeScore(podReqs *PodRequirements, nodeTopology *CarbonNodeTopology, node *v1.Node) *NodeScore {
	score := &NodeScore{
		NodeName:        node.Name,
		CarbonIntensity: nodeTopology.CarbonIntensity,
		QoSClass:        nodeTopology.QoSClass,
	}

	// Calculate carbon score (lower intensity = higher score)
	score.CarbonScore = p.calculateCarbonScore(nodeTopology.CarbonIntensity)

	// Calculate QoS score
	score.QoSScore = p.calculateQoSScore(podReqs, nodeTopology)

	// Calculate topology score
	score.TopologyScore = p.calculateTopologyScore(podReqs, nodeTopology)

	// Calculate resource efficiency score
	score.ResourceEffScore = p.calculateResourceEfficiencyScore(nodeTopology)

	// Calculate weighted total score
	score.TotalScore = (score.CarbonScore * CarbonIntensityWeight +
		score.QoSScore * QoSAffinityWeight +
		score.TopologyScore * TopologyWeight +
		score.ResourceEffScore * ResourceEffWeight) / 100.0

	// Apply carbon weight multiplier
	score.TotalScore *= p.config.CarbonWeightMultiplier

	// Check if migration should be recommended
	score.RecommendMigration = p.shouldRecommendMigration(podReqs, nodeTopology)

	return score
}

// calculateCarbonScore calculates the carbon intensity score
func (p *KatalystCarbonPlugin) calculateCarbonScore(carbonIntensity float64) float64 {
	// Inverse relationship: lower carbon intensity = higher score
	if carbonIntensity <= GreenThreshold {
		return 100.0 // Perfect green energy
	} else if carbonIntensity <= MixedThreshold {
		// Linear interpolation between 100 and 60
		return 100.0 - ((carbonIntensity - GreenThreshold) / (MixedThreshold - GreenThreshold) * 40.0)
	} else if carbonIntensity <= DirtyThreshold {
		// Linear interpolation between 60 and 20
		return 60.0 - ((carbonIntensity - MixedThreshold) / (DirtyThreshold - MixedThreshold) * 40.0)
	} else {
		// Very high carbon intensity
		return math.Max(0.0, 20.0 - (carbonIntensity - DirtyThreshold) / 100.0)
	}
}

// calculateQoSScore calculates the QoS compatibility score
func (p *KatalystCarbonPlugin) calculateQoSScore(podReqs *PodRequirements, nodeTopology *CarbonNodeTopology) float64 {
	qosProfile, exists := p.qosController.GetQoSProfile(podReqs.QoSProfile)
	if !exists {
		return 50.0 // Default score
	}

	score := 50.0

	// Bonus for matching QoS class
	if nodeTopology.QoSClass == podReqs.QoSProfile {
		score += 30.0
	}

	// Bonus for carbon threshold compatibility
	if nodeTopology.CarbonIntensity <= qosProfile.CarbonThreshold {
		score += 20.0
	}

	// Penalty for exceeding carbon threshold
	if nodeTopology.CarbonIntensity > qosProfile.CarbonThreshold {
		excess := (nodeTopology.CarbonIntensity - qosProfile.CarbonThreshold) / qosProfile.CarbonThreshold
		score -= excess * 30.0
	}

	return math.Max(0.0, math.Min(100.0, score))
}

// calculateTopologyScore calculates the topology optimization score
func (p *KatalystCarbonPlugin) calculateTopologyScore(podReqs *PodRequirements, nodeTopology *CarbonNodeTopology) float64 {
	if !p.config.EnableTopologyAware {
		return 50.0 // Neutral score if topology awareness is disabled
	}

	score := 50.0

	// Score based on NUMA topology
	if len(nodeTopology.TopologyInfo.NUMANodes) > 0 {
		score += 20.0 // Bonus for NUMA awareness
		
		// Additional bonus for balanced NUMA nodes
		totalMemory := int64(0)
		for _, numa := range nodeTopology.TopologyInfo.NUMANodes {
			totalMemory += numa.Memory
		}
		avgMemory := totalMemory / int64(len(nodeTopology.TopologyInfo.NUMANodes))
		
		balanced := true
		for _, numa := range nodeTopology.TopologyInfo.NUMANodes {
			if math.Abs(float64(numa.Memory - avgMemory)) / float64(avgMemory) > 0.2 {
				balanced = false
				break
			}
		}
		
		if balanced {
			score += 10.0
		}
	}

	// Score based on topology affinity requirements
	for _, affinity := range podReqs.TopologyAffinity {
		switch affinity {
		case "numa-local":
			if len(nodeTopology.TopologyInfo.NUMANodes) > 0 {
				score += 15.0
			}
		case "gpu-accelerated":
			if len(nodeTopology.TopologyInfo.GPUs) > 0 {
				score += 20.0
			}
		case "high-bandwidth":
			for _, netDev := range nodeTopology.TopologyInfo.NetworkDevices {
				if netDev.BandwidthMbps >= 1000 {
					score += 15.0
					break
				}
			}
		}
	}

	return math.Max(0.0, math.Min(100.0, score))
}

// calculateResourceEfficiencyScore calculates the resource efficiency score
func (p *KatalystCarbonPlugin) calculateResourceEfficiencyScore(nodeTopology *CarbonNodeTopology) float64 {
	score := 50.0

	// Score based on CPU efficiency
	if nodeTopology.ResourceProfile.CPUEfficiency > 10.0 {
		score += 15.0
	}

	// Score based on memory efficiency
	if nodeTopology.ResourceProfile.MemoryEfficiency > 0.1 {
		score += 15.0
	}

	// Score based on power usage efficiency
	for _, powerZone := range nodeTopology.TopologyInfo.PowerZones {
		if powerZone.Efficiency <= 1.3 { // Good PUE
			score += 20.0
		} else if powerZone.Efficiency <= 1.5 { // Acceptable PUE
			score += 10.0
		}
		break // Use first power zone for scoring
	}

	return math.Max(0.0, math.Min(100.0, score))
}

// shouldRecommendMigration determines if migration should be recommended
func (p *KatalystCarbonPlugin) shouldRecommendMigration(podReqs *PodRequirements, nodeTopology *CarbonNodeTopology) bool {
	// Recommend migration if carbon intensity is very high and pod doesn't tolerate it
	if !podReqs.TolerateHighCarbon && nodeTopology.CarbonIntensity > DirtyThreshold {
		return true
	}

	// Recommend migration if energy efficiency is poor
	for _, powerZone := range nodeTopology.TopologyInfo.PowerZones {
		if powerZone.Efficiency > podReqs.EnergyEfficiency * 1.5 {
			return true
		}
	}

	return false
}

// estimatePodCarbonConsumption estimates the daily carbon consumption for a pod
func (p *KatalystCarbonPlugin) estimatePodCarbonConsumption(podReqs *PodRequirements, nodeTopology *CarbonNodeTopology) float64 {
	// Estimate based on resource profile and carbon intensity
	// This is a simplified calculation - in practice, this would be more sophisticated
	
	basePowerW := nodeTopology.ResourceProfile.PowerUsage * 0.1 // Assume pod uses 10% of node power
	dailyEnergyKWh := basePowerW * 24 / 1000 // Convert to kWh per day
	carbonKg := dailyEnergyKWh * nodeTopology.CarbonIntensity / 1000 // Convert gCO2 to kg
	
	return carbonKg
}

// getCarbonClass returns the carbon classification for intensity
func (p *KatalystCarbonPlugin) getCarbonClass(intensity float64) string {
	if intensity <= GreenThreshold {
		return "green"
	} else if intensity <= MixedThreshold {
		return "mixed"
	}
	return "dirty"
}

// updateScoringMetrics updates Prometheus metrics after scoring
func (p *KatalystCarbonPlugin) updateScoringMetrics(nodeScore *NodeScore, podReqs *PodRequirements) {
	// Update carbon optimization score
	p.metrics.carbonOptimization.WithLabelValues(
		nodeScore.NodeName, "", nodeScore.QoSClass,
	).Set(nodeScore.CarbonScore)

	// Update topology optimization score
	p.metrics.topologyOptimization.WithLabelValues(
		nodeScore.NodeName, "0", "carbon_aware",
	).Set(nodeScore.TopologyScore)
}

// recommendMigration logs and metrics for migration recommendations
func (p *KatalystCarbonPlugin) recommendMigration(pod *v1.Pod, node *v1.Node, nodeScore *NodeScore) {
	klog.V(2).Infof("Recommending migration for pod %s/%s from node %s due to carbon/efficiency concerns", 
		pod.Namespace, pod.Name, node.Name)

	p.metrics.migrationRecommendations.WithLabelValues(
		node.Name, "", "carbon_efficiency",
	).Inc()
}

// ScoreExtensions returns the score extensions
func (p *KatalystCarbonPlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

// Ensure the plugin implements the required interfaces
var _ framework.FilterPlugin = &KatalystCarbonPlugin{}
var _ framework.ScorePlugin = &KatalystCarbonPlugin{}