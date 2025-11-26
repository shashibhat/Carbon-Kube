#!/usr/bin/env python3
"""
Enhanced RL Tuner for Carbon-Kube with Katalyst Integration

This module extends the original RL tuner to work with Katalyst-core's QoS model
and resource management, providing carbon-aware optimization that considers
topology, QoS profiles, and energy efficiency.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class KatalystQoSMetrics:
    """Katalyst QoS metrics for carbon-aware optimization"""
    node_name: str
    qos_class: str
    carbon_intensity: float
    energy_efficiency: float  # PUE
    resource_utilization: Dict[str, float]
    topology_score: float
    numa_efficiency: float
    power_consumption: float
    thermal_state: str
    migration_cost: float
    sla_compliance: float
    timestamp: datetime

@dataclass
class CarbonOptimizationState:
    """State representation for carbon optimization"""
    carbon_intensity: np.ndarray  # Per zone carbon intensity
    qos_metrics: Dict[str, KatalystQoSMetrics]
    resource_demand: np.ndarray  # Resource demand forecast
    energy_prices: np.ndarray  # Energy price forecast
    renewable_forecast: np.ndarray  # Renewable energy forecast
    topology_constraints: Dict[str, Any]
    migration_history: List[Dict]
    sla_violations: int
    carbon_budget_remaining: float
    time_of_day: float
    day_of_week: int

@dataclass
class OptimizationAction:
    """Action representation for carbon optimization"""
    carbon_threshold_adjustments: np.ndarray  # Per QoS class threshold adjustments
    migration_decisions: Dict[str, str]  # Pod -> target node mappings
    qos_profile_updates: Dict[str, str]  # Node -> QoS profile mappings
    power_management: Dict[str, float]  # Node -> power cap adjustments
    topology_preferences: Dict[str, List[str]]  # Workload -> preferred topology

class KatalystCarbonEnvironment(gym.Env):
    """
    Gymnasium environment for Katalyst-aware carbon optimization
    
    This environment integrates with Katalyst's QoS model to optimize
    carbon emissions while maintaining performance and SLA compliance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.k8s_client = None
        self.current_state = None
        self.episode_length = config.get('episode_length', 288)  # 24 hours in 5-min intervals
        self.current_step = 0
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.k8s_client = client.ApiClient()
        self.core_v1 = client.CoreV1Api()
        self.custom_api = client.CustomObjectsApi()
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Metrics tracking
        self.metrics_history = []
        self.carbon_savings = 0.0
        self.sla_violations = 0
        self.migration_count = 0
        
        logger.info("Katalyst Carbon Environment initialized")
    
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Observation space: carbon intensity, QoS metrics, resource utilization, etc.
        obs_dim = (
            10 +  # Carbon intensity per zone (max 10 zones)
            50 +  # QoS metrics (10 nodes * 5 metrics each)
            20 +  # Resource utilization metrics
            10 +  # Energy and topology metrics
            5     # Time and context features
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: threshold adjustments, migration decisions, QoS updates
        action_dim = (
            3 +   # Carbon threshold adjustments per QoS class
            10 +  # Migration decisions (binary per node)
            10 +  # QoS profile updates (categorical per node)
            10 +  # Power management adjustments
            5     # Topology preference weights
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.carbon_savings = 0.0
        self.sla_violations = 0
        self.migration_count = 0
        
        # Get initial state from Kubernetes cluster
        self.current_state = self._get_cluster_state()
        
        observation = self._state_to_observation(self.current_state)
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        # Parse action
        parsed_action = self._parse_action(action)
        
        # Apply action to the cluster
        reward, info = self._apply_action(parsed_action)
        
        # Update state
        self.current_state = self._get_cluster_state()
        observation = self._state_to_observation(self.current_state)
        
        # Check if episode is done
        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        # Update metrics
        self._update_metrics(parsed_action, reward, info)
        
        return observation, reward, terminated, truncated, info
    
    def _get_cluster_state(self) -> CarbonOptimizationState:
        """Get current cluster state from Kubernetes and Katalyst"""
        try:
            # Get carbon intensity data
            carbon_data = self._get_carbon_intensity_data()
            
            # Get Katalyst QoS metrics
            qos_metrics = self._get_katalyst_qos_metrics()
            
            # Get resource demand forecast
            resource_demand = self._get_resource_demand_forecast()
            
            # Get energy and pricing data
            energy_prices = self._get_energy_prices()
            renewable_forecast = self._get_renewable_forecast()
            
            # Get topology constraints
            topology_constraints = self._get_topology_constraints()
            
            # Get migration history
            migration_history = self._get_migration_history()
            
            # Calculate remaining carbon budget
            carbon_budget_remaining = self._calculate_carbon_budget_remaining()
            
            # Get time context
            now = datetime.now()
            time_of_day = now.hour + now.minute / 60.0
            day_of_week = now.weekday()
            
            return CarbonOptimizationState(
                carbon_intensity=carbon_data,
                qos_metrics=qos_metrics,
                resource_demand=resource_demand,
                energy_prices=energy_prices,
                renewable_forecast=renewable_forecast,
                topology_constraints=topology_constraints,
                migration_history=migration_history,
                sla_violations=self.sla_violations,
                carbon_budget_remaining=carbon_budget_remaining,
                time_of_day=time_of_day,
                day_of_week=day_of_week
            )
            
        except Exception as e:
            logger.error(f"Error getting cluster state: {e}")
            return self._get_default_state()
    
    def _get_carbon_intensity_data(self) -> np.ndarray:
        """Get carbon intensity data from ConfigMap"""
        try:
            cm = self.core_v1.read_namespaced_config_map(
                name="carbon-scores", namespace="default"
            )
            
            carbon_data = json.loads(cm.data.get("carbon-intensity", "{}"))
            
            # Convert to numpy array (pad/truncate to 10 zones)
            intensities = []
            zones = sorted(carbon_data.keys())[:10]  # Max 10 zones
            
            for i in range(10):
                if i < len(zones):
                    intensities.append(carbon_data[zones[i]])
                else:
                    intensities.append(200.0)  # Default value
            
            return np.array(intensities, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"Could not get carbon intensity data: {e}")
            return np.full(10, 200.0, dtype=np.float32)  # Default values
    
    def _get_katalyst_qos_metrics(self) -> Dict[str, KatalystQoSMetrics]:
        """Get Katalyst QoS metrics from cluster"""
        qos_metrics = {}
        
        try:
            # Get nodes
            nodes = self.core_v1.list_node()
            
            for node in nodes.items[:10]:  # Max 10 nodes
                node_name = node.metadata.name
                
                # Get QoS class from node labels/annotations
                qos_class = node.metadata.labels.get(
                    "carbon-kube.io/qos-class", "mixed-burstable"
                )
                
                # Get carbon intensity for node's zone
                zone = node.metadata.labels.get("topology.kubernetes.io/zone", "default")
                carbon_intensity = self._get_zone_carbon_intensity(zone)
                
                # Get resource utilization (simplified)
                resource_util = self._get_node_resource_utilization(node_name)
                
                # Create QoS metrics object
                qos_metrics[node_name] = KatalystQoSMetrics(
                    node_name=node_name,
                    qos_class=qos_class,
                    carbon_intensity=carbon_intensity,
                    energy_efficiency=1.5,  # Default PUE
                    resource_utilization=resource_util,
                    topology_score=75.0,  # Default score
                    numa_efficiency=80.0,  # Default efficiency
                    power_consumption=200.0,  # Default watts
                    thermal_state="normal",
                    migration_cost=10.0,  # Default cost
                    sla_compliance=95.0,  # Default compliance
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            logger.error(f"Error getting Katalyst QoS metrics: {e}")
        
        return qos_metrics
    
    def _get_zone_carbon_intensity(self, zone: str) -> float:
        """Get carbon intensity for a specific zone"""
        try:
            cm = self.core_v1.read_namespaced_config_map(
                name="carbon-scores", namespace="default"
            )
            
            carbon_data = json.loads(cm.data.get("carbon-intensity", "{}"))
            return carbon_data.get(zone, 200.0)
            
        except:
            return 200.0  # Default value
    
    def _get_node_resource_utilization(self, node_name: str) -> Dict[str, float]:
        """Get resource utilization for a node"""
        # In practice, this would query metrics server or Prometheus
        return {
            "cpu": np.random.uniform(0.3, 0.8),
            "memory": np.random.uniform(0.4, 0.7),
            "network": np.random.uniform(0.2, 0.6),
            "storage": np.random.uniform(0.1, 0.5)
        }
    
    def _get_resource_demand_forecast(self) -> np.ndarray:
        """Get resource demand forecast"""
        # Simplified forecast - in practice, this would use historical data
        base_demand = 0.6
        time_factor = np.sin(2 * np.pi * datetime.now().hour / 24) * 0.2
        noise = np.random.normal(0, 0.1)
        
        demand = base_demand + time_factor + noise
        return np.array([max(0.1, min(1.0, demand))] * 4, dtype=np.float32)  # CPU, Memory, Network, Storage
    
    def _get_energy_prices(self) -> np.ndarray:
        """Get energy price forecast"""
        # Simplified pricing model
        hour = datetime.now().hour
        if 9 <= hour <= 17:  # Peak hours
            base_price = 0.15
        elif 18 <= hour <= 22:  # Evening peak
            base_price = 0.20
        else:  # Off-peak
            base_price = 0.10
        
        return np.array([base_price] * 24, dtype=np.float32)  # 24-hour forecast
    
    def _get_renewable_forecast(self) -> np.ndarray:
        """Get renewable energy forecast"""
        # Simplified renewable forecast
        hour = datetime.now().hour
        if 6 <= hour <= 18:  # Daylight hours (solar)
            renewable_pct = 0.4 + 0.3 * np.sin(np.pi * (hour - 6) / 12)
        else:  # Night hours (wind)
            renewable_pct = 0.2 + 0.1 * np.random.random()
        
        return np.array([renewable_pct] * 24, dtype=np.float32)
    
    def _get_topology_constraints(self) -> Dict[str, Any]:
        """Get topology constraints from Katalyst"""
        return {
            "numa_affinity": True,
            "gpu_locality": True,
            "network_topology": "leaf-spine",
            "power_domains": 2,
            "cooling_zones": 1
        }
    
    def _get_migration_history(self) -> List[Dict]:
        """Get recent migration history"""
        # In practice, this would query migration events from Kubernetes events
        return []
    
    def _calculate_carbon_budget_remaining(self) -> float:
        """Calculate remaining carbon budget for the day"""
        # Simplified budget calculation
        daily_budget = 100.0  # kg CO2
        hour = datetime.now().hour
        used_budget = daily_budget * (hour / 24.0) * 0.8  # Assume 80% usage rate
        
        return max(0.0, daily_budget - used_budget)
    
    def _get_default_state(self) -> CarbonOptimizationState:
        """Get default state when cluster state is unavailable"""
        return CarbonOptimizationState(
            carbon_intensity=np.full(10, 200.0, dtype=np.float32),
            qos_metrics={},
            resource_demand=np.array([0.6, 0.5, 0.4, 0.3], dtype=np.float32),
            energy_prices=np.full(24, 0.15, dtype=np.float32),
            renewable_forecast=np.full(24, 0.3, dtype=np.float32),
            topology_constraints={},
            migration_history=[],
            sla_violations=0,
            carbon_budget_remaining=50.0,
            time_of_day=12.0,
            day_of_week=1
        )
    
    def _state_to_observation(self, state: CarbonOptimizationState) -> np.ndarray:
        """Convert state to observation vector"""
        obs = []
        
        # Carbon intensity (10 values)
        obs.extend(state.carbon_intensity)
        
        # QoS metrics (50 values: 10 nodes * 5 metrics)
        qos_values = []
        node_names = sorted(state.qos_metrics.keys())[:10]
        
        for i in range(10):
            if i < len(node_names):
                metrics = state.qos_metrics[node_names[i]]
                qos_values.extend([
                    metrics.carbon_intensity / 1000.0,  # Normalize
                    metrics.energy_efficiency / 3.0,    # Normalize PUE
                    metrics.topology_score / 100.0,     # Normalize score
                    metrics.numa_efficiency / 100.0,    # Normalize efficiency
                    metrics.power_consumption / 500.0   # Normalize power
                ])
            else:
                qos_values.extend([0.2, 0.5, 0.75, 0.8, 0.4])  # Default values
        
        obs.extend(qos_values)
        
        # Resource demand (4 values)
        obs.extend(state.resource_demand)
        
        # Energy prices (average of 24-hour forecast)
        obs.append(np.mean(state.energy_prices))
        
        # Renewable forecast (average of 24-hour forecast)
        obs.append(np.mean(state.renewable_forecast))
        
        # Additional metrics (18 values to reach target dimension)
        obs.extend([
            state.sla_violations / 100.0,  # Normalize violations
            state.carbon_budget_remaining / 100.0,  # Normalize budget
            state.time_of_day / 24.0,  # Normalize time
            state.day_of_week / 7.0,   # Normalize day
            len(state.migration_history) / 10.0,  # Normalize migration count
        ])
        
        # Pad to reach exact observation dimension (95 values)
        while len(obs) < 95:
            obs.append(0.0)
        
        return np.array(obs[:95], dtype=np.float32)
    
    def _parse_action(self, action: np.ndarray) -> OptimizationAction:
        """Parse action vector into structured action"""
        idx = 0
        
        # Carbon threshold adjustments (3 values for 3 QoS classes)
        threshold_adj = action[idx:idx+3] * 100.0  # Scale to ±100 gCO2/kWh
        idx += 3
        
        # Migration decisions (10 binary decisions)
        migration_raw = action[idx:idx+10]
        migration_decisions = {}
        # Convert to binary decisions (>0 means migrate)
        for i, val in enumerate(migration_raw):
            if val > 0.5:
                migration_decisions[f"node-{i}"] = f"target-node-{(i+1)%10}"
        idx += 10
        
        # QoS profile updates (10 categorical decisions)
        qos_raw = action[idx:idx+10]
        qos_profiles = ["green-guaranteed", "mixed-burstable", "dirty-besteffort"]
        qos_updates = {}
        for i, val in enumerate(qos_raw):
            profile_idx = int((val + 1) * 1.5) % 3  # Map [-1,1] to [0,2]
            qos_updates[f"node-{i}"] = qos_profiles[profile_idx]
        idx += 10
        
        # Power management (10 power cap adjustments)
        power_adj = action[idx:idx+10] * 0.2  # Scale to ±20% power adjustment
        power_management = {f"node-{i}": adj for i, adj in enumerate(power_adj)}
        idx += 10
        
        # Topology preferences (5 preference weights)
        topo_prefs = action[idx:idx+5]
        topology_preferences = {
            "cpu-intensive": ["numa-local"] if topo_prefs[0] > 0 else [],
            "memory-intensive": ["numa-local", "high-bandwidth"] if topo_prefs[1] > 0 else [],
            "gpu-workload": ["gpu-accelerated"] if topo_prefs[2] > 0 else [],
            "network-intensive": ["high-bandwidth"] if topo_prefs[3] > 0 else [],
            "storage-intensive": ["numa-local"] if topo_prefs[4] > 0 else []
        }
        
        return OptimizationAction(
            carbon_threshold_adjustments=threshold_adj,
            migration_decisions=migration_decisions,
            qos_profile_updates=qos_updates,
            power_management=power_management,
            topology_preferences=topology_preferences
        )
    
    def _apply_action(self, action: OptimizationAction) -> Tuple[float, Dict]:
        """Apply action to the cluster and calculate reward"""
        reward = 0.0
        info = {}
        
        try:
            # Apply carbon threshold adjustments
            carbon_reward = self._apply_carbon_thresholds(action.carbon_threshold_adjustments)
            reward += carbon_reward
            
            # Apply migration decisions
            migration_reward = self._apply_migrations(action.migration_decisions)
            reward += migration_reward
            
            # Apply QoS profile updates
            qos_reward = self._apply_qos_updates(action.qos_profile_updates)
            reward += qos_reward
            
            # Apply power management
            power_reward = self._apply_power_management(action.power_management)
            reward += power_reward
            
            # Update topology preferences
            topo_reward = self._apply_topology_preferences(action.topology_preferences)
            reward += topo_reward
            
            # Calculate penalty for SLA violations
            sla_penalty = self._calculate_sla_penalty()
            reward -= sla_penalty
            
            info = {
                "carbon_reward": carbon_reward,
                "migration_reward": migration_reward,
                "qos_reward": qos_reward,
                "power_reward": power_reward,
                "topology_reward": topo_reward,
                "sla_penalty": sla_penalty,
                "total_reward": reward
            }
            
        except Exception as e:
            logger.error(f"Error applying action: {e}")
            reward = -10.0  # Penalty for failed actions
            info = {"error": str(e)}
        
        return reward, info
    
    def _apply_carbon_thresholds(self, adjustments: np.ndarray) -> float:
        """Apply carbon threshold adjustments and return reward"""
        reward = 0.0
        
        try:
            # Get current QoS profiles
            qos_profiles = ["green-guaranteed", "mixed-burstable", "dirty-besteffort"]
            
            for i, adjustment in enumerate(adjustments):
                if i < len(qos_profiles):
                    profile_name = qos_profiles[i]
                    
                    # Update ConfigMap with new threshold
                    cm_name = f"carbon-qos-{profile_name}"
                    try:
                        cm = self.core_v1.read_namespaced_config_map(
                            name=cm_name, namespace="default"
                        )
                        
                        profile_data = json.loads(cm.data["profile.json"])
                        old_threshold = profile_data["carbonThreshold"]
                        new_threshold = max(50.0, old_threshold + adjustment)
                        profile_data["carbonThreshold"] = new_threshold
                        
                        cm.data["profile.json"] = json.dumps(profile_data)
                        self.core_v1.patch_namespaced_config_map(
                            name=cm_name, namespace="default", body=cm
                        )
                        
                        # Reward for reducing thresholds (encouraging green energy)
                        if adjustment < 0:
                            reward += abs(adjustment) / 100.0
                        
                        logger.info(f"Updated {profile_name} threshold: {old_threshold} -> {new_threshold}")
                        
                    except ApiException as e:
                        if e.status == 404:
                            logger.warning(f"QoS profile ConfigMap {cm_name} not found")
                        else:
                            raise
            
        except Exception as e:
            logger.error(f"Error applying carbon thresholds: {e}")
            reward = -1.0
        
        return reward
    
    def _apply_migrations(self, migrations: Dict[str, str]) -> float:
        """Apply migration decisions and return reward"""
        reward = 0.0
        
        # In a real implementation, this would trigger pod migrations
        # For now, we'll simulate the effect
        
        for source_node, target_node in migrations.items():
            # Calculate migration benefit (simplified)
            source_carbon = self._get_node_carbon_intensity(source_node)
            target_carbon = self._get_node_carbon_intensity(target_node)
            
            if target_carbon < source_carbon:
                # Reward for migrating to lower carbon node
                carbon_improvement = (source_carbon - target_carbon) / source_carbon
                reward += carbon_improvement * 2.0
                self.migration_count += 1
                
                logger.info(f"Beneficial migration: {source_node} -> {target_node} "
                           f"(carbon reduction: {carbon_improvement:.2%})")
            else:
                # Penalty for migrating to higher carbon node
                reward -= 0.5
        
        # Penalty for excessive migrations
        if len(migrations) > 3:
            reward -= (len(migrations) - 3) * 0.5
        
        return reward
    
    def _get_node_carbon_intensity(self, node_name: str) -> float:
        """Get carbon intensity for a node"""
        if self.current_state and node_name in self.current_state.qos_metrics:
            return self.current_state.qos_metrics[node_name].carbon_intensity
        return 200.0  # Default value
    
    def _apply_qos_updates(self, qos_updates: Dict[str, str]) -> float:
        """Apply QoS profile updates and return reward"""
        reward = 0.0
        
        for node_name, qos_profile in qos_updates.items():
            try:
                # Update node annotation with new QoS profile
                # In practice, this would patch the node object
                node_carbon = self._get_node_carbon_intensity(node_name)
                
                # Reward for matching QoS profile to carbon intensity
                if qos_profile == "green-guaranteed" and node_carbon <= 100:
                    reward += 1.0
                elif qos_profile == "mixed-burstable" and 100 < node_carbon <= 300:
                    reward += 0.5
                elif qos_profile == "dirty-besteffort" and node_carbon > 300:
                    reward += 0.2
                else:
                    reward -= 0.3  # Penalty for mismatch
                
                logger.debug(f"Updated QoS profile for {node_name}: {qos_profile}")
                
            except Exception as e:
                logger.error(f"Error updating QoS profile for {node_name}: {e}")
                reward -= 0.1
        
        return reward
    
    def _apply_power_management(self, power_adjustments: Dict[str, float]) -> float:
        """Apply power management adjustments and return reward"""
        reward = 0.0
        
        for node_name, adjustment in power_adjustments.items():
            # Reward for reducing power consumption
            if adjustment < 0:  # Power reduction
                reward += abs(adjustment) * 2.0
            else:  # Power increase
                reward -= adjustment * 0.5
            
            logger.debug(f"Power adjustment for {node_name}: {adjustment:.2%}")
        
        return reward
    
    def _apply_topology_preferences(self, preferences: Dict[str, List[str]]) -> float:
        """Apply topology preferences and return reward"""
        reward = 0.0
        
        # Reward for enabling topology optimizations
        for workload_type, prefs in preferences.items():
            if prefs:  # Non-empty preferences
                reward += len(prefs) * 0.1
                logger.debug(f"Topology preferences for {workload_type}: {prefs}")
        
        return reward
    
    def _calculate_sla_penalty(self) -> float:
        """Calculate penalty for SLA violations"""
        # Simulate SLA compliance check
        violation_probability = 0.05  # 5% base violation rate
        
        if np.random.random() < violation_probability:
            self.sla_violations += 1
            return 5.0  # High penalty for SLA violations
        
        return 0.0
    
    def _update_metrics(self, action: OptimizationAction, reward: float, info: Dict):
        """Update metrics tracking"""
        metrics = {
            "timestamp": datetime.now(),
            "reward": reward,
            "carbon_savings": self.carbon_savings,
            "sla_violations": self.sla_violations,
            "migration_count": self.migration_count,
            "action_summary": {
                "threshold_adjustments": len(action.carbon_threshold_adjustments),
                "migrations": len(action.migration_decisions),
                "qos_updates": len(action.qos_profile_updates),
                "power_adjustments": len(action.power_management)
            }
        }
        
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def _get_info(self) -> Dict:
        """Get environment info"""
        return {
            "episode_step": self.current_step,
            "carbon_savings": self.carbon_savings,
            "sla_violations": self.sla_violations,
            "migration_count": self.migration_count
        }

class KatalystCarbonCallback(BaseCallback):
    """Callback for monitoring Katalyst Carbon RL training"""
    
    def __init__(self, eval_env, eval_freq=1000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_rewards = []
        self.carbon_savings = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current policy
            obs = self.eval_env.reset()[0]
            total_reward = 0
            total_carbon_savings = 0
            
            for _ in range(100):  # Short evaluation episode
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                total_reward += reward
                total_carbon_savings += info.get("carbon_reward", 0)
                
                if terminated or truncated:
                    break
            
            self.eval_rewards.append(total_reward)
            self.carbon_savings.append(total_carbon_savings)
            
            if self.verbose > 0:
                logger.info(f"Eval episode {len(self.eval_rewards)}: "
                           f"reward={total_reward:.2f}, "
                           f"carbon_savings={total_carbon_savings:.2f}")
        
        return True

class KatalystRLTuner:
    """
    Enhanced RL Tuner for Carbon-Kube with Katalyst Integration
    
    This class provides carbon-aware optimization using reinforcement learning
    while integrating with Katalyst's QoS model and topology management.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.env = None
        self.model = None
        self.training_history = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize Prometheus metrics
        self.registry = CollectorRegistry()
        self._setup_metrics()
        
        self.logger.info("Katalyst RL Tuner initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            "environment": {
                "episode_length": 288,  # 24 hours in 5-min intervals
                "max_zones": 10,
                "max_nodes": 10
            },
            "training": {
                "total_timesteps": 100000,
                "learning_rate": 3e-4,
                "batch_size": 256,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "train_freq": 1,
                "gradient_steps": 1,
                "target_update_interval": 1,
                "tau": 0.005
            },
            "evaluation": {
                "eval_freq": 1000,
                "eval_episodes": 10
            },
            "monitoring": {
                "prometheus_gateway": "localhost:9091",
                "metrics_interval": 60
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge configurations
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
    def _setup_metrics(self):
        """Setup Prometheus metrics"""
        self.metrics = {
            "training_reward": Gauge(
                "katalyst_rl_training_reward",
                "Current training reward",
                registry=self.registry
            ),
            "carbon_savings": Gauge(
                "katalyst_rl_carbon_savings",
                "Cumulative carbon savings in kg CO2",
                registry=self.registry
            ),
            "sla_violations": Counter(
                "katalyst_rl_sla_violations_total",
                "Total SLA violations",
                registry=self.registry
            ),
            "migration_events": Counter(
                "katalyst_rl_migration_events_total",
                "Total migration events",
                registry=self.registry
            ),
            "model_updates": Counter(
                "katalyst_rl_model_updates_total",
                "Total model updates",
                registry=self.registry
            ),
            "optimization_score": Gauge(
                "katalyst_rl_optimization_score",
                "Current optimization score",
                registry=self.registry
            )
        }
    
    def initialize_environment(self):
        """Initialize the Katalyst Carbon environment"""
        env_config = self.config["environment"]
        self.env = KatalystCarbonEnvironment(env_config)
        
        # Create vectorized environment for training
        self.vec_env = make_vec_env(
            lambda: self.env, n_envs=1, seed=42
        )
        
        self.logger.info("Environment initialized")
    
    def initialize_model(self, model_path: Optional[str] = None):
        """Initialize or load the SAC model"""
        if model_path and os.path.exists(model_path):
            self.model = SAC.load(model_path, env=self.vec_env)
            self.logger.info(f"Loaded model from {model_path}")
        else:
            # Create new model
            policy_kwargs = dict(
                net_arch=[256, 256, 256],  # 3-layer network
                activation_fn=torch.nn.ReLU
            )
            
            self.model = SAC(
                "MlpPolicy",
                self.vec_env,
                learning_rate=self.config["training"]["learning_rate"],
                buffer_size=self.config["training"]["buffer_size"],
                learning_starts=self.config["training"]["learning_starts"],
                batch_size=self.config["training"]["batch_size"],
                tau=self.config["training"]["tau"],
                gamma=0.99,
                train_freq=self.config["training"]["train_freq"],
                gradient_steps=self.config["training"]["gradient_steps"],
                target_update_interval=self.config["training"]["target_update_interval"],
                policy_kwargs=policy_kwargs,
                verbose=1,
                seed=42,
                device="auto"
            )
            
            self.logger.info("Created new SAC model")
    
    def train(self, save_path: Optional[str] = None):
        """Train the RL model"""
        if not self.model or not self.env:
            raise ValueError("Model and environment must be initialized before training")
        
        # Setup callback for evaluation
        eval_env = KatalystCarbonEnvironment(self.config["environment"])
        callback = KatalystCarbonCallback(
            eval_env=eval_env,
            eval_freq=self.config["evaluation"]["eval_freq"],
            verbose=1
        )
        
        self.logger.info("Starting training...")
        
        # Train the model
        self.model.learn(
            total_timesteps=self.config["training"]["total_timesteps"],
            callback=callback,
            log_interval=100,
            progress_bar=True
        )
        
        # Save the trained model
        if save_path:
            self.model.save(save_path)
            self.logger.info(f"Model saved to {save_path}")
        
        # Store training history
        self.training_history = {
            "eval_rewards": callback.eval_rewards,
            "carbon_savings": callback.carbon_savings,
            "training_timesteps": self.config["training"]["total_timesteps"]
        }
        
        self.logger.info("Training completed")
    
    def optimize(self, duration_hours: int = 24):
        """Run optimization for a specified duration"""
        if not self.model or not self.env:
            raise ValueError("Model and environment must be initialized before optimization")
        
        self.logger.info(f"Starting optimization for {duration_hours} hours...")
        
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)
        
        obs = self.env.reset()[0]
        total_reward = 0
        step_count = 0
        
        while time.time() < end_time:
            # Get action from trained model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            step_count += 1
            
            # Update metrics
            self._update_prometheus_metrics(reward, info)
            
            # Log progress
            if step_count % 100 == 0:
                self.logger.info(f"Step {step_count}: reward={reward:.2f}, "
                               f"total_reward={total_reward:.2f}")
            
            # Reset environment if episode ended
            if terminated or truncated:
                obs = self.env.reset()[0]
            
            # Sleep to maintain real-time operation
            time.sleep(5)  # 5-second intervals
        
        self.logger.info(f"Optimization completed. Total reward: {total_reward:.2f}")
        return total_reward
    
    def _update_prometheus_metrics(self, reward: float, info: Dict):
        """Update Prometheus metrics"""
        self.metrics["training_reward"].set(reward)
        
        if "carbon_reward" in info:
            self.metrics["carbon_savings"].set(info["carbon_reward"])
        
        if "sla_violations" in info:
            self.metrics["sla_violations"]._value._value = info["sla_violations"]
        
        if "migration_count" in info:
            self.metrics["migration_events"]._value._value = info["migration_count"]
        
        # Push metrics to Prometheus gateway
        try:
            gateway = self.config["monitoring"]["prometheus_gateway"]
            push_to_gateway(gateway, job="katalyst-rl-tuner", registry=self.registry)
        except Exception as e:
            self.logger.warning(f"Failed to push metrics to Prometheus: {e}")
    
    def evaluate(self, episodes: int = 10) -> Dict[str, float]:
        """Evaluate the current model"""
        if not self.model or not self.env:
            raise ValueError("Model and environment must be initialized before evaluation")
        
        self.logger.info(f"Evaluating model over {episodes} episodes...")
        
        rewards = []
        carbon_savings = []
        sla_violations = []
        
        for episode in range(episodes):
            obs = self.env.reset()[0]
            episode_reward = 0
            episode_carbon = 0
            episode_sla = 0
            
            for step in range(100):  # Limit episode length
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_carbon += info.get("carbon_reward", 0)
                episode_sla += info.get("sla_violations", 0)
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
            carbon_savings.append(episode_carbon)
            sla_violations.append(episode_sla)
        
        results = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_carbon_savings": np.mean(carbon_savings),
            "total_sla_violations": sum(sla_violations)
        }
        
        self.logger.info(f"Evaluation results: {results}")
        return results
    
    def visualize_training(self, save_path: Optional[str] = None):
        """Visualize training progress"""
        if not self.training_history:
            self.logger.warning("No training history available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Katalyst Carbon RL Training Progress", fontsize=16)
        
        # Plot evaluation rewards
        axes[0, 0].plot(self.training_history["eval_rewards"])
        axes[0, 0].set_title("Evaluation Rewards")
        axes[0, 0].set_xlabel("Evaluation Episode")
        axes[0, 0].set_ylabel("Total Reward")
        axes[0, 0].grid(True)
        
        # Plot carbon savings
        axes[0, 1].plot(self.training_history["carbon_savings"])
        axes[0, 1].set_title("Carbon Savings")
        axes[0, 1].set_xlabel("Evaluation Episode")
        axes[0, 1].set_ylabel("Carbon Savings")
        axes[0, 1].grid(True)
        
        # Plot reward distribution
        axes[1, 0].hist(self.training_history["eval_rewards"], bins=20, alpha=0.7)
        axes[1, 0].set_title("Reward Distribution")
        axes[1, 0].set_xlabel("Reward")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].grid(True)
        
        # Plot carbon savings distribution
        axes[1, 1].hist(self.training_history["carbon_savings"], bins=20, alpha=0.7)
        axes[1, 1].set_title("Carbon Savings Distribution")
        axes[1, 1].set_xlabel("Carbon Savings")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training visualization saved to {save_path}")
        
        plt.show()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive optimization report"""
        if not self.env:
            return {"error": "Environment not initialized"}
        
        # Get current environment state
        current_state = self.env.current_state
        
        # Calculate summary statistics
        report = {
            "timestamp": datetime.now().isoformat(),
            "environment_info": {
                "total_nodes": len(current_state.qos_metrics) if current_state else 0,
                "carbon_budget_remaining": current_state.carbon_budget_remaining if current_state else 0,
                "sla_violations": self.env.sla_violations,
                "migration_count": self.env.migration_count
            },
            "carbon_metrics": {
                "average_intensity": np.mean(current_state.carbon_intensity) if current_state else 0,
                "min_intensity": np.min(current_state.carbon_intensity) if current_state else 0,
                "max_intensity": np.max(current_state.carbon_intensity) if current_state else 0,
                "renewable_percentage": np.mean(current_state.renewable_forecast) * 100 if current_state else 0
            },
            "qos_metrics": {},
            "optimization_recommendations": []
        }
        
        # Add QoS metrics per node
        if current_state and current_state.qos_metrics:
            for node_name, qos_metrics in current_state.qos_metrics.items():
                report["qos_metrics"][node_name] = {
                    "qos_class": qos_metrics.qos_class,
                    "carbon_intensity": qos_metrics.carbon_intensity,
                    "energy_efficiency": qos_metrics.energy_efficiency,
                    "topology_score": qos_metrics.topology_score,
                    "sla_compliance": qos_metrics.sla_compliance
                }
        
        # Generate optimization recommendations
        if current_state:
            recommendations = self._generate_recommendations(current_state)
            report["optimization_recommendations"] = recommendations
        
        # Add training history if available
        if self.training_history:
            report["training_summary"] = {
                "total_episodes": len(self.training_history["eval_rewards"]),
                "best_reward": max(self.training_history["eval_rewards"]),
                "average_reward": np.mean(self.training_history["eval_rewards"]),
                "total_carbon_savings": sum(self.training_history["carbon_savings"])
            }
        
        return report
    
    def _generate_recommendations(self, state: CarbonOptimizationState) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on current state"""
        recommendations = []
        
        # Check for high carbon intensity nodes
        for node_name, qos_metrics in state.qos_metrics.items():
            if qos_metrics.carbon_intensity > 400:  # High carbon threshold
                recommendations.append({
                    "type": "carbon_reduction",
                    "priority": "high",
                    "node": node_name,
                    "description": f"Node {node_name} has high carbon intensity ({qos_metrics.carbon_intensity:.1f} gCO2/kWh)",
                    "action": "Consider migrating workloads to greener nodes or updating QoS profile"
                })
        
        # Check for poor energy efficiency
        for node_name, qos_metrics in state.qos_metrics.items():
            if qos_metrics.energy_efficiency > 2.0:  # Poor PUE
                recommendations.append({
                    "type": "energy_efficiency",
                    "priority": "medium",
                    "node": node_name,
                    "description": f"Node {node_name} has poor energy efficiency (PUE: {qos_metrics.energy_efficiency:.2f})",
                    "action": "Investigate cooling and power management optimizations"
                })
        
        # Check carbon budget
        if state.carbon_budget_remaining < 10:  # Low budget
            recommendations.append({
                "type": "carbon_budget",
                "priority": "high",
                "description": f"Carbon budget critically low ({state.carbon_budget_remaining:.1f} kg CO2 remaining)",
                "action": "Implement aggressive carbon reduction measures"
            })
        
        # Check for renewable energy opportunities
        current_renewable = np.mean(state.renewable_forecast)
        if current_renewable > 0.6:  # High renewable availability
            recommendations.append({
                "type": "renewable_opportunity",
                "priority": "low",
                "description": f"High renewable energy availability ({current_renewable:.1%})",
                "action": "Consider scheduling energy-intensive workloads now"
            })
        
        return recommendations

async def main():
    """Main function for running the Katalyst RL Tuner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Katalyst Carbon RL Tuner")
    parser.add_argument("--mode", choices=["train", "optimize", "evaluate"], 
                       default="optimize", help="Operation mode")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--model-path", type=str, help="Model file path")
    parser.add_argument("--duration", type=int, default=24, 
                       help="Optimization duration in hours")
    parser.add_argument("--episodes", type=int, default=10, 
                       help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    # Initialize tuner
    tuner = KatalystRLTuner(config_path=args.config)
    tuner.initialize_environment()
    tuner.initialize_model(model_path=args.model_path)
    
    if args.mode == "train":
        logger.info("Starting training mode...")
        tuner.train(save_path="katalyst_carbon_model.zip")
        tuner.visualize_training(save_path="training_progress.png")
        
    elif args.mode == "optimize":
        logger.info("Starting optimization mode...")
        total_reward = tuner.optimize(duration_hours=args.duration)
        
        # Generate and save report
        report = tuner.generate_report()
        with open("optimization_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Optimization completed with total reward: {total_reward:.2f}")
        
    elif args.mode == "evaluate":
        logger.info("Starting evaluation mode...")
        results = tuner.evaluate(episodes=args.episodes)
        
        print("\nEvaluation Results:")
        print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean Carbon Savings: {results['mean_carbon_savings']:.2f}")
        print(f"Total SLA Violations: {results['total_sla_violations']}")

if __name__ == "__main__":
    asyncio.run(main())