#!/usr/bin/env python3
"""
Reinforcement Learning Tuner for Carbon-Kube

This component uses offline reinforcement learning to optimize migration thresholds
based on historical migration events and their outcomes.
"""

import json
import logging
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MigrationEvent:
    """Represents a migration event for RL training."""
    job_id: str
    from_zone: str
    to_zone: str
    intensity_delta: float  # Difference in carbon intensity
    latency_delta: float    # Change in job completion time (seconds)
    saved_co2: float       # CO2 saved in kg
    timestamp: int
    threshold_used: float
    success: bool


@dataclass
class RLState:
    """State representation for the RL environment."""
    intensity_delta: float      # Current vs target zone intensity difference
    latency_risk: float        # Estimated latency impact (0-1)
    current_threshold: float   # Current migration threshold
    time_of_day: float        # Hour of day (0-23)
    zone_load: float          # Current zone utilization (0-1)


class CarbonMigrationEnv(gym.Env):
    """
    Gym environment for carbon-aware migration decisions.
    
    State: [intensity_delta, latency_risk, current_threshold, time_of_day, zone_load]
    Action: 0 (hold), 1 (migrate)
    Reward: saved_co2 - (latency_penalty * latency_delta)
    """
    
    def __init__(self, replay_buffer: deque, penalty_factor: float = 10.0):
        super().__init__()
        
        self.replay_buffer = replay_buffer
        self.penalty_factor = penalty_factor
        self.current_event_idx = 0
        
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(2)  # 0: hold, 1: migrate
        
        # Observation space: [intensity_delta, latency_risk, threshold, time_of_day, zone_load]
        self.observation_space = gym.spaces.Box(
            low=np.array([-1000.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([1000.0, 1.0, 500.0, 23.0, 1.0]),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if not self.replay_buffer:
            # Default state if no replay data
            self.state = np.array([0.0, 0.5, 200.0, 12.0, 0.5], dtype=np.float32)
        else:
            self.current_event_idx = np.random.randint(0, len(self.replay_buffer))
            event = self.replay_buffer[self.current_event_idx]
            self.state = self._event_to_state(event)
        
        return self.state, {}
    
    def step(self, action):
        """Execute action and return next state, reward, done, info."""
        if not self.replay_buffer:
            # No replay data available
            return self.state, 0.0, True, False, {}
        
        event = self.replay_buffer[self.current_event_idx]
        
        # Calculate reward based on action
        if action == 1:  # Migrate
            reward = event.saved_co2 - (self.penalty_factor * abs(event.latency_delta))
            if not event.success:
                reward -= 50.0  # Penalty for failed migrations
        else:  # Hold
            reward = 0.0  # No immediate reward for holding
            if event.intensity_delta > 100:  # Missed opportunity
                reward -= event.intensity_delta * 0.1
        
        # Move to next event
        self.current_event_idx = (self.current_event_idx + 1) % len(self.replay_buffer)
        next_event = self.replay_buffer[self.current_event_idx]
        self.state = self._event_to_state(next_event)
        
        # Episode ends after processing all events or randomly
        done = np.random.random() < 0.1  # 10% chance to end episode
        
        info = {
            'saved_co2': event.saved_co2 if action == 1 else 0.0,
            'latency_delta': event.latency_delta if action == 1 else 0.0,
            'migration_success': event.success if action == 1 else True
        }
        
        return self.state, reward, done, False, info
    
    def _event_to_state(self, event: MigrationEvent) -> np.ndarray:
        """Convert migration event to RL state."""
        # Extract time of day from timestamp
        dt = datetime.fromtimestamp(event.timestamp)
        time_of_day = dt.hour + dt.minute / 60.0
        
        # Estimate zone load (simplified)
        zone_load = 0.5 + 0.3 * np.sin(2 * np.pi * time_of_day / 24)  # Sinusoidal pattern
        zone_load = np.clip(zone_load, 0.0, 1.0)
        
        return np.array([
            event.intensity_delta,
            event.latency_delta / 3600.0,  # Normalize to hours
            event.threshold_used,
            time_of_day,
            zone_load
        ], dtype=np.float32)


class RLTuner:
    """Main RL tuner class for optimizing migration thresholds."""
    
    def __init__(self, model_path: str = "/tmp/carbon_rl_model.pkl"):
        self.model_path = model_path
        self.replay_buffer = deque(maxlen=1000)  # Store last 1000 events
        self.model = None
        self.env = None
        
        # RL hyperparameters
        self.penalty_factor = 10.0
        self.learning_rate = 3e-4
        self.training_episodes = 100
        self.update_frequency = 3600  # Update every hour
        
        # Threshold bounds
        self.min_threshold = 50.0
        self.max_threshold = 500.0
        self.current_threshold = 200.0
        
        # Initialize Kubernetes client
        try:
            config.load_incluster_config()
        except config.ConfigException:
            try:
                config.load_kube_config()
            except config.ConfigException:
                logger.error("Could not load Kubernetes configuration")
                return
        
        self.k8s_client = client.CoreV1Api()
        self.apps_client = client.AppsV1Api()
        
        # Load existing model if available
        self._load_model()
        
        logger.info("RL Tuner initialized")
    
    def _load_model(self):
        """Load existing RL model from disk."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.replay_buffer = model_data.get('replay_buffer', deque(maxlen=1000))
                    self.current_threshold = model_data.get('threshold', 200.0)
                
                # Recreate environment and model
                if self.replay_buffer:
                    self.env = CarbonMigrationEnv(self.replay_buffer, self.penalty_factor)
                    self.model = PPO.load(self.model_path.replace('.pkl', '_ppo.zip'))
                    logger.info(f"Loaded existing model with {len(self.replay_buffer)} events")
                else:
                    self._initialize_model()
            else:
                self._initialize_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize new RL model."""
        self.env = CarbonMigrationEnv(self.replay_buffer, self.penalty_factor)
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            verbose=1,
            tensorboard_log="/tmp/carbon_rl_logs/"
        )
        logger.info("Initialized new RL model")
    
    def _save_model(self):
        """Save RL model and replay buffer to disk."""
        try:
            # Save model
            if self.model:
                self.model.save(self.model_path.replace('.pkl', '_ppo.zip'))
            
            # Save replay buffer and metadata
            model_data = {
                'replay_buffer': self.replay_buffer,
                'threshold': self.current_threshold,
                'last_updated': time.time()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Saved RL model and replay buffer")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def collect_migration_events(self) -> List[MigrationEvent]:
        """Collect migration events from Kubernetes events and logs."""
        events = []
        
        try:
            # Watch for pod events that indicate migrations
            v1 = client.CoreV1Api()
            
            # Get recent events (last hour)
            field_selector = f"involvedObject.kind=Pod"
            time_threshold = datetime.utcnow() - timedelta(hours=1)
            
            k8s_events = v1.list_event_for_all_namespaces(
                field_selector=field_selector,
                limit=100
            )
            
            for event in k8s_events.items:
                if event.reason in ['Scheduled', 'FailedScheduling', 'Preempted']:
                    migration_event = self._parse_k8s_event(event)
                    if migration_event:
                        events.append(migration_event)
            
            logger.info(f"Collected {len(events)} migration events")
            
        except Exception as e:
            logger.error(f"Failed to collect migration events: {e}")
        
        return events
    
    def _parse_k8s_event(self, k8s_event) -> Optional[MigrationEvent]:
        """Parse Kubernetes event into MigrationEvent."""
        try:
            # Extract information from event
            pod_name = k8s_event.involved_object.name
            namespace = k8s_event.involved_object.namespace
            
            # Get pod details
            try:
                pod = self.k8s_client.read_namespaced_pod(pod_name, namespace)
                
                # Extract zone information
                node_name = pod.spec.node_name
                if node_name:
                    node = self.k8s_client.read_node(node_name)
                    zone = node.metadata.labels.get('topology.kubernetes.io/zone', 'unknown')
                else:
                    zone = 'unknown'
                
                # Simulate migration data (in real implementation, this would come from metrics)
                intensity_delta = np.random.uniform(-200, 200)  # Placeholder
                latency_delta = np.random.uniform(-300, 300)    # Placeholder
                saved_co2 = max(0, intensity_delta * 0.001)     # Placeholder calculation
                
                return MigrationEvent(
                    job_id=pod_name,
                    from_zone='unknown',  # Would need to track previous zone
                    to_zone=zone,
                    intensity_delta=intensity_delta,
                    latency_delta=latency_delta,
                    saved_co2=saved_co2,
                    timestamp=int(k8s_event.first_timestamp.timestamp()),
                    threshold_used=self.current_threshold,
                    success=k8s_event.reason == 'Scheduled'
                )
                
            except ApiException:
                return None
                
        except Exception as e:
            logger.error(f"Failed to parse K8s event: {e}")
            return None
    
    def update_replay_buffer(self, events: List[MigrationEvent]):
        """Update replay buffer with new events."""
        for event in events:
            self.replay_buffer.append(event)
        
        logger.info(f"Updated replay buffer: {len(self.replay_buffer)} total events")
    
    def train_model(self):
        """Train the RL model on replay buffer data."""
        if len(self.replay_buffer) < 10:
            logger.warning("Not enough data for training (need at least 10 events)")
            return
        
        try:
            # Update environment with new replay buffer
            self.env = CarbonMigrationEnv(self.replay_buffer, self.penalty_factor)
            
            if self.model is None:
                self._initialize_model()
            else:
                # Update model environment
                self.model.set_env(self.env)
            
            # Train model
            logger.info(f"Training RL model on {len(self.replay_buffer)} events")
            self.model.learn(total_timesteps=self.training_episodes * 100)
            
            # Evaluate and update threshold
            self._update_threshold()
            
            # Save model
            self._save_model()
            
            logger.info("RL model training completed")
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
    
    def _update_threshold(self):
        """Update migration threshold based on model predictions."""
        if not self.model or len(self.replay_buffer) < 5:
            return
        
        try:
            # Sample states from replay buffer
            sample_states = []
            for _ in range(min(50, len(self.replay_buffer))):
                event = self.replay_buffer[np.random.randint(len(self.replay_buffer))]
                state = self.env._event_to_state(event)
                sample_states.append(state)
            
            sample_states = np.array(sample_states)
            
            # Get model predictions
            actions, _ = self.model.predict(sample_states, deterministic=True)
            
            # Calculate optimal threshold based on action patterns
            migrate_decisions = actions == 1
            if np.any(migrate_decisions):
                migrate_states = sample_states[migrate_decisions]
                # Use average intensity delta of migration decisions as new threshold
                new_threshold = np.mean(migrate_states[:, 0])  # intensity_delta column
                
                # Bound the threshold
                new_threshold = np.clip(new_threshold, self.min_threshold, self.max_threshold)
                
                # Smooth the update
                alpha = 0.1  # Learning rate for threshold updates
                self.current_threshold = (1 - alpha) * self.current_threshold + alpha * new_threshold
                
                logger.info(f"Updated threshold: {self.current_threshold:.1f} gCO2/kWh")
            
        except Exception as e:
            logger.error(f"Failed to update threshold: {e}")
    
    def update_configmap_threshold(self):
        """Update the threshold in Kubernetes ConfigMap."""
        try:
            # Get existing ConfigMap
            configmap = self.k8s_client.read_namespaced_config_map(
                name="carbon-scores",
                namespace="default"
            )
            
            # Update threshold
            if configmap.data is None:
                configmap.data = {}
            
            configmap.data["threshold"] = str(self.current_threshold)
            configmap.data["rl_last_updated"] = datetime.utcnow().isoformat()
            
            # Patch ConfigMap
            self.k8s_client.patch_namespaced_config_map(
                name="carbon-scores",
                namespace="default",
                body=configmap
            )
            
            logger.info(f"Updated ConfigMap threshold to {self.current_threshold:.1f}")
            
        except Exception as e:
            logger.error(f"Failed to update ConfigMap threshold: {e}")
    
    def run_training_cycle(self):
        """Run one complete training cycle."""
        logger.info("Starting RL training cycle")
        
        try:
            # Collect new migration events
            events = self.collect_migration_events()
            
            # Update replay buffer
            self.update_replay_buffer(events)
            
            # Train model if we have enough data
            if len(self.replay_buffer) >= 10:
                self.train_model()
                
                # Update ConfigMap with new threshold
                self.update_configmap_threshold()
            else:
                logger.info(f"Waiting for more data: {len(self.replay_buffer)}/10 events")
            
            logger.info("RL training cycle completed")
            
        except Exception as e:
            logger.error(f"RL training cycle failed: {e}")
    
    def run_continuous(self):
        """Run continuous RL tuning."""
        logger.info(f"Starting continuous RL tuning (update every {self.update_frequency}s)")
        
        while True:
            try:
                self.run_training_cycle()
                time.sleep(self.update_frequency)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down")
                break
            except Exception as e:
                logger.error(f"Unexpected error in RL tuner: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying


def main():
    """Main entry point."""
    tuner = RLTuner()
    
    # Check if running as one-shot or continuous
    run_once = os.getenv('RUN_ONCE', 'false').lower() == 'true'
    
    if run_once:
        tuner.run_training_cycle()
    else:
        tuner.run_continuous()


if __name__ == "__main__":
    main()