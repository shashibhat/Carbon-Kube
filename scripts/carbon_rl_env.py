#!/usr/bin/env python3
"""
Carbon RL Environment Module

This module provides the Gymnasium environment for reinforcement learning
in carbon-aware workload migration decisions.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import logging

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
    Gymnasium environment for carbon-aware migration decisions.
    
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
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        if not self.replay_buffer:
            # Return default response if no replay data
            return self.state, 0.0, True, False, {}
        
        event = self.replay_buffer[self.current_event_idx]
        
        # Calculate reward
        reward = 0.0
        if action == 1:  # Migrate
            if event.success:
                # Positive reward for successful migration
                reward = event.saved_co2 - (self.penalty_factor * max(0, event.latency_delta))
            else:
                # Negative reward for failed migration
                reward = -event.latency_delta * self.penalty_factor
        else:  # Hold
            # Small negative reward for not migrating when beneficial
            if event.intensity_delta > 50:  # Significant intensity difference
                reward = -event.intensity_delta * 0.01
        
        # Move to next event
        self.current_event_idx = (self.current_event_idx + 1) % len(self.replay_buffer)
        event = self.replay_buffer[self.current_event_idx]
        self.state = self._event_to_state(event)
        
        # Done if we've cycled through all events
        done = self.current_event_idx == 0
        truncated = False
        
        return self.state, reward, done, truncated, {}
    
    def _event_to_state(self, event: MigrationEvent) -> np.ndarray:
        """Convert migration event to state vector."""
        # Extract time of day from timestamp
        from datetime import datetime
        time_of_day = datetime.fromtimestamp(event.timestamp).hour
        
        # Estimate latency risk (0-1) based on latency delta
        latency_risk = min(1.0, max(0.0, abs(event.latency_delta) / 3600.0))  # Normalize by hour
        
        # Estimate zone load (placeholder - would come from actual metrics)
        zone_load = 0.5  # Default moderate load
        
        return np.array([
            event.intensity_delta,
            latency_risk,
            event.threshold_used,
            float(time_of_day),
            zone_load
        ], dtype=np.float32)

