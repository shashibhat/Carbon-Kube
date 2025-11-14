from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple


State = Tuple[float, float]  # (current_score, latency_risk)
Action = int  # -1, 0, +1 adjustment steps


@dataclass
class RLTunerConfig:
    """Configuration values for the RL tuner."""

    buffer_size: int = 1000
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 0.1
    threshold_values: List[float] | None = None

    def __post_init__(self) -> None:
        if self.threshold_values is None:
            self.threshold_values = [100.0, 200.0, 300.0]


class RLTuner:
    """Tabular Q-learning tuner for the migration threshold."""

    def __init__(self, config: RLTunerConfig | None = None) -> None:
        self._cfg = config or RLTunerConfig()
        self._buffer: Deque[Tuple[State, Action, float, State]] = deque(
            maxlen=self._cfg.buffer_size
        )
        self._q_table: Dict[Tuple[State, Action], float] = {}
        self._threshold_index: int = 1  # default middle

    @property
    def current_threshold(self) -> float:
        return self._cfg.threshold_values[self._threshold_index]

    def select_action(self, state: State) -> Action:
        """Epsilon-greedy action selection."""
        if random.random() < self._cfg.epsilon:
            return random.choice([-1, 0, 1])

        best_action = 0
        best_value = float("-inf")
        for action in (-1, 0, 1):
            key = (state, action)
            value = self._q_table.get(key, 0.0)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def observe(self, state: State, action: Action, reward: float, next_state: State) -> None:
        """Record a transition and update the Q-table."""
        self._buffer.append((state, action, reward, next_state))
        self._update_q(state, action, reward, next_state)
        self._apply_action(action)

    def _apply_action(self, action: Action) -> None:
        new_index = self._threshold_index + action
        new_index = max(0, min(len(self._cfg.threshold_values) - 1, new_index))
        self._threshold_index = new_index

    def _update_q(self, state: State, action: Action, reward: float, next_state: State) -> None:
        key = (state, action)
        old_value = self._q_table.get(key, 0.0)
        next_values = [self._q_table.get((next_state, a), 0.0) for a in (-1, 0, 1)]
        best_next = max(next_values) if next_values else 0.0
        new_value = old_value + self._cfg.learning_rate * (
            reward + self._cfg.discount_factor * best_next - old_value
        )
        self._q_table[key] = new_value
