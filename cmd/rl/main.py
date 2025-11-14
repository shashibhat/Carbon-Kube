"""Entrypoint for a simple RL tuning loop.

In a real deployment, this service would subscribe to events
(migrations, latency metrics) and update the threshold used by the
scheduler (e.g., via a shared Config CRD or ConfigMap).
"""

import logging
import random
import time

from rl.rl_tuner import RLTuner, RLTunerConfig

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("carbon-kube-rl")


def main() -> None:
    tuner = RLTuner(RLTunerConfig())

    LOGGER.info("Starting RL tuner demo loop (Ctrl+C to stop)")
    try:
        while True:
            # Fake state: (current_score, latency_risk)
            state = (random.uniform(50, 500), random.uniform(0.0, 0.5))
            action = tuner.select_action(state)
            # Fake reward: prefer lower emissions and low latency risk.
            reward = -state[0] * 0.01 - state[1]
            next_state = (state[0] * 0.9, state[1] * 0.9)
            tuner.observe(state, action, reward, next_state)
            LOGGER.info(
                "state=%s action=%d reward=%.2f new_threshold=%.2f",
                state,
                action,
                reward,
                tuner.current_threshold,
            )
            time.sleep(2.0)
    except KeyboardInterrupt:
        LOGGER.info("RL tuner exiting")


if __name__ == "__main__":
    main()
