import random
from typing import Dict, Any

class MORLModel:
    def __init__(self):
        self.alpha = 0.4
        self.beta = 0.2
        self.gamma = 0.2
        self.delta = 0.2

    def select_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        lr = state.get("learningRate", 0.05)
        er = state.get("explorationRate", 0.1)
        if random.random() < er:
            self.alpha = max(0.0, min(1.0, self.alpha + (random.random() - 0.5) * lr))
            self.beta = max(0.0, min(1.0, self.beta + (random.random() - 0.5) * lr))
            self.gamma = max(0.0, min(1.0, self.gamma + (random.random() - 0.5) * lr))
            self.delta = max(0.0, min(1.0, self.delta + (random.random() - 0.5) * lr))
        action = {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "temporalShiftMinutes": 15,
            "regionHint": state.get("preferredRegion", "")
        }
        return action

