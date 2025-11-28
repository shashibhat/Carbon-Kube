import json
import time
from typing import Dict, Any
from .model import MORLModel

class Agent:
    def __init__(self):
        self.model = MORLModel()

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        action = self.model.select_action(state)
        return action

    def run(self, get_state, apply_update, interval_sec: int = 300):
        while True:
            state = get_state()
            action = self.step(state)
            apply_update(action)
            time.sleep(interval_sec)

