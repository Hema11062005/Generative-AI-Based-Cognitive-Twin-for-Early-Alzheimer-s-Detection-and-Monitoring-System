import numpy as np


class CognitiveTwin:

    def __init__(self):
        self.stage_map = {
            "NonDemented": 0,
            "VeryMildDemented": 1,
            "MildDemented": 2,
            "ModerateDemented": 3
        }
        # Progression rate per stage (higher stage = faster progression)
        self.rate_map = {
            "NonDemented": 0.08,
            "VeryMildDemented": 0.18,
            "MildDemented": 0.28,
            "ModerateDemented": 0.05   # already severe, slow further change
        }

    def simulate_progression(self, stage):
        severity = self.stage_map.get(stage, 0)
        rate = self.rate_map.get(stage, 0.15)

        years = np.linspace(0, 10, 100)

        # Sigmoid-style progression: starts at current severity, curves toward 3
        progression = 3 / (1 + np.exp(-rate * (years - (5 - severity * 1.2))))

        # Shift so it starts exactly at the current stage severity
        offset = progression[0] - severity
        progression = progression - offset

        progression = np.clip(progression, 0, 3)

        return years, progression