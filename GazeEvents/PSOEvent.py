import numpy as np

import Config.experiment_config as cnfg
from GazeEvents.BaseEvent import BaseEvent


class PSOEvent(BaseEvent):
    _EVENT_LABEL = cnfg.EVENT_LABELS.PSO

    def is_high(self, threshold: float) -> bool:
        if np.isnan(self.peak_velocity_px):
            return False
        return self.peak_velocity_px > threshold

