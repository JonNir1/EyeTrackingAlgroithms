import numpy as np

import Config.constants as cnst
from GazeEvents.BaseEvent import BaseEvent


class PSOEvent(BaseEvent):
    _EVENT_LABEL = cnst.EVENT_LABELS.PSO

    def is_high(self, threshold: float) -> bool:
        if np.isnan(self.peak_velocity):
            return False
        return self.peak_velocity > threshold

