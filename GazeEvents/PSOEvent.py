import numpy as np

import Config.constants as cnst
from GazeEvents.BaseGazeEvent import BaseGazeEvent


class PSOEvent(BaseGazeEvent):
    _EVENT_TYPE = cnst.EVENTS.PSO

    def is_high(self, threshold: float) -> bool:
        if np.isnan(self.peak_velocity):
            return False
        return self.peak_velocity > threshold

