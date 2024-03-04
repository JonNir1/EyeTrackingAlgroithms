import numpy as np
import pandas as pd
from typing import Tuple

import Config.constants as cnst
import Utils.pixel_utils as pixel_utils
import Utils.visual_angle_utils as visang_utils
from GazeEvents.BaseGazeEvent import BaseGazeEvent


class PSOEvent(BaseGazeEvent):
    _EVENT_TYPE = cnst.EVENTS.PSO

    def is_high(self, threshold: float) -> bool:
        if np.isnan(self.peak_velocity):
            return False
        return self.peak_velocity > threshold

