import numpy as np
import pandas as pd
from typing import Tuple

import Config.constants as cnst
import Utils.pixel_utils as pixel_utils
import Utils.visual_angle_utils as visang_utils
from GazeEvents.BaseGazeEvent import BaseGazeEvent


class SaccadeEvent(BaseGazeEvent):
    _EVENT_LABEL = cnst.EVENT_LABELS.SACCADE
