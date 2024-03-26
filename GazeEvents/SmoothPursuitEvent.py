import Config.constants as cnst
from GazeEvents.BaseGazeEvent import BaseGazeEvent


class SmoothPursuitEvent(BaseGazeEvent):
    _EVENT_LABEL = cnst.EVENT_LABELS.SMOOTH_PURSUIT
