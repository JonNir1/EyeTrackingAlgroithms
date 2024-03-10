import Config.constants as cnst
from GazeEvents.BaseGazeEvent import BaseGazeEvent


class SmoothPursuitEvent(BaseGazeEvent):
    _EVENT_TYPE = cnst.EVENTS.SMOOTH_PURSUIT
