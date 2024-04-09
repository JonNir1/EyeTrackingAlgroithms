import Config.constants as cnst
from GazeEvents.BaseEvent import BaseEvent


class SmoothPursuitEvent(BaseEvent):
    _EVENT_LABEL = cnst.EVENT_LABELS.SMOOTH_PURSUIT
