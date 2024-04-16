import Config.experiment_config as cnfg
from GazeEvents.BaseEvent import BaseEvent


class SmoothPursuitEvent(BaseEvent):
    _EVENT_LABEL = cnfg.EVENT_LABELS.SMOOTH_PURSUIT
