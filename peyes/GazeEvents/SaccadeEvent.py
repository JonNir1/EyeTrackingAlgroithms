import peyes.Config.experiment_config as cnfg
from peyes.GazeEvents.BaseEvent import BaseEvent


class SaccadeEvent(BaseEvent):
    _EVENT_LABEL = cnfg.EVENT_LABELS.SACCADE
