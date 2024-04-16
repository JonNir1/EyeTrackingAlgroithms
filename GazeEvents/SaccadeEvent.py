import Config.experiment_config as cnfg
from GazeEvents.BaseEvent import BaseEvent


class SaccadeEvent(BaseEvent):
    _EVENT_LABEL = cnfg.EVENT_LABELS.SACCADE
