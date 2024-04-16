import Config.experiment_config as cnfg
from GazeEvents.BaseEvent import BaseEvent


class BlinkEvent(BaseEvent):
    _EVENT_LABEL = cnfg.EVENT_LABELS.BLINK

