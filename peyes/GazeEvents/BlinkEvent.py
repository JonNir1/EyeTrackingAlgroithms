import peyes.Config.experiment_config as cnfg
from peyes.GazeEvents.BaseEvent import BaseEvent


class BlinkEvent(BaseEvent):
    _EVENT_LABEL = cnfg.EVENT_LABELS.BLINK

