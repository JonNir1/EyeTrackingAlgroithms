import Config.constants as cnst
from GazeEvents.BaseEvent import BaseEvent
import Utils.pixel_utils as pixel_utils


class SaccadeEvent(BaseEvent):
    _EVENT_LABEL = cnst.EVENT_LABELS.SACCADE
