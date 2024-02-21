import Config.constants as cnst
from GazeEvents.BaseEvent import BaseEvent


class BlinkEvent(BaseEvent):
    _EVENT_TYPE = cnst.EVENTS.BLINK

