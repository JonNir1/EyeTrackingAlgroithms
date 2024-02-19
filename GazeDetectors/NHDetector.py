import numpy as np

from GazeDetectors.BaseDetector import BaseDetector
from Config.GazeEventTypeEnum import GazeEventTypeEnum


class NHDetector(BaseDetector):

    def __init__(self,
                 missing_value=BaseDetector.DEFAULT_MISSING_VALUE,
                 viewer_distance: float = BaseDetector.DEFAULT_VIEWER_DISTANCE,
                 pixel_size: float = BaseDetector.DEFAULT_PIXEL_SIZE,
                 minimum_event_duration: float = BaseDetector.DEFAULT_MINIMUM_EVENT_DURATION,
                 pad_blinks_by: float = BaseDetector.DEFAULT_BLINK_PADDING):
        super().__init__(missing_value=missing_value,
                         viewer_distance=viewer_distance,
                         pixel_size=pixel_size,
                         minimum_event_duration=minimum_event_duration,
                         pad_blinks_by=pad_blinks_by)
