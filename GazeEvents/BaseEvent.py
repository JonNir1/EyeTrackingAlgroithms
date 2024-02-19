import numpy as np
import pandas as pd
from abc import ABC
from typing import List, final

import constants as cnst
import Config.experiment_config as cnfg
from Config.GazeEventTypeEnum import GazeEventTypeEnum


class BaseEvent(ABC):
    _EVENT_TYPE: GazeEventTypeEnum

    def __init__(self, timestamps: np.ndarray):
        if len(timestamps) < cnst.MINIMUM_SAMPLES_IN_EVENT:
            raise ValueError(f"{self.__class__.__name__} must be at least {cnst.MINIMUM_SAMPLES_IN_EVENT} samples long")
        if np.isnan(timestamps).any() or np.isinf(timestamps).any():
            raise ValueError("array `timestamps` must not contain NaN or infinite values")
        if np.any(timestamps < 0):
            raise ValueError("array `timestamps` must not contain negative values")
        self._timestamps = timestamps

    @final
    @property
    def start_time(self) -> float:
        # Event's start time in milliseconds
        return self._timestamps[0]

    @final
    @property
    def end_time(self) -> float:
        # Event's end time in milliseconds
        return self._timestamps[-1]

    @final
    @property
    def duration(self) -> float:
        # Event's duration in milliseconds
        return self.end_time - self.start_time

    @final
    @property
    def is_outlier(self) -> bool:
        return len(self.get_outlier_reasons()) > 0

    def get_outlier_reasons(self) -> List[str]:
        reasons = []
        if self.duration < self.get_min_duration():
            reasons.append(f"min_{cnst.DURATION}")
        if self.duration > self.get_max_duration():
            reasons.append(f"max_{cnst.DURATION}")
        return reasons

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of event information.
        :return: a pd.Series with the following index:
            - start_time: event's start time in milliseconds
            - end_time: event's end time in milliseconds
            - duration: event's duration in milliseconds
            - is_outlier: boolean indicating whether the event is an outlier or not
            - outlier_reasons: a list of strings indicating the reasons why the event is an outlier
        """
        return pd.Series(data=[self._EVENT_TYPE.name, self.start_time, self.end_time, self.duration, self.is_outlier,
                               self.get_outlier_reasons()],
                         index=["event_type", "start_time", "end_time", "duration", "is_outlier", "outlier_reasons"])

    @final
    def get_timestamps(self, round_decimals: int = 1, zero_corrected: bool = True) -> np.ndarray:
        """
        Returns the timestamps of the event, rounded to the specified number of decimals.
        If zero_corrected is True, the timestamps will be relative to the first timestamp of the event.
        """
        timestamps = self._timestamps  # timestamps in milliseconds
        if zero_corrected:
            timestamps = timestamps - timestamps[0]  # start from 0
        timestamps = np.round(timestamps, decimals=round_decimals)
        return timestamps

    @classmethod
    @final
    def event_type(cls) -> GazeEventTypeEnum:
        return cls._EVENT_TYPE

    @classmethod
    @final
    def get_min_duration(cls) -> float:
        return cnfg.EVENT_DURATIONS[cls._EVENT_TYPE][0]

    @classmethod
    @final
    def set_min_duration(cls, min_duration: float):
        event_type = cls._EVENT_TYPE.name.capitalize()
        if min_duration < 0:
            raise ValueError(f"min_duration for {event_type} must be a positive number")
        max_duration = cnfg.EVENT_DURATIONS[cls._EVENT_TYPE][1]
        if min_duration > max_duration:
            raise ValueError(f"min_duration for {event_type} must be less than or equal to max_duration")
        cnfg.EVENT_DURATIONS[cls._EVENT_TYPE] = (min_duration, max_duration)

    @classmethod
    @final
    def get_max_duration(cls) -> float:
        return cnfg.EVENT_DURATIONS[cls._EVENT_TYPE][1]

    @classmethod
    @final
    def set_max_duration(cls, max_duration: float):
        event_type = cls._EVENT_TYPE.name.capitalize()
        if max_duration < 0:
            raise ValueError(f"max_duration for {event_type} must be a positive number")
        min_duration = cnfg.EVENT_DURATIONS[cls._EVENT_TYPE][0]
        if max_duration < min_duration:
            raise ValueError(f"max_duration for {event_type} must be greater than or equal to min_duration")
        cnfg.EVENT_DURATIONS[cls._EVENT_TYPE] = (min_duration, max_duration)

    def __repr__(self):
        event_type = self._EVENT_TYPE.name.capitalize()
        return f"{event_type} ({self.duration:.1f} ms)"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self._timestamps.shape != other._timestamps.shape:
            return False
        if not np.allclose(self._timestamps, other._timestamps):
            return False
        return True

