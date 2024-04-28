from abc import ABC
from typing import final, Tuple, List

import numpy as np
import pandas as pd

from Config import constants as cnst
import Config.experiment_config as cnfg
import Utils.pixel_utils as pixel_utils
import Utils.visual_angle_utils as visang_utils


class BaseEvent(ABC):
    """ Base class for eye tracking events. """

    _EVENT_LABEL: cnfg.EVENT_LABELS

    def __init__(self,
                 timestamps: np.ndarray,
                 x: np.ndarray,
                 y: np.ndarray,
                 pupil: np.ndarray,
                 viewer_distance: float,
                 pixel_size: float):
        if timestamps is None or len(timestamps) < cnfg.MINIMUM_SAMPLES_IN_EVENT:
            raise ValueError(f"{self.__class__.__name__} must be at least {cnfg.MINIMUM_SAMPLES_IN_EVENT} samples long")
        if np.isnan(timestamps).any() or np.isinf(timestamps).any():
            raise ValueError("array `timestamps` must not contain NaN or infinite values")
        if np.any(timestamps < 0):
            raise ValueError("array `timestamps` must not contain negative values")
        if x is None or len(x) != len(timestamps):
            raise ValueError("Array `x` must have the same length as `timestamps`")
        if y is None or len(y) != len(timestamps):
            raise ValueError("Array `y` must have the same length as `timestamps`")
        if pupil is None or len(pupil) != len(timestamps):
            raise ValueError("Array `pupil` must have the same length as `timestamps`")
        if viewer_distance is None or not np.isfinite(viewer_distance) or viewer_distance <= 0:
            raise ValueError("viewer_distance must be a positive finite number")
        if pixel_size is None or not np.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError("pixel_size must be a positive finite number")
        self._timestamps = timestamps
        self._x = x
        self._y = y
        self._pupil = pupil
        self._viewer_distance = viewer_distance  # in cm
        self._pixel_size = pixel_size  # in cm
        self._velocities = pixel_utils.calculate_velocities(xs=self._x,
                                                            ys=self._y,
                                                            timestamps=self._timestamps)  # units: px / ms

    def get_outlier_reasons(self) -> List[str]:
        reasons = []
        if self.duration < self.get_min_duration():
            reasons.append(f"min_{cnst.DURATION}")
        if self.duration > self.get_max_duration():
            reasons.append(f"max_{cnst.DURATION}")
        # TODO: check min, max velocity, acceleration, dispersion
        # TODO: check if inside the screen
        return reasons

    @final
    def overlap_time(self, other: "BaseEvent", normalized: bool = True) -> float:
        if self.start_time > other.end_time:
            return 0
        if other.start_time > self.end_time:
            return 0
        overlap = min(self.end_time, other.end_time) - max(self.start_time, other.start_time)
        if normalized:
            return overlap / min(self.duration, other.duration)
        return overlap

    @final
    def l2_timing_offset(self, other: "BaseEvent") -> float:
        """
        Event-matching metric: L2 norm of the timing offset between two events.
        See Kothari et al. (2020) for more details.
        """
        onset_diff = self.start_time - other.start_time
        offset_diff = self.end_time - other.end_time
        return np.sqrt(onset_diff ** 2 + offset_diff ** 2)

    @final
    def intersection_over_union(self, other: "BaseEvent") -> float:
        """
        Calculate the intersection over union (IoU) between two events.
        See Startsev & Zemblys (2023) for more information.
        """
        intersection = self.overlap_time(other, normalized=False)
        union = self.duration + other.duration
        return intersection / union

    @final
    def get(self, feature: str, safe=False) -> float:
        feature = feature.lower().replace(" ", "_").replace("-", "_")
        if not hasattr(self, feature):
            if safe:
                return np.nan
            raise ValueError(f"Feature '{feature}' is not available for {self.__class__.__name__}")
        return getattr(self, feature)

    @final
    @property
    def event_label(self) -> cnfg.EVENT_LABELS:
        return self.__class__._EVENT_LABEL

    @final
    @property
    def start_time(self) -> float:
        # Event's start time in milliseconds
        return float(self._timestamps[0])

    @final
    @property
    def end_time(self) -> float:
        # Event's end time in milliseconds
        return float(self._timestamps[-1])

    @final
    @property
    def duration(self) -> float:
        # Event's duration in milliseconds
        return self.end_time - self.start_time

    @final
    @property
    def is_outlier(self) -> bool:
        return len(self.get_outlier_reasons()) > 0

    @final
    @property
    def start_point(self) -> Tuple[float, float]:
        """ returns the saccade's start point as a tuple of the X,Y coordinates """
        x = float(self._x[0])
        y = float(self._y[0])
        return x, y

    @final
    @property
    def end_point(self) -> Tuple[float, float]:
        """ returns the saccade's end point as a tuple of the X,Y coordinates """
        x = float(self._x[-1])
        y = float(self._y[-1])
        return x, y

    @final
    @property
    def distance(self) -> float:
        """ returns the distance of the saccade in pixels """
        x_start, y_start = self.start_point
        x_end, y_end = self.end_point
        return np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)

    @final
    @property
    def amplitude(self) -> float:
        return visang_utils.pixels_to_visual_angle(num_px=self.distance,
                                                   d=self._viewer_distance,
                                                   pixel_size=self._pixel_size,
                                                   use_radians=False)

    @final
    @property
    def azimuth(self) -> float:
        """ returns the azimuth of the saccade in degrees """
        return pixel_utils.calculate_azimuth(p1=self.start_point, p2=self.end_point, use_radians=False)

    @final
    @property
    def peak_velocity_px(self) -> float:
        """ Returns the maximum velocity of the event in pixels per second """
        return float(np.nanmax(self._velocities))

    @final
    @property
    def peak_velocity(self) -> float:
        """ Returns the maximum velocity of the event in degrees per second """
        px_vel = self.peak_velocity_px
        return visang_utils.pixels_to_visual_angle(num_px=px_vel,
                                                   d=self._viewer_distance,
                                                   pixel_size=self._pixel_size,
                                                   use_radians=False)

    @final
    @property
    def mean_velocity_px(self) -> float:
        """ Returns the mean velocity of the event in pixels per second """
        return float(np.nanmean(self._velocities))

    @final
    @property
    def mean_velocity(self) -> float:
        """ Returns the mean velocity of the event in degrees per second """
        px_vel = self.mean_velocity_px
        return visang_utils.pixels_to_visual_angle(num_px=px_vel,
                                                   d=self._viewer_distance,
                                                   pixel_size=self._pixel_size,
                                                   use_radians=False)

    @final
    @property
    def mean_pupil_size(self) -> float:
        """ returns the mean pupil size during the fixation (in mm) """
        return float(np.nanmean(self._pupil))

    @final
    @property
    def std_pupil_size(self) -> float:
        """ returns the standard deviation of the pupil size during the fixation (in mm) """
        return float(np.nanstd(self._pupil))

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

    @final
    def get_velocities(self) -> np.ndarray:
        """
        Returns the velocities of the event in pixels per millisecond
        """
        return self._velocities

    @final
    def get_pupil_sizes(self) -> np.ndarray:
        """ returns the pupil size during the fixation (in mm) """
        return self._pupil

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of saccade information.
        :return: a pd.Series with the same values as super().to_series() and the following additional values:
            - event_label: the event's label
            - start_time: event's start time in milliseconds
            - end_time: event's end time in milliseconds
            - duration: event's duration in milliseconds
            - start_point: event's start point (2D pixel coordinates)
            - end_point: event's end point (2D pixel coordinates)
            - distance: event's distance (in pixels)
            - amplitude: event's visual angle (in degrees)
            - azimuth: event's azimuth (in degrees)
            - peak_velocity_px: the maximum velocity of the event in pixels per second
            - mean_velocity_px: the mean velocity of the event in pixels per second
            - mean_pupil_size: mean pupil size during the fixation (in mm)
            - std_pupil_size: standard deviation of the pupil size during the fixation (in mm)
            - is_outlier: boolean indicating whether the event is an outlier or not
            - outlier_reasons: a list of strings indicating the reasons why the event is an outlier
        """
        return pd.Series(data=[
            self._EVENT_LABEL.name, self.start_time, self.end_time, self.duration, self.start_point, self.end_point,
            self.distance, self.amplitude, self.azimuth, self.peak_velocity_px, self.mean_velocity_px, self.mean_pupil_size,
            self.std_pupil_size, self.is_outlier, self.get_outlier_reasons()
        ],
            index=[
                "event_label", "start_time", "end_time", "duration", "start_point", "end_point", "distance",
                "amplitude", "azimuth", "peak_velocity_px", "mean_velocity_px", "mean_pupil_size", "std_pupil_size",
                "is_outlier", "outlier_reasons"
            ])

    @classmethod
    @final
    def get_min_duration(cls) -> float:
        return cnfg.EVENT_MAPPING[cls._EVENT_LABEL][cnst.MIN_DURATION]

    @classmethod
    @final
    def set_min_duration(cls, min_duration: float):
        event_type = cls._EVENT_LABEL.name.capitalize()
        if min_duration < 0:
            raise ValueError(f"min_duration for {event_type} must be a positive number")
        max_duration = cnfg.EVENT_MAPPING[cls._EVENT_LABEL][cnst.MAX_DURATION]
        if min_duration > max_duration:
            raise ValueError(f"min_duration for {event_type} must be less than or equal to max_duration")
        cnfg.EVENT_MAPPING[cls._EVENT_LABEL][cnst.MIN_DURATION] = min_duration

    @classmethod
    @final
    def get_max_duration(cls) -> float:
        return cnfg.EVENT_MAPPING[cls._EVENT_LABEL][cnst.MIN_DURATION]

    @classmethod
    @final
    def set_max_duration(cls, max_duration: float):
        event_type = cls._EVENT_LABEL.name.capitalize()
        if max_duration < 0:
            raise ValueError(f"max_duration for {event_type} must be a positive number")
        min_duration = cnfg.EVENT_MAPPING[cls._EVENT_LABEL][cnst.MIN_DURATION]
        if max_duration < min_duration:
            raise ValueError(f"max_duration for {event_type} must be greater than or equal to min_duration")
        cnfg.EVENT_MAPPING[cls._EVENT_LABEL][cnst.MAX_DURATION] = max_duration

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self._EVENT_LABEL != other._EVENT_LABEL:
            return False
        if self._timestamps.shape != other._timestamps.shape:
            return False
        if not np.allclose(self._timestamps, other._timestamps):
            return False
        if self._viewer_distance != other._viewer_distance:
            return False
        if self._pixel_size != other._pixel_size:
            return False
        if not np.array_equal(self._x, other._x, equal_nan=True):
            return False
        if not np.array_equal(self._y, other._y, equal_nan=True):
            return False
        if not np.array_equal(self._pupil, other._pupil, equal_nan=True):
            return False
        return True

    def __str__(self):
        event_type = self._EVENT_LABEL.name.upper()
        return f"{event_type[:3]}({self.duration:.1f} ms)"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(
            (self._EVENT_LABEL,
             self._timestamps.tobytes(),
             self._x.tobytes(),
             self._y.tobytes(),
             self._pupil.tobytes(),
             self._viewer_distance,
             self._pixel_size)
        )
