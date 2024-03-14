import numpy as np
import pandas as pd
from typing import final, Tuple

import Utils.pixel_utils as pixel_utils
import Utils.visual_angle_utils as visang_utils
from GazeEvents.BaseEvent import BaseEvent


class BaseGazeEvent(BaseEvent):
    """
    Base class for events that contain gaze data (x,y coordinates).
    """

    def __init__(self,
                 timestamps: np.ndarray,
                 x: np.ndarray,
                 y: np.ndarray,
                 pupil: np.ndarray,
                 viewer_distance: float,
                 pixel_size: float):
        super().__init__(timestamps=timestamps)
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
        self._viewer_distance = viewer_distance  # in cm
        self._pixel_size = pixel_size  # in cm
        self._x = x
        self._y = y
        self._pupil = pupil
        self._velocities = pixel_utils.calculate_velocities(xs=self._x,
                                                            ys=self._y,
                                                            timestamps=self._timestamps)  # units: px / ms

    def get_outlier_reasons(self):
        reasons = super().get_outlier_reasons()
        # TODO: check min, max velocity, acceleration, dispersion
        # TODO: check if inside the screen
        return reasons

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
    def peak_velocity(self) -> float:
        """ Returns the maximum velocity of the event in pixels per second """
        return float(np.nanmax(self._velocities))

    @final
    @property
    def peak_velocity_deg(self) -> float:
        """ Returns the maximum velocity of the event in degrees per second """
        px_vel = self.peak_velocity
        return visang_utils.pixels_to_visual_angle(num_px=px_vel,
                                                   d=self._viewer_distance,
                                                   pixel_size=self._pixel_size,
                                                   use_radians=False)

    @final
    @property
    def mean_velocity(self) -> float:
        """ Returns the mean velocity of the event in pixels per second """
        return float(np.nanmean(self._velocities))

    @final
    @property
    def mean_velocity_deg(self) -> float:
        """ Returns the mean velocity of the event in degrees per second """
        px_vel = self.mean_velocity
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
            - start_point: saccade's start point (2D pixel coordinates)
            - end_point: saccade's end point (2D pixel coordinates)
            - distance: saccade's distance (in pixels)
            - amplitude: saccade's visual angle (in degrees)
            - azimuth: saccade's azimuth (in degrees)
            - peak_velocity: the maximum velocity of the event in pixels per second
            - mean_velocity: the mean velocity of the event in pixels per second
            - mean_pupil_size: mean pupil size during the fixation (in mm)
            - std_pupil_size: standard deviation of the pupil size during the fixation (in mm)
        """
        series = super().to_series()
        series["start_point"] = self.start_point
        series["end_point"] = self.end_point
        series["distance"] = self.distance
        series["amplitude"] = self.amplitude
        series["azimuth"] = self.azimuth
        series["peak_velocity"] = self.peak_velocity
        series["mean_velocity"] = self.mean_velocity
        series["mean_pupil_size"] = self.mean_pupil_size
        series["std_pupil_size"] = self.std_pupil_size
        return series

    def __eq__(self, other):
        if not super().__eq__(other):
            return False
        if self._viewer_distance != other._viewer_distance:
            return False
        if self._pixel_size != other._pixel_size:
            return False
        if not np.array_equal(self._x, other._x, equal_nan=True):
            return False
        if not np.array_equal(self._y, other._y, equal_nan=True):
            return False
        return True

    def __hash__(self):
        super_hash = super().__hash__()
        return hash((super_hash, self._viewer_distance, self._pixel_size, self._x.tobytes(), self._y.tobytes()))
