import traceback
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import final

import Config.constants as cnst
import Config.experiment_config as cnfg
import Utils.array_utils as arr_utils


class BaseDetector(ABC):
    """
    Base class for gaze event detectors, objects that indicate the type of gaze event (fixation, saccade, blink) at each
    sample in the gaze data. All inherited classes must implement the `_detect_impl` method, which is the core of the
    gaze event detection process.

    Detection process:
    1. Detect blinks, including padding them by the amount in `pad_blinks_by`
    2. Set x and y to nan where blinks are detected
    3. Detect gaze events (using the class-specific logic, implemented in `_detect_impl` method)
    4. Ignore chunks of gaze-events that are shorter than `minimum_event_duration`
    5. Merge chunks of the same type that are separated by less than `minimum_event_duration`

    :param missing_value: the value that indicates missing data in the gaze data. Default is np.nan
    :param viewer_distance: the distance from the viewer to the screen, in centimeters. Default is 60 cm
    :param pixel_size: the size of a single pixel on the screen, in centimeters. Default is the pixel size of the
        screen monitor
    :param minimum_event_duration: the minimum duration of a gaze event, in milliseconds. Default is 5 ms
    :param pad_blinks_by: the amount of time to pad blinks by, in milliseconds. Default is 0 ms (no padding)
    """

    def __init__(self, **kwargs):
        self._missing_value = kwargs.get('missing_value', cnfg.DEFAULT_MISSING_VALUE)
        self._sr = np.nan  # sampling rate
        self._candidates = None  # event candidates
        self.data: dict = {}  # gaze data

        self._viewer_distance = kwargs.get('viewer_distance', cnfg.DEFAULT_VIEWER_DISTANCE)  # cm
        if self._viewer_distance <= 0:
            raise ValueError("viewer_distance must be positive")

        self._pixel_size = kwargs.get('pixel_size', cnfg.SCREEN_MONITOR.pixel_size)  # cm
        if self._pixel_size <= 0:
            raise ValueError("pixel_size must be positive")

        self._pad_blinks_by = kwargs.get('pad_blinks_by', cnfg.DEFAULT_BLINK_PADDING)  # ms
        if self._pad_blinks_by < 0:
            raise ValueError("pad_blinks_by must be non-negative")

    @final
    def detect(self,
               t: np.ndarray,
               x: np.ndarray,
               y: np.ndarray,
               vd: float = cnfg.DEFAULT_VIEWER_DISTANCE,
               ps: float = cnfg.SCREEN_MONITOR.pixel_size) -> dict:
        if not np.isfinite(vd) or vd <= 0:
            raise ValueError("viewer distance must be a positive finite number")
        if not np.isfinite(ps) or ps <= 0:
            raise ValueError("pixel size must be a positive finite number")
        t, x, y = self._verify_inputs(t, x, y)
        self._candidates = np.full_like(t, cnst.EVENTS.UNDEFINED)
        self.data[cnst.GAZE] = pd.DataFrame({cnst.T: t, cnst.X: x, cnst.Y: y, cnst.EVENT_TYPE: self._candidates})
        self._viewer_distance = vd
        self._pixel_size = ps
        try:
            self._sr = self._calculate_sampling_rate(t)

            # detect blinks and remove blink samples from the data
            self._candidates = self._detect_blinks(x, y)
            x_copy, y_copy = x.copy(), y.copy()
            x_copy[self._candidates == cnst.EVENTS.BLINK] = np.nan
            y_copy[self._candidates == cnst.EVENTS.BLINK] = np.nan

            # detect gaze-event candidates
            candidates = self._detect_impl(t, x, y)
            self._candidates = self._merge_close_events(candidates)

            # add important values to self.data
            self.data[cnst.SAMPLING_RATE] = self._sr
            self.data[cnst.GAZE][cnst.EVENT_TYPE] = self._candidates    # update the event-type column
            self.data[cnst.VIEWER_DISTANCE] = self._viewer_distance
            self.data[cnst.PIXEL_SIZE] = self._pixel_size
        except ValueError as e:
            trace = traceback.format_exc()
            print(f"Failed to detect gaze-event candidates:\t{e.__class__.__name__}\n\t{trace}")
        return self.data

    @abstractmethod
    def _detect_impl(self, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @final
    def _detect_blinks(self,
                       x: np.ndarray,
                       y: np.ndarray) -> np.ndarray:
        """
        Detects blink candidates in the given gaze data:
        1. Identifies samples where x or y are missing as blinks
        2. Ignores chunks of blinks that are shorter than the minimum event duration
        3. Merges consecutive blink chunks separated by less than the minimum event duration
        4. Pads the blink candidates by the amount in `self._pad_blinks_by`

        :param x: x-coordinates
        :param y: y-coordinates

        :return: array of candidates
        """
        candidates = np.asarray(self._candidates, dtype=cnst.EVENTS).copy()

        # identify samples where x or y are missing as blinks
        is_missing_x = np.array([self._is_missing_value(xi) for xi in x])
        is_missing_y = np.array([self._is_missing_value(yi) for yi in y])
        candidates[is_missing_x | is_missing_y] = cnst.EVENTS.BLINK
        candidates = self._merge_close_events(candidates)  # ignore short blinks and merge consecutive blinks

        # pad blinks by the given amount
        if self._pad_blinks_by == 0:
            return candidates
        pad_samples = self._calc_num_samples(self._pad_blinks_by)
        for i, c in enumerate(candidates):
            if c == cnst.EVENTS.BLINK:
                start = max(0, i - pad_samples)
                end = min(len(candidates), i + pad_samples)
                candidates[start:end] = cnst.EVENTS.BLINK
        return candidates

    def _verify_inputs(self, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        if not arr_utils.is_one_dimensional(t):
            raise ValueError("`t` must be one-dimensional")
        if not arr_utils.is_one_dimensional(x):
            raise ValueError("`x` must be one-dimensional")
        if not arr_utils.is_one_dimensional(y):
            raise ValueError("`y` must be one-dimensional")
        t = t.reshape(-1)
        x = x.reshape(-1)
        y = y.reshape(-1)
        if len(t) != len(x) or len(t) != len(y) or len(x) != len(y):
            raise ValueError("t, x and y must have the same length")
        return t, x, y

    @final
    def _is_missing_value(self, value: float) -> bool:
        if np.isnan(self._missing_value):
            return np.isnan(value)
        return value == self._missing_value

    @final
    def _merge_close_events(self, candidates) -> np.ndarray:
        """
        1. Splits the candidates array into chunks of identical event-types
        2. If two chunks of the same event-type are separated by a chunk of a different event-type, and the middle
            chunk is shorter than the minimum event duration, it is set to the same event-type as the other two chunks
        3. Sets chunks that are shorter than the minimum event duration to cnst.EVENTS.UNDEFINED
        """
        # calculate number of samples in the minimum event duration
        min_event_duration = min([cnfg.EVENT_MAPPING[e]["min_duration"] for e in cnfg.EVENT_MAPPING.keys()
                                  if e != cnst.EVENTS.UNDEFINED])
        ns = self._calc_num_samples(min_event_duration)
        min_samples = max(ns, cnst.MINIMUM_SAMPLES_IN_EVENT)
        cand = arr_utils.merge_close_chunks(candidates, min_samples, cnst.EVENTS.UNDEFINED)
        return cand

    @final
    def _calc_num_samples(self, duration: float) -> int:
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("duration must be a non-negative finite number")
        if not np.isfinite(self._sr) or self._sr <= 0:
            raise ValueError("sampling rate must be a positive finite number")
        return round(duration * self._sr / cnst.MILLISECONDS_PER_SECOND)

    @staticmethod
    @final
    def _calculate_sampling_rate(ms: np.ndarray) -> float:
        """
        Calculates the sampling rate of the given timestamps in Hz.
        :param ms: timestamps in milliseconds (floating-point, not integer)
        """
        if len(ms) < 2:
            raise ValueError("timestamps must be of length at least 2")
        sr = cnst.MILLISECONDS_PER_SECOND / np.median(np.diff(ms))
        if not np.isfinite(sr):
            raise RuntimeError("Error calculating sampling rate")
        return sr
