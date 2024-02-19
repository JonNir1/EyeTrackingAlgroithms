import traceback
import numpy as np
from abc import ABC, abstractmethod
from typing import final

import constants as cnst
from Config import experiment_config as cnfg
import Utils.array_utils as arr_utils
from Config.GazeEventTypeEnum import GazeEventTypeEnum


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

    DEFAULT_MISSING_VALUE = np.nan
    DEFAULT_VIEWER_DISTANCE = 60  # cm
    DEFAULT_PIXEL_SIZE = cnfg.SCREEN_MONITOR.pixel_size   # cm
    DEFAULT_MINIMUM_EVENT_DURATION = 5  # ms
    DEFAULT_BLINK_PADDING = 0  # ms
    __DEFAULT_MINIMUM_EVENT_SAMPLES = 2  # events cannot be shorter than 2 samples

    def __init__(self,
                 missing_value=DEFAULT_MISSING_VALUE,
                 viewer_distance: float = DEFAULT_VIEWER_DISTANCE,
                 pixel_size: float = DEFAULT_PIXEL_SIZE,
                 minimum_event_duration: float = DEFAULT_MINIMUM_EVENT_DURATION,
                 pad_blinks_by: float = DEFAULT_BLINK_PADDING):
        self._missing_value = np.nan if missing_value is None else missing_value
        self._viewer_distance = viewer_distance if viewer_distance is not None else self.DEFAULT_VIEWER_DISTANCE
        if self._viewer_distance <= 0:
            raise ValueError("viewer_distance must be positive")
        self._pixel_size = pixel_size if pixel_size is not None else self.DEFAULT_PIXEL_SIZE
        if pixel_size <= 0:
            raise ValueError("pixel_size must be positive")
        self._minimum_event_duration = minimum_event_duration if minimum_event_duration is not None else self.DEFAULT_MINIMUM_EVENT_DURATION
        if minimum_event_duration < 0:
            raise ValueError("minimum_event_duration must be non-negative")
        self._minimum_event_duration = minimum_event_duration   # ms
        if pad_blinks_by < 0:
            raise ValueError("pad_blinks_by must be non-negative")
        self._pad_blinks_by = pad_blinks_by                     # ms

    def minimum_event_samples(self, sr: float) -> int:
        ns = self._calc_num_samples(self._minimum_event_duration, sr)
        return max(ns, self.__DEFAULT_MINIMUM_EVENT_SAMPLES)

    @final
    def detect(self, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        t, x, y = self._verify_inputs(t, x, y)
        candidates = np.full_like(t, GazeEventTypeEnum.UNDEFINED)
        try:
            # detect blinks
            candidates = self.detect_blinks(t, x, y, candidates)
            # set x and y to nan where blinks are detected
            x_copy, y_copy = x.copy(), y.copy()
            x_copy[candidates == GazeEventTypeEnum.BLINK] = np.nan
            y_copy[candidates == GazeEventTypeEnum.BLINK] = np.nan
            # detect gaze events
            candidates = self._detect_impl(t, x_copy, y_copy, candidates)
            sr = self._calculate_sampling_rate(t)
            candidates = self._merge_consecutive_chunks(candidates, sr)
        except ValueError as e:
            trace = traceback.format_exc()
            print(f"Failed to detect gaze-event candidates:\t{e.__class__.__name__}\n\t{trace}")
        return candidates

    @final
    def detect_blinks(self,
                      t: np.ndarray,
                      x: np.ndarray,
                      y: np.ndarray,
                      candidates: np.ndarray) -> np.ndarray:
        """
        Detects blink candidates in the given gaze data:
        1. Identifies samples where x or y are missing as blinks
        2. Ignores chunks of blinks that are shorter than the minimum event duration
        3. Merges consecutive blink chunks separated by less than the minimum event duration
        4. Pads the blink candidates by the amount in `self._pad_blinks_by`

        :param t: timestamps in milliseconds
        :param x: x-coordinates
        :param y: y-coordinates
        :param candidates: array of event candidates

        :return: array of blink candidates
        """
        if len(candidates) != len(t) or len(candidates) != len(x) or len(candidates) != len(y):
            raise ValueError("arrays t, x, y and candidates must have the same length")

        # identify samples where x or y are missing as blinks
        is_missing_x = np.array([self._is_missing_value(xi) for xi in x])
        is_missing_y = np.array([self._is_missing_value(yi) for yi in y])
        candidates[is_missing_x | is_missing_y] = GazeEventTypeEnum.BLINK

        # ignore short blinks and merge consecutive blinks
        sr = self._calculate_sampling_rate(t)
        candidates = self._merge_consecutive_chunks(candidates, sr)

        # pad blinks by the given amount
        if self._pad_blinks_by == 0:
            return candidates
        pad_samples = self._calc_num_samples(self._pad_blinks_by, sr)
        for i, c in enumerate(candidates):
            if c == GazeEventTypeEnum.BLINK:
                start = max(0, i - pad_samples)
                end = min(len(candidates), i + pad_samples)
                candidates[start:end] = GazeEventTypeEnum.BLINK
        return candidates

    @abstractmethod
    def _detect_impl(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        raise NotImplementedError

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

    def _is_missing_value(self, value: float) -> bool:
        if np.isnan(self._missing_value):
            return np.isnan(value)
        return value == self._missing_value

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

    def _merge_consecutive_chunks(self, candidates: np.ndarray, sr: float) -> np.ndarray:
        """
        1. Splits the candidates array into chunks of identical values
        2. Sets chunks that are shorter than the minimum event duration to GazeEventTypeEnum.UNDEFINED
        3. Merges consecutive chunks of the same type into a single chunk with the same type

        :param candidates: array of event candidates
        :param sr: sampling rate in Hz

        :return: array of merged event candidates
        """
        min_samples = self.minimum_event_samples(sr)
        cand_copy = np.asarray(candidates).copy()

        # set short chunks to undefined
        chunk_indices = arr_utils.get_chunk_indices(candidates)
        for chunk_idxs in chunk_indices:
            if len(chunk_idxs) < min_samples:
                cand_copy[chunk_idxs] = GazeEventTypeEnum.UNDEFINED

        # merge consecutive events of the same type
        chunk_indices = arr_utils.get_chunk_indices(candidates)  # re-calculate chunks after setting short chunks to undefined
        for i, middle_chunk_idxs in enumerate(chunk_indices):
            if i == 0 or i == len(chunk_indices) - 1:
                # don't attempt to merge the first or last chunk
                continue
            if len(middle_chunk_idxs) >= min_samples:
                # skip chunks that are long enough
                continue
            prev_chunk_value = cand_copy[chunk_indices[i - 1][-1]]  # value of the previous chunk
            next_chunk_value = cand_copy[chunk_indices[i + 1][0]]  # value of the next chunk
            if prev_chunk_value == next_chunk_value:
                # set the middle chunk to the same value as the previous and next chunks (essentially merging them)
                cand_copy[middle_chunk_idxs] = prev_chunk_value
        return cand_copy

    @staticmethod
    def _calc_num_samples(duration: float, sampling_rate: float) -> int:
        if duration < 0:
            raise ValueError("duration must be non-negative")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        return round(duration * sampling_rate / cnst.MILLISECONDS_PER_SECOND)
