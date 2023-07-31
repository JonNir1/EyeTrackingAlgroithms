import numpy as np
from abc import ABC, abstractmethod
from typing import final, List, Tuple, Set

import constants as cnst
import Utils.array_utils as arr_utils
from GazeEvents.GazeEventTypeEnum import GazeEventTypeEnum


class BaseDetector(ABC):
    """
    Base class for gaze event detectors, that segment eye-tracking data into separate events, such as blinks, saccades,
    fixations, etc.
    The detection process is implemented in detect_candidates_monocular() and detect_candidates_binocular() and is the
    same for all detectors. Detection steps are as follows:
    1. Verify the input is valid
    2. Detecting blink candidates based on missing data in the recorded gaze data
    3. Detecting event candidates using unique algorithms for each detector (implemented in _identify_event_candidates())
    4. Filling short chunks of event candidates with GazeEventTypeEnum.UNDEFINED
    5. Merging chunks of identical event candidates that are close to each other
    6. If binocular data is available, candidates from both eyes are merged into a single list of candidates based on
    pre-defined logic (e.g. both eyes must detect a candidate for it to be considered a binocular candidate).
    """

    _MINIMUM_TIME_WITHIN_EVENT: float = 5  # min duration of single event (in milliseconds)
    _MINIMUM_TIME_BETWEEN_IDENTICAL_EVENTS: float = 5  # min duration between identical events (in milliseconds)

    def __init__(self, sr: float):
        self._sr = sr  # sampling rate in Hz

    @final
    def detect_candidates_monocular(self, x: np.ndarray, y: np.ndarray) -> List[GazeEventTypeEnum]:
        """
        Detects event-candidates in the given gaze data from a single eye. Detection steps:
        1. Verify that x and y are valid inputs
        2. Detect blink candidates when there is missing gaze data
        3. Find event candidates based on each Detector's implementation of _identify_event_candidates()
        4. Fill short chunks of event candidates with GazeEventTypeEnum.UNDEFINED
        5. Merge chunks of identical event candidates that are close to each other

        :param x: x-coordinates of gaze data from a single eye
        :param y: y-coordinates of gaze data from a single eye

        :return: list of GazeEventTypeEnum values, where each value indicates the type of event that is detected at the
            corresponding index in the given gaze data
        """
        x, y = self._verify_inputs(x, y)
        x, y, candidates = self._identify_blink_candidates(x, y)
        candidates = self._identify_gaze_event_candidates(x, y, candidates)
        candidates = self._set_short_chunks_as_undefined(candidates)
        candidates = self._merge_proximal_chunks_of_identical_values(candidates)
        return candidates

    @final
    def detect_candidates_binocular(self,
                                    x_l: np.ndarray, y_l: np.ndarray,
                                    x_r: np.ndarray, y_r: np.ndarray,
                                    detect_by: str = 'both') -> List[GazeEventTypeEnum]:
        left_candidates = self.detect_candidates_monocular(x=x_l, y=y_l)
        right_candidates = self.detect_candidates_monocular(x=x_r, y=y_r)

        detect_by = detect_by.lower()
        if detect_by == cnst.LEFT:
            return left_candidates
        if detect_by == cnst.RIGHT:
            return right_candidates

        assert len(left_candidates) == len(right_candidates)
        if detect_by in ["both", "and"]:
            # only keep candidates that are detected by both eyes
            both_candidates = [left_cand if left_cand == right_cand else GazeEventTypeEnum.UNDEFINED
                               for left_cand, right_cand in zip(left_candidates, right_candidates)]
            return both_candidates
        if detect_by in ["either", "or"]:
            either_candidates = [left_cand or right_cand for left_cand, right_cand
                                 in zip(left_candidates, right_candidates)]
            return either_candidates

        # TODO: support more complex logic: fixations & blinks are monocular, saccades are binocular, etc.

        raise ValueError(f"invalid value for `detect_by`: {detect_by}")

    def _verify_inputs(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not arr_utils.is_one_dimensional(x):
            raise ValueError("x must be one-dimensional")
        if not arr_utils.is_one_dimensional(y):
            raise ValueError("y must be one-dimensional")
        x = x.reshape(-1)
        y = y.reshape(-1)
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        return x, y

    @final
    def _identify_blink_candidates(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[GazeEventTypeEnum]]:
        """
        Identifies blink candidates in the given gaze data from a single eye, and removes them from the gaze data.
        Returns the modified gaze data and a list of event candidates including where the blinks were detected.
        """
        candidates = [GazeEventTypeEnum.UNDEFINED] * len(x)
        candidates[np.isnan(x) | np.isnan(y)] = GazeEventTypeEnum.BLINK

        # TODO: add blink correction before/after NaNs

        candidates_arr = np.array(candidates)
        x[candidates_arr == GazeEventTypeEnum.BLINK] = np.nan
        y[candidates_arr == GazeEventTypeEnum.BLINK] = np.nan
        return x, y, candidates

    @abstractmethod
    def _identify_gaze_event_candidates(self, x: np.ndarray, y: np.ndarray,
                                        candidates: List[GazeEventTypeEnum]) -> List[GazeEventTypeEnum]:
        """
        Identifies gaze-event (fixations, saccades, etc.) candidates in the given gaze data from a single eye
        """
        raise NotImplementedError

    @final
    def _set_short_chunks_as_undefined(self, arr) -> List[GazeEventTypeEnum]:
        """
        If a "chunk" of identical values is shorter than `self._minimum_samples_within_event`, we fill the indices of
        said chunk with value GazeEventTypeEnum.UNDEFINED.
        """
        arr_copy = np.copy(arr)
        chunk_indices = arr_utils.get_chunk_indices(arr)
        for chunk_idx in chunk_indices:
            if len(chunk_idx) < self._minimum_samples_within_event:
                arr_copy[chunk_idx] = GazeEventTypeEnum.UNDEFINED
        return arr_copy.tolist()

    @final
    def _merge_proximal_chunks_of_identical_values(self, arr,
                                                   allow_short_chunks_of: Set = None) -> List[GazeEventTypeEnum]:
        """
        If two "chunks" of identical values are separated by a short "chunk" of other values, merges the two chunks into
        one chunk by filling the middle chunk with the value of the left chunk.
        Chunks with values specified in `allow_short_chunks_of` are not merged.
        """
        if allow_short_chunks_of is None or len(allow_short_chunks_of) == 0:
            allow_short_chunks_of = set()

        arr_copy = np.asarray(arr).copy()
        chunk_indices = arr_utils.get_chunk_indices(arr)
        for i, middle_chunk in enumerate(chunk_indices):
            if i == 0 or i == len(chunk_indices) - 1:
                # ignore the first and last chunk
                continue
            if len(middle_chunk) >= self._minimum_samples_between_identical_events:
                # ignore chunks that are long enough
                continue
            middle_chunk_value = arr_copy[middle_chunk[0]]
            if middle_chunk_value in allow_short_chunks_of:
                # ignore chunks of the specified types
                continue
            left_chunk_value = arr_copy[chunk_indices[i - 1][0]]
            right_chunk_value = arr_copy[chunk_indices[i + 1][0]]
            if left_chunk_value != right_chunk_value:
                # ignore middle chunks if the left and right chunks are not identical
                continue

            # reached here if the middle chunk is short, its value is not allowed to be short, and left and right chunks
            # are identical. merge the left and right chunks by filling `middle_chunk` with the value of `left_chunk`.
            arr_copy[middle_chunk] = left_chunk_value
        return arr_copy.tolist()

    @property
    @final
    def _minimum_samples_within_event(self) -> int:
        """ minimum number of samples within a single event """
        return round(self._MINIMUM_TIME_WITHIN_EVENT * self._sr / 1000)

    @property
    @final
    def _minimum_samples_between_identical_events(self) -> int:
        """ minimum number of samples between identical events """
        return round(self._MINIMUM_TIME_BETWEEN_IDENTICAL_EVENTS * self._sr / 1000)