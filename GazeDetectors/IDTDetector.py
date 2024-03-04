import numpy as np

import Config.constants as cnst
from GazeDetectors.BaseDetector import BaseDetector
from Utils import visual_angle_utils as vis_utils


class IDTDetector(BaseDetector):
    """
    Implements the I-DT (dispersion threshold) gaze event detection algorithm, as described in:
        Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades in eye-tracking protocols.
        In Proceedings of the Symposium on Eye Tracking Research & Applications (pp. 71-78).

    General algorithm:
    1. Initialize a window spanning the first `window_duration` milliseconds
    2. Calculate the dispersion of the gaze data in the window
    3. If the dispersion is below the threshold, label all samples in the window as fixation and expand the window by a
        single sample. Otherwise, label the current sample as saccade and start a new window in the next sample
    4. Repeat until the end of the gaze data

    :param dispersion_threshold: the threshold for dispersion, in degrees. Default is 0.5 degrees, as suggested in the
        paper "One algorithm to rule them all? An evaluation and discussion of ten eye movement event-detection
        algorithms" (2016), Andersson et al.
    :param window_duration: the duration of the window in milliseconds. Default is 100 ms, as suggested in the paper
        "One algorithm to rule them all? An evaluation and discussion of ten eye movement event-detection algorithms"
        (2016), Andersson et al.
    """

    __DEFAULT_DISPERSION_THRESHOLD = 0.5    # visual degrees
    __DEFAULT_WINDOW_DURATION = 100         # ms

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs.get('dispersion_threshold', self.__DEFAULT_DISPERSION_THRESHOLD) <= 0:
            raise ValueError("dispersion_threshold must be positive")
        if kwargs.get('window_duration', self.__DEFAULT_WINDOW_DURATION) <= 0:
            raise ValueError("window_duration must be positive")
        self._dispersion_threshold = kwargs.get('dispersion_threshold', self.__DEFAULT_DISPERSION_THRESHOLD)
        self._window_duration = kwargs.get('window_duration', self.__DEFAULT_WINDOW_DURATION)

    def _detect_impl(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, vd: float, ps: float) -> np.ndarray:
        candidates = np.asarray(self._candidates, dtype=cnst.EVENTS).copy()
        ws = self._calculate_window_size(t)
        start_idx, end_idx = 0, ws
        is_fixation = False
        while end_idx <= len(t):
            dispersion = self._calculate_dispersion(x, y, start_idx, end_idx, vd, ps)
            if dispersion < self._dispersion_threshold:
                # label all samples in the window as fixation and expand window to the right
                is_fixation = True
                candidates[start_idx: end_idx] = cnst.EVENTS.FIXATION
                end_idx += 1
            elif is_fixation:
                # start new window in the end of the old one
                start_idx = end_idx - 1
                end_idx = start_idx + ws
                is_fixation = False
            else:
                # label current sample as saccade and start new window in the next sample
                candidates[start_idx] = cnst.EVENTS.SACCADE
                start_idx += 1
                end_idx += 1
        return candidates

    def _calculate_window_size(self, t: np.ndarray) -> int:
        ws = self._calc_num_samples(self._window_duration)
        if ws < 2:
            raise ValueError(f"window_duration={ws} is too short for the given sampling rate")
        if ws >= len(t):
            raise ValueError(f"window_duration={ws} is too long for the given input data")
        return ws

    @staticmethod
    def _calculate_dispersion(x: np.ndarray,
                              y: np.ndarray,
                              start_idx: int,
                              end_idx: int,
                              vd: float,
                              ps: float) -> float:
        window_x, window_y = x[start_idx: end_idx], y[start_idx: end_idx]
        dispersion = max(window_x) - min(window_x) + max(window_y) - min(window_y)
        ang_dispersion = vis_utils.pixels_to_visual_angle(num_px=dispersion, d=vd, pixel_size=ps)
        return ang_dispersion

    @staticmethod
    def _calculate_dispersion_area(x: np.ndarray,
                                   y: np.ndarray,
                                   start_idx: int,
                                   end_idx: int,
                                   vd: float,
                                   ps: float) -> float:
        # TODO: check if yields better results
        window_x, window_y = x[start_idx: end_idx + 1], y[start_idx: end_idx + 1]
        horiz_axis = 0.5 * (max(window_x) - min(window_x))
        ang_horiz_axis = vis_utils.pixels_to_visual_angle(num_px=horiz_axis, d=vd, pixel_size=ps)
        vert_axis = 0.5 * (max(window_y) - min(window_y))
        ang_vert_axis = vis_utils.pixels_to_visual_angle(num_px=vert_axis, d=vd, pixel_size=ps)
        return np.pi * ang_horiz_axis * ang_vert_axis
