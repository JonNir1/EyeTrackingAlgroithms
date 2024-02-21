import numpy as np

import Config.constants as cnst
from GazeDetectors.BaseDetector import BaseDetector
from Utils import visual_angle_utils as vis_utils


class IVTDetector(BaseDetector):
    """
    Implements the I-VT (velocity threshold) gaze event detection algorithm, as described in:
        Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades in eye-tracking protocols.
        In Proceedings of the Symposium on Eye Tracking Research & Applications (pp. 71-78).

    General algorithm:
    1. Calculate the angular velocity of the gaze data
    2. Identify saccade candidates as samples with angular velocity greater than the threshold
    3. Assume undefined (non-blink) samples are fixations

    :param velocity_threshold: the threshold for angular velocity, in degrees per second. Default is 45 degrees per-second,
        as suggested in the paper "One algorithm to rule them all? An evaluation and discussion of ten eye
        movement event-detection algorithms" (2016), Andersson et al.
    """

    __DEFAULT_VELOCITY_THRESHOLD = 45  # degrees per second

    def __init__(self,
                 velocity_threshold: float = __DEFAULT_VELOCITY_THRESHOLD,
                 missing_value=BaseDetector.DEFAULT_MISSING_VALUE,
                 viewer_distance: float = BaseDetector.DEFAULT_VIEWER_DISTANCE,
                 pixel_size: float = BaseDetector.DEFAULT_PIXEL_SIZE,
                 pad_blinks_by: float = BaseDetector.DEFAULT_BLINK_PADDING):
        super().__init__(missing_value=missing_value,
                         viewer_distance=viewer_distance,
                         pixel_size=pixel_size,
                         pad_blinks_by=pad_blinks_by)
        if velocity_threshold <= 0:
            raise ValueError("velocity_threshold must be positive")
        self._velocity_threshold = velocity_threshold

    def _detect_impl(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        angular_velocities = vis_utils.calculates_angular_velocities_from_pixels(xs=x, ys=y, timestamps=t,
                                                                                 d=self._viewer_distance,
                                                                                 pixel_size=self._pixel_size)
        assert len(angular_velocities) == len(x), (f"angular velocities (shape {angular_velocities.shape}) do not " +
                                                   f"match the length of x (shape {x.shape})")

        # assume undefined (non-blink) samples are fixations, unless angular velocity is above threshold
        candidates_copy = np.asarray(candidates, dtype=cnst.EVENTS).copy()
        candidates_copy[candidates_copy == cnst.EVENTS.UNDEFINED] = cnst.EVENTS.FIXATION
        candidates_copy[angular_velocities >= self._velocity_threshold] = cnst.EVENTS.SACCADE
        return candidates_copy
