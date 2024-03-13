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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs.get('velocity_threshold', self.__DEFAULT_VELOCITY_THRESHOLD) <= 0:
            raise ValueError("velocity_threshold must be positive")
        self._velocity_threshold = kwargs.get('velocity_threshold', self.__DEFAULT_VELOCITY_THRESHOLD)

    def _detect_impl(self, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        candidates = np.asarray(self._candidates, dtype=cnst.EVENT_LABELS).copy()
        angular_velocities = vis_utils.calculates_angular_velocities_from_pixels(xs=x, ys=y, timestamps=t,
                                                                                 d=self._viewer_distance,
                                                                                 pixel_size=self._pixel_size)
        assert len(angular_velocities) == len(x), (f"angular velocities (shape {angular_velocities.shape}) do not " +
                                                   f"match the length of x (shape {x.shape})")

        # assume undefined (non-blink) samples are fixations, unless angular velocity is above threshold
        candidates[candidates == cnst.EVENT_LABELS.UNDEFINED] = cnst.EVENT_LABELS.FIXATION
        candidates[angular_velocities >= self._velocity_threshold] = cnst.EVENT_LABELS.SACCADE
        return candidates
