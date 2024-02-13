import numpy as np
from overrides import override

from Detectors.BaseDetector import BaseDetector


class IVTDetector(BaseDetector):
    __DEFAULT_VELOCITY_THRESHOLD = 0.5

    def __init__(self,
                 missing_value: float = BaseDetector._MISSING_VALUE,
                 velocity_threshold: float = __DEFAULT_VELOCITY_THRESHOLD):
        super().__init__(missing_value)
        self._velocity_threshold = velocity_threshold

    @override
    def _identify_gaze_event_candidates(self, x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        # TODO: implement
        raise NotImplementedError

    def detect(self, x_coords, y_coords) -> np.ndarray:
        # calculate velocity by difference between coordinates
        diff_x = np.diff(x_coords)
        diff_y = np.diff(y_coords)

        # we assume that the frequency is 500Hz so there is 2ms gap between every two samples
        # FIXME: can't assume frequency, need to get it from the data
        velocity = np.sqrt(np.power(diff_x, 2) + np.power(diff_y, 2)) / 2

        # velocities below threshold = fixation (label 1), above = saccade (label 2)
        labels = np.ndarray(shape=velocity.shape)
        labels[velocity < self._velocity_threshold] = 1
        labels[velocity >= self._velocity_threshold] = 2

        return labels
