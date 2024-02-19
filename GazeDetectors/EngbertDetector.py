import numpy as np
from overrides import override

import constants as cnst
from GazeDetectors.BaseDetector import BaseDetector
from Config.GazeEventTypeEnum import GazeEventTypeEnum


class EngbertDetector(BaseDetector):
    """
    Implements the algorithm described by Engbert, Kliegl, and Mergenthaler in
        "Microsaccades uncover the orientation of covert attention" (2003)
        "Microsaccades are triggered by low retinal image slip" (2006)

    Implementation is based on the following repositories:
        - https://shorturl.at/lyBE2
        - https://shorturl.at/DHJZ6

    General algorithm:
        1. Calculate the velocity of the gaze data in both axes
        2. Calculate the median-standard-deviation of the velocity in both axes
        3. Calculate the noise threshold as the multiple of the median-standard-deviation with the constant `lambda_noise_threshold`
        4. Identify saccade candidates as samples with velocity greater than the noise threshold

    :param lambda_noise_threshold: the threshold for the noise, as a multiple of the median-standard-deviation. Default
        is 5, as suggested in the original paper
    :param derivation_window_size: the size of the window used to calculate the velocity. Default is 2, as suggested in
        the original paper
    """

    __LAMBDA_NOISE_THRESHOLD = 5
    __DERIVATION_WINDOW_SIZE = 2

    def __init__(self,
                 lambda_noise_threshold: float = __LAMBDA_NOISE_THRESHOLD,
                 derivation_window_size: int = __DERIVATION_WINDOW_SIZE,
                 missing_value=BaseDetector.DEFAULT_MISSING_VALUE,
                 viewer_distance: float = BaseDetector.DEFAULT_VIEWER_DISTANCE,
                 pixel_size: float = BaseDetector.DEFAULT_PIXEL_SIZE,
                 minimum_event_duration: float = BaseDetector.DEFAULT_MINIMUM_EVENT_DURATION,
                 pad_blinks_by: float = BaseDetector.DEFAULT_BLINK_PADDING):
        super().__init__(missing_value=missing_value,
                         viewer_distance=viewer_distance,
                         pixel_size=pixel_size,
                         minimum_event_duration=minimum_event_duration,
                         pad_blinks_by=pad_blinks_by)
        if lambda_noise_threshold <= 1:
            raise ValueError("lambda_noise_threshold must be greater than 1")
        self._lambda_noise_threshold = lambda_noise_threshold
        if derivation_window_size <= 0:
            raise ValueError("derivation_window_size must be positive")
        self._derivation_window_size = round(derivation_window_size)

    def _detect_impl(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        candidates_copy = np.asarray(candidates, dtype=GazeEventTypeEnum).copy()

        # Calculate the velocity of the gaze data in both axes
        sr = self._calculate_sampling_rate(t)
        x_velocity = self._calculate_axial_velocity(x, sr)
        thresh_x = self._median_standard_deviation(x_velocity) * self._lambda_noise_threshold
        y_velocity = self._calculate_axial_velocity(y, sr)
        thresh_y = self._median_standard_deviation(y_velocity) * self._lambda_noise_threshold

        # Identify saccade candidates as samples with velocity greater than the noise threshold
        ellipse = (x_velocity / thresh_x) ** 2 + (y_velocity / thresh_y) ** 2
        candidates_copy[ellipse < 1] = GazeEventTypeEnum.FIXATION
        candidates_copy[ellipse >= 1] = GazeEventTypeEnum.SACCADE
        return candidates_copy

    def _calculate_axial_velocity(self, arr, sr) -> np.ndarray:
        """
        Calculates the velocity along a single axis, based on the algorithm described in the original paper:
        1. Sum values in a window of size window_size, *before* the current sample:
            sum_before = arr(t-1) + arr(t-2) + ... + arr(t-ws)
        2. Sum values in a window of size window_size, *after* the current sample
            sum_after = arr(t+1) + arr(t+2) + ... + arr(t+ws)
        3. Calculate the difference between the two sums
            diff = sum_after - sum_before
        4. Divide by the time-difference, calculated as `sampling_rate` / (2 * `window_size`)
            velocity = diff * (sampling_rate / (2 * (window_size + 1))
        5. For the first and last `window_size` samples, the velocity is np.nan
        """
        arr_copy = np.copy(arr)
        ws = self._derivation_window_size
        velocities = np.full_like(arr_copy, np.nan)
        for t in range(ws, len(arr_copy) - ws):
            sum_before = np.sum(arr_copy[t - ws:t])
            sum_after = np.sum(arr_copy[t + 1:t + ws + 1])
            diff = sum_after - sum_before
            velocities[t] = diff * (sr / (2 * (ws + 1)))
        return velocities

    @override
    def _verify_inputs(self, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        t, x, y = super()._verify_inputs(t, x, y)
        if len(x) < 2 * self._derivation_window_size:
            raise ValueError(f"derivation window size ({self._derivation_window_size}) is too large for the given data")
        return t, x, y

    @staticmethod
    def _median_standard_deviation(arr) -> float:
        """
        Calculates the median standard deviation of the given array
        """
        squared_median = np.power(np.nanmedian(arr), 2)
        median_of_squares = np.nanmedian(np.power(arr, 2))
        sd = np.sqrt(median_of_squares - squared_median)
        return float(np.nanmax([sd, cnst.EPSILON]))
