import numpy as np
from overrides import override

from Config import constants as cnst
from GazeDetectors.BaseDetector import BaseDetector


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

    :param lambda: the threshold for the noise, as a multiple of the median-standard-deviation. Default
        is 5, as suggested in the original paper
    :param window_size: the size of the window used to calculate the velocity. Default is 2, as suggested in
        the original paper
    """

    __DEFAULT_LAMBDAA = 5       # lambda noise threshold
    __DEFAULT_WINDOW_SIZE = 2   # (half) number of samples used to calculate axial-velocity

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs.get('lambdaa', EngbertDetector.__DEFAULT_LAMBDAA) <= 0:
            raise ValueError("lambdaa must be positive")
        if kwargs.get('window_size', EngbertDetector.__DEFAULT_WINDOW_SIZE) <= 0:
            raise ValueError("window_size must be positive")
        self._lambda = kwargs.get('lambdaa', self.__DEFAULT_LAMBDAA)
        self._window_size = kwargs.get('window_size', self.__DEFAULT_WINDOW_SIZE)

    def _detect_impl(self, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        candidates = np.asarray(self._candidates, dtype=cnst.EVENT_LABELS).copy()

        # Calculate the velocity of the gaze data in both axes
        x_velocity = self._calculate_axial_velocity(x)
        thresh_x = self._median_standard_deviation(x_velocity) * self._lambda
        y_velocity = self._calculate_axial_velocity(y)
        thresh_y = self._median_standard_deviation(y_velocity) * self._lambda

        # Identify saccade candidates as samples with velocity greater than the noise threshold
        ellipse = (x_velocity / thresh_x) ** 2 + (y_velocity / thresh_y) ** 2
        candidates[ellipse < 1] = cnst.EVENT_LABELS.FIXATION
        candidates[ellipse >= 1] = cnst.EVENT_LABELS.SACCADE

        # add important values to self.data
        df = self.data[cnst.GAZE]
        df[f"{cnst.X}_{cnst.VELOCITY}"] = x_velocity
        df[f"{cnst.Y}_{cnst.VELOCITY}"] = y_velocity
        self.data[cnst.GAZE] = df
        self.data[f'thresh_V{cnst.X}'] = thresh_x
        self.data[f'thresh_V{cnst.Y}'] = thresh_y
        return candidates

    def _calculate_axial_velocity(self, arr) -> np.ndarray:
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
        ws = self._window_size
        velocities = np.full_like(arr_copy, np.nan)
        for t in range(ws, len(arr_copy) - ws):
            sum_before = np.sum(arr_copy[t - ws:t])
            sum_after = np.sum(arr_copy[t + 1:t + ws + 1])
            diff = sum_after - sum_before
            velocities[t] = diff * (self._sr / (2 * (ws + 1)))
        return velocities

    @override
    def _verify_inputs(self, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        t, x, y = super()._verify_inputs(t, x, y)
        if len(x) < 2 * self._window_size:
            raise ValueError(f"derivation window size ({self._window_size}) is too large for the given data")
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
