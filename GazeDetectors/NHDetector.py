import numpy as np
from scipy.signal import savgol_filter

import Config.experiment_config as cnfg
from GazeDetectors.BaseDetector import BaseDetector
import Utils.visual_angle_utils as vis_utils


class NHDetector(BaseDetector):
    """
    Implements the algorithm described by Nyström & Holmqvist in
        Nyström, M., Holmqvist, K. An adaptive algorithm for fixation, saccade, and glissade detection in eyetracking
        data. Behavior Research Methods 42, 188–204 (2010).

    General algorithm:
        1. Calculate angular velocity & acceleration
        2. Denoise the data using SAVGOL filter
        3. Saccade Detection:
            3a. Detect velocity peaks
            3b. Detect saccade onset and offset surrounding each peak
            3c. Ignore saccades that are too short
        4. PSO (Glissade) Detection:
            4a. Detect samples with velocity exceeding the PSO threshold, that shortly follow a saccade offset
            4b. Find PSO offset
        5. Fixation Detection:
            5a. Detect samples that are not part of a saccade, PSO or noise
            5b. Ignore fixations that are too short
    """
    __DEFAULT_FILTER_DURATION = 10  # ms
    __DEFAULT_FILTER_POLYORDER = 2
    __DEFAULT_MAX_SACCADE_VELOCITY, __DEFAULT_MAX_SACCADE_ACCELERATION = 1000, 100000  # deg/s, deg/s^2

    def __init__(self,
                 filter_duration: float = __DEFAULT_FILTER_DURATION,
                 filter_polyorder: int = __DEFAULT_FILTER_POLYORDER,
                 max_saccade_velocity: float = __DEFAULT_MAX_SACCADE_VELOCITY,
                 max_saccade_acceleration: float = __DEFAULT_MAX_SACCADE_ACCELERATION,
                 missing_value=BaseDetector.DEFAULT_MISSING_VALUE,
                 viewer_distance: float = BaseDetector.DEFAULT_VIEWER_DISTANCE,
                 pixel_size: float = BaseDetector.DEFAULT_PIXEL_SIZE,
                 pad_blinks_by: float = BaseDetector.DEFAULT_BLINK_PADDING):
        super().__init__(missing_value=missing_value,
                         viewer_distance=viewer_distance,
                         pixel_size=pixel_size,
                         pad_blinks_by=pad_blinks_by)
        if filter_duration <= 0:
            raise ValueError("filter_duration must be positive")
        self._filter_duration = filter_duration
        if filter_polyorder < 0:
            raise ValueError("filter_polyorder must be non-negative")
        self._filter_polyorder = filter_polyorder
        if max_saccade_velocity <= 0:
            raise ValueError("max_saccade_velocity must be positive")
        self._max_saccade_velocity = max_saccade_velocity
        if max_saccade_acceleration <= 0:
            raise ValueError("max_saccade_acceleration must be positive")
        self._max_saccade_acceleration = max_saccade_acceleration

    def _detect_impl(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        candidates_copy = np.asarray(candidates, dtype=cnfg.EVENTS).copy()

        # detect noise
        sr = self._calculate_sampling_rate(t)
        v, a = self._calculate_velocity_and_acceleration(x, y, sr)
        is_noise = self._detect_noise(v, a)

        # denoise the data
        x_copy, y_copy, v_copy, a_copy = x.copy(), y.copy(), v.copy(), a.copy()
        x_copy[is_noise] = np.nan
        y_copy[is_noise] = np.nan
        v_copy[is_noise] = np.nan
        a_copy[is_noise] = np.nan


        return None

    def _calculate_velocity_and_acceleration(self, x: np.ndarray, y: np.ndarray, sr: float) -> (np.ndarray, np.ndarray):
        """
        Calculates the angular velocity and acceleration of the gaze data using SAVGOL filter for denoising.
        :param x: 1D array of x coordinates
        :param y: 1D array of y coordinates
        :param sr: sampling rate of the data
        :return: angular velocity and acceleration of each point (first is NaN)
        """
        pixel_to_angle_constant = vis_utils.pixels_to_visual_angle(1, self._viewer_distance, self._pixel_size)
        window_size = self._calc_num_samples(self._filter_duration, sr)
        order = self._filter_polyorder
        if window_size <= order:
            raise RuntimeError(f"Cannot compute {order} order SAVGOL filter with duration {self._filter_duration}ms " +
                               f"on data with sampling rate {sr}Hz")

        # calculate angular velocity: v = sr * sqrt((x')^2 + (y')^2) * pixel-to-angle-constant
        dx = savgol_filter(x, window_size, order, deriv=1)
        dy = savgol_filter(y, window_size, order, deriv=1)
        velocity = sr * np.sqrt(dx ** 2 + dy ** 2) * pixel_to_angle_constant

        # calculate angular acceleration: a = sr * sqrt((x'')^2 + (y'')^2) * pixel-to-angle-constant
        ddx = savgol_filter(x, window_size, order, deriv=2)
        ddy = savgol_filter(y, window_size, order, deriv=2)
        acceleration = sr * np.sqrt(ddx ** 2 + ddy ** 2) * pixel_to_angle_constant
        return velocity, acceleration

    def _detect_noise(self, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Detects noise in the gaze data based on the angular velocity and acceleration.
        :param v: angular velocity of the gaze data
        :param a: angular acceleration of the gaze data
        :return: boolean array indicating noise samples
        """
        is_noise = np.zeros(len(v), dtype=bool)
        is_noise[v > self._max_saccade_velocity] = True
        is_noise[a > self._max_saccade_acceleration] = True

        # expand noise periods to include surrounding samples up to the median overall velocity
        median_v = np.nanmedian(v)
        noise_idxs = np.where(is_noise)[0]
        for idx in noise_idxs:
            start, end = idx, idx
            while start > 0 and v[start] > median_v:
                start -= 1
            while end < len(v) and v[end] > median_v:
                end += 1
            is_noise[start:end] = True
        return is_noise

    def _detect_saccades(self, v: np.ndarray, sr: float) -> np.ndarray:
        # find saccade-peak samples
        pt = self._find_saccade_peak_threshold(v)
        is_peak_idxs = np.where(v > pt)[0]

        # for each peak, find the index of the onset: the first sample preceding the peak with velocity below the
        # onset threshold (OnT = mean(v) + 3 * std(v) for v < PT), AND is a local minimum
        onset_threshold = np.nanmean(v[v < pt]) + 3 * np.nanstd(v[v < pt])  # global onset threshold
        start_idxs = []
        for idx in is_peak_idxs:
            start = idx - 1
            while start > 0:
                if v[start] < onset_threshold and v[start] < v[start + 1] and v[start] < v[start - 1]:
                    # sample is below OnT and is local minimum
                    start_idxs.append(start)
                    break
                start -= 1
        assert len(start_idxs) == len(is_peak_idxs), "Failed to find saccade onset for all peaks"  # sanity

        # for each peak, find the index of the offset: the first sample following the peak with velocity below the
        # offset threshold (OfT = a * OnT + b * OtT), AND is a local minimum
        # note the locally adaptive term: OtT = mean(v) + 3 * std(v) for the min_fixation_samples prior to saccade onset
        min_fixation_duration = cnfg.EVENT_DURATIONS[cnfg.EVENTS.FIXATION][0]
        min_fixation_samples = self._calc_num_samples(min_fixation_duration, sr)
        # TODO: start here



        # for every peak, expand it backwards and forwards to find the onset and offset
        for idx in is_peak_idxs:
            start, end = idx, idx

            # find saccade onset index
            while start > 0 and v[start] > onset_threshold:
                start -= 1

            # find saccade offset threshold: OfT = a * OnT + b * OtT
            # where OtT = mean(v) + 3 * std(v) for the min_fixation_samples preceding the peak
            while end < len(v) and v[end] > onset_threshold:
                end += 1
        return None

    @staticmethod
    def _find_saccade_peak_threshold(v: np.ndarray, max_iters: int = 100) -> float:
        """
        Finds threshold velocity (PT) for detecting saccade peaks, using an iterative algorithm:
        1. Start with PT_1 = max(300, 250, 200, 150, 100) s.t. there is at least 1 sample with higher velocity
        2. Calculate mean & std of velocity below PT
        3. Update PT = mean + 6 * std
        4. Repeat steps 2-3 until PT converges

        :param v: angular velocity of the gaze data
        :param max_iters: maximum number of iterations
        :return: the threshold velocity for detecting saccade peaks
        """
        # find the starting PT value, by making sure there are at least 1 peak with higher velocity
        start_criteria = np.arange(300, 99, -50)
        pt = start_criteria[0]
        for i, pt in enumerate(start_criteria):
            if any(v > pt):
                break
            if i == len(start_criteria) - 1:
                raise RuntimeError("Could not find a suitable PT_1 value for saccade detection")

        # iteratively update PT value until convergence
        pt_prev = 0
        while abs(pt - pt_prev) > 1 and max_iters > 0:
            max_iters -= 1
            pt_prev = pt
            # calculate mean & std of velocity below PT
            mu = np.nanmean(v[v <= pt])
            sigma = np.nanstd(v[v <= pt])
            # update PT
            pt = mu + 6 * sigma
        if max_iters == 0:
            raise RuntimeError("Failed to converge on PT_1 value for saccade detection")
        return pt

