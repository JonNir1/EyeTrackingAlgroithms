import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, Tuple, List

import Config.constants as cnst
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
    __DEFAULT_ALPHA = 0.7   # weight for saccade onset threshold when calculating saccade_offset_threshold
    __DEFAULT_BETA = 0.3    # weight for locally adaptive threshold when calculating saccade_offset_threshold

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs.get('filter_duration', self.__DEFAULT_FILTER_DURATION) <= 0:
            raise ValueError("filter_duration must be positive")
        if kwargs.get('filter_polyorder', self.__DEFAULT_FILTER_POLYORDER) < 0:
            raise ValueError("filter_polyorder must be non-negative")
        if kwargs.get('max_saccade_velocity', self.__DEFAULT_MAX_SACCADE_VELOCITY) <= 0:
            raise ValueError("max_saccade_velocity must be positive")
        if kwargs.get('max_saccade_acceleration', self.__DEFAULT_MAX_SACCADE_ACCELERATION) <= 0:
            raise ValueError("max_saccade_acceleration must be positive")
        if not 0 <= kwargs.get('alpha', self.__DEFAULT_ALPHA) <= 1:
            raise ValueError("parameter alpha must be in the range [0, 1]")
        if not 0 <= kwargs.get('beta', self.__DEFAULT_BETA) <= 1:
            raise ValueError("parameter beta must be in the range [0, 1]")

        self._filter_duration = kwargs.get('filter_duration', self.__DEFAULT_FILTER_DURATION)
        self._filter_polyorder = kwargs.get('filter_polyorder', self.__DEFAULT_FILTER_POLYORDER)
        self._max_saccade_velocity = kwargs.get('max_saccade_velocity', self.__DEFAULT_MAX_SACCADE_VELOCITY)
        self._max_saccade_acceleration = kwargs.get('max_saccade_acceleration', self.__DEFAULT_MAX_SACCADE_ACCELERATION)
        self._alpha = kwargs.get('alpha', self.__DEFAULT_ALPHA)
        self._beta = kwargs.get('beta', self.__DEFAULT_BETA)
        self._detect_high_psos = kwargs.get('detect_high_psos', False)

    def _detect_impl(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
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

        # detect saccades
        peak_threshold = self._find_saccade_peak_threshold(v_copy)  # threshold velocity for detecting saccade-peaks
        onset_threshold = np.nanmean(v_copy[v_copy < peak_threshold]) + 3 * np.nanstd(v_copy[v_copy < peak_threshold])  # global saccade-onset threshold velocity
        saccades_info = self._detect_saccades(v_copy, sr, peak_threshold, onset_threshold)  # peak_idx -> (start_idx, end_idx, offset_threshold)
        is_saccades = np.zeros(len(t), dtype=bool)
        for _p_idx, (start_idx, end_idx, _) in saccades_info.items():
            is_saccades[start_idx: end_idx] = True

        # detect PSOs
        psos_info = self._detect_psos(v_copy, saccades_info, sr, peak_threshold, onset_threshold)
        is_psos = np.zeros(len(t), dtype=bool)
        for start_idx, end_idx in psos_info:
            is_psos[start_idx: end_idx] = True
        assert not np.any(is_saccades & is_psos), "PSO and saccade overlap"  # sanity  # TODO: remove

        # mark events on the candidates array
        candidates_copy = np.asarray(candidates, dtype=cnst.EVENTS).copy()
        is_blinks = candidates_copy == cnst.EVENTS.BLINK
        candidates_copy[is_saccades] = cnst.EVENTS.SACCADE
        candidates_copy[is_psos] = cnst.EVENTS.PSO
        candidates_copy[~(is_noise | is_saccades | is_psos | is_blinks)] = cnst.EVENTS.FIXATION
        return candidates_copy

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
            raise RuntimeError(f"Cannot compute {order}-order SAVGOL filter with duration {self._filter_duration}ms " +
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

    def _detect_saccades(self, v: np.ndarray, sr: float, pt: float, ont: float) -> Dict[int, Tuple[int, int, float]]:
        """
        Detects saccades in the gaze data based on the angular velocity:
        1. Detect samples with velocity exceeding the saccade peak threshold (PT)
        2. Find the 1st sample preceding each peak with velocity below the onset threshold (OnT) and is a local minimum
        3. Find the 1st sample following each peak with velocity below the offset threshold (OfT) and is a local minimum
        4. Ignore saccades whose preceding samples have mean velocity above PT
        5. Match each saccade peak-idx with its onset-idx, offset-idx and offset-threshold-velocity

        :param v: angular velocity of the gaze data
        :param sr: sampling rate of the data
        :param pt: saccades' peak threshold velocity
        :param ont: saccades' onset threshold velocity

        :return: dictionary of saccade peak-idx -> (onset-idx, offset-idx, offset-threshold-velocity)
        """
        # find saccades' onsets by moving backwards from a peak until finding a local minimum below the onset threshold
        is_peak_idxs = (np.where(v > pt)[0]).astype(int)
        start_idxs = [self.__find_local_minimum_index(v, idx, ont, move_back=True) for idx in is_peak_idxs]

        # for each peak, find the index of the offset: the first sample following the peak with velocity below the
        # offset threshold (OfT = a * OnT + b * OtT), AND is a local minimum
        # note the locally adaptive term: OtT = mean(v) + 3 * std(v) for the min_fixation_samples prior to saccade onset
        min_fixation_duration = cnfg.EVENT_DURATIONS[cnst.EVENTS.FIXATION][0]
        min_fixation_samples = self._calc_num_samples(min_fixation_duration, sr)

        saccades_info = {}  # peak_idx -> (start_idx, end_idx, offset_threshold)
        for saccade_id, peak_idx in enumerate(is_peak_idxs):
            # calculate velocity in the window prior to saccade onset
            saccade_start_idx = start_idxs[saccade_id]
            window_start_idx = max(0, saccade_start_idx - min_fixation_samples)
            window = v[window_start_idx: saccade_start_idx]
            window_mean, window_std = np.nanmean(window), np.nanstd(window)
            if window_mean > pt:
                # exclude saccades whose preceding window has mean velocity above PT
                continue

            # calculate offset threshold
            locally_adaptive_threshold = window_mean + 3 * window_std
            locally_adaptive_threshold = locally_adaptive_threshold if np.isfinite(locally_adaptive_threshold) else ont  # if window is empty, use global threshold
            offset_threshold = self._alpha * ont + self._beta * locally_adaptive_threshold

            # find saccade offset index
            end = is_peak_idxs[saccade_id] + 1
            end = self.__find_local_minimum_index(v, end, offset_threshold, move_back=False)

            # save saccade info
            saccades_info[peak_idx] = (saccade_start_idx, end, offset_threshold)
        return saccades_info

    def _detect_psos(self,
                     v: np.ndarray,
                     saccade_info: Dict[int, Tuple[int, int, float]],
                     sr: float,
                     pt: float,
                     ont: float) -> List[Tuple[int, int]]:
        """
        Detects PSOs in the gaze data based on the angular velocity:
        1. Determine what velocity threshold to use for PSO detection (depending on value of self._detect_high_psos)
        2. Check if a window of length min fixation duration, succeeding each saccade, has samples above AND below the
              threshold. If so, there is a PSO.
        3. Identify the end of the PSO - the first local velocity-minimum after the last sample above the threshold
        4. Ignore PSOs with amplitude exceeding the preceding saccade

        :param v: angular velocity of the gaze data
        :param saccade_info: dictionary of saccade peak-idx -> (onset-idx, offset-idx, offset-threshold-velocity)
        :param sr: sampling rate of the data
        :param pt: saccades' peak threshold velocity (used for determining PSOs' threshold velocity)
        :param ont: saccades' onset threshold velocity (used for determining PSOs' threshold velocity)

        :return: list of PSO start & end idxs
        """
        # calculate the size of the window where PSO may occur after each saccade
        min_fixation_duration = cnfg.EVENT_DURATIONS[cnst.EVENTS.FIXATION][0]
        min_fixation_samples = self._calc_num_samples(min_fixation_duration, sr)

        # find PSO start & end idxs after each saccade
        saccade_info_list = sorted(saccade_info.items(), key=lambda x: x[0])
        pso_idxs = []
        for i, (sac_peak_idx, (sac_start_idx, sac_end_idx, sac_offset_threshold)) in enumerate(saccade_info_list):
            # determine which threshold to use for PSO detection
            possible_thresh = [pt, ont, sac_offset_threshold]
            v_thresh = max(possible_thresh) if self._detect_high_psos else min(possible_thresh)

            # if a window succeeding a saccade has samples above AND below the threshold, there is a PSO
            window = v[sac_end_idx + 1: sac_end_idx + 1 + min_fixation_samples]
            is_above = window > v_thresh
            is_below = window < v_thresh
            if not any(is_above) or not any(is_below):
                # no PSO in the window
                continue
            if max(window) > v_thresh:
                # Ignore PSOs with amplitude exceeding the preceding saccade
                continue

            # PSO's start index is the first sample after a saccade offset
            pso_start_idx = sac_end_idx + 1

            # PSO's end index is the *first* local minimum after the *last* sample above the threshold
            # PSO must end before the next saccade starts
            is_last_saccade = i == len(saccade_info_list) - 1
            next_saccade_start_idx = saccade_info_list[i + 1][1][0] if not is_last_saccade else len(v)
            pso_end_idx_boundary = min([next_saccade_start_idx - 1, len(v) - 1])

            # find last sample above threshold within the PSO boundary
            is_above = v[pso_start_idx: pso_end_idx_boundary + 1] > v_thresh
            assert any(is_above), "Error in PSO detection: no sample above threshold"  # sanity  # TODO: remove
            last_peak_idx = pso_start_idx + np.argmax(is_above)

            # find the first local minimum after the last sample above the threshold
            window = v[last_peak_idx: pso_end_idx_boundary + 1]
            window_minimum_idx = self.__find_local_minimum_index(window, 0, min_thresh=v_thresh, move_back=False)
            pso_end_idx = last_peak_idx + window_minimum_idx

            # save PSO info
            pso_idxs.append((pso_start_idx, pso_end_idx))
        return pso_idxs

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
        start_criteria = np.arange(300, 74, -25)
        pt = start_criteria[0]
        for i, pt in enumerate(start_criteria):
            if any(v > pt):
                break
            if i == len(start_criteria) - 1:
                # raise RuntimeError("Could not find a suitable PT_1 value for saccade detection")
                pt = np.median(start_criteria)

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

    @staticmethod
    def __find_local_minimum_index(arr: np.ndarray, idx: int, min_thresh=np.inf, move_back=False) -> int:
        """
        Finds a local minimum in the array (an element that is smaller than its neighbors) starting from the given index.
        :param arr: the array to search in
        :param idx: the starting index
        :param min_thresh: the minimum value for a local minimum    (default: infinity)
        :param move_back: whether to move back or forward from the starting index   (default: False)
        :return: the index of the local minimum
        """
        # if not 0 <= idx < len(arr):
        #     raise IndexError(f"Index {idx} is out of bounds for array of length {len(arr)}")
        while 0 < idx < len(arr):
            if arr[idx] < min_thresh and arr[idx] < arr[idx + 1] and arr[idx] < arr[idx - 1]:
                # idx is a local minimum
                return idx
            idx = idx - 1 if move_back else idx + 1
        return idx

