import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, Tuple, List

import Config.constants as cnst
import Config.experiment_config as cnfg
from GazeDetectors.BaseDetector import BaseDetector
import Utils.visual_angle_utils as vis_utils
import Utils.array_utils as arr_utils


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
    __DEFAULT_FILTER_DURATION = cnfg.EVENT_MAPPING[cnst.EVENTS.SACCADE]["min_duration"] * 2  # default: 10*2 ms
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

    @property
    def _minimum_fixation_samples(self) -> int:
        min_duration = cnfg.EVENT_MAPPING[cnst.EVENTS.FIXATION]["min_duration"]
        return self._calc_num_samples(min_duration)

    def _detect_impl(self, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # detect noise
        v, a = self._calculate_velocity_and_acceleration(x, y)
        is_noise = self._detect_noise(v, a)

        # denoise the data
        x_copy, y_copy, v_copy, a_copy = x.copy(), y.copy(), v.copy(), v.copy()
        x_copy[is_noise] = np.nan
        y_copy[is_noise] = np.nan
        v_copy[is_noise] = np.nan
        a_copy[is_noise] = np.nan

        # detect saccades
        peak_threshold, onset_threshold = self._estimate_saccade_thresholds(v_copy)  # global velocity thresholds
        saccades_info = self._detect_saccades(v_copy, peak_threshold, onset_threshold)  # peak_idx -> (start_idx, end_idx, offset_threshold)
        is_saccades = np.zeros(len(t), dtype=bool)
        for _p_idx, (start_idx, end_idx, _) in saccades_info.items():
            is_saccades[start_idx: min([end_idx + 1, len(is_saccades)])] = True

        # detect PSOs
        psos_info = self._detect_psos(v_copy, saccades_info, peak_threshold, onset_threshold)
        is_psos = np.zeros(len(t), dtype=bool)
        for start_idx, end_idx in psos_info:
            is_psos[start_idx: end_idx] = True
        assert not np.any(is_saccades & is_psos), "PSO and saccade overlap"  # sanity  # TODO: remove

        # mark events on the candidates array
        candidates_copy = np.asarray(self._candidates, dtype=cnst.EVENTS).copy()
        is_blinks = candidates_copy == cnst.EVENTS.BLINK
        candidates_copy[is_saccades] = cnst.EVENTS.SACCADE
        candidates_copy[is_psos] = cnst.EVENTS.PSO
        candidates_copy[~(is_noise | is_saccades | is_psos | is_blinks)] = cnst.EVENTS.FIXATION

        # save important values to self.data
        df = self.data[cnst.GAZE]
        df[cnst.VELOCITY] = v
        df[cnst.ACCELERATION] = a
        self.data[cnst.GAZE] = df
        self.data["saccade_peak_threshold"] = peak_threshold
        self.data["saccade_onset_threshold"] = onset_threshold
        return candidates_copy

    def _calculate_velocity_and_acceleration(self, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Calculates the 1st and 2nd derivatives of the gaze data, using SAVGOL filter. Then calculates the angular
        velocity and acceleration from the derivatives.

        Note the original paper calculates the velocity and acceleration using:
            v = sr * sqrt[(x')^2 + (y')^2] * pixel-to-angle-constant
            a = sr * sqrt[(x'')^2 + (y'')^2] * pixel-to-angle-constant
        We use `delta=1/self._sr` when calculating the derivatives, to account for sampling time, so we don't need to
        multiply by sr when computing `v` and `a`. See more in the scipy documentation and in the following links:
            - https://stackoverflow.com/q/56168730/8543025
            - https://github.com/scipy/scipy/issues/9910

        :param x: 1D array of x coordinates
        :param y: 1D array of y coordinates
        :return: angular velocity and acceleration of each point
        """
        pixel_to_angle_constant = vis_utils.pixels_to_visual_angle(1, self._viewer_distance, self._pixel_size)
        window_size = self._calc_num_samples(self._filter_duration)
        if window_size <= self._filter_polyorder:
            raise RuntimeError(f"Cannot compute {self._filter_polyorder}-order SAVGOL filter with duration " +
                               f"{self._filter_duration}ms on data with sampling rate {self._sr}Hz")

        # calculate angular velocity (deg/s): v = sqrt((x')^2 + (y')^2) * pixel-to-angle-constant
        dx = savgol_filter(x, window_size, self._filter_polyorder, deriv=1, delta=1/self._sr)
        dy = savgol_filter(y, window_size, self._filter_polyorder, deriv=1, delta=1/self._sr)
        velocity = np.sqrt(dx ** 2 + dy ** 2) * pixel_to_angle_constant

        # calculate angular acceleration (deg/s^2): a = sqrt((x'')^2 + (y'')^2) * pixel-to-angle-constant
        ddx = savgol_filter(x, window_size, self._filter_polyorder, deriv=2, delta=1/self._sr)
        ddy = savgol_filter(y, window_size, self._filter_polyorder, deriv=2, delta=1/self._sr)
        acceleration = np.sqrt(ddx ** 2 + ddy ** 2) * pixel_to_angle_constant
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

    def _estimate_saccade_thresholds(self,
                                     v: np.ndarray,
                                     max_iters: int = 100,
                                     enforce_min_dur: bool = True) -> (float, float):
        """
        Finds threshold velocity (PT) for detecting saccade peaks, using an iterative algorithm:
        1. Start with PT_1 = max(300, 250, 200, 150, 100) s.t. there is at least 1 sample with higher velocity (deg / s)
        2. Find indices of samples with velocity below PT
        3. If enforce_min_dur is True, ignores samples that aren't part of a chunk of length >= least min_fixation_samples
        4. Calculate mean & std of velocity below PT
        5. Update PT = mean + 6 * std
        6. Repeat steps 2-5 until PT converges

        :param v: angular velocity of the gaze data
        :param max_iters: maximum number of iterations
        :param enforce_min_dur: if true, calculates thresholds only based on chunks of consecutive samples that are
            longer than the minimum duration of a fixation. This is meant to help convergence of the peak threshold.
            See more details in https://shorturl.at/wyCH7.

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
        is_below_pt = v <= pt
        pt_prev = 0
        while abs(pt - pt_prev) > 1 and max_iters > 0:
            max_iters -= 1
            pt_prev = pt

            # find indices of samples with velocity below PT
            is_below_pt = v <= pt
            if enforce_min_dur:
                chunks_below_pt = [ch for ch in arr_utils.get_chunk_indices(is_below_pt)
                                   if is_below_pt[ch[0]] and len(ch) >= self._minimum_fixation_samples]
                # calc number of samples to ignore at the edges of each chunk (to avoid contamination from saccades)
                num_edge_idxs = self._calc_num_samples(cnfg.EVENT_MAPPING[cnst.EVENTS.SACCADE]["min_duration"] // 3)
                chunks_below_pt = [ch[num_edge_idxs: -num_edge_idxs] for ch in chunks_below_pt]
                is_below_pt = np.concatenate(chunks_below_pt)

            # calculate mean & std of velocity below PT
            mu = np.nanmean(v[is_below_pt])
            sigma = np.nanstd(v[is_below_pt])
            # update PT
            pt = mu + 6 * sigma
        if max_iters == 0:
            raise RuntimeError("Failed to converge on PT_1 value for saccade detection")

        ont = np.nanmean(v[is_below_pt]) + 3 * np.nanstd(v[is_below_pt])
        return pt, ont

    def _detect_saccades(self,
                         v: np.ndarray,
                         pt: float,
                         ont: float,
                         min_peak_samples: int = 2) -> Dict[int, Tuple[int, int, float]]:
        """
        Detects saccades in the gaze data based on the angular velocity:
        1. Detect samples with velocity exceeding the saccade peak threshold (PT)
        2. Find the 1st sample preceding each peak with velocity below the onset threshold (OnT) and is a local minimum
        3. Find the 1st sample following each peak with velocity below the offset threshold (OfT) and is a local minimum
        4. Ignore saccades whose preceding samples have mean velocity above PT
        5. Match each saccade peak-idx with its onset-idx, offset-idx and offset-threshold-velocity

        :param v: angular velocity of the gaze data
        :param pt: saccades' peak threshold velocity
        :param ont: saccades' onset threshold velocity
        :param min_peak_samples: minimum number of samples for a peak to be considered a saccade, otherwise ignored

        :return: dictionary of saccade peak-idx -> (onset-idx, offset-idx, offset-threshold-velocity)
        """
        # find idxs of samples above PT
        is_above_pt = v > pt
        chunks_above_pt = [ch for ch in arr_utils.get_chunk_indices(is_above_pt)
                           if is_above_pt[ch[0]] and len(ch) >= min_peak_samples]   # assume very short peaks are noise

        # each chunk is a possible saccade, find the onset and offset of each saccade
        saccades_info = {}  # peak_idx -> (start_idx, end_idx, offset_threshold)
        for chunk in chunks_above_pt:
            onset_idx = self.__find_local_minimum_index(v, chunk[0], ont, move_back=True)

            # calculate the offset threshold: OfT = a * OnT + b * OtT
            # note the locally adaptive term: OtT = mean(v) + 3 * std(v) for the min_fixation_samples before saccade onset
            window_start_idx = max(0, onset_idx - self._minimum_fixation_samples)
            window_vel = v[window_start_idx: onset_idx]
            local_threshold = np.nanmean(window_vel) + 3 * np.nanstd(window_vel)
            # check if the local velocity threshold exceeds the peak velocity threshold
            if np.isfinite(local_threshold) and local_threshold < pt:
                offset_threshold = self._alpha * ont + self._beta * local_threshold
            else:
                offset_threshold = ont

            # save saccade info
            offset_idx = self.__find_local_minimum_index(v, chunk[-1] + 1, offset_threshold, move_back=False)
            saccades_info[chunk[0]] = (onset_idx, offset_idx, offset_threshold)
        return saccades_info

    def _detect_psos(self,
                     v: np.ndarray,
                     saccade_info: Dict[int, Tuple[int, int, float]],
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
        :param pt: saccades' peak threshold velocity (used for determining PSOs' threshold velocity)
        :param ont: saccades' onset threshold velocity (used for determining PSOs' threshold velocity)

        :return: list of PSO start & end idxs
        """
        # find PSO start & end idxs after each saccade
        saccade_info_list = sorted(saccade_info.items(), key=lambda x: x[0])
        pso_idxs = []
        for i, (sac_peak_idx, (sac_start_idx, sac_end_idx, sac_offset_threshold)) in enumerate(saccade_info_list):
            # determine which threshold to use for PSO detection
            possible_thresh = [pt, ont, sac_offset_threshold]
            v_thresh = max(possible_thresh) if self._detect_high_psos else min(possible_thresh)

            # PSO's start index is the first sample after a saccade offset
            pso_start_idx = sac_end_idx + 1

            # if a window succeeding a saccade has samples above AND below the threshold, there is a PSO
            window = v[pso_start_idx : pso_start_idx + self._minimum_fixation_samples]
            is_above = window > v_thresh
            is_below = window < v_thresh
            if not any(is_above) or not any(is_below):
                # no PSO in the window
                continue
            if max(window) > v_thresh:
                # Ignore PSOs with amplitude exceeding the preceding saccade
                continue

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

