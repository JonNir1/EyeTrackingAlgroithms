import numpy as np
from scipy.signal import savgol_filter
from typing import Dict, Tuple

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
    The code is based on the Matlab implementation available in https://github.com/dcnieho/NystromHolmqvist2010, which
    was developed for the following article:
        Niehorster, D. C., Siu, W. W., & Li, L. (2015). Manual tracking enhances smooth pursuit eye movements. Journal
        of vision, 15(15), 11-11.

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
    __DEFAULT_FILTER_DURATION = cnfg.EVENT_MAPPING[cnfg.EVENT_LABELS.SACCADE][cnst.MIN_DURATION] * 2  # default: 10*2 ms
    __DEFAULT_FILTER_POLYORDER = 2
    __DEFAULT_MAX_SACCADE_VELOCITY, __DEFAULT_MAX_SACCADE_ACCELERATION = 1000, 100000  # deg/s, deg/s^2
    __DEFAULT_ALPHA = 0.7   # weight for saccade onset threshold when calculating saccade_offset_threshold
    __DEFAULT_BETA = 0.3    # weight for locally adaptive threshold when calculating saccade_offset_threshold

    def __init__(
            self,
            missing_value=cnfg.DEFAULT_MISSING_VALUE,
            viewer_distance=cnfg.DEFAULT_VIEWER_DISTANCE,
            pixel_size=cnfg.SCREEN_MONITOR.pixel_size,
            dilate_nans_by=cnfg.DEFAULT_NAN_PADDING,
            filter_duration=__DEFAULT_FILTER_DURATION,
            filter_polyorder=__DEFAULT_FILTER_POLYORDER,
            max_saccade_velocity=__DEFAULT_MAX_SACCADE_VELOCITY,
            max_saccade_acceleration=__DEFAULT_MAX_SACCADE_ACCELERATION,
            alpha=__DEFAULT_ALPHA,
            beta=__DEFAULT_BETA,
            allow_high_psos=True,
    ):
        super().__init__(missing_value, viewer_distance, pixel_size, dilate_nans_by)
        self._filter_duration = filter_duration
        if self._filter_duration <= 0:
            raise ValueError("filter_duration must be positive")
        self._filter_polyorder = filter_polyorder
        if self._filter_polyorder < 0:
            raise ValueError("filter_polyorder must be non-negative")
        self._max_saccade_velocity = max_saccade_velocity
        if self._max_saccade_velocity <= 0:
            raise ValueError("max_saccade_velocity must be positive")
        self._max_saccade_acceleration = max_saccade_acceleration
        if self._max_saccade_acceleration <= 0:
            raise ValueError("max_saccade_acceleration must be positive")
        self._alpha = alpha
        if not 0 <= self._alpha <= 1:
            raise ValueError("parameter alpha must be in the range [0, 1]")
        self._beta = beta
        if not 0 <= self._beta <= 1:
            raise ValueError("parameter beta must be in the range [0, 1]")
        self._allow_high_psos = allow_high_psos

    @property
    def _minimum_fixation_samples(self) -> int:
        min_duration = cnfg.EVENT_MAPPING[cnfg.EVENT_LABELS.FIXATION][cnst.MIN_DURATION]
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

        # detect saccades and PSOs
        peak_threshold, onset_threshold = self._estimate_saccade_thresholds(v_copy)  # global velocity thresholds
        saccades_info = self._detect_saccades(v_copy, peak_threshold, onset_threshold)  # saccade id -> (onset-idx, peak-idx, offset-idx, offset-threshold-velocity)
        psos_info = self._detect_psos(v_copy, saccades_info, peak_threshold, onset_threshold)  # saccade id -> (start_idx, end_idx, pso_type)

        # save results
        candidates = self._classify_samples(is_noise, saccades_info, psos_info)
        df = self.data[cnst.GAZE]
        df[cnst.VELOCITY] = v
        df[cnst.ACCELERATION] = a
        self.data[cnst.GAZE] = df
        self.data["saccade_peak_threshold"] = peak_threshold
        self.data["saccade_onset_threshold"] = onset_threshold
        return candidates

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
                num_edge_idxs = self._calc_num_samples(cnfg.EVENT_MAPPING[cnfg.EVENT_LABELS.SACCADE][cnst.MIN_DURATION] // 3)
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
                         min_peak_samples: int = 2) -> Dict[int, Tuple[int, int, int, float]]:
        """
        Detects saccades in the gaze data based on the angular velocity:
        1. Detect samples with velocity exceeding the saccade peak threshold (PT)
        2. Find the 1st sample preceding each peak with velocity below the onset threshold (OnT) and is a local minimum
        3. Find the 1st sample following each peak with velocity below the offset threshold (OfT) and is a local minimum
        4. Match each saccade peak-idx with its onset-idx, offset-idx and offset-threshold-velocity

        :param v: angular velocity of the gaze data
        :param pt: saccades' peak threshold velocity
        :param ont: saccades' onset threshold velocity
        :param min_peak_samples: minimum number of samples for a peak to be considered a saccade, otherwise ignored

        :return: dictionary of saccade id -> (onset-idx, peak-idx, offset-idx, offset-threshold-velocity)
        """
        # find idxs of samples above PT
        is_above_pt = v > pt
        chunks_above_pt = [ch for ch in arr_utils.get_chunk_indices(is_above_pt)
                           if is_above_pt[ch[0]] and len(ch) >= min_peak_samples]   # assume very short peaks are noise

        # each chunk is a possible saccade, find the onset and offset of each saccade
        saccades_info = {}  # peak_idx -> (start_idx, end_idx, offset_threshold)
        for i, chunk in enumerate(chunks_above_pt):
            peak_idx: int = chunk[0]
            onset_idx = self.__find_local_minimum_index(v, peak_idx, ont, move_back=True)

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
            last_peak_idx: int = chunk[-1]
            offset_idx = self.__find_local_minimum_index(v, last_peak_idx, offset_threshold, move_back=False)
            saccades_info[i] = (onset_idx, peak_idx, offset_idx, offset_threshold)
        return saccades_info

    def _detect_psos(self,
                     v: np.ndarray,
                     saccade_info: Dict[int, Tuple[int, int, int, float]],
                     pt: float,
                     ont: float) -> Dict[int, Tuple[int, int, bool]]:
        """
        Detects PSOs in the gaze data based on the angular velocity:
        1. Determine what velocity threshold to use for PSO detection (depending on value of self._detect_high_psos)
        2. Check if a window of length min fixation duration, succeeding each saccade, has samples above AND below the
              threshold. If so, there is a PSO.
        3. Identify the end of the PSO - the first local velocity-minimum after the last sample above the threshold
        4. Ignore PSOs with amplitude exceeding the preceding saccade

        :param v: angular velocity of the gaze data
        :param saccade_info: dictionary of saccade id -> (onset-idx, peak-idx, offset-idx, offset-threshold-velocity)
        :param pt: saccades' peak threshold velocity (used for determining PSOs' threshold velocity)
        :param ont: saccades' onset threshold velocity (used for determining PSOs' threshold velocity)

        :return: dict matching saccade id with PSO start-idx, end-idx and PSO type (high or low)
        """
        max_pso_samples = self._calc_num_samples(cnfg.EVENT_MAPPING[cnfg.EVENT_LABELS.PSO][cnst.MAX_DURATION])
        pso_info = {}  # saccade_id -> (start_idx, end_idx, pso_type)

        # extract saccade indices to ready-to-use lists
        saccade_info_list = sorted(saccade_info.items(), key=lambda x: x[0])
        saccade_onsets = [info[1][0] for info in saccade_info_list]
        saccade_peaks = [info[1][1] for info in saccade_info_list]
        saccade_offsets = [info[1][2] for info in saccade_info_list]

        # find PSO start & end idxs after each saccade
        i = 0
        while i < len(saccade_info):
            sac_onset_idx, sac_peak_idx, sac_offset_idx, sac_offset_threshold = saccade_info[i]
            # if a window succeeding a saccade has samples above AND below the offset threshold, there is a PSO
            start_idx, end_idx = sac_offset_idx + 1, min([sac_offset_idx + self._minimum_fixation_samples + 1, len(v)])
            window = v[start_idx: end_idx]
            is_above = window > sac_offset_threshold
            is_below = window < sac_offset_threshold
            if not any(is_above) or not any(is_below):
                # no PSO in the window
                i += 1
                continue

            # find last sample above the threshold
            last_above_oft_idx = np.where(is_above)[0][-1]
            end_idx = start_idx + last_above_oft_idx
            is_high_pso = False

            # if the window contains samples with velocity above saccades' peak threshold, but below the previous
            # saccade's max velocity, it is considered "high" PSO
            is_peak_in_window = [start_idx <= p < end_idx for p in saccade_peaks]
            if any(is_peak_in_window):
                last_peak_idx = np.where(is_peak_in_window)[0][-1]
                last_offset_idx = saccade_offsets[last_peak_idx]
                if v[start_idx: last_offset_idx].max() < v[sac_onset_idx: sac_offset_idx].max():
                    # only allow high PSO if its max velocity is below the previous saccade's max velocity
                    is_high_pso = True
                    end_idx = max([end_idx, last_offset_idx])

            # move forward from end_idx to find the first local minimum
            window = v[end_idx: len(v)]
            window_minimum_idx = self.__find_local_minimum_index(window, 0, min_thresh=sac_offset_threshold, move_back=False)
            end_idx += window_minimum_idx

            # ignore PSOs with duration exceeding the max duration for PSOs
            if end_idx - start_idx > max_pso_samples:
                i += 1
                continue

            pso_info[i] = (start_idx, end_idx, is_high_pso)

            # move to the next saccade
            next_sac_idx = np.where(saccade_onsets > end_idx)[0]
            if len(next_sac_idx):
                i = next_sac_idx[0]
            else:
                break
        return pso_info

    def _classify_samples(self,
                          is_noise: np.ndarray,
                          saccade_info: Dict[int, Tuple[int, int, int, float]],
                          pso_info: Dict[int, Tuple[int, int, bool]]) -> (np.ndarray, np.ndarray):
        """
        Classifies each sample as either noise, saccade, PSO, fixation or blink. Samples that are not classified as
        noise, blink, saccade or PSO are considered fixations.

        If we allow high PSOs we override the saccade classification when a high PSO was also detected (i.e. we consider
        a saccade immediately following a previous saccade as high-PSO).
        Note the matlab implementation does the opposite, and first merges subsequent saccades that are only separated
        by a few PSO samples, and classifies the union as a saccade. We don't do this here (see in https://shorturl.at/EMOQR)

        :param is_noise: boolean array indicating noise samples
        :param saccade_info: dict of saccade id -> (onset-idx, peak-idx, offset-idx, offset-threshold-velocity)
        :param pso_info: dict of saccade -> (PSO start-idx, PSO end-idx and PSO type (high or low))

        :return: array of classified samples
        """
        candidates_copy = np.asarray(self._candidates, dtype=cnfg.EVENT_LABELS).copy()
        for val in saccade_info.values():
            onset_idx, _, offset_idx, _ = val
            candidates_copy[onset_idx: offset_idx] = cnfg.EVENT_LABELS.SACCADE
        for val in pso_info.values():
            start_idx, end_idx, is_high = val
            if is_high and not self._allow_high_psos:
                # high PSO are essentially saccades that immediately follow a previous saccade
                continue
            candidates_copy[start_idx: end_idx] = cnfg.EVENT_LABELS.PSO

        is_blinks = candidates_copy == cnfg.EVENT_LABELS.BLINK
        is_saccade = candidates_copy == cnfg.EVENT_LABELS.SACCADE
        is_pso = candidates_copy == cnfg.EVENT_LABELS.PSO
        candidates_copy[~(is_noise | is_saccade | is_pso | is_blinks)] = cnfg.EVENT_LABELS.FIXATION
        return candidates_copy

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
