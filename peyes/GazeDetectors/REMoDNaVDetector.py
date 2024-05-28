import numpy as np
import remodnav
from typing import List, Dict

import peyes.Config.constants as cnst
import peyes.Config.experiment_config as cnfg
from peyes.GazeDetectors.BaseDetector import BaseDetector


class REMoDNaVDetector(BaseDetector):
    """
    This is a wrapper class that uses an implementation of the REMoDNaV algorithm to detect gaze events in the gaze
        data. This algorithm is based on the NHDetector algorithm, but extends and improves it by adding a more
        sophisticated saccade/pso detection algorithm, and by adding a smooth pursuit detection algorithm.

    See the REMoDNaV paper:
        Dar AH, Wagner AS, Hanke M. REMoDNaV: robust eye-movement classification for dynamic stimulation. Behav Res
        Methods. 2021 Feb;53(1):399-414. doi: 10.3758/s13428-020-01428-x
    See the NH Detector paper:
        Nyström, M., Holmqvist, K. An adaptive algorithm for fixation, saccade, and glissade detection in eyetracking
        data. Behavior Research Methods 42, 188–204 (2010).

    See the REMoDNaV algorithm documentation & implementation:
        https://github.com/psychoinformatics-de/remodnav/tree/master
    """

    __DEFAULT_SACCADE_INITIAL_VELOCITY_THRESHOLD = 300          # deg/s
    __DEFAULT_SACCADE_CONTEXT_WINDOW_DURATION = 1               # s
    __DEFAULT_SACCADE_INITIAL_MAX_FREQ = 2.0                    # Hz
    __DEFAULT_SACCADE_ONSET_THRESHOLD_NOISE_FACTOR = 5.0        # unitless
    __DEFAULT_SMOOTH_PURSUIT_DRIFT_VELOCITY_THRESHOLD = 2.0     # deg/s
    __DEFAULT_SMOOTH_PURSUIT_LOWPASS_CUTOFF_FREQ = 4.0          # Hz
    __DEFAULT_MEDIAN_FILTER_LENGTH = 0.05                       # s
    __DEFAULT_SAVGOL_LENGTH = 0.019                             # s
    __DEFAULT_SAVGOL_POLYORD = 2                                # unitless
    __DEFAULT_MAX_VELOCITY = 1000                               # deg/s

    def __init__(
            self,
            missing_value=cnfg.DEFAULT_MISSING_VALUE,
            viewer_distance=cnfg.DEFAULT_VIEWER_DISTANCE,
            pixel_size=cnfg.SCREEN_MONITOR.pixel_size,
            dilate_nans_by=cnfg.DEFAULT_NAN_PADDING,
            saccade_initial_velocity_threshold=__DEFAULT_SACCADE_INITIAL_VELOCITY_THRESHOLD,
            saccade_context_window_duration=__DEFAULT_SACCADE_CONTEXT_WINDOW_DURATION,
            saccade_initial_max_freq=__DEFAULT_SACCADE_INITIAL_MAX_FREQ,
            saccade_onset_threshold_noise_factor=__DEFAULT_SACCADE_ONSET_THRESHOLD_NOISE_FACTOR,
            smooth_pursuit_drift_velocity_threshold=__DEFAULT_SMOOTH_PURSUIT_DRIFT_VELOCITY_THRESHOLD,
            smooth_pursuit_lowpass_cutoff_freq=__DEFAULT_SMOOTH_PURSUIT_LOWPASS_CUTOFF_FREQ,
            median_filter_length=__DEFAULT_MEDIAN_FILTER_LENGTH,
            savgol_length=__DEFAULT_SAVGOL_LENGTH,
            savgol_polyord=__DEFAULT_SAVGOL_POLYORD,
            max_velocity=__DEFAULT_MAX_VELOCITY,
    ):
        super().__init__(missing_value, viewer_distance, pixel_size, dilate_nans_by)
        self._saccade_initial_velocity_threshold = saccade_initial_velocity_threshold
        if self._saccade_initial_velocity_threshold <= 0:
            raise ValueError("saccade_initial_velocity_threshold must be positive")
        self._saccade_context_window_duration = saccade_context_window_duration
        if self._saccade_context_window_duration <= 0:
            raise ValueError("saccade_context_window_duration must be positive")
        self._saccade_initial_max_freq = saccade_initial_max_freq
        if self._saccade_initial_max_freq <= 0:
            raise ValueError("saccade_initial_max_freq must be positive")
        self._saccade_onset_threshold_noise_factor = saccade_onset_threshold_noise_factor
        if self._saccade_onset_threshold_noise_factor <= 0:
            raise ValueError("saccade_onset_threshold_noise_factor must be positive")
        self._smooth_pursuit_drift_velocity_threshold = smooth_pursuit_drift_velocity_threshold
        if self._smooth_pursuit_drift_velocity_threshold <= 0:
            raise ValueError("smooth_pursuit_velocity_threshold must be positive")
        self._smooth_pursuit_lowpass_cutoff_freq = smooth_pursuit_lowpass_cutoff_freq
        if self._smooth_pursuit_lowpass_cutoff_freq <= 0:
            raise ValueError("smooth_pursuit_lowpass_cutoff_freq must be positive")
        self._median_filter_length = median_filter_length
        if self._median_filter_length <= 0:
            raise ValueError("median_filter_length must be positive")
        self._savgol_length = savgol_length
        if self._savgol_length <= 0:
            raise ValueError("savgol_length must be positive")
        self._savgol_polyord = savgol_polyord
        if self._savgol_polyord <= 0:
            raise ValueError("savgol_polyord must be positive")
        self._max_velocity = max_velocity
        if self._max_velocity <= 0:
            raise ValueError("max_velocity must be positive")

    def _detect_impl(self, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        classifier = remodnav.EyegazeClassifier(
            px2deg=self._pixel_size,
            sampling_rate=self._sr,
            min_saccade_duration=cnfg.EVENT_MAPPING[cnfg.EVENT_LABELS.SACCADE][cnst.MIN_DURATION] / cnst.MILLISECONDS_PER_SECOND,
            min_pursuit_duration=cnfg.EVENT_MAPPING[cnfg.EVENT_LABELS.SMOOTH_PURSUIT][cnst.MIN_DURATION] / cnst.MILLISECONDS_PER_SECOND,
            min_fixation_duration=cnfg.EVENT_MAPPING[cnfg.EVENT_LABELS.FIXATION][cnst.MIN_DURATION] / cnst.MILLISECONDS_PER_SECOND,
            max_pso_duration=cnfg.EVENT_MAPPING[cnfg.EVENT_LABELS.PSO][cnst.MAX_DURATION] / cnst.MILLISECONDS_PER_SECOND,
            min_intersaccade_duration=1/self._sr,  # allow saccades to be detected immediately after a previous saccade
            velthresh_startvelocity=self._saccade_initial_velocity_threshold,
            saccade_context_window_length=self._saccade_context_window_duration,
            max_initial_saccade_freq=self._saccade_initial_max_freq,
            noise_factor=self._saccade_onset_threshold_noise_factor,
            pursuit_velthresh=self._smooth_pursuit_drift_velocity_threshold,
            lowpass_cutoff_freq=self._smooth_pursuit_lowpass_cutoff_freq,
        )

        xy = np.rec.fromarrays([x, y], names="{},{}".format(cnst.X, cnst.Y), formats="<f8,<f8")
        pp = classifier.preproc(xy,
                                min_blink_duration=cnfg.EVENT_MAPPING[cnfg.EVENT_LABELS.BLINK][cnst.MIN_DURATION] / cnst.MILLISECONDS_PER_SECOND,
                                dilate_nan=cnfg.DEFAULT_NAN_PADDING,
                                median_filter_length=self._median_filter_length,
                                savgol_length=self._savgol_length,
                                savgol_polyord=self._savgol_polyord,
                                max_vel=self._max_velocity)

        # save preprocessed gaze data
        df = self.data[cnst.GAZE]
        df["preproc_x"] = pp[cnst.X]
        df["preproc_y"] = pp[cnst.Y]
        df[cnst.VELOCITY] = pp["vel"]
        df[cnst.ACCELERATION] = pp["accel"]
        self.data[cnst.GAZE] = df

        # detect events and parse results into candidates array
        # TODO: classifier overwrites pre-detected blinks. find a way to merge the two
        detected_events = classifier(pp, classify_isp=True, sort_events=True) # list of dicts, each contains a single event's information
        self.data["detected_events"] = detected_events
        self._candidates = self._parse_events(detected_events)
        return self._candidates

    def _parse_events(self, events: List[Dict[str, float]]) -> np.ndarray:
        events_map = {'FIXA': cnfg.EVENT_LABELS.FIXATION,
                      'SACC': cnfg.EVENT_LABELS.SACCADE,
                      'ISAC': cnfg.EVENT_LABELS.SACCADE,
                      'HPSO': cnfg.EVENT_LABELS.PSO,
                      'IHPS': cnfg.EVENT_LABELS.PSO,
                      'LPSO': cnfg.EVENT_LABELS.PSO,
                      'ILPS': cnfg.EVENT_LABELS.PSO,
                      'PURS': cnfg.EVENT_LABELS.SMOOTH_PURSUIT,
                      'BLNK': cnfg.EVENT_LABELS.BLINK}
        candidates = np.full_like(self._candidates, cnfg.EVENT_LABELS.UNDEFINED)
        for i, event in enumerate(events):
            start_sample = round(event["start_time"] * self._sr)
            end_sample = round(event["end_time"] * self._sr)
            label = events_map[event["label"]]
            candidates[start_sample:end_sample+1] = label
        return candidates
