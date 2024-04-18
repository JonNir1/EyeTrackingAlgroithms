import os
import numpy as np
import pandas as pd
from pymatreader import read_mat
import scipy.signal as signal
import pywt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import ProjectTAV.tav_helpers as tavh
from ProjectTAV.Subject import Subject
from ProjectTAV.tav_participant import Participant


pio.renderers.default = "browser"

#################################

s101 = Subject(101)
p101 = Participant(101)

s101.plot_eyetracker_saccade_detection()
p101.plot_saccade_detection()

#################################

reog = s101.get_eeg_channel('reog')
reog_srp_filtered = tavh.apply_filter(s101.get_eeg_channel('reog'), 'srp')
is_reog_saccade_onset = s101.calculate_reog_saccade_onset_channel(filter_name='srp', snr=3.5, enforce_trials=True)
is_reog_saccade_onset_idxs = np.where(is_reog_saccade_onset)[0]

#################################


def hit_rate(gt_idxs, pred_idxs, half_window: int = 8):
    # todo: move to helpers
    pred_window_idxs = np.array([np.arange(i - half_window, i + half_window + 1) for i in pred_idxs])
    hit_count = np.sum(np.isin(gt_idxs, pred_window_idxs))
    return hit_count / len(gt_idxs)


# Tav's code:
def count_hits(self: Participant):
    as_strided = np.lib.stride_tricks.as_strided
    r = 9
    above_threshold_ext = np.concatenate((np.full(r, np.nan), self.above_threshold, np.full(r, np.nan)))
    windows = as_strided(above_threshold_ext,
                         (above_threshold_ext.shape[0], 2 * r + 1),
                         above_threshold_ext.strides * 2)
    windows = windows[self.eye_tracker_sacc_idx]
    detected = np.count_nonzero(np.sum(windows, axis=1))
    sacc_count = np.sum(self.eye_tracker_sacc_vec)  # len(s101._saccade_onset_idxs)
    ## p101.above_threshold is s101.is_reog_saccade_onset
    ## p101.eye_tracker_sacc_idx is s101._saccade_onset_idxs
    return detected / sacc_count

########

s101_hr = hit_rate(s101._saccade_onset_idxs, is_reog_saccade_onset_idxs, half_window=9)  # use 9 instead of 8 to match Tav's code
p101_hr = count_hits(p101)

# s101_hr == p101_hr  # True

#################################
# TODO: re-implement Tav's `false_alarm_count` so that we divide only by trial-windows and not all windows


def false_alarm_rate(self: Subject, gt_idxs, pred_idxs, half_window: int = 8, enforce_trials: bool = True):
    window_size = 2 * half_window + 1
    gt_channel = self.create_boolean_event_channel(gt_idxs, enforce_trials=False)
    gt_channel_windows = np.split(gt_channel, np.arange(window_size, gt_channel.size, window_size))
    gt_channel_windows_no_event = np.array(list(map(lambda x: not any(x), gt_channel_windows)))
    if enforce_trials:
        is_trial_window = np.array(list(map(any, np.split(
            self._is_trial, np.arange(window_size, self._is_trial.size, window_size)))))
        gt_channel_windows_no_event = gt_channel_windows_no_event[is_trial_window]

    return None


# Tav's code:
def count_false_alarms(self: Participant):
    # above_threshold_trials = self.filter_trials(self.above_threshold)
    # eye_tracker_sacc_trials = self.filter_trials(self.eye_tracker_sacc_vec)

    section_len = 19
    N = self.above_threshold.shape[0] // section_len * section_len


    sacc_windows = np.reshape(self.eye_tracker_sacc_vec[:N], (-1, section_len))
    above_threshold_windows = np.reshape(self.above_threshold[:N], (-1, section_len))

    is_sacc = sacc_windows.sum(axis=1)
    count_no_sacc = np.count_nonzero(is_sacc == 0)

    is_sp = above_threshold_windows.sum(axis=1)
    is_sp = np.where(is_sacc == 0, is_sp, 0)
    count_sp_no_sacc = np.count_nonzero(is_sp)

    return count_sp_no_sacc / count_no_sacc

#################################
