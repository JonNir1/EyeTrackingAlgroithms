import os
import warnings

import numpy as np
import pandas as pd
from pymatreader import read_mat
import scipy.signal as signal
import pywt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import ProjectTAV.tav_helpers as tavh
from ProjectTAV.Subject import Subject
from ProjectTAV.TavParticipant import TavParticipant


pio.renderers.default = "browser"

#################################
# load the data

# This is too costly with memory!
# with warnings.catch_warnings(action="ignore"):
#     subjects = [Subject(i) for i in range(101, 111)]
#     participants = [TavParticipant(i) for i in range(101, 111)]

#################################
# verify same results:

for i in range(101, 111):
    print(f"Subject {i}:")
    with warnings.catch_warnings(action="ignore"):
        s = Subject(i)
        p = TavParticipant(i)

    # compare REOG channels
    # s.plot_eyetracker_saccade_detection()
    # p.plot_saccade_detection()

    # calculate REOG saccades for Subject
    is_reog_saccade_onset = s.calculate_reog_saccade_onset_channel(filter_name='srp', snr=3.5, enforce_trials=True)
    is_reog_saccade_onset_idxs = np.where(is_reog_saccade_onset)[0]

    # compare hit rate
    s_hr = tavh.hit_rate(s._saccade_onset_idxs, is_reog_saccade_onset_idxs, half_window=9)  # use 9 instead of 8 to match Tav's code
    p_hr = p.count_hits()
    print(f"\tHit Rate:\t {np.isclose(s_hr, p_hr)}")

    # compare false alarm rate
    s_fr = tavh.false_alarm_rate(s._saccade_onset_idxs, is_reog_saccade_onset_idxs, s.num_samples, half_window=9)  # use 9 instead of 8 to match Tav's code
    p_fr = p.count_false_alarms()
    print(f"\tFalse Alarm Rate:\t {np.isclose(s_fr, p_fr)}")

del i, s, p, is_reog_saccade_onset, is_reog_saccade_onset_idxs, s_hr, p_hr, s_fr, p_fr

#################################
# compare different window sizes:

COL_NAMES = ['Subject HR', 'Subject FR']

statistics = []
figures = []

window_sizes = np.arange(11)
for i in range(101, 111):
    print(f"Subject {i}:")
    with warnings.catch_warnings(action="ignore"):
        s = Subject(i)

    is_reog_saccade_onset = s.calculate_reog_saccade_onset_channel(filter_name='srp', snr=3.5, enforce_trials=True)
    is_reog_saccade_onset_idxs = np.where(is_reog_saccade_onset)[0]
    stats = np.zeros((len(window_sizes), 2))
    for j, ws in enumerate(window_sizes):
        s_hr = tavh.hit_rate(s._saccade_onset_idxs, is_reog_saccade_onset_idxs, half_window=ws)
        s_fr = tavh.false_alarm_rate(s._saccade_onset_idxs, is_reog_saccade_onset_idxs, s.num_samples, half_window=ws)
        stats[j] = [s_hr, s_fr]
    stats = pd.DataFrame(stats, columns=COL_NAMES)
    statistics.append(stats)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Hit Rate", "False Alarm Rate"))
    for c in COL_NAMES:
        fig.add_trace(go.Scatter(x=window_sizes, y=stats[c], mode='lines+markers', name=c),
                      col=1, row=1 if c == 'Subject HR' else 2)
    fig.update_layout(title=f"Subject {i}",
                      xaxis_title="Window Size (ms)",
                      yaxis_title="Rate",
                      showlegend=True)
    figures.append(fig)
    fig.show()

del i, s, p, is_reog_saccade_onset, is_reog_saccade_onset_idxs, j, ws, s_hr, s_fr, p_hr, p_fr, stats, fig


mean_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Mean Hit Rate", "Mean False Alarm Rate"))
for c in COL_NAMES:
    mean_stats = np.mean([s[c] for s in statistics], axis=0)
    mean_fig.add_trace(go.Scatter(x=window_sizes, y=mean_stats, mode='lines+markers', name=c),
                       col=1, row=1 if c == 'Subject HR' else 2)
mean_fig.update_layout(title="Mean Statistics",
                       xaxis_title="Window Size (ms)",
                       yaxis_title="Rate",
                       showlegend=True)
mean_fig.show()

#################################

s101 = Subject(101)
p101 = TavParticipant(101)

s101.plot_eyetracker_saccade_detection()
p101.plot_saccade_detection()

reog = s101.get_eeg_channel('reog')
reog_srp_filtered = tavh.apply_filter(s101.get_eeg_channel('reog'), 'srp')
is_reog_saccade_onset = s101.calculate_reog_saccade_onset_channel(filter_name='srp', snr=3.5, enforce_trials=True)
is_reog_saccade_onset_idxs = np.where(is_reog_saccade_onset)[0]

s101_hr = tavh.hit_rate(s101._saccade_onset_idxs, is_reog_saccade_onset_idxs, half_window=9)  # use 9 instead of 8 to match Tav's code
p101_hr = p101.count_hits()
print(f"Same Hit Rate:\t {s101_hr == p101_hr}")  # True

s101_fr = tavh.false_alarm_rate(s101._saccade_onset_idxs, is_reog_saccade_onset_idxs, s101.num_samples, half_window=9)
p101_fr = p101.count_false_alarms()

print(f"Same False Alarm Rate:\t {np.isclose(s101_fr, p101_fr)}")  # True

#################################
