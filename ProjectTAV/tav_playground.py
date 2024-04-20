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
from ProjectTAV.TavParticipant import TavParticipant


pio.renderers.default = "browser"

#################################
# Difference between original SRP filter and TAV's SRP filter
#################################

srp = tavh.create_filter('srp')
srp_tav = tavh.create_filter('srp_tav')

fig = go.Figure()
fig.add_trace(go.Scatter(y=srp[0], mode='lines', name='SRP'))
fig.add_trace(go.Scatter(y=srp_tav[0], mode='lines', name='SRP TAV'))
fig.show()

#################################

s101 = Subject(101)
p101 = TavParticipant(101)

s101.plot_eyetracker_saccade_detection()
p101.plot_saccade_detection()

#################################

reog = s101.get_eeg_channel('reog')
reog_srp_filtered = tavh.apply_filter(s101.get_eeg_channel('reog'), 'srp')
is_reog_saccade_onset = s101.calculate_reog_saccade_onset_channel(filter_name='srp', snr=3.5, enforce_trials=True)
is_reog_saccade_onset_idxs = np.where(is_reog_saccade_onset)[0]

#################################

s101_hr = tavh.hit_rate(s101._saccade_onset_idxs, is_reog_saccade_onset_idxs, half_window=9)  # use 9 instead of 8 to match Tav's code
p101_hr = p101.count_hits()

print(f"Same Hit Rate:\t {s101_hr == p101_hr}")  # True

#################################

s101_fr = tavh.false_alarm_rate(s101._saccade_onset_idxs, is_reog_saccade_onset_idxs, s101.num_samples, half_window=9)
p101_fr = p101.count_false_alarms()

print(f"Same False Alarm Rate:\t {np.isclose(s101_fr, p101_fr)}")  # True

#################################


