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


pio.renderers.default = "browser"

#################################
# Difference between original SRP filter and TAV's SRP filter
#################################

# srp = tavh.create_filter('srp')
# srp_tav = tavh.create_filter('srp_tav')
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(y=srp[0], mode='lines', name='SRP'))
# fig.add_trace(go.Scatter(y=srp_tav[0], mode='lines', name='SRP TAV'))
# fig.show()

#################################

s101 = Subject(101)
timestamp_idxs = np.arange(s101.num_samples)
reog = s101.calculate_radial_eog()
srp_reog = tavh.apply_filter(reog, 'srp_tav')
reog_saccades = s101.find_reog_saccades(filter_name='srp_tav', snr=3.5)


# %%add_to Participant
# def plot_saccade_detection(self):
#     mask = ((self.r >= self.start_idx[:, None]) & (self.r < self.end_idx[:, None])).any(0)
#     raw_object_data = np.vstack([self.reog_channel[mask], self.SRPed_data[mask], self.eye_tracker_sacc_vec[mask]])
#     raw_object_ch = ['REOG', 'REOG_filtered', 'ET_SACC']
#     raw_object_ch_types = ['eeg'] * 2 + ['stim']
#     raw_object_info = mne.create_info(raw_object_ch, sfreq=1024, ch_types=raw_object_ch_types)
#     raw_object = mne.io.RawArray(data=raw_object_data,
#                                  info=raw_object_info)
#     events = mne.find_events(raw_object, stim_channel='ET_SACC')
#     scalings = dict(eeg=5e2, stim=1e10)
#     fig = raw_object.plot(n_channels=2, events=events, scalings=scalings, event_color={1: 'r'}, show=False)
#     fig.suptitle(f"Figure 1",  y=1.01)
#     plt.show()
