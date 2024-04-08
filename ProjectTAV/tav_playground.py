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

timestamp_idxs = np.arange(s101.num_samples)
reog = s101.get_eeg_channel('reog')
srp_reog = tavh.apply_filter(s101.get_eeg_channel('reog'), 'srp')
is_reog_saccade = s101.calculate_reog_saccade_onset_channel(filter_name='srp', snr=3.5, enforce_trial=True)

print((is_reog_saccade == p101.above_threshold).all())
