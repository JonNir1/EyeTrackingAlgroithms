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

srp = tavh.create_filter('srp')
srp_tav = tavh.create_filter('srp_tav')

fig = go.Figure()
fig.add_trace(go.Scatter(y=srp[0], mode='lines', name='SRP'))
fig.add_trace(go.Scatter(y=srp_tav[0], mode='lines', name='SRP TAV'))
fig.show()

#################################

s101 = Subject(101)
timestamp_idxs = np.arange(s101.num_samples)
reog = s101.calculate_radial_eog()
