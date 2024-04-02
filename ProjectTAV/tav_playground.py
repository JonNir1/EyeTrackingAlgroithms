import os
import numpy as np
import pandas as pd
from pymatreader import read_mat
import scipy.signal as signal
import pywt
import plotly.express as px
import plotly.io as pio

import ProjectTAV.tav_helpers as tavh
from ProjectTAV.Subject import Subject


pio.renderers.default = "browser"

#################################

s101 = Subject(101)
timestamp_idxs = np.arange(s101.num_samples)
reog = s101.calculate_radial_eog()
