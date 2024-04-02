import os
import time

import numpy as np
import plotly.io as pio
from pymatreader import read_mat

import Config.constants as cnst

pio.renderers.default = "browser"

######################################

path = r"C:\Users\nirjo\Desktop\EEGEYENET"
DOTS, VSS = "dots_data", "processing_speed_data"
SYNCHRONISED_MIN, SYNCHRONISED_MAX = "synchronised_min", "synchronised_max"

vss_min_example = read_mat(os.path.join(path, VSS, SYNCHRONISED_MIN, "AB7", "AB7_WI2_EEG.mat"))["sEEG"]
timestamps_vss_min = vss_min_example["times"]
data_vss_min = vss_min_example["data"]
channel_locations_vss_min = vss_min_example["chanlocs"]
events_vss_min = vss_min_example["event"]
ur_events_vss_min = vss_min_example["urevent"]
sampling_rate_vss_min = vss_min_example["srate"]

vss_max_example = read_mat(os.path.join(path, VSS, SYNCHRONISED_MAX, "AB7", "AB7_WI2_EEG.mat"))["sEEG"]
timestamps_vss_max = vss_max_example["times"]
data_vss_max = vss_max_example["data"]
channel_locations_vss_max = vss_max_example["chanlocs"]
events_vss_max = vss_max_example["event"]
ur_events_vss_max = vss_max_example["urevent"]
sampling_rate_vss_max = vss_max_example["srate"]

## DIFFERENCE BETWEEN MIN AND MAX:
# Min - cleaned from artifacts
# Max - cleaned from artifacts + ICA to remove eye movements
# the only actual difference is in channels 0-128 of the `data` arrays
