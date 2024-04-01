import os
import numpy as np
import pandas as pd
from pymatreader import read_mat
import pywt
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"

BASE_PATH = r'C:\Users\nirjo\Desktop\TAV'
DATA, EPOCHS = 'data', 'epochs'
START_CODE, END_CODE, ET_CODE = 11, 12, 1

fname = os.path.join(BASE_PATH, DATA, f"{101}_info.csv")
df = pd.read_csv(fname, header=0)

start_times = df[df["Codes"] == START_CODE]['latency'].to_numpy()
end_times = df[df["Codes"] == END_CODE]['latency'].to_numpy()
et_saccade_onset_times = df[df["Codes"] == ET_CODE]['SacOnset'].to_numpy().astype('int64')
et_erp_onset_times = df[(df["NewCodes"] // 10000 % 10 == 2) & (df["NewCodes"] // 1000 % 10 == 0)][
    'latency'].to_numpy().astype('int64')
et_frp_saccade_onset_times = df[
    (df["NewCodes"] // 10000 % 10 == 1) & (df["NewCodes"] // 10 % 10 == 1) & (df["NewCodes"] // 1000 % 10 == 0)][
    'SacOnset'].to_numpy().astype('int64')
et_frp_fixation_onset_times = df[
    (df["NewCodes"] // 10000 % 10 == 1) & (df["NewCodes"] // 10 % 10 == 1) &
    (df["NewCodes"] // 1000 % 10 == 0)]['latency'].to_numpy().astype('int64')


fname_blinks = os.path.join(BASE_PATH, DATA, f"S{101}_data_ica_onlyBlinks.mat")
only_blinks_data = read_mat(fname_blinks)['dat']
r = np.arange(only_blinks_data.shape[0])
only_blinks_data = np.swapaxes(only_blinks_data, 0, 1)

fname_interp = os.path.join(BASE_PATH, DATA, f"S{101}_data_interp.mat")
interp_data = read_mat(fname_interp)['data']
interp_data = np.swapaxes(interp_data, 0, 1)

eye_tracker_sacc_vec = np.zeros_like(only_blinks_data[0, :])
np.put(eye_tracker_sacc_vec, ind=et_saccade_onset_times, v=1)


# b, a = signal.butter(6, Wn=np.array([30, 100]) / (1024 / 2), btype='bandpass')
#
# wavelet = pywt.ContinuousWavelet("gaus1")
# phi, psi = wavelet.wavefun(level=3)

