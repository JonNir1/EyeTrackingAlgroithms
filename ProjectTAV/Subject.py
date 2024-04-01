import posixpath as path
from typing import Dict

import numpy as np
import pandas as pd
from pymatreader import read_mat


class Subject:
    _BASE_PATH = r'C:\Users\nirjo\Desktop\TAV'
    _REFERENCE_CHANNEL = "Pz"
    _PLOTTING_CHANNEL = "O2"

    def __init__(self, idx: int):
        self.idx = idx
        self._channels_map = self.__read_channels_map(idx)
        self._trial_starts, self._trial_ends = self.__read_trial_data(idx)
        self._saccade_onsets, self._erp_onsets, self._frp_saccade_onsets, self._frp_fixation_onsets = self.__read_eyetracking_data(idx)
        self._eyemovement_removed_data, self._blinks_removed = self.__read_eeg_data(idx)

    def calculate_radial_eog(self) -> np.ndarray:
        mean_eog = np.nanmean(np.vstack([self._blinks_removed[self._channels_map['LHEOG']],
                                         self._blinks_removed[self._channels_map['RHEOG']],
                                         self._blinks_removed[self._channels_map['RVEOGS']],
                                         self._blinks_removed[self._channels_map['RVEOGI']]]),
                              axis=0)
        ref_channel = self._blinks_removed[self._channels_map[Subject._REFERENCE_CHANNEL]]
        return mean_eog - ref_channel

    @staticmethod
    def __read_channels_map(idx: int) -> Dict[str, int]:
        fname = path.join(Subject._BASE_PATH, "data", f"{idx}_channels.csv")
        df = pd.read_csv(fname, header=0, index_col=0)
        channels_series = df[df.columns[-1]]
        channels_dict = channels_series.to_dict()
        return {v: k for k, v in channels_dict.items()}

    @staticmethod
    def __read_trial_data(idx: int) -> (np.ndarray, np.ndarray):
        START_CODE, END_CODE = 11, 12
        fname = path.join(Subject._BASE_PATH, "data", f"{idx}_info.csv")
        df = pd.read_csv(fname, header=0)
        trial_start_times = df[df["Codes"] == START_CODE]['latency'].to_numpy()
        trial_end_times = df[df["Codes"] == END_CODE]['latency'].to_numpy()
        return trial_start_times, trial_end_times

    @staticmethod
    def __read_eyetracking_data(idx: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        ET_CODE = 1
        fname = path.join(Subject._BASE_PATH, "data", f"{idx}_info.csv")
        df = pd.read_csv(fname, header=0)
        saccade_onset_times = df[df["Codes"] == ET_CODE]['SacOnset'].to_numpy().astype('int64')
        erp_onset_times = df[(df["NewCodes"] // 10000 % 10 == 2) & (df["NewCodes"] // 1000 % 10 == 0)][
            'latency'].to_numpy().astype('int64')
        frp_saccade_onset_times = df[
            (df["NewCodes"] // 10000 % 10 == 1) & (df["NewCodes"] // 10 % 10 == 1) & (
                        df["NewCodes"] // 1000 % 10 == 0)][
            'SacOnset'].to_numpy().astype('int64')
        frp_fixation_onset_times = df[
            (df["NewCodes"] // 10000 % 10 == 1) & (df["NewCodes"] // 10 % 10 == 1) &
            (df["NewCodes"] // 1000 % 10 == 0)]['latency'].to_numpy().astype('int64')
        return saccade_onset_times, erp_onset_times, frp_saccade_onset_times, frp_fixation_onset_times

    @staticmethod
    def __read_eeg_data(idx: int) -> (np.ndarray, np.ndarray):
        fname_interp = path.join(Subject._BASE_PATH, "data", f"S{idx}_data_interp.mat")
        eyemovement_removed_data = read_mat(fname_interp)['data']
        eyemovement_removed_data = np.swapaxes(eyemovement_removed_data, 0, 1)

        fname_blinks = path.join(Subject._BASE_PATH, "data", f"S{idx}_data_ica_onlyBlinks.mat")
        blinks_removed_data = read_mat(fname_blinks)['dat']
        blinks_removed_data = np.swapaxes(blinks_removed_data, 0, 1)
        return eyemovement_removed_data, blinks_removed_data

