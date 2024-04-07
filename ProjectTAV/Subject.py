import os
from typing import Dict

import numpy as np
import pandas as pd
import scipy as sp
from pymatreader import read_mat

import ProjectTAV.tav_helpers as tavh


class Subject:
    # _BASE_PATH = r'C:\Users\nirjo\Desktop\TAV'
    _BASE_PATH = r'E:\Tav'
    _REFERENCE_CHANNEL = "Pz"
    _PLOTTING_CHANNEL = "O2"

    def __init__(self, idx: int):
        self.idx = idx

        # read eeg data
        self._eeg_no_eyemovements = self.__read_eeg_no_eyemovements(idx)
        self._eeg_no_blinks = self.__read_eeg_no_blinks(idx)
        self._num_channels, self._num_samples = self._eeg_no_eyemovements.shape
        assert self._eeg_no_blinks.shape == self._eeg_no_eyemovements.shape

        # read channel map
        self._channels_map = self.__read_channels_map(idx)

        # read trial start & end times
        trial_starts, trial_ends = self.__read_trial_data(idx)
        ts = np.arange(self._num_samples)
        self._is_trial = (ts >= trial_starts[:, None]) & (ts < trial_ends[:, None]).any(axis=0)

        # read eyetracking data
        events = self.__read_eyetracking_events(idx)
        self._saccade_onsets, self._erp_onsets, self._frp_saccade_onsets, self._frp_fixation_onsets = events

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def find_reog_saccades(self, filter_name: str = 'srp', snr: float = 3.5) -> np.ndarray:
        assert snr > 0, "Signal-to-noise ratio must be positive"
        reog = self.calculate_radial_eog()
        filtered = tavh.apply_filter(reog, filter_name)
        min_peak_height = filtered.mean() + snr * filtered.std()
        peak_idxs, _ = sp.signal.find_peaks(filtered, height=min_peak_height)
        return peak_idxs

    def calculate_radial_eog(self) -> np.ndarray:
        mean_eog = np.nanmean(np.vstack([self._eeg_no_blinks[self._channels_map['LHEOG']],
                                         self._eeg_no_blinks[self._channels_map['RHEOG']],
                                         self._eeg_no_blinks[self._channels_map['RVEOGS']],
                                         self._eeg_no_blinks[self._channels_map['RVEOGI']]]),
                              axis=0)
        ref_channel = self._eeg_no_blinks[self._channels_map[Subject._REFERENCE_CHANNEL]]
        return mean_eog - ref_channel

    @staticmethod
    def __read_eeg_no_eyemovements(idx: int) -> np.ndarray:
        fname = os.path.join(Subject._BASE_PATH, "data", f"S{idx}_data_interp.mat")
        eeg_no_eyemovements = read_mat(fname)['data']
        eeg_no_eyemovements = np.swapaxes(eeg_no_eyemovements, 0, 1)
        return eeg_no_eyemovements

    @staticmethod
    def __read_eeg_no_blinks(idx: int) -> np.ndarray:
        fname = os.path.join(Subject._BASE_PATH, "data", f"S{idx}_data_ica_onlyBlinks.mat")
        eeg_no_blinks = read_mat(fname)['dat']
        eeg_no_blinks = np.swapaxes(eeg_no_blinks, 0, 1)
        return eeg_no_blinks

    @staticmethod
    def __read_channels_map(idx: int) -> Dict[str, int]:
        fname = os.path.join(Subject._BASE_PATH, "data", f"{idx}_channels.csv")
        df = pd.read_csv(fname, header=0, index_col=0)
        channels_series = df[df.columns[-1]]
        channels_dict = channels_series.to_dict()
        return {v: k for k, v in channels_dict.items()}

    @staticmethod
    def __read_trial_data(idx: int) -> (np.ndarray, np.ndarray):
        START_CODE, END_CODE = 11, 12
        fname = os.path.join(Subject._BASE_PATH, "data", f"{idx}_info.csv")
        df = pd.read_csv(fname, header=0)
        trial_start_times = df[df["Codes"] == START_CODE]['latency'].to_numpy()
        trial_end_times = df[df["Codes"] == END_CODE]['latency'].to_numpy()
        return trial_start_times, trial_end_times

    @staticmethod
    def __read_eyetracking_events(idx: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        ET_CODE = 1
        fname = os.path.join(Subject._BASE_PATH, "data", f"{idx}_info.csv")
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

