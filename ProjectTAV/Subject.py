import os
from typing import Dict

import numpy as np
import pandas as pd
import scipy as sp
import mne
from pymatreader import read_mat

import ProjectTAV.tav_helpers as tavh


class Subject:
    _BASE_PATH = r'C:\Users\nirjo\Desktop\TAV'
    # _BASE_PATH = r'E:\Tav'

    _REFERENCE_CHANNEL = "Pz"
    _PLOTTING_CHANNEL = "O2"

    def __init__(self, idx: int):
        self.idx = idx

        # read eeg data
        self._eeg_no_eyemovements = self.__read_eeg_no_eyemovements(idx)  # all eye movements removed using ICA
        self._eeg_no_blinks = self.__read_eeg_no_blinks(idx)  # only blinks removed using ICA
        self._num_channels, self._num_samples = self._eeg_no_eyemovements.shape
        assert self._eeg_no_blinks.shape == self._eeg_no_eyemovements.shape

        # read channel map
        self._channels_map = self.__read_channels_map(idx)

        # read trial start & end times
        trial_starts, trial_ends = self.__read_trial_data(idx)
        ts = self.get_sample_indices()
        self._is_trial = ((ts >= trial_starts[:, None]) & (ts < trial_ends[:, None])).any(axis=0)

        # read eyetracking data
        self._saccade_onset_idxs, self._erp_onset_idxs, self._frp_saccade_onset_idxs, self._frp_fixation_onset_idxs = self.__read_eyetracking_events(idx)

        # calculate radial EOG
        self._reog_channel = self._calculate_radial_eog()

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def get_sample_indices(self) -> np.ndarray:
        return np.arange(self.num_samples)

    def get_eeg_channel(self, channel_name: str, full_ica: bool = False) -> np.ndarray:
        channel_name = channel_name.upper().strip()
        if channel_name == "REOG":
            return self._reog_channel
        if full_ica:
            return self._eeg_no_eyemovements[self._channels_map[channel_name]]
        return self._eeg_no_blinks[self._channels_map[channel_name]]


    def create_boolean_event_channel(self, event_idxs: np.ndarray, enforce_trials: bool = True) -> np.ndarray:
        """
        Creates a boolean array with length equal to the number of samples, where True values indicate the presence
        of an event at the corresponding index. If `enforce_trial` is True, only events that occur during a trial are
        marked as True.
        """
        is_event = np.zeros(self.num_samples, dtype=bool)
        is_event[event_idxs] = True
        if enforce_trials:
            is_event &= self._is_trial
        return is_event

    def calculate_reog_saccade_onset_channel(self,
                                             filter_name: str = 'srp',
                                             snr: float = 3.5,
                                             enforce_trials: bool = True) -> np.ndarray:
        """
        Detects saccade onsets in the radial EOG channel using the specified filter and signal-to-noise ratio.
        If `enforce_trials` is True, only saccades that occur during a trial are marked as True.
        Returns a boolean array with length equal to the number of samples, where True values indicate saccade-onsets,
        detected from the radial EOG channel.
        """
        assert snr > 0, "Signal-to-noise ratio must be positive"
        filtered = tavh.apply_filter(self._reog_channel, filter_name)
        min_peak_height = filtered.mean() + snr * filtered.std()
        peak_idxs, _ = sp.signal.find_peaks(filtered, height=min_peak_height)
        return self.create_boolean_event_channel(peak_idxs, enforce_trials)

    def get_eyetracking_event_channels(self, enforce_trials: bool = True) -> Dict[str, np.ndarray]:
        return {
            "ET_SACCADE_ONSET": self.create_boolean_event_channel(self._saccade_onset_idxs, enforce_trials),
            "ERP_SACCADE_ONSET": self.create_boolean_event_channel(self._erp_onset_idxs, enforce_trials),
            "FRP_SACCADE_ONSET": self.create_boolean_event_channel(self._frp_saccade_onset_idxs, enforce_trials),
            "FRP_FIXATION_ONSET": self.create_boolean_event_channel(self._frp_fixation_onset_idxs, enforce_trials)
        }

    def plot_eyetracker_saccade_detection(self):
        # extract channels
        # TODO: add reog with butter/wavelet filters as well
        reog = self._reog_channel
        reog_srp_filtered = tavh.apply_filter(reog, "srp")
        is_et_saccade_channel = self.create_boolean_event_channel(self._saccade_onset_idxs, enforce_trials=False)

        # create mne object
        raw_object_data = np.vstack([self._reog_channel[self._is_trial],
                                     reog_srp_filtered[self._is_trial],
                                     is_et_saccade_channel[self._is_trial]])
        raw_object_info = mne.create_info(ch_names=['REOG', 'REOG_srp_filtered', 'ET_SACC'],
                                          ch_types=['eeg'] * 2 + ['stim'],
                                          sfreq=tavh.SAMPLING_FREQUENCY)
        raw_object = mne.io.RawArray(data=raw_object_data,
                                     info=raw_object_info)
        events = mne.find_events(raw_object, stim_channel='ET_SACC')
        scalings = dict(eeg=5e2, stim=1e10)
        fig = raw_object.plot(n_channels=2, events=events, scalings=scalings, event_color={1: 'r'}, show=False)
        fig.suptitle(f"ET Saccade Detection", y=0.99)
        fig.show()

    def _calculate_radial_eog(self) -> np.ndarray:
        mean_eog = np.nanmean(np.vstack([self._eeg_no_blinks[self._channels_map['LHEOG']],
                                         self._eeg_no_blinks[self._channels_map['RHEOG']],
                                         self._eeg_no_blinks[self._channels_map['RVEOGS']],
                                         self._eeg_no_blinks[self._channels_map['RVEOGI']]]),
                              axis=0)
        ref_channel = self._eeg_no_blinks[self._channels_map[Subject._REFERENCE_CHANNEL.upper()]]
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
        return {v.upper(): k for k, v in channels_dict.items()}

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

