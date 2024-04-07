import numpy as np
import pandas as pd
import scipy.signal as signal
import pywt


SAMPLING_FREQUENCY = 1024  # eeg sampling frequency
SRP_FILTER = np.array([
    0.000e+00, -0.000e+00, -1.000e-04, -2.000e-04, -2.000e-04, -1.000e-04, 1.000e-04, 3.000e-04, 7.000e-04, 1.500e-03,
    2.800e-03, 5.000e-03, 8.000e-03, 1.140e-02, 1.510e-02, 1.880e-02, 2.170e-02, 2.410e-02, 2.670e-02, 2.720e-02,
    2.710e-02, 2.870e-02, 3.290e-02, 3.910e-02, 4.620e-02, 5.440e-02, 6.050e-02, 6.020e-02, 4.470e-02, 3.000e-03,
    -6.720e-02, -1.615e-01, -2.631e-01, -3.490e-01, -3.965e-01, -3.834e-01, -3.045e-01, -1.706e-01, -1.090e-02,
    1.349e-01, 2.355e-01, 2.789e-01, 2.707e-01, 2.271e-01, 1.683e-01, 1.100e-01, 6.310e-02, 3.190e-02, 1.740e-02,
    1.420e-02, 1.930e-02, 2.740e-02, 3.120e-02, 3.030e-02, 2.570e-02, 1.830e-02, 8.800e-03, -7.000e-04, -8.600e-03,
    -1.520e-02, -1.980e-02, -2.210e-02, -2.290e-02, -2.300e-02, -2.190e-02, -1.990e-02, -1.790e-02, -1.570e-02,
    -1.290e-02, -1.010e-02, -7.000e-03, -4.200e-03, -2.000e-03, -3.000e-04, 9.000e-04, 1.300e-03, 1.300e-03, 1.100e-03,
    8.000e-04, 5.000e-04, 2.000e-04, 1.000e-04, 0.000e+00, 0.000e+00
])
SRP_FILTER_TAV = np.array([
    0, -0.0000, -0.0001, -0.0002, -0.0002, -0.0001, 0.0001, 0.0003, 0.0007, 0.0015, 0.0028, 0.0050, 0.0080, 0.0114,
    0.0151, 0.0188, 0.0217, 0.0241, 0.0267, 0.0272, 0.0271, 0.0287, 0.0329, 0.0391, 0.0462, 0.0544, 0.0605, 0.0602,
    0.0447, 0.0030, -0.0672, -0.1615, -0.2631, -0.3490, -0.3965, -0.3834, -0.3045, -0.1706, -0.0109, 0.1349, 0.2355,
    0.2789, 0.2707, 0.2271, 0.1683, 0.1100, 0.0631, 0.0319, 0.0174, 0.0142, 0.0193, 0.0274, 0.0312, 0.0303, 0.0257,
    0.0183, 0.0088, -0.0007, -0.0086, -0.0152, -0.0198, -0.0221, -0.0229, -0.0230 - 0.0219, -0.0199, -0.0179, -0.0157,
    -0.0129, -0.0101, -0.0070, -0.0042, -0.0020, -0.0003, 0.0009, 0.0013, 0.0013, 0.0011, 0.0008, 0.0005, 0.0002,
    0.0001, 0.0000, 0
])


def create_filter(name: str) -> (np.ndarray, np.ndarray):
    name = name.lower()
    if name == 'butter':
        b, a = signal.butter(6, Wn=np.array([30, 100]), fs=SAMPLING_FREQUENCY, btype='bandpass')
        return b, a
    if name == 'wavelet':
        wavelet = pywt.ContinuousWavelet("gaus1", dtype=float)
        phi, psi, _x = wavelet.wavefun(level=3)
        return phi, psi
    if name == 'srp':
        return SRP_FILTER, np.ones_like(SRP_FILTER)
    if name == 'srp_tav':
        return SRP_FILTER_TAV, np.ones_like(SRP_FILTER_TAV)
    raise ValueError(f"Filter {name} not recognized")


def apply_filter(data: np.ndarray, filter_name: str) -> np.ndarray:
    filter_name = filter_name.lower()
    if filter_name == 'butter':
        # see https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
        b, a = create_filter(filter_name)
        return signal.lfilter(b, a, data)
    if filter_name == 'wavelet':
        # phi, psi = create_filter(filter_name)
        # return signal.convolve(data, psi, mode='same')
        # see https://scicoding.com/introduction-to-wavelet-transform-using-python/
        raise NotImplementedError
    if filter_name == 'srp_tav':
        # copying implementation from Tav's code (see `filterSRP` function in Tav's code)
        srp_filter, _ = create_filter(filter_name)
        n, SPOnset = len(srp_filter), 28
        reog_convolved = np.convolve(data, srp_filter[::-1])
        reog_convolved = reog_convolved[n - SPOnset: 2 - SPOnset]
        reog_convolved = reog_convolved[:-1]  # todo: remove last element while indexing in the above line
        return reog_convolved
    raise ValueError(f"Filter {filter_name} not recognized")