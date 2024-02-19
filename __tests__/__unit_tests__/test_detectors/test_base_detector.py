import unittest
import numpy as np

import constants as cnst
from GazeDetectors.EngbertDetector import EngbertDetector
from Config.GazeEventTypeEnum import GazeEventTypeEnum


class TestBaseDetector(unittest.TestCase):
    _SR = 500
    DETECTOR = EngbertDetector(lambda_noise_threshold=5, derivation_window_size=2)

    def test__verify_inputs(self):
        t = x = y = np.arange(10)
        self.assertTrue(np.array_equal(self.DETECTOR._verify_inputs(t, x, y), (t, x, y)))

        y = np.arange(9)
        with self.assertRaises(ValueError):
            self.DETECTOR._verify_inputs(t, x, y)

    def test__calculate_sampling_rate(self):
        t = np.arange(10)
        self.assertEqual(self.DETECTOR._calculate_sampling_rate(t), cnst.MILLISECONDS_PER_SECOND)
        t = np.arange(0, 11, 2)
        self.assertEqual(self.DETECTOR._calculate_sampling_rate(t), cnst.MILLISECONDS_PER_SECOND / 2)
        t = np.arange(10) * cnst.MILLISECONDS_PER_SECOND
        self.assertEqual(self.DETECTOR._calculate_sampling_rate(t), 1)
        t = np.setdiff1d(np.arange(10), [2, 5, 8])
        self.assertEqual(self.DETECTOR._calculate_sampling_rate(t), cnst.MILLISECONDS_PER_SECOND / 1.5)

    def test__merge_consecutive_chunks(self):
        arr = np.array([])
        self.assertTrue(np.array_equal(self.DETECTOR._merge_consecutive_chunks(arr, self._SR), arr))

        arr = np.array([GazeEventTypeEnum.SACCADE] * 1)
        expected = np.array([GazeEventTypeEnum.UNDEFINED] * 1)
        res = self.DETECTOR._merge_consecutive_chunks(arr, 500)
        self.assertTrue(np.array_equal(res, expected))

        arr = np.array([GazeEventTypeEnum.FIXATION] * 3 +
                       [GazeEventTypeEnum.SACCADE] * 1 +
                       [GazeEventTypeEnum.FIXATION] * 2)
        expected = np.array([GazeEventTypeEnum.FIXATION] * 6)
        res = self.DETECTOR._merge_consecutive_chunks(arr, 500)
        self.assertTrue(np.array_equal(res, expected))

        arr = np.array([GazeEventTypeEnum.FIXATION] * 3 +
                       [GazeEventTypeEnum.SACCADE] * 1 +
                       [GazeEventTypeEnum.BLINK] * 2)
        expected = np.array([GazeEventTypeEnum.FIXATION] * 3 +
                       [GazeEventTypeEnum.UNDEFINED] * 1 +
                       [GazeEventTypeEnum.BLINK] * 2)
        res = self.DETECTOR._merge_consecutive_chunks(arr, 500)
        self.assertTrue(np.array_equal(res, expected))
