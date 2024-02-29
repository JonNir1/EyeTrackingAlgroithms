import unittest
import numpy as np

from Config import constants as cnst
from GazeDetectors.EngbertDetector import EngbertDetector
from Config.GazeEventTypeEnum import GazeEventTypeEnum


class TestBaseDetector(unittest.TestCase):
    DETECTOR = EngbertDetector(lambda_noise_threshold=5, derivation_window_size=2)
    DETECTOR._sr = 500

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

    def test__merge_close_events(self):
        self.DETECTOR._candidates = np.array([])
        self.assertTrue(np.array_equal(self.DETECTOR._merge_close_events(), np.array([])))

        self.DETECTOR._candidates = np.array([GazeEventTypeEnum.SACCADE] * 1)
        expected = np.array([GazeEventTypeEnum.UNDEFINED] * 1)
        res = self.DETECTOR._merge_close_events()
        self.assertTrue(np.array_equal(res, expected))

        self.DETECTOR._candidates = np.array([GazeEventTypeEnum.FIXATION] * 6 +
                                             [GazeEventTypeEnum.SACCADE] * 2 +
                                             [GazeEventTypeEnum.BLINK] * 4)
        expected = np.array([GazeEventTypeEnum.FIXATION] * 6 + [GazeEventTypeEnum.UNDEFINED] * 6)
        res = self.DETECTOR._merge_close_events()
        self.assertTrue(np.array_equal(res, expected))

        self.DETECTOR._candidates = np.array([GazeEventTypeEnum.FIXATION] * 6 +
                                             [GazeEventTypeEnum.SACCADE] * 2 +
                                             [GazeEventTypeEnum.FIXATION] * 4)
        expected = np.array([GazeEventTypeEnum.FIXATION] * 12)
        res = self.DETECTOR._merge_close_events()
        self.assertTrue(np.array_equal(res, expected))

        self.DETECTOR._candidates = np.array([GazeEventTypeEnum.FIXATION] * 6 +
                                             [GazeEventTypeEnum.SACCADE] * 2 +
                                             [GazeEventTypeEnum.BLINK] * 6)
        expected = np.array([GazeEventTypeEnum.FIXATION] * 6 +
                            [GazeEventTypeEnum.UNDEFINED] * 2 +
                            [GazeEventTypeEnum.BLINK] * 6)
        res = self.DETECTOR._merge_close_events()
        self.assertTrue(np.array_equal(res, expected))
