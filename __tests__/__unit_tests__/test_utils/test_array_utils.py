import unittest

import numpy as np

import Utils.array_utils as arr_utils


class TestIOUtils(unittest.TestCase):

    def test_is_one_dimensional(self):
        self.assertTrue(arr_utils.is_one_dimensional([1, 2, 3]))
        self.assertTrue(arr_utils.is_one_dimensional([[1], [2], [3]]))
        self.assertTrue(arr_utils.is_one_dimensional([[1, 2, 3]]))
        self.assertFalse(arr_utils.is_one_dimensional([[1, 2], [3, 4]]))
        self.assertRaises(ValueError, arr_utils.is_one_dimensional, [[1, 2], [3]])

    def test_get_chunk_indices(self):
        arr = [1, 1, 1, 2, 2, 3, 3, 3, 3]
        obs = arr_utils.get_chunk_indices(arr)
        exp = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5, 6, 7, 8])]
        self.assertTrue(all([np.array_equal(o, e, equal_nan=True) for o, e in zip(obs, exp)]))

        arr[2] = np.nan
        obs = arr_utils.get_chunk_indices(arr)
        exp = [np.array([0, 1]), np.array([2]), np.array([3, 4]), np.array([5, 6, 7, 8])]
        self.assertTrue(all([np.array_equal(o, e, equal_nan=True) for o, e in zip(obs, exp)]))

        arr = np.arange(-5, 5)
        obs = arr_utils.get_chunk_indices(arr)
        exp = [np.array([i]) for i in range(10)]
        self.assertTrue(all([np.array_equal(o, e, equal_nan=True) for o, e in zip(obs, exp)]))
