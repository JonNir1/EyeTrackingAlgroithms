import warnings as w
import numpy as np
import pandas as pd
from typing import List, Tuple

from Config import constants as cnst


def is_one_dimensional(arr) -> bool:
    """ Returns true if the array's shape is (n,) or (1, n) or (n, 1) """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return True
    if arr.ndim == 2 and min(arr.shape) == 1:
        return True
    return False


def numeric_derivative(y, x, deg: int = 1, mul_const: float = 1) -> np.ndarray:
    """
    Calculates the `deg`-th derivative of the given numeric function `y` with respect to `x`.
    Argument `mul_const` is a constant multiplier for the derivative.
    """
    if not is_one_dimensional(y) or not is_one_dimensional(x):
        raise ValueError("`y` and `x` must be one-dimensional")
    if len(y) != len(x):
        raise ValueError("`y` and `x` must be of the same length")
    if deg < 0:
        raise ValueError("`deg` must be non-negative")
    if deg == 0:
        return y
    dy = np.concatenate([np.array([0]), np.diff(y)])        # first element is 0
    dx = np.concatenate([np.array([np.nan]), np.diff(x)])   # first element is NaN
    dy_dx = np.divide(dy, dx) * mul_const
    return numeric_derivative(dy_dx, x, deg=deg-1, mul_const=mul_const)


def extract_column_safe(data: pd.DataFrame, colname: str, warn: bool = True) -> np.ndarray:
    try:
        return data[colname].values
    except KeyError:
        if warn:
            w.warn(f"Column {colname} not found in the given DataFrame")
        return np.full(shape=data.shape[0], fill_value=np.nan)


def get_chunk_indices(arr) -> List[np.ndarray]:
    """
    Given a 1D array with multiple values, returns a list of arrays, where each array contains the indices of
    a different "chunk", i.e. a sequence of the same value.
    """
    if not is_one_dimensional(arr):
        raise ValueError("arr must be one-dimensional")
    indices = np.arange(len(arr))
    split_on = np.nonzero(np.diff(arr))[0] + 1  # +1 because we want to include the last index of each chunk
    chunk_indices = np.split(indices, split_on)
    return chunk_indices


def merge_close_chunks(arr: np.ndarray, min_size: int, default_value) -> np.ndarray:
    """
    1. Splits the array into chunks of identical values
    2. If two chunks of the same value are separated by a chunk of a different value, and the middle chunk is shorter
         than `min_size`, it is set to the same value as the other two chunks
    3. Sets chunks that are shorter than the minimum size to default value
    :return: array of merged chunks
    """
    if not is_one_dimensional(arr):
        raise ValueError("arr must be one-dimensional")
    if min_size < 0:
        raise ValueError("min_size must be at non-negative")

    arr_copy = arr.copy()
    # merge consecutive chunks of the same value
    chunk_indices = get_chunk_indices(arr_copy)
    for i, middle_chunk_idxs in enumerate(chunk_indices):
        if i == 0 or i == len(chunk_indices) - 1:
            # don't attempt to merge the first or last chunk
            continue
        if len(middle_chunk_idxs) >= min_size:
            # skip chunks that are long enough
            continue
        prev_chunk_value = arr_copy[chunk_indices[i - 1][-1]]  # value of the previous chunk
        next_chunk_value = arr_copy[chunk_indices[i + 1][0]]  # value of the next chunk
        if prev_chunk_value == next_chunk_value:
            # set the middle chunk to the same value as the previous and next chunks (essentially merging them)
            arr_copy[middle_chunk_idxs] = prev_chunk_value

    # set short chunks to default value
    chunk_indices = get_chunk_indices(arr_copy)     # re-calculating the chunks after the previous step
    for chunk_idxs in chunk_indices:
        if len(chunk_idxs) < min_size:
            arr_copy[chunk_idxs] = default_value
    return arr_copy


def find_sequences_in_sparse_array(sparse_array: np.ndarray, sequence: np.ndarray) -> List[Tuple[int, int]]:
    """
    Finds all occurrences of the given sequence in the given sparse array, while ignoring intermediate NaN values.
    :param sparse_array: array to search in, may contain NaN values
    :param sequence: sequence to search for
    :return: list of (start_idx, end_idx) tuples for each occurrence of the sequence in the array

    see examples in https://stackoverflow.com/a/76812495/8543025
    """
    from numpy.lib.stride_tricks import sliding_window_view as swv
    if not is_one_dimensional(sparse_array):
        raise ValueError("arr must be one-dimensional")
    n = len(sequence)
    non_nan_idxs = np.where(~np.isnan(sparse_array))[0]
    if len(non_nan_idxs) < n:
        return []
    swv_non_nan_array = swv(sparse_array[non_nan_idxs], n)
    is_sequence = np.all(swv_non_nan_array == sequence, axis=1)
    start_end_idxs = list(zip(non_nan_idxs[:1-n][is_sequence], non_nan_idxs[n-1:][is_sequence]))
    return start_end_idxs
