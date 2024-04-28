import itertools
from typing import Callable, Optional, Union, List, Tuple

import pandas as pd


def apply_on_column_pairs(data: pd.DataFrame,
                          func: Callable,
                          is_symmetric: bool = True) -> pd.DataFrame:
    """
    Applies the `func` on each pair of columns in the given `data`. If `is_symmetric` is True, only calculate the
    function once for each (unordered-)pair of columns, e.g, (A, B) and (B, A) will be the same. If False, calculate
    the function for all ordered-pairs of columns.


    Applies the `matching_func` on each pair of columns in the given `data`, where cells contain sequences of
    gaze-events detected by different raters/detectors. If `is_symmetric` is True, only calculate the measure once
    for each (unordered-)pair of columns, e.g, (A, B) and (B, A) will be the same. If False, calculate the measure
    for all ordered-pairs of columns.

    :param data: The DataFrame to calculate the function on its columns.
    :param func: The function to calculate the measure between two columns.
    :param is_symmetric: Determines whether to calculate the measure for ordered or unordered pairs of columns.
    :return: A DataFrame with the same index as the input data, columns as the pairs of columns of the input data,
        and values in the DataFrame are the result of the function applied to the corresponding pair of columns.
    """
    cols = sorted(data.columns)
    if is_symmetric:
        column_pairs = list(itertools.combinations(cols, 2))
    else:
        column_pairs = [pair for pair in itertools.product(cols, repeat=2) if pair[0] != pair[1]]
    res = {}
    for idx in data.index:
        res[idx] = {}
        for pair in column_pairs:
            vals1, vals2 = data.loc[idx, pair[0]], data.loc[idx, pair[1]]
            if len(vals1) == 0 or pd.isnull(vals1).all():
                res[idx][pair] = None
            elif len(vals2) == 0 or pd.isnull(vals2).all():
                res[idx][pair] = None
            else:
                res[idx][pair] = func(vals1, vals2)
    res = pd.DataFrame.from_dict(res, orient="index")
    res.index.names = data.index.names
    return res


def group_and_aggregate(data: pd.DataFrame,
                        group_by: Optional[Union[str, List[str]]]) -> pd.DataFrame:
    """ Group the data by the given criteria and aggregate the values in each group. """
    if group_by is None:
        return data
    grouped_vals = data.groupby(level=group_by).agg(list).map(lambda group: pd.Series(group).explode().to_list())
    if len(grouped_vals.index) == 1:
        return grouped_vals
    # there is more than one group, so add a row for "all" groups
    group_all = pd.Series([data[col].explode().to_list() for col in data.columns], index=data.columns, name="all")
    grouped_vals = pd.concat([grouped_vals.T, group_all], axis=1).T  # add "all" row
    return grouped_vals


def extract_rater_detector_pairs(data: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Extracts pairs of (human-rater, human-rater) and (human-rater, detector) columns from the given DataFrame.
    :param data: DataFrame where each column is either a human-rater or a detector.
        human raters are columns with two-letter names, and detectors are columns with "det" in their name.
    :return: a list of pairs of (human-rater, human-rater) and (human-rater, detector) column names.
    """
    rater_names = sorted([col.upper() for col in data.columns if len(col) == 2])
    rater_rater_pairs = [(r1, r2) for r1, r2 in itertools.product(rater_names, repeat=2) if r1 != r2]
    detector_names = sorted([col for col in data.columns if "det" in col.lower()])
    rater_detector_pairs = [(rater, detector) for rater in rater_names for detector in detector_names]
    return rater_rater_pairs + rater_detector_pairs
