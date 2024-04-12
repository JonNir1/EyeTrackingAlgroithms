from typing import List, Optional, Union

import pandas as pd

from GazeDetectors.BaseDetector import BaseDetector


def get_default_detectors() -> Union[BaseDetector, List[BaseDetector]]:
    from GazeDetectors.IVTDetector import IVTDetector
    from GazeDetectors.IDTDetector import IDTDetector
    from GazeDetectors.EngbertDetector import EngbertDetector
    from GazeDetectors.NHDetector import NHDetector
    from GazeDetectors.REMoDNaVDetector import REMoDNaVDetector
    return [IVTDetector(), IDTDetector(), EngbertDetector(), NHDetector(), REMoDNaVDetector()]


def group_and_aggregate(data: pd.DataFrame, group_by: Optional[Union[str, List[str]]]) -> pd.DataFrame:
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
