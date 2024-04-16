import time
import warnings
import itertools
from abc import ABC, abstractmethod
from typing import List, Union, final

import pandas as pd

from GazeDetectors.BaseDetector import BaseDetector
from DataSetLoaders.DataSetFactory import DataSetFactory
from Analysis.EventMatcher import EventMatcher as Matcher


class BaseAnalyzer(ABC):
    _DEFAULT_EVENT_MATCHING_PARAMS = {
        "match_by": "onset",
        "max_onset_latency": 15,
        "allow_cross_matching": False,
        "ignore_events": None,
    }

    @staticmethod
    @abstractmethod
    def analyze(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    @final
    def preprocess_dataset(dataset_name: str,
                           detectors: List[BaseDetector] = None,
                           verbose=False,
                           **kwargs):
        """
        Preprocess the dataset by:
            1. Loading the dataset
            2. Detecting events using the given detectors
            3. Match events detected by each pair of detectors
            4. Extract pairs of (human-rater, detector) for future analysis

        :param dataset_name: The name of the dataset to load and preprocess.
        :param detectors: A list of detectors to use for detecting events. If None, the default detectors will be used.
        :param verbose: Whether to print the progress of the preprocessing.
        :keyword column_mapper: A function to map the column names of the samples, events, and detector results DataFrames.
        :additional keyword arguments: Additional parameters to pass to the event matching algorithm (see EventMatcher).

        :return: the preprocessed samples, events, raw detector results, event matches, and comparison columns.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if verbose:
                print(f"Preprocessing dataset `{dataset_name}`...")
            start = time.time()
            detectors = BaseAnalyzer._get_default_detectors() if detectors is None else detectors
            samples_df, events_df, detector_results_df = DataSetFactory.load_and_detect(dataset_name, detectors)

            # rename columns
            column_mapper = kwargs.pop("column_mapper", lambda col: col)
            samples_df.rename(columns=column_mapper, inplace=True)
            events_df.rename(columns=column_mapper, inplace=True)
            detector_results_df.rename(columns=column_mapper, inplace=True)

            # match events
            kwargs = {**BaseAnalyzer._DEFAULT_EVENT_MATCHING_PARAMS, **kwargs}
            matches = Matcher.match_events(events_df, is_symmetric=True, **kwargs)

            # extract column-pairs to compare
            rater_names = [col.upper() for col in samples_df.columns if len(col) == 2]
            detector_names = [col for col in samples_df.columns if "det" in col.lower()]
            rater_rater_pairs = list(itertools.combinations(sorted(rater_names), 2))
            rater_detector_pairs = [(rater, detector) for rater in rater_names for detector in detector_names]
            comparison_columns = rater_rater_pairs + rater_detector_pairs
            end = time.time()
            if verbose:
                print(f"\tPreprocessing:\t{end - start:.2f}s")
        return samples_df, events_df, detector_results_df, matches, comparison_columns

    @staticmethod
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

    @staticmethod
    def _get_default_detectors() -> Union[BaseDetector, List[BaseDetector]]:
        from GazeDetectors.IVTDetector import IVTDetector
        from GazeDetectors.IDTDetector import IDTDetector
        from GazeDetectors.EngbertDetector import EngbertDetector
        from GazeDetectors.NHDetector import NHDetector
        from GazeDetectors.REMoDNaVDetector import REMoDNaVDetector
        return [IVTDetector(), IDTDetector(), EngbertDetector(), NHDetector(), REMoDNaVDetector()]


