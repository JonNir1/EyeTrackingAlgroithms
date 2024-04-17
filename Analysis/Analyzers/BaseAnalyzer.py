import time
import warnings
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Set, Dict, final

import numpy as np
import pandas as pd
import scipy.stats as stat

import Config.constants as cnst
import Config.experiment_config as cnfg
import Analysis.helpers as hlp
from DataSetLoaders.DataSetFactory import DataSetFactory
from GazeDetectors.BaseDetector import BaseDetector
from GazeEvents.BaseEvent import BaseEvent


class BaseAnalyzer(ABC):

    EVENT_FEATURES = {
        "Count", "Micro-Saccade Ratio", "Amplitude", "Duration", "Azimuth", "Peak Velocity"
    }
    EVENT_FEATURES_STR = "Event Features"

    @staticmethod
    def preprocess_dataset(dataset_name: str,
                           detectors: List[BaseDetector] = None,
                           verbose=False,
                           **kwargs) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Preprocess the dataset by:
            1. Loading the dataset
            2. Detecting events using the given detectors
            3. Renaming the columns of the samples, events, and detector results DataFrames.

        :param dataset_name: The name of the dataset to load and preprocess.
        :param detectors: A list of detectors to use for detecting events. If None, the default detectors will be used.
        :param verbose: Whether to print the progress of the preprocessing.
        :keyword column_mapper: A function to map the column names of the samples, events, and detector results DataFrames.

        :return: the preprocessed samples, events and raw detector results DataFrames.
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

            end = time.time()
            if verbose:
                print(f"\tPreprocessing:\t{end - start:.2f}s")
        return samples_df, events_df, detector_results_df

    @classmethod
    @final
    def analyze(cls,
                events_df: pd.DataFrame,
                test_name: str,
                ignore_events: Set[cnfg.EVENT_LABELS] = None,
                verbose: bool = False,
                **kwargs) -> (Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.time()
            ignore_events = ignore_events or set()

            # extract observations
            if verbose:
                print("Calculating Observations...")
            observations_start = time.time()
            obs_dict = cls.calculate_observed_data(events_df, ignore_events=ignore_events, **kwargs)
            observations_end = time.time()
            if verbose:
                print(f"Observations Time:\t{observations_end - observations_start:.2f}s")

            # perform statistical analysis
            if verbose:
                print("Performing Statistical Analysis...")
            stat_start = time.time()
            stats_dict = cls.statistical_analysis(observations=obs_dict, test_name=test_name, **kwargs)
            stat_end = time.time()
            if verbose:
                print(f"Statistical Analysis Time:\t{stat_end - stat_start:.2f}s")

            # conclude analysis
            end = time.time()
            if verbose:
                print(f"Total Analysis Time:\t{end - start:.2f}s")
        return obs_dict, stats_dict

    @classmethod
    @abstractmethod
    def calculate_observed_data(cls,
                                data: pd.DataFrame,
                                ignore_events: Set[cnfg.EVENT_LABELS] = None,
                                verbose: bool = False,
                                **kwargs) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def statistical_analysis(cls,
                             observations: Dict[str, pd.DataFrame],
                             **kwargs) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    @staticmethod
    def group_and_aggregate(data: pd.DataFrame,
                            group_by: Optional[Union[str, List[str]]] = cnst.STIMULUS) -> pd.DataFrame:
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

    @staticmethod
    def _get_statistical_test_func(test_name: str):
        test_name = test_name.lower().replace("_", " ").replace("-", " ").strip()
        if test_name in {"u", "u test", "mann whitney", "mann whitney u", "mannwhitneyu"}:
            return stat.mannwhitneyu
        elif test_name in {"rank sum", "ranksum", "ranksums", "wilcoxon rank sum"}:
            return stat.ranksums
        elif test_name in {"wilcoxon", "wilcoxon signed rank", "signed rank"}:
            return stat.wilcoxon
        else:
            raise ValueError(f"Unknown test name: {test_name}")

    @staticmethod
    def _rearrange_statistical_results(stat_results: pd.DataFrame) -> pd.DataFrame:
        # remap stat_results to a DataFrame with multi-index columns
        statistics = {(col1, col2, cnst.STATISTIC): {vk: vv[0] for vk, vv in vals.items()}
                      for (col1, col2), vals in stat_results.items()}
        p_values = {(col1, col2, cnst.P_VALUE): {vk: vv[1] for vk, vv in vals.items()}
                    for (col1, col2), vals in stat_results.items()}
        results = {**statistics, **p_values}
        results = pd.DataFrame(results)

        # reorder columns to group by the first column, then the second column, and finally by the statistic type
        column_order = {triplet: (
            len([trp for trp in results.keys() if trp[0] == triplet[0]]),
            len([trp for trp in results.keys() if trp[0] == triplet[1]]),
            1 if triplet[2] == cnst.STATISTIC else 0
        ) for triplet in results.keys()}
        ordered_columns = sorted(results.columns, key=lambda col: column_order[col], reverse=True)
        results = results[ordered_columns]
        return results

