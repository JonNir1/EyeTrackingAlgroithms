import time
import warnings
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Set, Dict

import numpy as np
import pandas as pd

import Config.constants as cnst
import Config.experiment_config as cnfg
from GazeDetectors.BaseDetector import BaseDetector
from GazeEvents.BaseEvent import BaseEvent


class BaseAnalyzer(ABC):

    EVENT_FEATURES = {
        "Count", "Micro-Saccade Ratio", "Amplitude", "Duration", "Azimuth", "Peak Velocity"
    }
    EVENT_FEATURES_STR = "Event Features"

    @staticmethod
    @abstractmethod
    def preprocess_dataset(dataset_name: str,
                           verbose=False,
                           **kwargs):
        raise NotImplementedError

    @staticmethod
    def analyze(events_df: pd.DataFrame,
                ignore_events: Set[cnst.EVENT_LABELS] = None,
                verbose: bool = False,
                **kwargs):
        """
        Analyze the given events DataFrame and extract the features of the detected events.

        :param events_df: A DataFrame containing the detected events of each rater/detector.
        :param ignore_events: A set of event labels to ignore when extracting the features.
        :param verbose: Whether to print the progress of the analysis.
        :param kwargs: placeholder for additional parameters used by inherited classes.

        :return: A dictionary mapping a feature name to a DataFrame containing the extracted features.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.time()
            ignore_events = ignore_events or set()
            results = {BaseAnalyzer.EVENT_FEATURES_STR: BaseAnalyzer._extract_event_features(events_df,
                                                                                             ignore_events=ignore_events,
                                                                                             verbose=verbose)}
            end = time.time()
            if verbose:
                print(f"Total Analysis Time:\t{end - start:.2f}s")
        return results

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
    def _extract_event_features(events_df: pd.DataFrame,
                                ignore_events: Set[cnst.EVENT_LABELS] = None,
                                verbose=False) -> Dict[str, pd.DataFrame]:
        global_start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ignore_events = ignore_events or set()
            results = {}
            for feature in BaseAnalyzer.EVENT_FEATURES:
                start = time.time()
                if feature == "Count":
                    grouped = BaseAnalyzer.__event_counts_impl(events_df, ignore_events=ignore_events)
                elif feature == "Micro-Saccade Ratio":
                    grouped = BaseAnalyzer.__microsaccade_ratio_impl(events_df)
                else:
                    attr = feature.lower().replace(" ", "_")
                    feature_df = events_df.map(lambda cell: [getattr(e, attr) for e in cell if
                                                             e.event_label not in ignore_events and hasattr(e, attr)])
                    grouped = BaseAnalyzer.group_and_aggregate(feature_df)
                results[feature] = grouped
                end = time.time()
                if verbose:
                    print(f"\tExtracting {feature}s:\t{end - start:.2f}s")
        global_end = time.time()
        if verbose:
            print(f"Total time:\t{global_end - global_start:.2f}s\n")
        return results

    @staticmethod
    def __event_counts_impl(events: pd.DataFrame, ignore_events: Set[cnst.EVENT_LABELS] = None, ) -> pd.DataFrame:
        """
        Counts the number of detected events for each detector by type of event, and groups the results by the stimulus.
        :param events: A DataFrame containing the detected events of each rater/detector.
        :return: A DataFrame containing the count of events detected by each rater/detector (cols), grouped by the given
            criteria (rows).
        """

        def count_event_labels(data: List[Union[BaseEvent, cnst.EVENT_LABELS]]) -> pd.Series:
            labels = pd.Series([e.event_label if isinstance(e, BaseEvent) else e for e in data])
            counts = labels.value_counts()
            if counts.empty:
                return pd.Series({l: 0 for l in cnst.EVENT_LABELS})
            if len(counts) == len(cnst.EVENT_LABELS):
                return counts
            missing_labels = pd.Series({l: 0 for l in cnst.EVENT_LABELS if l not in counts.index})
            return pd.concat([counts, missing_labels]).sort_index()

        ignore_events = ignore_events or set()
        events = events.map(lambda cell: [e for e in cell if e.event_label not in ignore_events])
        event_counts = events.map(count_event_labels)
        grouped_vals = event_counts.groupby(level=cnst.STIMULUS).agg(list).map(sum)
        if len(grouped_vals.index) == 1:
            return grouped_vals
        # there is more than one group, so add a row for "all" groups
        group_all = pd.Series(event_counts.sum(axis=0), index=event_counts.columns, name="all")
        grouped_vals = pd.concat([grouped_vals.T, group_all], axis=1).T  # add "all" row
        return grouped_vals

    @staticmethod
    def __microsaccade_ratio_impl(events: pd.DataFrame,
                                  threshold_amplitude: float = cnfg.MICROSACCADE_AMPLITUDE_THRESHOLD) -> pd.DataFrame:
        saccades = events.map(lambda cell: [e for e in cell if e.event_label == cnst.EVENT_LABELS.SACCADE])
        saccades_count = saccades.map(len).to_numpy()
        microsaccades = saccades.map(lambda cell: [e for e in cell if e.amplitude < threshold_amplitude])
        microsaccades_count = microsaccades.map(len).to_numpy()

        ratios = np.divide(microsaccades_count, saccades_count,
                           out=np.full_like(saccades_count, fill_value=np.nan, dtype=float),  # fill NaN if denom is 0
                           where=saccades_count != 0)
        ratios = pd.DataFrame(ratios, index=events.index, columns=events.columns)
        return BaseAnalyzer.group_and_aggregate(ratios)
