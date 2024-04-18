import time
import warnings
from typing import Set, Dict, List, Tuple

import numpy as np
import pandas as pd

import Config.experiment_config as cnfg
from Analysis.Analyzers.BaseAnalyzer import BaseAnalyzer
from GazeDetectors.BaseDetector import BaseDetector
from Analysis.EventMatcher import EventMatcher as Matcher


class MatchedEventsAnalyzer(BaseAnalyzer):

    SINGLE_EVENT_FEATURES = {"Start Time", "End Time", "Duration", "Amplitude", "Azimuth", "Peak Velocity"}
    DUAL_EVENT_FEATURES = {
        "Match Ratio", "Onset Jitter", "Offset Jitter", "L2 Timing Difference", "IoU", "Overlap Time",
        # "Duration Difference", "Amplitude Difference", "Azimuth Difference", "Peak Velocity Difference"  # uninteresting
    }

    _DEFAULT_EVENT_MATCHING_PARAMS = {
        "match_by": "onset",
        "max_onset_latency": 15,
        "allow_cross_matching": False,
        "ignore_events": None,
    }
    _DEFAULT_PAIRED_TEST = "Wilcoxon"
    _DEFAULT_SINGLE_TEST = "Wilcoxon"

    @staticmethod
    def preprocess_dataset(dataset_name: str,
                           detectors: List[BaseDetector] = None,
                           verbose=False,
                           **kwargs) -> (pd.DataFrame, pd.DataFrame, List[Tuple[str, str]]):
        """
        Preprocess the dataset by:
            1. Loading the dataset
            2. Detecting events using the given detectors
            3. Match events detected by each pair of detectors
            4. Extract pairs of (human-rater, detector) for future analysis

        :param dataset_name: The name of the dataset to load and preprocess.
        :param detectors: A list of detectors to use for detecting events. If None, the default detectors will be used.
        :param verbose: Whether to print the progress of the preprocessing.
        :keyword arguments:
            - column_mapper: A function to map the column names of the samples, events, and detector results DataFrames.
            - event_matching_params: Additional parameters to pass to the event matching algorithm (see EventMatcher).

        :return: the preprocessed samples, events, raw detector results, event matches, and comparison columns.
        """
        if verbose:
            print(f"Preprocessing dataset `{dataset_name}`...")
        start = time.time()
        _, events_df, _ = super(MatchedEventsAnalyzer, MatchedEventsAnalyzer).preprocess_dataset(
            dataset_name, detectors, False, **kwargs
        )
        # match events
        event_matching_params = kwargs.pop("event_matching_params",
                                           MatchedEventsAnalyzer._DEFAULT_EVENT_MATCHING_PARAMS)
        matches_df = Matcher.match_events(events_df, is_symmetric=True, **event_matching_params)
        # extract column-pairs to compare
        comparison_columns = MatchedEventsAnalyzer._extract_rater_detector_pairs(events_df)
        end = time.time()
        if verbose:
            print(f"\tPreprocessing:\t{end - start:.2f}s")
        return events_df, matches_df, comparison_columns

    @classmethod
    def calculate_observed_data(cls,
                                events_df: pd.DataFrame,
                                ignore_events: Set[cnfg.EVENT_LABELS] = None,
                                matches_df: pd.DataFrame = None,
                                verbose: bool = False,
                                **kwargs) -> Dict[str, pd.DataFrame]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if verbose:
                print("Extracting Matched-Events' Features...")
            # drop ignored events
            ignore_events = ignore_events or set()
            events_df = events_df.map(lambda cell: [e for e in cell if e.event_label not in ignore_events])
            matches_df = matches_df.map(
                lambda cell: {k: v for k, v in cell.items() if k.event_label not in ignore_events}
                if pd.notnull(cell) else np.nan
            )

            # calculate single-event features
            results = {}
            for feature in cls.SINGLE_EVENT_FEATURES:
                start = time.time()
                attr = feature.lower().replace(" ", "_")
                feature_df = matches_df.map(
                    lambda cell: [(getattr(k, attr), getattr(v, attr))
                                  for k, v in cell.items() if hasattr(k, attr) and hasattr(v, attr)]
                    if pd.notnull(cell) else np.nan
                )
                results[feature] = cls.group_and_aggregate(feature_df)
                end = time.time()
                if verbose:
                    print(f"\t{feature}:\t{end - start:.2f}s")

            # calculate dual-event features
            for feature in cls.DUAL_EVENT_FEATURES:
                start = time.time()
                if feature == "Match Ratio":
                    feature_df = cls._calculate_matching_ratio(events_df, matches_df)
                else:
                    feature_df = cls._calc_dual_feature(matches_df, feature)
                results[feature] = cls.group_and_aggregate(feature_df)
                end = time.time()
                if verbose:
                    print(f"\t{feature}:\t{end - start:.2f}s")
        return results

    @classmethod
    def statistical_analysis(cls,
                             matched_features_dict: Dict[str, pd.DataFrame],
                             paired_sample_test: str = _DEFAULT_PAIRED_TEST,
                             single_sample_test: str = _DEFAULT_SINGLE_TEST,
                             **kwargs) -> Dict[str, pd.DataFrame]:
        paired_test = cls._get_statistical_test_func(paired_sample_test)
        single_test = cls._get_statistical_test_func(single_sample_test)
        results = {}
        for feature_name, feature_df in matched_features_dict.items():
            feature_df = feature_df.map(lambda cell: [v for v in cell if not np.all(np.isnan(v))] if not np.all(pd.isna(cell)) else None)
            if feature_name in cls.SINGLE_EVENT_FEATURES:
                stat_res = feature_df.map(lambda cell: paired_test([v[0] for v in cell], [v[1] for v in cell]))
            elif feature_name in {"Onset Jitter", "Offset Jitter", "L2 Timing Difference"}:
                null_hypothesis_value = 0
                stat_res = feature_df.map(
                    lambda cell: single_test(cell, np.full_like(cell, null_hypothesis_value), zero_method="zsplit")
                    if not np.all(pd.isna(cell)) else None
                )  # use zsplit to include zero-differences
            elif feature_name in {"Match Ratio", "IoU", "Overlap Time"}:
                null_hypothesis_value = 1
                stat_res = feature_df.map(
                    lambda cell: single_test(cell, np.full_like(cell, null_hypothesis_value), zero_method="zsplit")
                    if not np.all(pd.isna(cell)) else None
                )  # use zsplit to include zero-differences
            else:
                raise ValueError(f"Unknown feature: {feature_name}")
            results[feature_name] = cls._rearrange_statistical_results(stat_res)
        return results

    @staticmethod
    def _calculate_matching_ratio(events_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
        event_counts = events_df.map(len)
        match_counts = matches_df.map(lambda cell: len(cell) if pd.notnull(cell) else np.nan)
        ratios = np.zeros_like(match_counts, dtype=float)
        for i in range(match_counts.index.size):
            for j in range(match_counts.columns.size):
                gt_col, _pred_col = match_counts.columns[j]
                ratios[i, j] = match_counts.iloc[i, j] / event_counts.iloc[i][gt_col]
        ratios = pd.DataFrame(ratios, index=match_counts.index, columns=match_counts.columns)
        return ratios

    @staticmethod
    def _calc_dual_feature(matches_df: pd.DataFrame, feature: str) -> pd.DataFrame:
        if feature == "Onset Jitter":
            feature_df = matches_df.map(
                lambda cell: [k.start_time - v.start_time for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "Offset Jitter":
            feature_df = matches_df.map(
                lambda cell: [k.end_time - v.end_time for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "L2 Timing Difference":
            feature_df = matches_df.map(
                lambda cell: [k.l2_timing_offset(v) for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "IoU":
            feature_df = matches_df.map(
                lambda cell: [k.intersection_over_union(v) for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "Overlap Time":
            feature_df = matches_df.map(
                lambda cell: [k.overlap_time(v) for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "Duration Difference":
            feature_df = matches_df.map(
                lambda cell: [k.duration - v.duration for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "Amplitude Difference":
            feature_df = matches_df.map(
                lambda cell: [k.amplitude - v.amplitude for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "Azimuth Difference":
            feature_df = matches_df.map(
                lambda cell: [k.azimuth - v.azimuth for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "Peak Velocity Difference":
            feature_df = matches_df.map(
                lambda cell: [k.peak_velocity_px - v.peak_velocity_px for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "Match Ratio":
            raise ValueError("Match Ratio feature should be calculated separately.")
        else:
            raise ValueError(f"Unknown feature: {feature}")
        return feature_df

