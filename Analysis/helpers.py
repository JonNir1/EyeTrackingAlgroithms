import time
import warnings
import itertools
from typing import List, Set, Dict, Optional, Union

import numpy as np
import pandas as pd

import Config.constants as cnst
import Utils.array_utils as au
import Utils.metrics as metrics
from GazeEvents.BaseEvent import BaseEvent
from GazeDetectors.BaseDetector import BaseDetector
from DataSetLoaders.DataSetFactory import DataSetFactory
from Analysis.EventMatcher import EventMatcher as matcher

_DEFAULT_EVENT_MATCHING_PARAMS = {
    "match_by": "onset",
    "max_onset_latency": 15,
    "allow_cross_matching": False,
    "ignore_events": None,
}

SAMPLE_METRICS = {
    "Accuracy": "acc",
    "Levenshtein Distance": "lev",
    "Cohen's Kappa": "kappa",
    "Mathew's Correlation": "mcc",
    "Transition Matrix l2-norm": "frobenius",
    "Transition Matrix KL-Divergence": "kl"
}
EVENT_FEATURES = {
    "Counts", "Amplitude", "Duration", "Azimuth", "Peak Velocity"
}

MATCHED_EVENT_FEATURES = {
    "Onset Jitter", "Offset Jitter", "L2 Timing Difference", "IoU", "Overlap Time", "Amplitude Difference",
    "Duration Difference", "Azimuth Difference", "Peak Velocity Difference"
}


def get_default_detectors() -> Union[BaseDetector, List[BaseDetector]]:
    from GazeDetectors.IVTDetector import IVTDetector
    from GazeDetectors.IDTDetector import IDTDetector
    from GazeDetectors.EngbertDetector import EngbertDetector
    from GazeDetectors.NHDetector import NHDetector
    from GazeDetectors.REMoDNaVDetector import REMoDNaVDetector
    return [IVTDetector(), IDTDetector(), EngbertDetector(), NHDetector(), REMoDNaVDetector()]


def preprocess_dataset(dataset_name: str,
                       detectors: List[BaseDetector] = None,
                       verbose=False,
                       **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if verbose:
            print(f"Preprocessing dataset `{dataset_name}`...")
        start = time.time()
        detectors = get_default_detectors() if detectors is None else detectors
        samples_df, events_df, detector_results_df = DataSetFactory.load_and_detect(dataset_name, detectors)

        # rename columns
        column_mapper = kwargs.get("column_mapper", lambda col: col)
        samples_df.rename(columns=column_mapper, inplace=True)
        events_df.rename(columns=column_mapper, inplace=True)
        detector_results_df.rename(columns=column_mapper, inplace=True)

        # match events
        kwargs = {**_DEFAULT_EVENT_MATCHING_PARAMS, **kwargs}
        matches = matcher.match_events(events_df, is_symmetric=True, **kwargs)

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


def calc_sample_metrics(samples_df: pd.DataFrame,
                        verbose=False):
    global_start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = {}
        for metric_name, metric_short in SAMPLE_METRICS.items():
            start = time.time()
            computed = _calc_sample_metric_impl(samples_df, metric_short)
            results[metric_name] = computed
            end = time.time()
            if verbose:
                print(f"\tCalculating `{metric_name}`:\t{end - start:.2f}s")
        global_end = time.time()
        if verbose:
            print(f"Total time:\t{global_end - global_start:.2f}s\n")
    return results


def extract_features(events_df: pd.DataFrame,
                     ignore_events: Set[cnst.EVENT_LABELS] = None,
                     verbose=False) -> Dict[str, pd.DataFrame]:
    global_start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ignore_events = ignore_events or set()
        results = {}
        for feature in EVENT_FEATURES:
            start = time.time()
            if feature == "Counts":
                grouped = _event_counts_impl(events_df, ignore_events=ignore_events)
            else:
                attr = feature.lower().replace(" ", "_")
                feature_df = events_df.map(lambda cell: [getattr(e, attr) for e in cell
                                                         if e not in ignore_events and hasattr(e, attr)])
                grouped = _group_and_aggregate(feature_df, group_by=cnst.STIMULUS)
            results[feature] = grouped
            end = time.time()
            if verbose:
                print(f"\tExtracting {feature}s:\t{end - start:.2f}s")
    global_end = time.time()
    if verbose:
        print(f"Total time:\t{global_end - global_start:.2f}s\n")
    return results


def calc_event_matching_ratios(events_df: pd.DataFrame,
                               matches_df: pd.DataFrame,
                               ignore_events: Set[cnst.EVENT_LABELS] = None,
                               verbose=False) -> Dict[str, pd.DataFrame]:
    global_start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ignore_events = ignore_events or set()
        event_counts = events_df.map(lambda cell: len([e for e in cell if e.event_label not in ignore_events]))
        match_counts = matches_df.map(
            lambda cell: len([k for k in cell.keys() if k.event_label not in ignore_events])
            if pd.notnull(cell) else np.nan
        )
        ratios = np.zeros_like(match_counts, dtype=float)
        for i in range(match_counts.index.size):
            for j in range(match_counts.columns.size):
                gt_col, _pred_col = match_counts.columns[j]
                ratios[i, j] = match_counts.iloc[i, j] / event_counts.iloc[i][gt_col]
        ratios = pd.DataFrame(100 * ratios, index=match_counts.index, columns=match_counts.columns)
        ratios = _group_and_aggregate(ratios, group_by=cnst.STIMULUS)
    global_end = time.time()
    if verbose:
        print(f"Total time:\t{global_end - global_start:.2f}s\n")
    return {"Match Ratio": ratios}


def calc_matched_events_feature(matches_df: pd.DataFrame,
                                ignore_events: Set[cnst.EVENT_LABELS] = None,
                                verbose=False) -> Dict[str, pd.DataFrame]:
    global_start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = {}
        for feature in MATCHED_EVENT_FEATURES:
            start = time.time()
            feature_diffs = _calc_matched_events_feature_impl(matches_df, feature, ignore_events)
            results[feature] = feature_diffs
            end = time.time()
            if verbose:
                print(f"\tCalculating Matched {feature}:\t{end - start:.2f}s")
    global_end = time.time()
    if verbose:
        print(f"Total time:\t{global_end - global_start:.2f}s\n")
    return results


def _calc_sample_metric_impl(samples: pd.DataFrame,
                             metric: str) -> pd.DataFrame:
    if metric == "acc" or metric == "accuracy" or metric == "balanced accuracy":
        res = au.apply_on_column_pairs(samples, metrics.balanced_accuracy)
    elif metric == "lev" or metric == "levenshtein":
        res = au.apply_on_column_pairs(samples, metrics.levenshtein_distance)
    elif metric == "kappa" or metric == "cohen kappa":
        res = au.apply_on_column_pairs(samples, metrics.cohen_kappa)
    elif metric == "mcc" or metric == "matthews correlation":
        res = au.apply_on_column_pairs(samples, metrics.matthews_correlation)
    elif metric == "fro" or metric == "frobenius" or metric == "l2":
        res = au.apply_on_column_pairs(samples,
                                       lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="fro"))
    elif metric == "kl" or metric == "kl divergence" or metric == "kullback leibler":
        res = au.apply_on_column_pairs(samples,
                                       lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="kl"))
    else:
        raise NotImplementedError(f"Unknown metric for samples:\t{metric}")
    return _group_and_aggregate(res, cnst.STIMULUS)


def _event_counts_impl(events: pd.DataFrame, ignore_events: Set[cnst.EVENT_LABELS] = None,) -> pd.DataFrame:
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


def _calc_matched_events_feature_impl(matches_df: pd.DataFrame,
                                      feature: str,
                                      ignore_events: Set[cnst.EVENT_LABELS] = None) -> pd.DataFrame:
    ignore_events = ignore_events or set()
    if feature == "Onset Jitter":
        diffs = matches_df.map(
            lambda cell: [k.start_time - v.start_time for k, v in cell.items() if k.event_label not in ignore_events]
            if pd.notnull(cell) else np.nan
        )
    elif feature == "Offset Jitter":
        diffs = matches_df.map(
            lambda cell: [k.end_time - v.end_time for k, v in cell.items() if k.event_label not in ignore_events]
            if pd.notnull(cell) else np.nan
        )
    elif feature == "L2 Timing Difference":
        diffs = matches_df.map(
            lambda cell: [k.l2_timing_offset(v) for k, v in cell.items() if k.event_label not in ignore_events]
            if pd.notnull(cell) else np.nan
        )
    elif feature == "IoU":
        diffs = matches_df.map(
            lambda cell: [k.intersection_over_union(v) for k, v in cell.items() if k.event_label not in ignore_events]
            if pd.notnull(cell) else np.nan
        )
    elif feature == "Overlap Time":
        diffs = matches_df.map(
            lambda cell: [k.overlap_time(v) for k, v in cell.items() if k.event_label not in ignore_events]
            if pd.notnull(cell) else np.nan
        )
    elif feature == "Duration Difference":
        diffs = matches_df.map(
            lambda cell: [k.duration - v.duration for k, v in cell.items() if k.event_label not in ignore_events]
            if pd.notnull(cell) else np.nan
        )
    elif feature == "Amplitude Difference":
        diffs = matches_df.map(
            lambda cell: [k.amplitude - v.amplitude for k, v in cell.items() if k.event_label not in ignore_events]
            if pd.notnull(cell) else np.nan
        )
    elif feature == "Azimuth Difference":
        diffs = matches_df.map(
            lambda cell: [k.azimuth - v.azimuth for k, v in cell.items() if k.event_label not in ignore_events]
            if pd.notnull(cell) else np.nan
        )
    elif feature == "Peak Velocity Difference":
        diffs = matches_df.map(
            lambda cell: [k.peak_velocity - v.peak_velocity for k, v in cell.items() if k.event_label not in ignore_events]
            if pd.notnull(cell) else np.nan
        )
    else:
        raise ValueError(f"Unknown feature: {feature}")
    return _group_and_aggregate(diffs, group_by=cnst.STIMULUS)


def _group_and_aggregate(data: pd.DataFrame, group_by: Optional[Union[str, List[str]]]) -> pd.DataFrame:
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
