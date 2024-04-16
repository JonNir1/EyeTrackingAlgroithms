import time
import warnings
import itertools
from abc import ABC
from typing import Set, Dict, Union, List

import numpy as np
import pandas as pd

import Config.constants as cnst
import Utils.metrics as metrics

from Analysis.BaseAnalyzer import BaseAnalyzer
from GazeEvents.BaseEvent import BaseEvent


class DetectorComparisonAnalyzer(BaseAnalyzer, ABC):
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

    MATCH_RATIO_STR = "Match Ratio"

    @staticmethod
    def analyze(events_df: pd.DataFrame,
                matches_df: pd.DataFrame,
                samples_df: pd.DataFrame = None,
                ignore_events: Set[cnst.EVENT_LABELS] = None,
                verbose=False):
        """
        Analyze the given events, event-matches and samples, by calculating various metrics and features.

        :param events_df: A DataFrame containing the detected events of each rater/detector.
        :param matches_df: A DataFrame containing the matched events between each pair of raters/detectors.
        :param samples_df: A DataFrame containing the detected samples of each rater/detector.
            Only used for sample-level metrics, when `ignore_events` is None or empty.
        :param ignore_events: A set of event labels to ignore when calculating metrics and features.
        :param verbose: Whether to print the progress of the analysis.

        :return: A dictionary containing the results of the analysis.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.time()
            ignore_events = ignore_events or set()
            results = {}
            if samples_df and not ignore_events:
                results["Sample Metrics"] = DetectorComparisonAnalyzer._calc_sample_metrics(samples_df, verbose=verbose)
            results["Event Features"] = DetectorComparisonAnalyzer._extract_features(events_df, ignore_events=ignore_events,
                                                                                     verbose=verbose)
            results["Event Matching Ratios"] = DetectorComparisonAnalyzer._calc_event_matching_ratios(events_df, matches_df,
                                                                                                      ignore_events=ignore_events,
                                                                                                      verbose=verbose)
            results["Matched Event Features"] = DetectorComparisonAnalyzer._calc_matched_events_feature(matches_df,
                                                                                                        ignore_events=ignore_events,
                                                                                                        verbose=verbose)
            end = time.time()
            if verbose:
                print(f"Total Analysis Time:\t{end - start:.2f}s")
        return results

    @staticmethod
    def _calc_sample_metrics(samples_df: pd.DataFrame,
                             verbose=False):
        global_start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = {}
            for metric_name, metric_short in DetectorComparisonAnalyzer.SAMPLE_METRICS.items():
                start = time.time()
                computed = DetectorComparisonAnalyzer.__calc_sample_metric_impl(samples_df, metric_short)
                results[metric_name] = computed
                end = time.time()
                if verbose:
                    print(f"\tCalculating `{metric_name}`:\t{end - start:.2f}s")
            global_end = time.time()
            if verbose:
                print(f"Total time:\t{global_end - global_start:.2f}s\n")
        return results

    @staticmethod
    def _extract_features(events_df: pd.DataFrame,
                          ignore_events: Set[cnst.EVENT_LABELS] = None,
                          verbose=False) -> Dict[str, pd.DataFrame]:
        global_start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ignore_events = ignore_events or set()
            results = {}
            for feature in DetectorComparisonAnalyzer.EVENT_FEATURES:
                start = time.time()
                if feature == "Counts":
                    grouped = DetectorComparisonAnalyzer.__event_counts_impl(events_df, ignore_events=ignore_events)
                else:
                    attr = feature.lower().replace(" ", "_")
                    feature_df = events_df.map(lambda cell: [getattr(e, attr) for e in cell
                                                             if e not in ignore_events and hasattr(e, attr)])
                    grouped = DetectorComparisonAnalyzer.group_and_aggregate(feature_df, group_by=cnst.STIMULUS)
                results[feature] = grouped
                end = time.time()
                if verbose:
                    print(f"\tExtracting {feature}s:\t{end - start:.2f}s")
        global_end = time.time()
        if verbose:
            print(f"Total time:\t{global_end - global_start:.2f}s\n")
        return results

    @staticmethod
    def _calc_event_matching_ratios(events_df: pd.DataFrame,
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
            ratios = DetectorComparisonAnalyzer.group_and_aggregate(ratios, group_by=cnst.STIMULUS)
        global_end = time.time()
        if verbose:
            print(f"Total time:\t{global_end - global_start:.2f}s\n")
        return {DetectorComparisonAnalyzer.MATCH_RATIO_STR: ratios}

    @staticmethod
    def _calc_matched_events_feature(matches_df: pd.DataFrame,
                                     ignore_events: Set[cnst.EVENT_LABELS] = None,
                                     verbose=False) -> Dict[str, pd.DataFrame]:
        global_start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = {}
            for feature in DetectorComparisonAnalyzer.MATCHED_EVENT_FEATURES:
                start = time.time()
                feature_diffs = DetectorComparisonAnalyzer.__calc_matched_events_feature_impl(matches_df, feature, ignore_events)
                results[feature] = feature_diffs
                end = time.time()
                if verbose:
                    print(f"\tCalculating Matched {feature}:\t{end - start:.2f}s")
        global_end = time.time()
        if verbose:
            print(f"Total time:\t{global_end - global_start:.2f}s\n")
        return results

    @staticmethod
    def __calc_sample_metric_impl(samples: pd.DataFrame,
                                  metric: str) -> pd.DataFrame:
        # extract the function to calculate the metric
        if metric == "acc" or metric == "accuracy" or metric == "balanced accuracy":
            measure_func = metrics.balanced_accuracy
        elif metric == "lev" or metric == "levenshtein":
            measure_func = metrics.levenshtein_distance
        elif metric == "kappa" or metric == "cohen kappa":
            measure_func = metrics.cohen_kappa
        elif metric == "mcc" or metric == "matthews correlation":
            measure_func = metrics.matthews_correlation
        elif metric == "fro" or metric == "frobenius" or metric == "l2":
            measure_func = lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="fro")
        elif metric == "kl" or metric == "kl divergence" or metric == "kullback leibler":
            measure_func = lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="kl")
        else:
            raise NotImplementedError(f"Unknown metric for samples:\t{metric}")

        # perform the calculation
        column_pairs = list(itertools.combinations(samples.columns, 2))
        res = {}
        for idx in samples.index:
            res[idx] = {}
            for pair in column_pairs:
                vals1, vals2 = samples.loc[idx, pair[0]], samples.loc[idx, pair[1]]
                if len(vals1) == 0 or pd.isnull(vals1).all():
                    res[idx][pair] = None
                elif len(vals2) == 0 or pd.isnull(vals2).all():
                    res[idx][pair] = None
                else:
                    res[idx][pair] = measure_func(vals1, vals2)
        res = pd.DataFrame.from_dict(res, orient="index")
        res.index.names = samples.index.names

        # aggregate over stimuli and return
        return DetectorComparisonAnalyzer.group_and_aggregate(res, cnst.STIMULUS)

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
    def __calc_matched_events_feature_impl(matches_df: pd.DataFrame,
                                           feature: str,
                                           ignore_events: Set[cnst.EVENT_LABELS] = None) -> pd.DataFrame:
        ignore_events = ignore_events or set()
        if feature == "Onset Jitter":
            diffs = matches_df.map(
                lambda cell: [k.start_time - v.start_time for k, v in cell.items() if
                              k.event_label not in ignore_events]
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
                lambda cell: [k.intersection_over_union(v) for k, v in cell.items() if
                              k.event_label not in ignore_events]
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
                lambda cell: [k.peak_velocity - v.peak_velocity for k, v in cell.items() if
                              k.event_label not in ignore_events]
                if pd.notnull(cell) else np.nan
            )
        else:
            raise ValueError(f"Unknown feature: {feature}")
        return DetectorComparisonAnalyzer.group_and_aggregate(diffs, group_by=cnst.STIMULUS)
