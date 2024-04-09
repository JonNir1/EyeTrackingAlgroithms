import time
import warnings
import itertools
from typing import List, Set

import numpy as np
import pandas as pd

import Config.constants as cnst
from DataSetLoaders.DataSetFactory import DataSetFactory
import Analysis.EventMatcher as matcher
import Analysis.comparisons as cmps
import Analysis.figures as figs

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
    "Onset Jitter", "Offset Jitter", "L2 Timing Jitter", "IoU", "Overlap Time", "Amplitude", "Duration", "Azimuth",
    "Peak Velocity"
}


def preprocess_dataset(dataset_name: str, verbose=False, **match_kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if verbose:
            print(f"Preprocessing dataset `{dataset_name}`...")
        start = time.time()
        samples_df, events_df, detector_results_df = DataSetFactory.load_and_process(dataset_name)
        samples_df.rename(columns=lambda col: col[:col.index("ector")] if "ector" in col else col, inplace=True)
        events_df.rename(columns=lambda col: col[:col.index("ector")] if "ector" in col else col, inplace=True)
        detector_results_df.rename(columns=lambda col: col[:col.index("ector")] if "ector" in col else col, inplace=True)

        # match events
        match_kwargs = {**_DEFAULT_EVENT_MATCHING_PARAMS, **match_kwargs}
        matches = matcher.match_events(events_df, is_symmetric=True, **match_kwargs)

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


def calculate_sample_metrics(samples_df: pd.DataFrame,
                             comparison_columns: List[str],
                             show_distributions=False,
                             verbose=False):
    global_start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = {}
        for metric_name, metric_short in SAMPLE_METRICS.items():
            start = time.time()
            # TODO: implement instead of cmps.compare_samples
            computed_metric = cmps.compare_samples(samples=samples_df, metric=metric_short, group_by=cnst.STIMULUS)
            results[metric_name] = computed_metric[comparison_columns]
            if show_distributions:
                distribution_fig = figs.distributions_grid(
                    computed_metric,
                    plot_type="violin",
                    title=f"Sample-Level `{metric_name.title()}` Distribution",
                    column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
                )
                distribution_fig.show()
            end = time.time()
            if verbose:
                print(f"\tCalculating `{metric_name}`:\t{end - start:.2f}s")
        global_end = time.time()
        if verbose:
            print(f"Total time:\t{global_end - global_start:.2f}s\n")
    return results


def extract_features(events_df: pd.DataFrame,
                     ignore_events: List[cnst.EVENT_LABELS] = None,
                     show_distributions=False,
                     verbose=False):
    global_start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = {}
        for feature in EVENT_FEATURES:
            start = time.time()
            if feature == "Counts":
                # TODO: implement instead of cmps.label_counts
                grouped = cmps.label_counts(events=events_df, group_by=cnst.STIMULUS)
            else:
                events_df = events_df.map(lambda cell: [getattr(e, feature.lower()) for e in cell
                                                        if e not in ignore_events or not hasattr(e, feature.lower())])
                grouped = cmps.group_and_aggregate(events_df, group_by=cnst.STIMULUS)
            results[feature] = grouped
            if show_distributions:
                distribution_fig = figs.distributions_grid(
                    grouped,
                    plot_type="violin",
                    title=f"Event-Level `{feature.title()}` Distribution",
                    column_title_mapper=lambda col: f"{col[0]}→{col[1]}",
                    show_counts=feature == "Counts",
                )
                distribution_fig.show()
            end = time.time()
            if verbose:
                print(f"\tExtracting `{feature}`s:\t{end - start:.2f}s")
    global_end = time.time()
    if verbose:
        print(f"Total time:\t{global_end - global_start:.2f}s\n")
    return results


def calc_event_matching_ratio(events_df: pd.DataFrame, matches_df: pd.DataFrame):
    # TODO implement instead of cmps.matched_event_ratios
    return None


def calc_matched_events_feature_diffs(matches_df: pd.DataFrame,
                                      comparison_columns: List[str],
                                      ignore_events: Set[cnst.EVENT_LABELS] = None,
                                      show_distributions=False,
                                      verbose=False):
    global_start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if ignore_events:
            matches_df = matches_df.map(
                lambda cell: {k: v for k, v in cell.items() if k.event_label not in ignore_events})
        results = {}
        for feature in MATCHED_EVENT_FEATURES:
            start = time.time()
            feature_diffs = _calc_matched_events_feature_diffs_impl(matches_df, feature)
            results[feature] = feature_diffs[comparison_columns]
            if show_distributions:
                distribution_fig = figs.distributions_grid(
                    feature_diffs,
                    plot_type="violin",
                    title=f"Matched-Event-Level `{feature}` Distribution",
                    column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
                )
                distribution_fig.show()
            end = time.time()
            if verbose:
                print(f"\tCalculating Matched-`{feature}`s:\t{end - start:.2f}s")
    global_end = time.time()
    if verbose:
        print(f"Total time:\t{global_end - global_start:.2f}s\n")
    return None


def _calc_matched_events_feature_diffs_impl(matches_df: pd.DataFrame, feature: str):
    if feature == "Onset Jitter":
        diffs = matches_df.map(
            lambda cell: [k.start_time - v.start_time for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    elif feature == "Offset Jitter":
        diffs = matches_df.map(
            lambda cell: [k.end_time - v.end_time for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    elif feature == "L2 Timing Jitter":
        diffs = matches_df.map(
            lambda cell: [k.l2_timing_offset(v) for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    elif feature == "IoU":
        diffs = matches_df.map(
            lambda cell: [k.intersection_over_union(v) for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    elif feature == "Overlap Time":
        diffs = matches_df.map(
            lambda cell: [k.overlap_time(v) for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    elif feature == "Duration":
        diffs = matches_df.map(
            lambda cell: [k.duration - v.duration for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    elif feature == "Amplitude":
        diffs = matches_df.map(
            lambda cell: [k.amplitude - v.amplitude for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    elif feature == "Azimuth":
        diffs = matches_df.map(
            lambda cell: [k.azimuth - v.azimuth for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    elif feature == "Peak Velocity":
        diffs = matches_df.map(
            lambda cell: [k.peak_velocity - v.peak_velocity for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    else:
        raise ValueError(f"Unknown feature: {feature}")
    return cmps.group_and_aggregate(diffs, group_by=cnst.STIMULUS)
