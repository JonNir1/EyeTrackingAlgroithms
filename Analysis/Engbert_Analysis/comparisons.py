import itertools
from typing import List, Callable, Optional, Union

import numpy as np
import pandas as pd

import Config.constants as cnst
import GazeEvents.helpers as hlp
import Utils.metrics as metrics

from GazeEvents.BaseEvent import BaseEvent
from Analysis.EventMatcher import EventMatcher


def label_counts(events: pd.DataFrame,
                 group_by: Optional[Union[str, List[str]]] = cnst.STIMULUS,
                 ignore_events: List[cnst.EVENT_LABELS] = None) -> pd.DataFrame:
    """
    Counts the number of event-labels detected by each rater/detector, and groups the results by the given criteria
    if specified. Ignores the specified event-labels if provided.

    :param events: A DataFrame containing the detected events of each rater/detector.
    :param group_by: The criteria to group the counts by.
    :param ignore_events: A set of event-labels to ignore during the counts.
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

    events = events.map(lambda cell: hlp.drop_events(cell, to_drop=ignore_events))
    event_counts = events.map(count_event_labels)
    group_all = pd.Series(event_counts.sum(axis=0), index=event_counts.columns, name="all")
    if group_by is None:
        return pd.DataFrame(group_all).T
    grouped_vals = event_counts.groupby(level=group_by).agg(list).map(sum)
    grouped_vals = pd.concat([grouped_vals.T, group_all], axis=1).T
    return grouped_vals


def event_features(events: pd.DataFrame,
                   feature: str,
                   group_by: Optional[Union[str, List[str]]] = cnst.STIMULUS,
                   ignore_events: List[cnst.EVENT_LABELS] = None) -> pd.DataFrame:
    """
    Extracts the required feature from events detected by each rater/detector, and groups the results by the given
    criteria if specified. Ignores the specified event-labels if provided.

    :param events: A DataFrame containing the detected events of each rater/detector.
    :param feature: The event feature to compare.
        Options: "duration", "amplitude", "azimuth", "peak velocity", "mean velocity", "mean pupil size"
    :param group_by: The criteria to group the contrast measure by.
    :param ignore_events: A set of event-labels to ignore during the contrast calculation.
    :return: A DataFrame containing the event features of all detected events (cols), grouped by the given criteria (rows).
    :raises NotImplementedError: If the comparison measure is unknown.
    """
    # TODO: replace "compare_by" with generic way to contrast event features
    events = events.map(lambda cell: hlp.drop_events(cell, to_drop=ignore_events))
    feature = feature.lower().replace("_", " ").replace("-", " ").strip()
    if feature in {"duration", "length"}:
        contrast = events.map(lambda cell: [e.duration for e in cell] if len(cell) else np.nan)
    elif feature in {"amplitude", "distance"}:
        contrast = events.map(
            lambda cell: [e.amplitude for e in cell] if len(cell) else np.nan)
    elif feature in {"azimuth", "direction"}:
        contrast = events.map(
            lambda cell: [e.azimuth for e in cell] if len(cell) else np.nan)
    elif feature in {"peak velocity", "max velocity"}:
        contrast = events.map(
            lambda cell: [e.peak_velocity for e in cell] if len(cell) else np.nan)
    elif feature in {"mean velocity", "avg velocity"}:
        contrast = events.map(
            lambda cell: [e.mean_velocity for e in cell] if len(cell) else np.nan)
    elif feature in {"mean pupil size", "pupil size"}:
        contrast = events.map(
            lambda cell: [e.mean_pupil_size for e in cell] if len(cell) else np.nan)
    else:
        raise NotImplementedError(f"Unknown contrast measure for matched events:\t{feature}")
    return group_and_aggregate(contrast, group_by)


def compare_samples(samples: pd.DataFrame,
                    metric: str,
                    group_by: Optional[Union[str, List[str]]] = cnst.STIMULUS,
                    ignore_events: List[cnst.EVENT_LABELS] = None) -> pd.DataFrame:
    """
    Calculate the comparison measure between the detected samples of each rater/detector pair, and group the results
    by the given criteria if specified. Ignore the specified event-labels during the contrast calculation.

    :param samples: A DataFrame containing the detected samples of each rater/detector.
    :param metric: The measure to calculate.
        Options:
            - "acc": Calculate the balanced accuracy between the sequence of labels.
            - "levenshtein": Calculate the (normalised) Levenshtein distance between the sequence of labels.
            - "kappa": Calculate the Cohen's Kappa coefficient between the sequence of labels.
            - "mcc": Calculate the Matthews correlation coefficient between the sequence of labels.
            - "frobenius": Calculate the Frobenius norm of the difference between the labels' transition matrices.
            - "kl": Calculate the Kullback-Leibler divergence between the labels' transition matrices.
    :param group_by: The criteria to group the contrast measure by.
    :param ignore_events: A set of event-labels to ignore during the contrast calculation.
    :return: A DataFrame containing the compared measure of all detected samples (cols), grouped by the given
        criteria (rows).
    :raises NotImplementedError: If the comparison measure is unknown.
    """
    samples = samples.map(lambda cell: hlp.drop_events(cell, to_drop=ignore_events))
    metric = metric.lower().replace("_", " ").replace("-", " ").strip()
    if metric == "acc" or metric == "accuracy" or metric == "balanced accuracy":
        contrast = _compare_columns(samples, metrics.balanced_accuracy)
    elif metric == "lev" or metric == "levenshtein":
        contrast = _compare_columns(samples, metrics.levenshtein_distance)
    elif metric == "kappa" or metric == "cohen kappa":
        contrast = _compare_columns(samples, metrics.cohen_kappa)
    elif metric == "mcc" or metric == "matthews correlation":
        contrast = _compare_columns(samples, metrics.matthews_correlation)
    elif metric == "fro" or metric == "frobenius" or metric == "l2":
        contrast = _compare_columns(samples,
                                    lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="fro"))
    elif metric == "kl" or metric == "kl divergence" or metric == "kullback leibler":
        contrast = _compare_columns(samples,
                                    lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="kl"))
    else:
        raise NotImplementedError(f"Unknown contrast measure for samples:\t{metric}")
    return group_and_aggregate(contrast, group_by)


def event_matching_ratio(events: pd.DataFrame,
                         match_by: str,
                         group_by: Optional[Union[str, List[str]]] = cnst.STIMULUS,
                         ignore_events: List[cnst.EVENT_LABELS] = None,
                         **match_kwargs) -> pd.DataFrame:
    """
    Match events between raters and detectors based on the given `match_by` criteria, and calculate the ratio of
    matched events to the total number of ground-truth events per trial (row) and detector/rater (column). Finally,
    group the results by the given `group_by` criteria if specified.
    Ignore the specified event-labels during the matching process.

    :param events: A DataFrame containing the detected events of each rater/detector.
    :param match_by: The matching criteria to use.
        Options: "first", "last", "max overlap", "longest match", "iou", "onset latency", "offset latency", "window"
    :param group_by: The criteria to group the results by.
    :param ignore_events: A set of event-labels to ignore during the matching process.
    :param match_kwargs: Additional keyword arguments to pass to the matching function.
    :return: A DataFrame containing the ratio of matched events
    """
    events = events.map(lambda cell: hlp.drop_events(cell, to_drop=ignore_events))
    matches = match_events(events, match_by, ignore_events, is_symmetric=True, **match_kwargs)
    event_counts = events.map(lambda cell: len(cell) if len(cell) else np.nan)
    match_counts = matches.map(lambda cell: len(cell) if pd.notnull(cell) else np.nan)
    ratios = np.zeros_like(match_counts, dtype=float)
    for i in range(match_counts.index.size):
        for j in range(match_counts.columns.size):
            gt_col, pred_col = match_counts.columns[j]
            ratios[i, j] = match_counts.iloc[i, j] / event_counts.iloc[i][gt_col]
    ratios = pd.DataFrame(ratios, index=match_counts.index, columns=match_counts.columns) * 100
    return group_and_aggregate(ratios, group_by)


def matched_events_feature_difference(matches: pd.DataFrame,
                                      feature: str,
                                      group_by: Optional[Union[str, List[str]]] = cnst.STIMULUS) -> pd.DataFrame:
    """
    Calculates the difference in feature values between each matched pair of events, and  groups the results by the
    given criteria if specified.

    :param matches: A DataFrame containing the matched events of each rater/detector.
    :param feature: The compared-measure to calculate.
        Options: "onset latency", "offset latency", "l2 timing jitter", "duration", "amplitude", "azimuth",
            "peak velocity", "mean velocity", "mean pupil size"
    :param group_by: The criteria to group the contrast measure by.
    :return: A DataFrame containing the comparison measure between matched events per trial (row) and detector/rater
        pair (column).
    :raises NotImplementedError: If the comparison measure is unknown.
    """
    # TODO: replace "feature" with generic way to contrast event features
    feature = feature.lower().replace("_", " ").replace("-", " ").strip()
    if feature in {"onset", "onset latency", "onset jitter"}:
        contrast = matches.map(
            lambda cell: [k.start_time - v.start_time for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    elif feature in {"offset", "offset latency", "offset jitter"}:
        contrast = matches.map(
            lambda cell: [k.end_time - v.end_time for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    elif feature in {"l2", "l2 timing", "l2 timing offset", "timing offset", "l2 timing jitter", "timing jitter"}:
        contrast = matches.map(
            lambda cell: [k.l2_timing_offset(v) for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    elif feature in {"duration", "length"}:
        contrast = matches.map(
            lambda cell: [k.duration - v.duration for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    elif feature in {"amplitude", "distance"}:
        contrast = matches.map(
            lambda cell: [k.amplitude - v.amplitude for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    elif feature in {"azimuth", "direction"}:
        contrast = matches.map(
            lambda cell: [k.azimuth - v.azimuth for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    elif feature in {"peak velocity", "max velocity"}:
        contrast = matches.map(
            lambda cell: [k.peak_velocity - v.peak_velocity for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    elif feature in {"mean velocity", "avg velocity"}:
        contrast = matches.map(
            lambda cell: [k.mean_velocity - v.mean_velocity for k, v in cell.items()] if pd.notnull(cell) else np.nan
        )
    elif feature in {"mean pupil size", "pupil size"}:
        contrast = matches.map(
            lambda cell: [k.mean_pupil_size - v.mean_pupil_size for k, v in cell.items()] if pd.notnull(
                cell) else np.nan
        )
    else:
        raise NotImplementedError(f"Unknown contrast measure for matched events:\t{feature}")
    return group_and_aggregate(contrast, group_by)


def match_events(events: pd.DataFrame,
                 match_by: str,
                 ignore_events: List[cnst.EVENT_LABELS] = None,
                 is_symmetric: bool = True,
                 **match_kwargs) -> pd.DataFrame:
    """
    Match events between raters and detectors based on the given matching criteria.
    Ignores the specified event-labels during the matching process.

    :param events: A DataFrame containing the detected events of each rater/detector.
    :param match_by: The matching criteria to use.
        Options: "first", "last", "max overlap", "longest match", "iou", "onset latency", "offset latency", "window"
    :param ignore_events: A set of event-labels to ignore during the matching process.
    :param is_symmetric: If true, only calculate the measure once for each (unordered-)pair of columns,
        e.g, (A, B) and (B, A) will be the same. If false, calculate the measure for all ordered-pairs of columns.
    :param match_kwargs: Additional keyword arguments to pass to the matching function.
    :return: A DataFrame containing the matched events per trial (row) and detector/rater pair (column).
    """
    events = events.map(lambda cell: hlp.drop_events(cell, to_drop=ignore_events))
    match_by = match_by.lower().replace("_", " ").replace("-", " ").strip()
    if match_by == "first" or match_by == "first overlap":
        return _compare_columns(events,
                                lambda seq1, seq2: EventMatcher.first_overlap(seq1, seq2, **match_kwargs),
                                is_symmetric=is_symmetric)
    if match_by == "last" or match_by == "last overlap":
        return _compare_columns(events,
                                lambda seq1, seq2: EventMatcher.last_overlap(seq1, seq2, **match_kwargs),
                                is_symmetric=is_symmetric)
    if match_by == "max" or match_by == "max overlap":
        return _compare_columns(events,
                                lambda seq1, seq2: EventMatcher.max_overlap(seq1, seq2, **match_kwargs),
                                is_symmetric=is_symmetric)
    if match_by == "longest" or match_by == "longest match":
        return _compare_columns(events,
                                lambda seq1, seq2: EventMatcher.longest_match(seq1, seq2, **match_kwargs),
                                is_symmetric=is_symmetric)
    if match_by == "iou" or match_by == "intersection over union":
        return _compare_columns(events, lambda seq1, seq2: EventMatcher.iou(seq1, seq2, **match_kwargs),
                                is_symmetric=is_symmetric)
    if match_by == "onset" or match_by == "onset latency":
        return _compare_columns(events,
                                lambda seq1, seq2: EventMatcher.onset_latency(seq1, seq2, **match_kwargs),
                                is_symmetric=is_symmetric)
    if match_by == "offset" or match_by == "offset latency":
        return _compare_columns(events,
                                lambda seq1, seq2: EventMatcher.offset_latency(seq1, seq2, **match_kwargs),
                                is_symmetric=is_symmetric)
    if match_by == "window" or match_by == "window based":
        return _compare_columns(events,
                                lambda seq1, seq2: EventMatcher.window_based(seq1, seq2, **match_kwargs),
                                is_symmetric=is_symmetric)
    return _compare_columns(events,
                            lambda seq1, seq2: EventMatcher.generic_matching(seq1, seq2, **match_kwargs),
                            is_symmetric=is_symmetric)


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


def _compare_columns(data: pd.DataFrame, measure: Callable, is_symmetric: bool = True) -> pd.DataFrame:
    """
    Calculate the compared-measure between all pairs of columns in the given data frame.

    :param data: The data frame to calculate the contrast measure on.
    :param measure: The function to calculate the contrast measure.
    :param is_symmetric: If true, only calculate the measure once for each (unordered-)pair of columns,
        e.g, (A, B) and (B, A) will be the same. If false, calculate the measure for all ordered-pairs of columns.
    :return: A data frame with the contrast measure between all pairs of columns.
    """
    if is_symmetric:
        column_pairs = list(itertools.combinations(data.columns, 2))
    else:
        column_pairs = list(itertools.product(data.columns, repeat=2))
        column_pairs = [pair for pair in column_pairs if pair[0] != pair[1]]
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
                res[idx][pair] = measure(vals1, vals2)
    res = pd.DataFrame.from_dict(res, orient="index")
    res.index.names = data.index.names
    return res
