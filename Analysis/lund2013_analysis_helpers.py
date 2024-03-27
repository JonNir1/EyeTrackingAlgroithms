import numpy as np
import pandas as pd
import itertools
from typing import Callable, Set

import Config.constants as cnst
import MetricCalculators.LevenshteinDistance as lev
import MetricCalculators.TransitionMatrix as tm
from GazeEvents.EventMatcher import EventMatcher


def calculate_distance(detected: pd.DataFrame,
                       distance: str,
                       ignore: Set[cnst.EVENT_LABELS] = None) -> pd.DataFrame:
    if ignore is not None and len(ignore) > 0:
        detected = detected.map(lambda cell: [e for e in cell if e not in ignore])
    distance = distance.lower().replace("_", " ").replace("-", " ").strip()
    if distance == "lev" or distance == "levenshtein":
        return _calculate_joint_measure(detected, lev.calculate_distance, is_ordered=False)
    if distance == "fro" or distance == "frobenius":
        transition_probabilities = detected.map(
            lambda cell: tm.transition_probabilities(cell) if all(cell.notnull()) else [np.nan])
        return _calculate_joint_measure(transition_probabilities,
                                        lambda m1, m2: tm.matrix_distance(m1, m2, norm="fro"),
                                        is_ordered=False)
    if distance == "kl" or distance == "kl divergence" or distance == "kullback leibler":
        transition_probabilities = detected.map(
            lambda cell: tm.transition_probabilities(cell) if all(cell.notnull()) else [np.nan])
        return _calculate_joint_measure(transition_probabilities,
                                        lambda m1, m2: tm.matrix_distance(m1, m2, norm="kl"),
                                        is_ordered=False)
    raise ValueError(f"Unknown distance measure: {distance}")


def _match_events(detected: pd.DataFrame, match_by: str, **match_kwargs) -> pd.DataFrame:
    match_by = match_by.lower().replace("_", " ").replace("-", " ").strip()
    if match_by == "first" or match_by == "first overlap":
        return _calculate_joint_measure(detected,
                                        lambda seq1, seq2: EventMatcher.first_overlap(seq1, seq2, **match_kwargs))
    if match_by == "last" or match_by == "last overlap":
        return _calculate_joint_measure(detected,
                                        lambda seq1, seq2: EventMatcher.last_overlap(seq1, seq2, **match_kwargs))
    if match_by == "max" or match_by == "max overlap":
        return _calculate_joint_measure(detected,
                                        lambda seq1, seq2: EventMatcher.max_overlap(seq1, seq2, **match_kwargs))
    if match_by == "longest" or match_by == "longest match":
        return _calculate_joint_measure(detected,
                                        lambda seq1, seq2: EventMatcher.longest_match(seq1, seq2, **match_kwargs))
    if match_by == "iou" or match_by == "intersection over union":
        return _calculate_joint_measure(detected, lambda seq1, seq2: EventMatcher.iou(seq1, seq2, **match_kwargs))
    if match_by == "onset" or match_by == "onset latency":
        return _calculate_joint_measure(detected,
                                        lambda seq1, seq2: EventMatcher.onset_latency(seq1, seq2, **match_kwargs))
    if match_by == "offset" or match_by == "offset latency":
        return _calculate_joint_measure(detected,
                                        lambda seq1, seq2: EventMatcher.offset_latency(seq1, seq2, **match_kwargs))
    if match_by == "window" or match_by == "window based":
        return _calculate_joint_measure(detected,
                                        lambda seq1, seq2: EventMatcher.window_based(seq1, seq2, **match_kwargs))
    return _calculate_joint_measure(detected, lambda seq1, seq2: EventMatcher.generic_matcher(seq1, seq2, **match_kwargs))


def _calculate_joint_measure(data: pd.DataFrame, measure: Callable, is_ordered: bool = True) -> pd.DataFrame:
    if is_ordered:
        column_pairs = list(itertools.product(data.columns, repeat=2))
    else:
        column_pairs = list(itertools.combinations_with_replacement(data.columns, 2))
    res = {}
    for idx in data.index:
        res[idx] = {}
        for pair in column_pairs:
            vals1, vals2 = data.loc[idx, pair[0]], data.loc[idx, pair[1]]
            if pd.isnull(vals1).all() or pd.isnull(vals2).all():
                res[idx][pair] = np.nan
            elif pd.isnull(vals1).any() or pd.isnull(vals2).any():
                raise AssertionError("Missing values in detected sequences.")
            else:
                res[idx][pair] = measure(vals1, vals2)
    res = pd.DataFrame.from_dict(res, orient="index")
    res.index.names = data.index.names
    return res
