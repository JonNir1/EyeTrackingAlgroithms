import numpy as np
import pandas as pd
import warnings
import itertools
from typing import Callable

import Config.constants as cnst
import DataSetLoaders.Lund2013DataSetLoader
from GazeEvents.EventFactory import EventFactory as EF
import MetricCalculators.LevenshteinDistance as lev
import MetricCalculators.TransitionMatrix as tm
import MetricCalculators.EventMatching as em

_DATASET = DataSetLoaders.Lund2013DataSetLoader.Lund2013DataSetLoader.load(should_save=False)
_INDEX_NAMES = [cnst.TRIAL, cnst.SUBJECT_ID, cnst.STIMULUS, f"{cnst.STIMULUS}_name"]


def detect(*detectors) -> (pd.DataFrame, pd.DataFrame):
    """
    Detects events in the Lund 2013 dataset using the provided detectors and the human-annotations in the dataset.
    Returns two dataframes:
         - The sequence of event-labels per sample for each trial and detector.
         - The sequence of event objects for each trial and detector.
    """
    samples_dict, event_dict = {}, {}
    for trial_num in _DATASET[cnst.TRIAL].unique():
        trial_data = _DATASET[_DATASET[cnst.TRIAL] == trial_num]
        labels, events = _detect_trial(trial_data, *detectors)
        subject_id, stimulus, stimulus_name = trial_data[_INDEX_NAMES[1:]].iloc[0]
        samples_dict[(trial_num, subject_id, stimulus, stimulus_name)] = labels
        event_dict[(trial_num, subject_id, stimulus, stimulus_name)] = events
    # create output dataframes
    samples_df = pd.DataFrame.from_dict(samples_dict, orient="index")
    samples_df.index.names = _INDEX_NAMES
    events_df = pd.DataFrame.from_dict(event_dict, orient="index")
    events_df.index.names = _INDEX_NAMES
    return samples_df, events_df


def event_matching(detected: pd.DataFrame, match_by: str, **match_kwargs) -> pd.DataFrame:
    match_by = match_by.lower().replace("_", " ").strip()
    if match_by == "first" or match_by == "first overlap":
        return _calculate_distance(detected, lambda seq1, seq2: em.first_overlap_matching(seq1, seq2, **match_kwargs))
    if match_by == "last" or match_by == "last overlap":
        return _calculate_distance(detected, lambda seq1, seq2: em.last_overlap_matching(seq1, seq2, **match_kwargs))
    if match_by == "max" or match_by == "max overlap":
        return _calculate_distance(detected, lambda seq1, seq2: em.max_overlap_matching(seq1, seq2, **match_kwargs))
    if match_by == "longest" or match_by == "longest match":
        return _calculate_distance(detected, lambda seq1, seq2: em.longest_match_matching(seq1, seq2, **match_kwargs))
    if match_by == "iou" or match_by == "intersection over union":
        return _calculate_distance(detected, lambda seq1, seq2: em.iou_matching(seq1, seq2, **match_kwargs))
    if match_by == "onset" or match_by == "onset latency":
        return _calculate_distance(detected, lambda seq1, seq2: em.onset_latency_matching(seq1, seq2, **match_kwargs))
    if match_by == "offset" or match_by == "offset latency":
        return _calculate_distance(detected, lambda seq1, seq2: em.offset_latency_matching(seq1, seq2, **match_kwargs))
    if match_by == "window" or match_by == "window based":
        return _calculate_distance(detected, lambda seq1, seq2: em.window_based_matching(seq1, seq2, **match_kwargs))
    return _calculate_distance(detected, lambda seq1, seq2: em.generic_matching(seq1, seq2, **match_kwargs))


def levenshtein_distances(detected: pd.DataFrame) -> pd.DataFrame:
    """ Calculate the levenshtein distance between all pairs of detectors in every trial. """
    return _calculate_distance(detected, lev.calculate_distance)


def transition_distances__frobinius(detected: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the transition-matrix of each event detector within each trial and calculate the frobinius distance
    between all pairs of detectors for every trial.
    """
    transition_probabilities = detected.map(lambda cell: tm.transition_probabilities(cell) if all(cell.notnull()) else [np.nan])
    return _calculate_distance(transition_probabilities, lambda m1, m2: tm.matrix_distance(m1, m2, norm="fro"))


def transition_distances__kl_divergence(detected: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the transition-matrix of each event detector within each trial and calculate the KL-divergence distance
    between all pairs of detectors for every trial.
    """
    transition_probabilities = detected.map(lambda cell: tm.transition_probabilities(cell) if all(cell.notnull()) else [np.nan])
    return _calculate_distance(transition_probabilities, lambda m1, m2: tm.matrix_distance(m1, m2, norm="kl"))


def _detect_trial(trial_data, *detectors):
    viewer_distance = trial_data["viewer_distance_cm"].to_numpy()[0]
    pixel_size = trial_data["pixel_size_cm"].to_numpy()[0]
    with warnings.catch_warnings(action="ignore"):
        labels = {
            "MN": trial_data["MN"],
            "RA": trial_data["RA"],
        }
        events = {
            "MN": EF.make_from_gaze_data(trial_data, vd=viewer_distance, ps=pixel_size,
                                         column_mapping={"MN": cnst.EVENT}),
            "RA": EF.make_from_gaze_data(trial_data, vd=viewer_distance, ps=pixel_size,
                                         column_mapping={"RA": cnst.EVENT})
        }
    for det in detectors:
        with warnings.catch_warnings(action="ignore"):
            res = det.detect(t=trial_data[cnst.T].to_numpy(),
                             x=trial_data[cnst.X].to_numpy(),
                             y=trial_data[cnst.Y].to_numpy(),
                             vd=viewer_distance,
                             ps=pixel_size)
        labels[det.name] = res[cnst.GAZE][cnst.EVENT]
        events[det.name] = res[cnst.EVENTS]
    return labels, events


def _calculate_distance(data: pd.DataFrame, distance_func: Callable) -> pd.DataFrame:
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
                res[idx][pair] = distance_func(vals1, vals2)
    res = pd.DataFrame.from_dict(res, orient="index")
    res.index.names = data.index.names
    return res
