import itertools
import numpy as np
import pandas as pd
from typing import List, Callable

import Config.constants as cnst
from GazeDetectors.BaseDetector import BaseDetector
from DataSetLoaders.DataSetFactory import DataSetFactory
from GazeEvents.EventMatcher import EventMatcher
import GazeEvents.helpers as hlp
import MetricCalculators.levenshtein_distance as lev
import MetricCalculators.transition_matrix as tm


class DetectorContrastCalculator:

    def __init__(self, dataset_name: str, raters: List[str], detectors: List[BaseDetector]):
        self._dataset_name = dataset_name
        self._raters = [r.upper() for r in raters]
        self._detectors = detectors
        samples_df, events_df = DataSetFactory.load_and_process(self._dataset_name, self._raters, self._detectors)
        self._detected_samples = samples_df
        self._detected_events = events_df

    def contrast_samples(self, contrast_by: str, ignore_events: List[cnst.EVENT_LABELS] = None) -> pd.DataFrame:
        samples = self._detected_samples.map(lambda cell: hlp.drop_events(cell, to_drop=ignore_events))
        contrast_by = contrast_by.lower().replace("_", " ").replace("-", " ").strip()
        if contrast_by == "lev" or contrast_by == "levenshtein":
            return self._contrast_columns(samples, lev.calculate_distance, is_symmetric=True)
        if contrast_by == "fro" or contrast_by == "frobenius" or contrast_by == "l2":
            transition_probabilities = samples.map(
                lambda cell: tm.transition_probabilities(cell) if pd.notnull(cell).all() else [np.nan]
            )
            return self._contrast_columns(transition_probabilities,
                                          lambda m1, m2: tm.matrix_distance(m1, m2, norm="fro"),
                                          is_symmetric=True)
        if contrast_by == "kl" or contrast_by == "kl divergence" or contrast_by == "kullback leibler":
            transition_probabilities = samples.map(
                lambda cell: tm.transition_probabilities(cell) if pd.notnull(cell).all() else [np.nan]
            )
            return self._contrast_columns(transition_probabilities,
                                          lambda m1, m2: tm.matrix_distance(m1, m2, norm="kl"),
                                          is_symmetric=True)
        raise ValueError(f"Unknown contrast measure for samples:\t{contrast_by}")

    def event_matching_ratio(self,
                             match_by: str,
                             ignore_events: List[cnst.EVENT_LABELS] = None,
                             **match_kwargs) -> pd.DataFrame:
        events = self._detected_events.map(lambda cell: hlp.drop_events(cell, to_drop=ignore_events))
        matches = self.match_events(match_by, ignore_events, **match_kwargs)
        event_counts = events.map(lambda cell: len(cell) if len(cell) else np.nan)
        match_counts = matches.map(lambda cell: len(cell) if pd.notnull(cell) else np.nan)
        ratios = np.zeros_like(match_counts, dtype=float)
        for i in range(match_counts.index.size):
            for j in range(match_counts.columns.size):
                gt_col, pred_col = match_counts.columns[j]
                ratios[i, j] = match_counts.iloc[i, j] / event_counts.iloc[i][gt_col]
        ratios = pd.DataFrame(ratios, index=match_counts.index, columns=match_counts.columns)
        return ratios * 100

    def contrast_matched_events(self,
                                match_by: str,
                                contrast_by: str,
                                ignore_events: List[cnst.EVENT_LABELS] = None,
                                **match_kwargs) -> pd.DataFrame:
        matches = self.match_events(match_by, ignore_events, **match_kwargs)
        contrast_by = contrast_by.lower().replace("_", " ").replace("-", " ").strip()
        if contrast_by in {"onset", "onset latency", "onset jitter"}:
            onset_diffs = matches.map(
                lambda cell: [k.start_time - v.start_time for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
            return onset_diffs
        if contrast_by in {"offset", "offset latency", "offset jitter"}:
            offset_diffs = matches.map(
                lambda cell: [k.end_time - v.end_time for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
            return offset_diffs
        if contrast_by in {"duration", "length"}:
            duration_diffs = matches.map(
                lambda cell: [k.duration - v.duration for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        raise ValueError(f"Unknown contrast measure for matched events:\t{contrast_by}")

    def match_events(self, match_by: str,
                     ignore_events: List[cnst.EVENT_LABELS] = None,
                     **match_kwargs) -> pd.DataFrame:
        events = self._detected_events.map(lambda cell: hlp.drop_events(cell, to_drop=ignore_events))
        match_by = match_by.lower().replace("_", " ").replace("-", " ").strip()
        if match_by == "first" or match_by == "first overlap":
            return self._contrast_columns(events,
                                          lambda seq1, seq2: EventMatcher.first_overlap(seq1, seq2, **match_kwargs),
                                          is_symmetric=False)
        if match_by == "last" or match_by == "last overlap":
            return self._contrast_columns(events,
                                          lambda seq1, seq2: EventMatcher.last_overlap(seq1, seq2, **match_kwargs),
                                          is_symmetric=False)
        if match_by == "max" or match_by == "max overlap":
            return self._contrast_columns(events,
                                          lambda seq1, seq2: EventMatcher.max_overlap(seq1, seq2, **match_kwargs),
                                          is_symmetric=False)
        if match_by == "longest" or match_by == "longest match":
            return self._contrast_columns(events,
                                          lambda seq1, seq2: EventMatcher.longest_match(seq1, seq2, **match_kwargs),
                                          is_symmetric=False)
        if match_by == "iou" or match_by == "intersection over union":
            return self._contrast_columns(events,
                                          lambda seq1, seq2: EventMatcher.iou(seq1, seq2, **match_kwargs),
                                          is_symmetric=False)
        if match_by == "onset" or match_by == "onset latency":
            return self._contrast_columns(events,
                                          lambda seq1, seq2: EventMatcher.onset_latency(seq1, seq2, **match_kwargs),
                                          is_symmetric=False)
        if match_by == "offset" or match_by == "offset latency":
            return self._contrast_columns(events,
                                          lambda seq1, seq2: EventMatcher.offset_latency(seq1, seq2, **match_kwargs),
                                          is_symmetric=False)
        if match_by == "window" or match_by == "window based":
            return self._contrast_columns(events,
                                          lambda seq1, seq2: EventMatcher.window_based(seq1, seq2, **match_kwargs),
                                          is_symmetric=False)
        return self._contrast_columns(events,
                                      lambda seq1, seq2: EventMatcher.generic_matcher(seq1, seq2, **match_kwargs),
                                      is_symmetric=False)

    @staticmethod
    def _contrast_columns(data: pd.DataFrame, measure: Callable, is_symmetric: bool = True) -> pd.DataFrame:
        """
        Calculate the contrast measure between all pairs of columns in the given data frame.
        :param data: The data frame to calculate the contrast measure on.
        :param measure: The function to calculate the contrast measure.
        :param is_symmetric: If true, only calculate the measure once for each (unordered-)pair of columns,
            e.g, (A, B) and (B, A) will be the same. If false, calculate the measure for all ordered-pairs of columns.
        :return: A data frame with the contrast measure between all pairs of columns.
        """
        if is_symmetric:
            column_pairs = list(itertools.combinations_with_replacement(data.columns, 2))
        else:
            column_pairs = list(itertools.product(data.columns, repeat=2))
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

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def raters(self):
        return self._raters

    @property
    def detectors(self):
        return self._detectors

    def __eq__(self, other):
        if not isinstance(other, DetectorContrastCalculator):
            return False
        if self.dataset_name != other.dataset_name:
            return False
        if self.raters != other.raters:
            return False
        if self.detectors != other.detectors:
            return False
        return True
