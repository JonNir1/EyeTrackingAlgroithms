import itertools
import numpy as np
import pandas as pd
from typing import List, Callable, Optional

import Config.constants as cnst
from GazeDetectors.BaseDetector import BaseDetector
from DataSetLoaders.DataSetFactory import DataSetFactory
from GazeEvents.EventMatcher import EventMatcher
import GazeEvents.helpers as hlp
import Analysis.metrics as metrics


class DetectorComparisonCalculator:

    def __init__(self, dataset_name: str, raters: List[str], detectors: List[BaseDetector]):
        self._dataset_name = dataset_name
        self._raters = [r.upper() for r in raters]
        self._detectors = detectors
        samples_df, events_df = DataSetFactory.load_and_process(self._dataset_name, self._raters, self._detectors)
        self._detected_samples = samples_df
        self._detected_events = events_df

    def compare_samples(self,
                        compare_by: str,
                        group_by: Optional[str] = cnst.STIMULUS,
                        ignore_events: List[cnst.EVENT_LABELS] = None) -> pd.DataFrame:
        """
        Calculate the comparison measure between the detected samples of each rater/detector pair, and group the results
        by the given criteria if specified. Ignore the specified event-labels during the contrast calculation.

        :param compare_by: The measure to calculate.
            Options:
                - "levenshtein": Calculate the Levenshtein distance between the sequence of labels.
                - "kappa": Calculate the Cohen's Kappa coefficient between the sequence of labels.
                - "frobenius": Calculate the Frobenius norm of the difference between the labels' transition matrices.
                - "kl": Calculate the Kullback-Leibler divergence between the labels' transition matrices.
        :param group_by: The criteria to group the contrast measure by.
        :param ignore_events: A set of event-labels to ignore during the contrast calculation.
        :return: A DataFrame containing the contrast measure between the detected samples per trial (row) and
            detector/rater pair (column).
        :raises NotImplementedError: If the contrast measure is unknown.
        """
        samples = self._detected_samples.map(lambda cell: hlp.drop_events(cell, to_drop=ignore_events))
        compare_by = compare_by.lower().replace("_", " ").replace("-", " ").strip()
        if compare_by == "lev" or compare_by == "levenshtein":
            contrast = self._compare_columns(samples, metrics.levenshtein_distance)
        elif compare_by == "kappa" or compare_by == "cohen kappa":
            contrast = self._compare_columns(samples, metrics.cohen_kappa)
        elif compare_by == "fro" or compare_by == "frobenius" or compare_by == "l2":
            contrast = self._compare_columns(samples,
                                             lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="fro"))
        elif compare_by == "kl" or compare_by == "kl divergence" or compare_by == "kullback leibler":
            contrast = self._compare_columns(samples,
                                             lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="kl"))
        else:
            raise NotImplementedError(f"Unknown contrast measure for samples:\t{compare_by}")

        if group_by is None:
            return contrast
        # Group by stimulus and calculate add row "all" (all stimuli)
        grouped_diffs = contrast.groupby(level=group_by).agg(list)
        all_stim = pd.Series([list(grouped_diffs[col].explode()) for col in grouped_diffs.columns],
                             index=grouped_diffs.columns, name="all")
        grouped_diffs = pd.concat([grouped_diffs.T, all_stim], axis=1).T
        return grouped_diffs

    def event_matching_ratio(self,
                             match_by: str,
                             group_by: Optional[str] = cnst.STIMULUS,
                             ignore_events: List[cnst.EVENT_LABELS] = None,
                             **match_kwargs) -> pd.DataFrame:
        """
        Match events between raters and detectors based on the given `match_by` criteria, and calculate the ratio of
        matched events to the total number of ground-truth events per trial (row) and detector/rater (column). Finally,
        group the results by the given `group_by` criteria if specified.
        Ignore the specified event-labels during the matching process.
        :param match_by: The matching criteria to use.
            Options: "first", "last", "max overlap", "longest match", "iou", "onset latency", "offset latency", "window"
        :param group_by: The criteria to group the results by.
        :param ignore_events: A set of event-labels to ignore during the matching process.
        :param match_kwargs: Additional keyword arguments to pass to the matching function.
        :return: A DataFrame containing the ratio of matched events
        """
        events = self._detected_events.map(lambda cell: hlp.drop_events(cell, to_drop=ignore_events))
        matches = self.match_events(match_by, ignore_events, is_symmetric=True, **match_kwargs)
        event_counts = events.map(lambda cell: len(cell) if len(cell) else np.nan)
        match_counts = matches.map(lambda cell: len(cell) if pd.notnull(cell) else np.nan)
        ratios = np.zeros_like(match_counts, dtype=float)
        for i in range(match_counts.index.size):
            for j in range(match_counts.columns.size):
                gt_col, pred_col = match_counts.columns[j]
                ratios[i, j] = match_counts.iloc[i, j] / event_counts.iloc[i][gt_col]
        ratios = pd.DataFrame(ratios, index=match_counts.index, columns=match_counts.columns) * 100

        if group_by is None:
            return ratios
        # Group by stimulus and calculate add row "all" (all stimuli)
        grouped_ratios = ratios.groupby(level=group_by).agg(list)
        all_stim = pd.Series([list(grouped_ratios[col].explode()) for col in grouped_ratios.columns],
                             index=grouped_ratios.columns, name="all")
        grouped_ratios = pd.concat([grouped_ratios.T, all_stim], axis=1).T
        return grouped_ratios

    def compare_matched_events(self,
                               match_by: str,
                               compare_by: str,
                               group_by: Optional[str] = cnst.STIMULUS,
                               ignore_events: List[cnst.EVENT_LABELS] = None,
                               **match_kwargs) -> pd.DataFrame:
        """
        Match events between raters and detectors based on the given matching criteria, while ignoring the specified
        event-labels. The compare measure is then calculated between each matched pair of events, and finally grouped
        by the given criteria if specified.

        :param match_by: The matching criteria to use.
            Options: "first", "last", "max overlap", "longest match", "iou", "onset latency", "offset latency", "window"
        :param compare_by: The compared-measure to calculate.
            Options: "onset latency", "offset latency", "duration", "amplitude"
        :param group_by: The criteria to group the contrast measure by.
        :param ignore_events: A set of event-labels to ignore during the matching process.
        :param match_kwargs: Additional keyword arguments to pass to the matching function.
        :return: A DataFrame containing the contrast measure between matched events per trial (row) and detector/rater
            pair (column).
        :raises NotImplementedError: If the contrast measure is unknown.
        """
        # TODO: replace "contrast_by" with generic way to contrast event features
        matches = self.match_events(match_by, ignore_events, is_symmetric=True, **match_kwargs)
        compare_by = compare_by.lower().replace("_", " ").replace("-", " ").strip()
        if compare_by in {"onset", "onset latency", "onset jitter"}:
            contrast = matches.map(
                lambda cell: [k.start_time - v.start_time for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif compare_by in {"offset", "offset latency", "offset jitter"}:
            contrast = matches.map(
                lambda cell: [k.end_time - v.end_time for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif compare_by in {"duration", "length"}:
            contrast = matches.map(
                lambda cell: [k.duration - v.duration for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif compare_by in {"amplitude", "distance"}:
            contrast = matches.map(
                lambda cell: [k.amplitude - v.amplitude for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        else:
            raise NotImplementedError(f"Unknown contrast measure for matched events:\t{compare_by}")

        if group_by is None:
            return contrast
        # Group by stimulus and calculate add row "all" (all stimuli)
        grouped_diffs = contrast.groupby(level=group_by).agg(lambda cell: pd.Series(cell).explode())
        all_stim = pd.Series([list(grouped_diffs[col].explode()) for col in grouped_diffs.columns],
                             index=grouped_diffs.columns, name="all")
        grouped_diffs = pd.concat([grouped_diffs.T, all_stim], axis=1).T
        return grouped_diffs

    def match_events(self,
                     match_by: str,
                     ignore_events: List[cnst.EVENT_LABELS] = None,
                     is_symmetric: bool = True,
                     **match_kwargs) -> pd.DataFrame:
        """
        Match events between raters and detectors based on the given matching criteria.
        Ignores the specified event-labels during the matching process.
        :param match_by: The matching criteria to use.
            Options: "first", "last", "max overlap", "longest match", "iou", "onset latency", "offset latency", "window"
        :param ignore_events: A set of event-labels to ignore during the matching process.
        :param is_symmetric: If true, only calculate the measure once for each (unordered-)pair of columns,
            e.g, (A, B) and (B, A) will be the same. If false, calculate the measure for all ordered-pairs of columns.
        :param match_kwargs: Additional keyword arguments to pass to the matching function.
        :return: A DataFrame containing the matched events per trial (row) and detector/rater pair (column).
        """
        events = self._detected_events.map(lambda cell: hlp.drop_events(cell, to_drop=ignore_events))
        match_by = match_by.lower().replace("_", " ").replace("-", " ").strip()
        if match_by == "first" or match_by == "first overlap":
            return self._compare_columns(events,
                                         lambda seq1, seq2: EventMatcher.first_overlap(seq1, seq2, **match_kwargs),
                                         is_symmetric=is_symmetric)
        if match_by == "last" or match_by == "last overlap":
            return self._compare_columns(events,
                                         lambda seq1, seq2: EventMatcher.last_overlap(seq1, seq2, **match_kwargs),
                                         is_symmetric=is_symmetric)
        if match_by == "max" or match_by == "max overlap":
            return self._compare_columns(events,
                                         lambda seq1, seq2: EventMatcher.max_overlap(seq1, seq2, **match_kwargs),
                                         is_symmetric=is_symmetric)
        if match_by == "longest" or match_by == "longest match":
            return self._compare_columns(events,
                                         lambda seq1, seq2: EventMatcher.longest_match(seq1, seq2, **match_kwargs),
                                         is_symmetric=is_symmetric)
        if match_by == "iou" or match_by == "intersection over union":
            return self._compare_columns(events, lambda seq1, seq2: EventMatcher.iou(seq1, seq2, **match_kwargs),
                                         is_symmetric=is_symmetric)
        if match_by == "onset" or match_by == "onset latency":
            return self._compare_columns(events,
                                         lambda seq1, seq2: EventMatcher.onset_latency(seq1, seq2, **match_kwargs),
                                         is_symmetric=is_symmetric)
        if match_by == "offset" or match_by == "offset latency":
            return self._compare_columns(events,
                                         lambda seq1, seq2: EventMatcher.offset_latency(seq1, seq2, **match_kwargs),
                                         is_symmetric=is_symmetric)
        if match_by == "window" or match_by == "window based":
            return self._compare_columns(events,
                                         lambda seq1, seq2: EventMatcher.window_based(seq1, seq2, **match_kwargs),
                                         is_symmetric=is_symmetric)
        return self._compare_columns(events,
                                     lambda seq1, seq2: EventMatcher.generic_matcher(seq1, seq2, **match_kwargs),
                                     is_symmetric=is_symmetric)

    @staticmethod
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
        if not isinstance(other, DetectorComparisonCalculator):
            return False
        if self.dataset_name != other.dataset_name:
            return False
        if self.raters != other.raters:
            return False
        if self.detectors != other.detectors:
            return False
        return True
