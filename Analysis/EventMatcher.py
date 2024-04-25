from abc import ABC
from typing import Set, Sequence, Dict, Union, Callable

import pandas as pd

import Config.experiment_config as cnfg
import Analysis.helpers as hlp
from GazeEvents.BaseEvent import BaseEvent


class EventMatcher(ABC):
    """
    Implementation of different methods to match two sequences of gaze-events, that may have been detected by different
    human annotators or detection algorithms, as discussed in section "Event Matching Methods" in the article:
        Startsev, M., Zemblys, R. Evaluating Eye Movement Event Detection: A Review of the State of the Art
        Behav Res 55, 1653–1714 (2023). https://doi.org/10.3758/s13428-021-01763-7
    """
    __TYPE_EVENT_MATCHES = Dict[BaseEvent, Union[BaseEvent, Sequence[BaseEvent]]]
    __TYPE_EVENT_MATCHING_FUNC = Callable[[Sequence[BaseEvent], Sequence[BaseEvent]], __TYPE_EVENT_MATCHES]

    @staticmethod
    def match_events(events: pd.DataFrame,
                     match_by: str,
                     ignore_events: Set[cnfg.EVENT_LABELS] = None,
                     is_symmetric: bool = True,
                     **match_kwargs) -> pd.DataFrame:
        """
        Match events based on the given matching criteria, ignoring specified event-labels.
        Matches can be one-to-one or one-to-many depending on the matching criteria and the specified parameters.

        :param events: A DataFrame containing detected events from different raters/detectors (columns) during
            different experimental trials (rows).
        :param match_by: The matching criteria to use. See options in the documentation of `generic_matching` method.
        :param ignore_events: A set of event-labels to ignore during the matching process.
        :param is_symmetric: If true, only calculate the measure once for each (unordered-)pair of columns,
            e.g, (A, B) and (B, A) will be the same. If false, calculate the measure for all ordered-pairs of columns.
        :param match_kwargs: Additional keyword arguments to pass to the matching function.
        :return: A DataFrame where each row corresponds to a trial and each column corresponds to a pair of detectors/raters.
            The cells of the DataFrame contain a dictionary matching each event from the first detector/rater to the second one.
        """
        ignore_events = ignore_events or set()
        events = events.map(lambda cell: [e for e in cell if e.event_label not in ignore_events])
        match_by = match_by.lower().replace("_", " ").replace("-", " ").strip()
        if match_by == "first" or match_by == "first overlap":
            return EventMatcher._apply_on_column_pairs(events,
                                                       lambda seq1, seq2: EventMatcher.first_overlap(seq1,
                                                                                                     seq2,
                                                                                                     **match_kwargs),
                                                       is_symmetric=is_symmetric)
        if match_by == "last" or match_by == "last overlap":
            return EventMatcher._apply_on_column_pairs(events,
                                                       lambda seq1, seq2: EventMatcher.last_overlap(seq1,
                                                                                                    seq2,
                                                                                                    **match_kwargs),
                                                       is_symmetric=is_symmetric)
        if match_by == "max" or match_by == "max overlap":
            return EventMatcher._apply_on_column_pairs(events,
                                                       lambda seq1, seq2: EventMatcher.max_overlap(seq1,
                                                                                                   seq2,
                                                                                                   **match_kwargs),
                                                       is_symmetric=is_symmetric)
        if match_by == "longest" or match_by == "longest match":
            return EventMatcher._apply_on_column_pairs(events,
                                                       lambda seq1, seq2: EventMatcher.longest_match(seq1,
                                                                                                     seq2,
                                                                                                     **match_kwargs),
                                                       is_symmetric=is_symmetric)
        if match_by == "iou" or match_by == "intersection over union":
            return EventMatcher._apply_on_column_pairs(events,
                                                       lambda seq1, seq2: EventMatcher.iou(seq1, seq2, **match_kwargs),
                                                       is_symmetric=is_symmetric)
        if match_by == "onset" or match_by == "onset latency":
            return EventMatcher._apply_on_column_pairs(events,
                                                       lambda seq1, seq2: EventMatcher.onset_latency(seq1,
                                                                                                     seq2,
                                                                                                     **match_kwargs),
                                                       is_symmetric=is_symmetric)
        if match_by == "offset" or match_by == "offset latency":
            return EventMatcher._apply_on_column_pairs(events,
                                                       lambda seq1, seq2: EventMatcher.offset_latency(seq1,
                                                                                                      seq2,
                                                                                                      **match_kwargs),
                                                       is_symmetric=is_symmetric)
        if match_by == "window" or match_by == "window based":
            return EventMatcher._apply_on_column_pairs(events,
                                                       lambda seq1, seq2: EventMatcher.window_based(seq1,
                                                                                                    seq2,
                                                                                                    **match_kwargs),
                                                       is_symmetric=is_symmetric)
        return EventMatcher._apply_on_column_pairs(events,
                                                   lambda seq1, seq2: EventMatcher.generic_matching(seq1,
                                                                                                    seq2,
                                                                                                    **match_kwargs),
                                                   is_symmetric=is_symmetric)

    @staticmethod
    def generic_matching(ground_truth: Sequence[BaseEvent],
                         predictions: Sequence[BaseEvent],
                         allow_cross_matching: bool,
                         min_overlap: float = - float("inf"),
                         min_iou: float = - float("inf"),
                         max_l2_timing_offset: float = float("inf"),
                         max_onset_latency: float = float("inf"),
                         max_offset_latency: float = float("inf"),
                         reduction: str = "all") -> __TYPE_EVENT_MATCHES:
        """
        Match each ground-truth event to a predicted event(s) that satisfies the specified criteria.

        :param ground_truth: sequence of ground-truth events
        :param predictions: sequence of predicted events
        :param allow_cross_matching: if True, a ground-truth event can match a predicted event of a different type
        :param min_overlap: minimum overlap time (in ms) to consider a possible match
        :param min_iou: minimum intersection-over-union to consider a possible match
        :param max_l2_timing_offset: maximum L2-timing-offset to consider a possible match
        :param max_onset_latency: maximum absolute difference (in ms) between the start times of the GT and predicted events
        :param max_offset_latency: maximum absolute difference (in ms) between the end times of the GT and predicted events
        :param reduction: name of reduction function used to choose a predicted event(s) from multiple matching ones:
            - 'all': return all matched events
            - 'first': return the first matched event
            - 'last': return the last matched event
            - 'longest': return the longest matched event
            - 'max overlap': return the matched event with maximum overlap with the GT event
            - 'iou': return the matched event with the maximum intersection-over-union with the GT event
            - 'onset latency': return the matched event with the least onset latency
            - 'offset latency': return the matched event with the least offset latency
        :return: dictionary, where keys are ground-truth events and values are their matched predicted event(s)
        :raises NotImplementedError: if the reduction function is not implemented
        """
        reduction = reduction.lower().replace("_", " ").replace("-", " ").strip()
        matches = {}
        matched_predictions = set()
        for gt in ground_truth:
            unmatched_predictions = [p for p in predictions if p not in matched_predictions]
            possible_matches = EventMatcher.__find_matches(gt=gt,
                                                           predictions=unmatched_predictions,
                                                           allow_cross_matching=allow_cross_matching,
                                                           min_overlap=min_overlap,
                                                           min_iou=min_iou,
                                                           max_l2_timing_offset=max_l2_timing_offset,
                                                           max_onset_latency=max_onset_latency,
                                                           max_offset_latency=max_offset_latency)
            p = EventMatcher.__choose_match(gt, possible_matches, reduction)
            if len(p):
                matches[gt] = p
            if reduction != "all":
                # If reduction is not 'all', cannot allow multiple matches for the same prediction
                matched_predictions.update(p)

        # verify output integrity
        if reduction != "all":
            assert all(len(v) == 1 for v in matches.values()), "Multiple matches for a GT event"
            matches = {k: v[0] for k, v in matches.items()}
            assert len(matches.values()) == len(set(matches.values())), "Matched predictions are not unique"
        return matches

    @staticmethod
    def first_overlap(ground_truth: Sequence[BaseEvent],
                      predictions: Sequence[BaseEvent],
                      min_overlap: float = 0,
                      allow_cross_matching: bool = True) -> Dict[BaseEvent, BaseEvent]:
        """
        Matches the first predicted event that overlaps with each ground-truth event, above a minimal overlap time.
        """
        return EventMatcher.generic_matching(ground_truth, predictions, allow_cross_matching, min_overlap=min_overlap,
                                             reduction="first")

    @staticmethod
    def last_overlap(ground_truth: Sequence[BaseEvent],
                     predictions: Sequence[BaseEvent],
                     min_overlap: float = 0,
                     allow_cross_matching: bool = True) -> Dict[BaseEvent, BaseEvent]:
        """
        Matches the last predicted event that overlaps with each ground-truth event, above a minimal overlap time.
        """
        return EventMatcher.generic_matching(ground_truth, predictions, allow_cross_matching, min_overlap=min_overlap,
                                             reduction="last")

    @staticmethod
    def longest_match(ground_truth: Sequence[BaseEvent],
                      predictions: Sequence[BaseEvent],
                      min_overlap: float = 0,
                      allow_cross_matching: bool = True) -> Dict[BaseEvent, BaseEvent]:
        """
        Matches the longest predicted event that overlaps with each ground-truth event, above a minimal overlap time.
        """
        return EventMatcher.generic_matching(ground_truth, predictions, allow_cross_matching, min_overlap=min_overlap,
                                             reduction="longest")

    @staticmethod
    def max_overlap(ground_truth: Sequence[BaseEvent],
                    predictions: Sequence[BaseEvent],
                    min_overlap: float = 0,
                    allow_cross_matching: bool = True) -> Dict[BaseEvent, BaseEvent]:
        """
        Matches the predicted event with maximum overlap with each ground-truth event, above a minimal overlap time.
        """
        return EventMatcher.generic_matching(ground_truth, predictions, allow_cross_matching, min_overlap=min_overlap,
                                             reduction="max overlap")

    @staticmethod
    def iou(ground_truth: Sequence[BaseEvent],
            predictions: Sequence[BaseEvent],
            min_iou: float = 0,
            allow_cross_matching: bool = True) -> Dict[BaseEvent, BaseEvent]:
        """
        Matches the predicted event with maximum intersection-over-union with each ground-truth event, above a minimal value.
        """
        return EventMatcher.generic_matching(ground_truth, predictions, allow_cross_matching, min_iou=min_iou,
                                             reduction="iou")

    @staticmethod
    def onset_latency(ground_truth: Sequence[BaseEvent],
                      predictions: Sequence[BaseEvent],
                      max_onset_latency: float = 0,
                      allow_cross_matching: bool = True) -> Dict[BaseEvent, BaseEvent]:
        """
        Matches the predicted event with least onset latency with each ground-truth event, below a maximum latency.
        """
        return EventMatcher.generic_matching(ground_truth, predictions, allow_cross_matching,
                                             max_onset_latency=max_onset_latency, reduction="onset latency")

    @staticmethod
    def offset_latency(ground_truth: Sequence[BaseEvent],
                       predictions: Sequence[BaseEvent],
                       max_offset_latency: float = 0,
                       allow_cross_matching: bool = True) -> Dict[BaseEvent, BaseEvent]:
        """
        Matches the predicted event with least offset latency with each ground-truth event, below a maximum latency.
        """
        return EventMatcher.generic_matching(ground_truth, predictions, allow_cross_matching,
                                             max_offset_latency=max_offset_latency, reduction="offset latency")

    @staticmethod
    def window_based(ground_truth: Sequence[BaseEvent],
                     predictions: Sequence[BaseEvent],
                     max_onset_latency: float = 0,
                     max_offset_latency: float = 0,
                     allow_cross_matching: bool = True,
                     reduction: str = "iou") -> Dict[BaseEvent, BaseEvent]:
        """
        Finds all predicted events with onset- and offset-latencies within a specified window for each ground-truth event,
        and chooses the best gt-prediction match based on the specified reduction function.
        """
        return EventMatcher.generic_matching(ground_truth, predictions, allow_cross_matching,
                                             max_onset_latency=max_onset_latency, max_offset_latency=max_offset_latency,
                                             reduction=reduction)

    @staticmethod
    def _apply_on_column_pairs(data: pd.DataFrame,
                               matching_func: __TYPE_EVENT_MATCHING_FUNC,
                               is_symmetric: bool = True) -> pd.DataFrame:
        """
        Applies the `matching_func` on each pair of columns in the given `data`, where cells contain sequences of
        gaze-events detected by different raters/detectors. If `is_symmetric` is True, only calculate the measure once
        for each (unordered-)pair of columns, e.g, (A, B) and (B, A) will be the same. If False, calculate the measure
        for all ordered-pairs of columns.

        :param data: The DataFrame to calculate the function on its columns. Each cell should contain a sequence of events.
        :param matching_func: The function to calculate the measure between two columns. Should take two sequences of
            gaze-events as input arguments, and return a dictionary where keys are ground-truth events and values are
            their matched predicted event(s).
        :param is_symmetric: Determines whether to calculate the measure for ordered or unordered pairs of columns.
        :return: A DataFrame with the same index as the input data, columns as the pairs of columns of the input data,
            and values in the DataFrame are dictionaries where keys are ground-truth events and values are their matched
            predicted event(s).
        """
        return hlp.apply_on_column_pairs(data, matching_func, is_symmetric)

    @staticmethod
    def __find_matches(gt: BaseEvent,
                       predictions: Sequence[BaseEvent],
                       allow_cross_matching: bool,
                       min_overlap: float,
                       min_iou: float,
                       max_l2_timing_offset: float,
                       max_onset_latency: float,
                       max_offset_latency: float, ) -> Sequence[BaseEvent]:
        """
        Find predicted events that are possible matches for the ground-truth event, based on the specified criteria.

        :param gt: ground-truth event
        :param predictions: sequence of predicted events
        :param allow_cross_matching: if True, a GT event can match a predicted event of a different type
        :param min_overlap: minimum overlap time to consider a possible match
        :param min_iou: minimum intersection-over-union to consider a possible match
        :param max_l2_timing_offset: maximum L2-timing-offset to consider a possible match
        :param max_onset_latency: maximum absolute difference between the start times of the GT and predicted events
        :param max_offset_latency: maximum absolute difference between the end times of the GT and predicted events
        :return: sequence of predicted events that are possible matches for the ground-truth event
        """
        if not allow_cross_matching:
            predictions = [p for p in predictions if p.event_label == gt.event_label]
        predictions = [p for p in predictions if
                       gt.overlap_time(p) >= min_overlap and
                       gt.intersection_over_union(p) >= min_iou and
                       gt.l2_timing_offset(p) <= max_l2_timing_offset and
                       abs(p.start_time - gt.start_time) <= max_onset_latency and
                       abs(p.end_time - gt.end_time) <= max_offset_latency]
        return predictions

    @staticmethod
    def __choose_match(gt: BaseEvent,
                       matches: Sequence[BaseEvent],
                       reduction: str) -> Sequence[BaseEvent]:
        """
        Choose predicted event(s) matching the ground-truth event, based on the reduction function.
        Possible reduction functions:
            - 'all': return all matched events
            - 'first': return the first matched event
            - 'last': return the last matched event
            - 'longest': return the longest matched event
            - 'max overlap': return the matched event with maximum overlap with the GT event
            - 'iou': return the matched event with the maximum intersection-over-union with the GT event
            - 'onset latency': return the matched event with the least onset latency
            - 'offset latency': return the matched event with the least offset latency

        :param gt: ground-truth event
        :param matches: sequence of predicted events matching with the GT event
        :param reduction: reduction function to choose a predicted event from multiple matching ones

        :return: predicted event(s) matching the ground-truth event
        :raises NotImplementedError: if the reduction function is not implemented
        """
        reduction = reduction.lower().replace("_", " ").replace("-", " ").strip()
        if len(matches) == 0:
            return []
        if len(matches) == 1:
            return [matches[0]]
        if reduction == "all":
            return matches
        if reduction == "first":
            return [min(matches, key=lambda e: e.start_time)]
        if reduction == "last":
            return [max(matches, key=lambda e: e.start_time)]
        if reduction == "longest":
            return [max(matches, key=lambda e: e.duration)]
        if reduction == "max overlap":
            return [max(matches, key=lambda e: gt.overlap_time(e))]
        if reduction == "iou":
            return [max(matches, key=lambda e: gt.intersection_over_union(e))]
        if reduction == "onset latency":
            return [min(matches, key=lambda e: abs(e.start_time - gt.start_time))]
        if reduction == "offset latency":
            return [min(matches, key=lambda e: abs(e.end_time - gt.end_time))]
        raise NotImplementedError(f"Reduction function '{reduction}' is not implemented")
