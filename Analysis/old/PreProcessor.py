import os
import time
import warnings
from abc import ABC
from typing import List, Union, Callable, Dict, Set

import numpy as np
import pandas as pd
import pickle as pkl

import Config.experiment_config as cnfg
import Analysis.helpers as hlp
import Utils.metrics as metrics
from DataSetLoaders.DataSetFactory import DataSetFactory
from GazeDetectors.BaseDetector import BaseDetector
from Analysis.EventMatcher import EventMatcher as Matcher
from GazeEvents.BaseEvent import BaseEvent


class PreProcessor(ABC):
    MATCHED_EVENT_FEATURES_WITHIN = {
        "Start Time", "End Time", "Duration", "Amplitude", "Azimuth", "Peak Velocity",
    }
    MATCHED_EVENT_FEATURES_BETWEEN = {
        "L2 Timing Difference", "IoU", "Overlap Time",
        # uninteresting: "Duration Difference", "Amplitude Difference", "Azimuth Difference",
        # "Peak Velocity Difference", "Onset Jitter", "Offset Jitter"
    }
    SAMPLE_METRICS = {
        "Accuracy",
        "Levenshtein Ratio",
        "Cohen's Kappa",
        "Mathew's Correlation",
        "Transition Matrix l2-norm",
        "Transition Matrix KL-Divergence",
    }
    EVENT_FEATURES = {
        "Count", "Micro-Saccade Ratio", "Amplitude", "Duration", "Azimuth", "Peak Velocity"
    }
    MATCHED_EVENT_FEATURES = MATCHED_EVENT_FEATURES_WITHIN | MATCHED_EVENT_FEATURES_BETWEEN

    @staticmethod
    def load_or_run(dataset_name: str, pipeline_name: str, verbose=False, **kwargs):
        start = time.time()
        if verbose:
            print(f"Dataset: {dataset_name}\tPipeline: {pipeline_name}")
        dataset_dir = os.path.join(cnfg.OUTPUT_DIR, dataset_name, pipeline_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        try:
            samples = pd.read_pickle(os.path.join(dataset_dir, "samples.pkl"))
            events = pd.read_pickle(os.path.join(dataset_dir, "events.pkl"))
            detector_results = pd.read_pickle(os.path.join(dataset_dir, "detector_results.pkl"))
        except FileNotFoundError:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                samples, events, detector_results = PreProcessor.load_and_detect(
                    dataset_name,
                    detectors=kwargs.get("detectors", None),
                    column_mapper=kwargs.get("column_mapper", lambda col: col),
                    verbose=verbose
                )
                samples.to_pickle(os.path.join(dataset_dir, "samples.pkl"), protocol=-1)
                events.to_pickle(os.path.join(dataset_dir, "events.pkl"), protocol=-1)
                detector_results.to_pickle(os.path.join(dataset_dir, "detector_results.pkl"), protocol=-1)

        try:
            matches = pd.read_pickle(os.path.join(dataset_dir, "matches.pkl"))
        except FileNotFoundError:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                matches = PreProcessor.match_events(
                    events,
                    matching_schemes=kwargs.get("matching_schemes", None),
                    xmatch=kwargs.get("xmatch", False),
                    verbose=verbose
                )
                with open(os.path.join(dataset_dir, "matches.pkl"), "wb") as f:
                    pkl.dump(matches, f, protocol=-1)

        try:
            sample_metrics = pd.read_pickle(os.path.join(dataset_dir, "sample_metrics.pkl"))
            event_features = pd.read_pickle(os.path.join(dataset_dir, "event_features.pkl"))
            match_ratios = pd.read_pickle(os.path.join(dataset_dir, "match_ratios.pkl"))
            matched_features = pd.read_pickle(os.path.join(dataset_dir, "matched_features.pkl"))
        except FileNotFoundError:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sample_metrics = PreProcessor.calculate_sample_metrics(samples, verbose=verbose)
                event_features = PreProcessor.calculate_event_features(events, verbose=verbose)
                match_ratios = PreProcessor.calculate_match_ratios(events, matches, verbose=verbose)
                matched_features = PreProcessor.calculate_matched_event_features(matches, verbose=verbose)
                with open(os.path.join(dataset_dir, "sample_metrics.pkl"), "wb") as f:
                    pkl.dump(sample_metrics, f, protocol=-1)
                with open(os.path.join(dataset_dir, "event_features.pkl"), "wb") as f:
                    pkl.dump(event_features, f, protocol=-1)
                with open(os.path.join(dataset_dir, "match_ratios.pkl"), "wb") as f:
                    pkl.dump(match_ratios, f, protocol=-1)
                with open(os.path.join(dataset_dir, "matched_features.pkl"), "wb") as f:
                    pkl.dump(matched_features, f, protocol=-1)

        end = time.time()
        if verbose:
            print(f"Preprocessing Completed:\t{end - start:.2f}s")
        return samples, events, detector_results, matches, sample_metrics, event_features, match_ratios, matched_features

    @staticmethod
    def load_and_detect(
            dataset_name: str,
            detectors: List[BaseDetector] = None,
            column_mapper: Callable[[str], str] = lambda col: col,
            verbose=False
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.time()
            if verbose:
                print(f"Preprocessing dataset `{dataset_name}`...")
            detectors = PreProcessor.__get_default_detectors() if detectors is None else detectors
            samples_df, events_df, detector_results_df = DataSetFactory.load_and_detect(dataset_name,
                                                                                        detectors=detectors,
                                                                                        column_mapper=column_mapper)

            end = time.time()
            if verbose:
                print(f"Preprocessing Completed:\t{end - start:.2f}s")
        return samples_df, events_df, detector_results_df

    @staticmethod
    def match_events(events_df: pd.DataFrame,
                     matching_schemes: Dict[str, Dict[str, float]] = None,
                     xmatch: bool = False,
                     verbose=False) -> Dict[str, pd.DataFrame]:
        if verbose:
            print(f"Matching Events...")
        start = time.time()
        matching_schemes = matching_schemes or PreProcessor.__get_default_matching_schemes()
        match_results = {}
        for scheme_name, scheme_kwargs in matching_schemes.items():
            sub_start = time.time()
            if verbose:
                print(f"\tMatching using \"{scheme_name.title()}\" matching-scheme...")
            scheme_name = scheme_name.lower()
            matched_events = Matcher.match_events(events_df,
                                                  match_by=scheme_kwargs.pop("match_by", scheme_name),
                                                  is_symmetric=False,
                                                  allow_cross_matching=xmatch,
                                                  **scheme_kwargs)
            match_results[scheme_name] = matched_events
            sub_end = time.time()
            if verbose:
                print(f"\tScheme \"{scheme_name.title()}\":\t{sub_end - sub_start:.2f}s")
        end = time.time()
        if verbose:
            print(f"Matching Completed:\t{end - start:.2f}s")
        return match_results

    @staticmethod
    def calculate_sample_metrics(
            samples_df: pd.DataFrame,
            metric_names: Set[str] = None,
            verbose=False
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates sample-level metrics for every pair of columns in the given DataFrame, and groups the results by the
        stimulus.
        :param samples_df: A DataFrame containing the label-per-sample of each rater/detector.
        :param metric_names: A set of metric names to calculate. If None, the default set of metrics will be calculated.
        :param verbose: Whether to print the progress of the metric calculation.
        :return: A dictionary mapping each metric to a DataFrame containing the calculated metric values.
        """
        start = time.time()
        if verbose:
            print("\tCalculating Sample Metrics...")
        metric_names = metric_names or PreProcessor.SAMPLE_METRICS
        sample_results = {}
        for metric in metric_names:
            met = metric.lower()
            if met in {"acc", "accuracy"}:
                metric_func = metrics.accuracy
            elif met in {"balanced accuracy"}:
                metric_func = metrics.balanced_accuracy
            elif met in {"lev", "levenshtein", "levenshtein distance", "nld", "1-nld", "complement nld"}:
                metric_func = metrics.complement_nld
            elif met in {"kappa", "cohen kappa", "cohen's kappa"}:
                metric_func = metrics.cohen_kappa
            elif met in {"mcc", "mathew's correlation", "mathews correlation"}:
                metric_func = metrics.matthews_correlation
            elif met in {"fro", "frobenius", "l2", "transition matrix l2-norm"}:
                metric_func = lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="fro")
            elif met in {"kl", "kl divergence", "kullback leibler", "transition matrix kl-divergence"}:
                metric_func = lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="kl")
            else:
                raise NotImplementedError(f"Unknown metric for samples:\t{metric}")
            computed = hlp.apply_on_column_pairs(samples_df, metric_func, is_symmetric=False)
            sample_results[metric] = computed
        end = time.time()
        if verbose:
            print(f"\tSample Metrics Calculated:\t{end - start:.2f}s")
        return sample_results

    @staticmethod
    def calculate_event_features(
            events_df: pd.DataFrame,
            feature_names: Set[str] = None,
            verbose=False
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates event-level features for each column in the given DataFrame.
        :param events_df: A DataFrame containing the detected events of each rater/detector.
        :param feature_names: A set of feature names to calculate. If None, the default set of features will be calculated.
        :param verbose: Whether to print the progress of the feature calculation.
        :return: A dictionary mapping each feature to a DataFrame containing the calculated feature values.
        """
        start = time.time()
        if verbose:
            print("\tCalculating Event Features...")
        feature_names = feature_names or PreProcessor.EVENT_FEATURES
        event_results = {}
        for feature in feature_names:
            feat = feature.lower()
            if feat in {"count", "counts", "event count", "event counts"}:
                computed = PreProcessor.__event_counts_impl(events_df)
            elif feat in {"micro-saccade ratio", "microsaccade ratio"}:
                computed = PreProcessor.__microsaccade_ratio_impl(events_df)
            else:
                attr = feat.lower().replace(" ", "_")
                computed = events_df.map(lambda cell: [getattr(e, attr) for e in cell if hasattr(e, attr)])
            event_results[feature] = computed
        end = time.time()
        if verbose:
            print(f"\tEvent Features Calculated:\t{end - start:.2f}s")
        return event_results

    @staticmethod
    def calculate_match_ratios(
            events_df: pd.DataFrame,
            matches: Dict[str, pd.DataFrame],
            verbose=False
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates the ratio of matched events to the total number of detected events for each detector, and groups the
        results by the stimulus.
        :param events_df: A DataFrame containing the detected events of each rater/detector.
        :param matches: A dictionary mapping each matching scheme to a DataFrame containing the matched events.
        :param verbose: Whether to print the progress of the match ratio calculation.
        :return: A dictionary mapping each matching scheme to a DataFrame containing the calculated match ratios.
        """
        start = time.time()
        if verbose:
            print("\tCalculating Match Ratios...")
        event_counts = events_df.map(
            lambda cell: [e for e in cell if e.event_label] if pd.notnull(cell).all() else np.nan
        ).map(len)
        match_ratios = {}
        for scheme, matches_df in matches.items():
            match_counts = matches_df.map(
                lambda cell: {k: v for k, v in cell.items() if k.event_label} if pd.notnull(cell) else np.nan
            ).map(lambda cell: len(cell) if pd.notnull(cell) else np.nan)
            ratios = np.zeros_like(match_counts, dtype=float)
            for i in range(match_counts.index.size):
                for j in range(match_counts.columns.size):
                    gt_col, _pred_col = match_counts.columns[j]
                    ratios[i, j] = match_counts.iloc[i, j] / event_counts.iloc[i][gt_col]
            ratios = pd.DataFrame(ratios, index=match_counts.index, columns=match_counts.columns)
            match_ratios[scheme] = ratios
        end = time.time()
        if verbose:
            print(f"\tMatch Ratios Calculated:\t{end - start:.2f}s")
        return match_ratios

    @staticmethod
    def calculate_matched_event_features(
            matches: Dict[str, pd.DataFrame],
            feature_names: Set[str] = None,
            verbose=False
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculates features for matched events for each matching scheme, and groups the results by the stimulus.
        :param matches: A dictionary mapping each matching scheme to a DataFrame containing the matched events.
        :param feature_names: A set of feature names to calculate. If None, the default set of features will be calculated.
        :param verbose: Whether to print the progress of the feature calculation.
        :return: A dictionary mapping each feature to a dictionary of matching schemes to a DataFrame containing the
            calculated feature values.
        """
        start = time.time()
        if verbose:
            print("\tCalculating Matched Event Features...")
        feature_names = feature_names or PreProcessor.MATCHED_EVENT_FEATURES
        matched_results = {}
        for feature in feature_names:
            matched_results[feature] = {}
            for scheme, matches_df in matches.items():
                if feature in PreProcessor.MATCHED_EVENT_FEATURES_WITHIN:
                    attr = feature.lower().replace(" ", "_")
                    computed = matches_df.map(
                        lambda cell: [(getattr(k, attr), getattr(v, attr)) for k, v in cell.items()
                                      if hasattr(k, attr) and hasattr(v, attr)]
                        if pd.notnull(cell) else np.nan
                    )
                elif feature in PreProcessor.MATCHED_EVENT_FEATURES_BETWEEN:
                    computed = PreProcessor.__calculate_dual_feature_impl(matches_df, feature)
                else:
                    raise NotImplementedError(f"Unknown feature for matched events:\t{feature}")
                matched_results[feature][scheme] = computed
        end = time.time()
        if verbose:
            print(f"\tMatched Event Features Calculated:\t{end - start:.2f}s")
        return matched_results

    @staticmethod
    def __get_default_detectors() -> Union[BaseDetector, List[BaseDetector]]:
        from GazeDetectors.IVTDetector import IVTDetector
        from GazeDetectors.IDTDetector import IDTDetector
        from GazeDetectors.EngbertDetector import EngbertDetector
        from GazeDetectors.NHDetector import NHDetector
        from GazeDetectors.REMoDNaVDetector import REMoDNaVDetector
        return [IVTDetector(), IDTDetector(), EngbertDetector(), NHDetector(), REMoDNaVDetector()]

    @staticmethod
    def __get_default_matching_schemes() -> Dict[str, Dict[str, float]]:
        return {
            "onset": {
                "match_by": "onset",        # events' onset-differences are within 15ms
                "max_onset_latency": 15
            },
            "window": {
                "match_by": "window",       # events' onset- and offset-differences are within 15ms *separately*
                "max_onset_latency": 15,
                "max_offset_latency": 15
            },
            "l2": {
                "match_by": "l2",           # events' onset- and offset-differences are within 15ms *together*
                "max_l2": 15
            },
            "iou": {
                "match_by": "iou",
                "min_iou": 0
            },
            "iou_1/3": {
                "match_by": "iou",          # events coincide for at least half of their duration
                "min_iou": 1/3
            },
            "max_overlap": {
                "match_by": "max_overlap",
                "min_overlap": 0
            },
            "max_overlap_1/2": {
                "match_by": "max_overlap",  # events coincide for at least half of their duration
                "min_overlap": 0.5
            },
        }

    @staticmethod
    def __event_counts_impl(events: pd.DataFrame) -> pd.DataFrame:
        """
        Counts the number of detected events for each detector by type of event.
        :param events: A DataFrame containing the detected events of each rater/detector.
        :return: A DataFrame containing the count of events detected by each rater/detector (cols), grouped by the given
            criteria (rows).
        """
        def count_event_labels(data: List[Union[BaseEvent, cnfg.EVENT_LABELS]]) -> pd.Series:
            labels = pd.Series([e.event_label if isinstance(e, BaseEvent) else e for e in data])
            counts = labels.value_counts()
            if counts.empty:
                return pd.Series({l: 0 for l in cnfg.EVENT_LABELS})
            if len(counts) == len(cnfg.EVENT_LABELS):
                return counts
            missing_labels = pd.Series({l: 0 for l in cnfg.EVENT_LABELS if l not in counts.index})
            return pd.concat([counts, missing_labels]).sort_index()
        event_counts = events.map(count_event_labels)
        return event_counts

    @staticmethod
    def __microsaccade_ratio_impl(events: pd.DataFrame,
                                  threshold_amplitude: float = cnfg.MICROSACCADE_AMPLITUDE_THRESHOLD) -> pd.DataFrame:
        saccades = events.map(lambda cell: [e for e in cell if e.event_label == cnfg.EVENT_LABELS.SACCADE])
        saccades_count = saccades.map(len).to_numpy()
        microsaccades = saccades.map(lambda cell: [e for e in cell if e.amplitude < threshold_amplitude])
        microsaccades_count = microsaccades.map(len).to_numpy()

        ratios = np.divide(microsaccades_count, saccades_count,
                           out=np.full_like(saccades_count, fill_value=np.nan, dtype=float),  # fill NaN if denom is 0
                           where=saccades_count != 0)
        ratios = pd.DataFrame(ratios, index=events.index, columns=events.columns)
        return ratios

    @staticmethod
    def __calculate_dual_feature_impl(matches_df: pd.DataFrame, feature: str) -> pd.DataFrame:
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
                lambda cell: [k.peak_velocity_px - v.peak_velocity_px for k, v in cell.items()] if pd.notnull(
                    cell) else np.nan
            )
        elif feature == "Match Ratio":
            raise ValueError("Match Ratio feature should be calculated separately.")
        else:
            raise ValueError(f"Unknown feature: {feature}")
        return feature_df
