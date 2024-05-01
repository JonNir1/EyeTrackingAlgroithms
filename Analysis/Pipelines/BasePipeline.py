import os
import time
import warnings
from abc import ABC, abstractmethod
from typing import List, Union, Callable, Set, Dict, Optional

import numpy as np
import pandas as pd

import Config.constants as cnst
import Config.experiment_config as cnfg
import Analysis.helpers as hlp
import Analysis.figures as figs
from DataSetLoaders.DataSetFactory import DataSetFactory
from GazeDetectors.BaseDetector import BaseDetector
from GazeEvents.BaseEvent import BaseEvent


class BasePipeline(ABC):

    _SCARFPLOTS_STR = "scarfplots"
    _SAMPLE_METRICS_STR = "sample_metrics"
    _FEATURES_STR = "features"
    _MATCHED_FEATURES_STR = "matched_features"

    _DEFAULT_SAMPLE_METRICS = {
        "Accuracy",
        "Levenshtein Ratio",
        "Cohen's Kappa",
        "Mathew's Correlation",
        "Transition Matrix l2-norm",
        "Transition Matrix KL-Divergence",
    }
    _DEFAULT_EVENT_FEATURES = {"Count", "Amplitude", "Duration", "Peak Velocity"}
    _DEFAULT_FIXATION_FEATURES = {"Duration", "Peak Velocity"}
    _DEFAULT_SACCADE_FEATURES = {"Micro-Saccade Ratio", "Amplitude", "Duration", "Azimuth", "Peak Velocity"}

    def __init__(self, dataset_name: str, reference_rater: str):
        self.dataset_name = dataset_name
        self.reference_rater = reference_rater
        self._dataset_dir = self._get_or_make_dataset_dir()
        self._rater_detector_pairs = []

    @abstractmethod
    def run(self, verbose=False):
        raise NotImplementedError

    def load_and_detect(
            self,
            detectors: List[BaseDetector] = None,
            column_mapper: Callable[[str], str] = lambda col: col,
            save=True,
            verbose=False
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.time()
            if verbose:
                print(f"Preprocessing dataset `{self.dataset_name}`...")
            try:
                samples = pd.read_pickle(os.path.join(self._dataset_dir, "samples.pkl"))
                events = pd.read_pickle(os.path.join(self._dataset_dir, "events.pkl"))
                detector_results = pd.read_pickle(os.path.join(self._dataset_dir, "detector_results.pkl"))
            except FileNotFoundError:
                detectors = self._get_default_detectors() if detectors is None else detectors
                samples, events, detector_results = DataSetFactory.load_and_detect(self.dataset_name,
                                                                                   detectors=detectors,
                                                                                   column_mapper=column_mapper)
                if save:
                    samples.to_pickle(os.path.join(self._dataset_dir, "samples.pkl"))
                    events.to_pickle(os.path.join(self._dataset_dir, "events.pkl"))
                    detector_results.to_pickle(os.path.join(self._dataset_dir, "detector_results.pkl"))
            self._rater_detector_pairs = hlp.extract_rater_detector_pairs(samples)
            end = time.time()
            if verbose:
                print(f"Preprocessing Completed:\t{end - start:.2f}s")
        return samples, events, detector_results

    def match_events(
            self,
            events_df: pd.DataFrame,
            matching_schemes: Dict[str, Dict[str, float]] = None,
            allow_cross_matching=False,
            verbose=False,
    ) -> Dict[str, pd.DataFrame]:
        from Analysis.Calculators.EventMatchesCalculator import EventMatchesCalculator
        start = time.time()
        if verbose:
            print(f"Matching Events...")
        matching_schemes = matching_schemes or self.__get_default_matching_schemes()
        for scheme in matching_schemes.keys():
            matching_schemes[scheme]["allow_cross_matching"] = allow_cross_matching
        matches = EventMatchesCalculator.calculate(file_path=os.path.join(self._dataset_dir, "matches.pkl"),
                                                   events_df=events_df, matching_schemes=matching_schemes,
                                                   verbose=False)
        end = time.time()
        if verbose:
            print(f"Matching Completed:\t{end - start:.2f}s")
        return matches

    def process_samples(
            self,
            samples_df: pd.DataFrame,
            metric_names: Set[str] = None,
            create_figures=False,
            verbose=False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates sample-level metrics for every pair of columns in the given DataFrame, and groups the results by the
            stimulus.
        :param samples_df: A DataFrame containing the label-per-sample of each rater/detector.
        :param metric_names: A set of metric names to calculate. If None, the default set of metrics will be calculated.
        :param create_figures: Whether to create and save figures.
        :param verbose: Whether to print the progress of the metric calculation.
        :return: A dictionary mapping each metric to a DataFrame containing the calculated metric values.
        """
        from Analysis.Calculators.SampleMetricsCalculator import SampleMetricsCalculator
        start = time.time()
        if verbose:
            print(f"Analyzing Samples...")
        metric_names = metric_names or self._DEFAULT_SAMPLE_METRICS
        sample_metrics = SampleMetricsCalculator.calculate(
            file_path=os.path.join(self._dataset_dir, "sample_metrics.pkl"),
            data=samples_df,
            metric_names=metric_names,
            verbose=False,
        )
        if create_figures:
            if verbose:
                print(f"Creating Sample Figures...")
            # create scarfplots
            scarfplot_dir = os.path.join(self._dataset_dir, f"{cnst.LABELS}", self._SCARFPLOTS_STR)
            if not os.path.exists(scarfplot_dir):
                os.makedirs(scarfplot_dir, exist_ok=True)
            _ = figs.create_comparison_scarfplots(samples_df, scarfplot_dir)
            # create sample-metric figures
            sample_metrics_dir = os.path.join(self._dataset_dir, f"{cnst.LABELS}", self._SAMPLE_METRICS_STR)
            if not os.path.exists(sample_metrics_dir):
                os.makedirs(sample_metrics_dir, exist_ok=True)
            _ = figs.create_sample_metric_distributions(
                sample_metrics, self.dataset_name, sample_metrics_dir, self._rater_detector_pairs
            )
        end = time.time()
        if verbose:
            print(f"Sample Analysis:\t{end - start:.2f}s")
        return sample_metrics

    def process_event_features(
            self,
            events_df: pd.DataFrame,
            event_label: Optional[cnfg.EVENT_LABELS] = None,
            feature_names: Set[str] = None,
            create_figures=False,
            verbose=False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates event-level features for each column in the given DataFrame.
        :param events_df: A DataFrame containing the detected events of each rater/detector.
        :param event_label: If provided, only features of this type of event are processed.
        :param feature_names: A set of feature names to calculate. If None, the default set of features will be calculated.
        :param create_figures: Whether to create and save figures.
        :param verbose: Whether to print the progress of the feature calculation.
        :return: A dictionary mapping each feature to a DataFrame containing the calculated feature values.
        """
        from Analysis.Calculators.EventFeaturesCalculator import EventFeaturesCalculator
        start = time.time()
        event_name = cnst.EVENT if event_label is None else event_label.name.lower()
        if verbose:
            print(f"Analyzing {event_name.capitalize()} Features...")
        data = events_df if event_label is None else events_df.map(
            lambda cell: [event for event in cell if event.event_label == event_label] if pd.notnull(cell).all() else None
        )
        feature_names = feature_names or self.__get_default_event_features(event_label)
        features = EventFeaturesCalculator.calculate(
            file_path=os.path.join(self._dataset_dir, f"{event_name}_features.pkl"),
            data=data,
            metric_names=feature_names,
            verbose=False,
        )
        if create_figures:
            if verbose:
                print(f"Creating {event_name.capitalize()} Feature Figures...")
            features_dir = os.path.join(self._dataset_dir, event_name, self._FEATURES_STR)
            if not os.path.exists(features_dir):
                os.makedirs(features_dir, exist_ok=True)
            _ = figs.create_event_feature_distributions(
                features, self.dataset_name, features_dir, self._rater_detector_pairs
            )
        end = time.time()
        if verbose:
            print(f"Features Analysis for {event_name.capitalize()}:\t{end - start:.2f}s")
        return features

    def process_match_ratios(
            self,
            events_df: pd.DataFrame,
            matches: Dict[str, pd.DataFrame],
            event_label: Optional[cnfg.EVENT_LABELS] = None,
            create_figures=False,
            verbose=False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates the ratio of matched events to the total number of detected events for each detector, and groups the
        results by the stimulus.
        :param events_df: A DataFrame containing the detected events of each rater/detector.
        :param matches: A dictionary mapping each matching scheme to a DataFrame containing the matched events.
        :param event_label: If provided, only features of this type of event are processed.
        :param create_figures: Whether to create and save figures.
        :param verbose: Whether to print the progress of the match ratio calculation.
        :return: A dictionary mapping each matching scheme to a DataFrame containing the calculated match ratios.
        """
        from Analysis.Calculators.MatchRatioCalculator import MatchRatioCalculator
        start = time.time()
        event_name = cnst.EVENT if event_label is None else event_label.name.lower()
        if verbose:
            print(f"Analyzing {event_name.capitalize()} Match Ratios...")
        # ignore events that are not of the required type
        if event_label is not None:
            events_df = events_df.map(
                lambda cell: [event for event in cell if event.event_label == event_label]
                if pd.notnull(cell).all() else None
            )
            for scheme, df in matches.items():
                df = df.map(
                    lambda cell: {gt: pred for gt, pred in cell.items() if gt.event_label == event_label}
                    if pd.notnull(cell).all() else None
                )
                matches[scheme] = df
        # calculate ratios & create figures
        ratios = MatchRatioCalculator.calculate(
            file_path=os.path.join(self._dataset_dir, f"{event_name}_match_ratios.pkl"),
            events=events_df,
            matches=matches,
            verbose=False,
        )
        if create_figures:
            if verbose:
                print(f"Creating {event_name.capitalize()} Match Ratio Figures...")
            matched_features_dir = os.path.join(self._dataset_dir, event_name, self._MATCHED_FEATURES_STR)
            if not os.path.exists(matched_features_dir):
                os.makedirs(matched_features_dir, exist_ok=True)
            _ = figs.create_matching_ratio_distributions(
                ratios, self.dataset_name, matched_features_dir, self._rater_detector_pairs
            )
        end = time.time()
        if verbose:
            print(f"Match Ratios Analysis for {event_name.capitalize()}:\t{end - start:.2f}s")
        return ratios

    def process_matched_features(
            self,
            matches: Dict[str, pd.DataFrame],
            event_label: Optional[cnfg.EVENT_LABELS] = None,
            feature_names: Set[str] = None,
            create_figures=False,
            verbose=False,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        from Analysis.Calculators.MatchedFeaturesCalculator import MatchedFeaturesCalculator
        start = time.time()
        event_name = cnst.EVENT if event_label is None else event_label.name.lower()
        if verbose:
            print(f"Analyzing {event_name.capitalize()} Matched Features...")
        # ignore events that are not of the required type
        if event_label is not None:
            for scheme, df in matches.items():
                df = df.map(
                    lambda cell: {gt: pred for gt, pred in cell.items() if gt.event_label == event_label}
                    if pd.notnull(cell).all() else None
                )
                matches[scheme] = df
        # calculate features & create figures
        feature_names = feature_names or MatchedFeaturesCalculator.MATCHED_EVENT_FEATURES_WITHIN | MatchedFeaturesCalculator.MATCHED_EVENT_FEATURES_BETWEEN
        features = MatchedFeaturesCalculator.calculate(
            file_path=os.path.join(self._dataset_dir, f"{event_name}_matched_features.pkl"),
            matches=matches,
            feature_names=feature_names,
            verbose=False,
        )
        if create_figures:
            if verbose:
                print(f"Creating {event_name.capitalize()} Matched-Features Figures...")
            matched_features_dir = os.path.join(self._dataset_dir, event_name, self._MATCHED_FEATURES_STR)
            if not os.path.exists(matched_features_dir):
                os.makedirs(matched_features_dir, exist_ok=True)
            _ = figs.create_matched_event_feature_distributions(
                features, self.dataset_name, matched_features_dir, self._rater_detector_pairs
            )
        end = time.time()
        if verbose:
            print(f"Matched Features Analysis for {event_name.capitalize()}:\t{end - start:.2f}s")
        return features

    @classmethod
    def _name(cls) -> str:
        classname = cls.__name__
        return classname[:classname.index("Pipeline")]

    def _get_or_make_dataset_dir(self) -> str:
        dataset_dir = os.path.join(cnfg.OUTPUT_DIR, self._name(), self.dataset_name)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        return dataset_dir

    def _get_or_make_subdir(self, subdir_name: str) -> str:
        subdir = os.path.join(self._dataset_dir, subdir_name)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        return subdir

    @classmethod
    def _get_default_detectors(cls) -> Union[BaseDetector, List[BaseDetector]]:
        from GazeDetectors.IVTDetector import IVTDetector
        from GazeDetectors.IDTDetector import IDTDetector
        from GazeDetectors.EngbertDetector import EngbertDetector
        from GazeDetectors.NHDetector import NHDetector
        from GazeDetectors.REMoDNaVDetector import REMoDNaVDetector
        return [IVTDetector(), IDTDetector(), EngbertDetector(), NHDetector(), REMoDNaVDetector()]


    def __get_default_event_features(self, label: Optional[cnfg.EVENT_LABELS]) -> Set[str]:
        if label is None:
            return self._DEFAULT_EVENT_FEATURES
        if label == cnfg.EVENT_LABELS.FIXATION:
            return self._DEFAULT_FIXATION_FEATURES
        if label == cnfg.EVENT_LABELS.SACCADE:
            return self._DEFAULT_SACCADE_FEATURES
        raise ValueError(f"No default features for {label.name} events.")

    def __calculate_event_features_impl(
            self,
            events_df: pd.DataFrame,
            feature_names: Set[str]
    ) -> Dict[str, pd.DataFrame]:
        event_features = {}
        for feature in feature_names:
            feat = feature.lower()
            if feat in {"count", "counts", "event count", "event counts"}:
                computed = self.__event_counts_impl(events_df)
            elif feat in {"micro-saccade ratio", "microsaccade ratio"}:
                computed = self.__microsaccade_ratio_impl(events_df)
            else:
                attr = feat.lower().replace(" ", "_")
                computed = events_df.map(lambda cell: [getattr(e, attr) for e in cell if hasattr(e, attr)])
            event_features[feature] = computed
        return event_features

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
    def __get_default_matching_schemes() -> Dict[str, Dict[str, float]]:
        return {
            "onset": {
                "match_by": "onset",  # events' onset-differences are within 15ms
                "max_onset_latency": 15
            },
            "window": {
                "match_by": "window",  # events' onset- and offset-differences are within 15ms *separately*
                "max_onset_latency": 15,
                "max_offset_latency": 15
            },
            "l2": {
                "match_by": "l2",  # events' onset- and offset-differences are within 15ms *together*
                "max_l2": 15
            },
            "iou": {
                "match_by": "iou",
                "min_iou": 0
            },
            "iou_1/3": {
                "match_by": "iou",  # events coincide for at least half of their duration
                "min_iou": 1 / 3
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
