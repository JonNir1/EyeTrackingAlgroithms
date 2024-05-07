import os
import time
import copy
from abc import ABC, abstractmethod
from typing import Set, Dict, Optional

import pandas as pd

import Config.constants as cnst
import Config.experiment_config as cnfg
import Analysis.figures as figs


class BasePipeline(ABC):

    _PIPELINE_STR = "Pipeline"
    _SCARFPLOTS_STR = "scarfplots"
    _SAMPLE_METRICS_STR = "sample_metrics"
    _FEATURES_STR = "features"
    _MATCHED_FEATURES_STR = "matched_features"

    def __init__(self, dataset_name: str, pipeline_name: Optional[str] = None):
        self.dataset_name = dataset_name
        self._name = pipeline_name
        self._output_dir = os.path.join(cnfg.OUTPUT_DIR, self.name, self.dataset_name)
        os.makedirs(self._output_dir, exist_ok=True)
        self._figure_columns = []  # a subset of columns to display in the figures

    def run(self, verbose=False, **kwargs):
        start = time.time()
        if verbose:
            print("\n====================================")
            print(f"Pipeline:\t{self.name.upper()}\t\tDataset:\t{self.dataset_name.upper()}")
        results = self._run_impl(verbose=verbose, **kwargs)
        end = time.time()
        if verbose:
            print(f"Pipeline Completed:\t{end - start:.2f}s")
            print("====================================\n")
        return results

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        classname = self.__class__.__name__
        if classname.endswith(self._PIPELINE_STR):
            return classname[:-len(self._PIPELINE_STR)]
        return classname

    @abstractmethod
    def _run_impl(self, verbose=False, **kwargs):
        raise NotImplementedError

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
        matches = EventMatchesCalculator.calculate(file_path=os.path.join(self._output_dir, "matches.pkl"),
                                                   events_df=events_df, matching_schemes=matching_schemes,
                                                   verbose=False)
        end = time.time()
        if verbose:
            print(f"Matching Completed:\t{end - start:.2f}s")
        return matches

    def process_samples(
            self,
            samples_df: pd.DataFrame,
            label: Optional[cnfg.EVENT_LABELS] = None,
            metric_names: Set[str] = None,
            create_figures=False,
            verbose=False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates sample-level metrics for every pair of columns in the given DataFrame, and groups the results by the
            stimulus.
        :param samples_df: A DataFrame containing the label-per-sample of each rater/detector.
        :param label: If provided, only metrics of this type of sample are processed.
        :param metric_names: A set of metric names to calculate. If None, the default set of metrics will be calculated.
        :param create_figures: Whether to create and save figures.
        :param verbose: Whether to print the progress of the metric calculation.
        :return: A dictionary mapping each metric to a DataFrame containing the calculated metric values.
        """
        from Analysis.Calculators.SampleMetricsCalculator import SampleMetricsCalculator
        start = time.time()
        label_name = cnst.EVENT if label is None else label.name.lower()
        if verbose:
            print(f"Analyzing {label_name.capitalize()} Samples...")
        data = samples_df if label is None else samples_df.map(
            lambda cell: [label if sample == label else cnfg.EVENT_LABELS.UNDEFINED for sample in cell] if pd.notnull(
                cell).all() else None
        )
        metric_names = metric_names or self.__get_default_sample_metrics(label)
        sample_metrics = SampleMetricsCalculator.calculate(
            file_path=os.path.join(self._output_dir, f"{label_name}_sample_metrics.pkl"),
            data=data,
            metric_names=metric_names,
            verbose=False,
        )
        if create_figures:
            if verbose:
                print(f"Creating Sample Figures...")
            # create scarfplots
            scarfplot_dir = os.path.join(self._output_dir, label_name, self._SCARFPLOTS_STR)
            if not os.path.exists(scarfplot_dir):
                os.makedirs(scarfplot_dir, exist_ok=True)
            _ = figs.create_comparison_scarfplots(samples_df, scarfplot_dir)
            # create sample-metric figures
            sample_metrics_dir = os.path.join(self._output_dir, label_name, self._SAMPLE_METRICS_STR)
            if not os.path.exists(sample_metrics_dir):
                os.makedirs(sample_metrics_dir, exist_ok=True)
            _ = figs.create_sample_metric_distributions(
                sample_metrics, self.dataset_name, sample_metrics_dir, self._figure_columns
            )
        end = time.time()
        if verbose:
            print(f"Sample Analysis for {label_name.capitalize()}:\t{end - start:.2f}s")
        return sample_metrics

    def process_event_features(
            self,
            events_df: pd.DataFrame,
            label: Optional[cnfg.EVENT_LABELS] = None,
            feature_names: Set[str] = None,
            create_figures=False,
            verbose=False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates event-level features for each column in the given DataFrame.
        :param events_df: A DataFrame containing the detected events of each rater/detector.
        :param label: If provided, only features of this type of event are processed.
        :param feature_names: A set of feature names to calculate. If None, the default set of features will be calculated.
        :param create_figures: Whether to create and save figures.
        :param verbose: Whether to print the progress of the feature calculation.
        :return: A dictionary mapping each feature to a DataFrame containing the calculated feature values.
        """
        from Analysis.Calculators.EventFeaturesCalculator import EventFeaturesCalculator
        start = time.time()
        label_name = cnst.EVENT if label is None else label.name.lower()
        if verbose:
            print(f"Analyzing {label_name.capitalize()} Features...")
        data = events_df if label is None else events_df.map(
            lambda cell: [event for event in cell if event.event_label == label] if pd.notnull(cell).all() else None
        )
        feature_names = feature_names or self.__get_default_event_features(label)
        features = EventFeaturesCalculator.calculate(
            file_path=os.path.join(self._output_dir, f"{label_name}_features.pkl"),
            data=data,
            metric_names=feature_names,
            verbose=False,
        )
        if create_figures:
            if verbose:
                print(f"Creating {label_name.capitalize()} Feature Figures...")
            features_dir = os.path.join(self._output_dir, label_name, self._FEATURES_STR)
            if not os.path.exists(features_dir):
                os.makedirs(features_dir, exist_ok=True)
            _ = figs.create_event_feature_distributions(
                features, self.dataset_name, features_dir, None
            )
        end = time.time()
        if verbose:
            print(f"Features Analysis for {label_name.capitalize()}:\t{end - start:.2f}s")
        return features

    def process_match_ratios(
            self,
            events_df: pd.DataFrame,
            matches: Dict[str, pd.DataFrame],
            label: Optional[cnfg.EVENT_LABELS] = None,
            create_figures=False,
            verbose=False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates the ratio of matched events to the total number of detected events for each detector, and groups the
        results by the stimulus.
        :param events_df: A DataFrame containing the detected events of each rater/detector.
        :param matches: A dictionary mapping each matching scheme to a DataFrame containing the matched events.
        :param label: If provided, only features of this type of event are processed.
        :param create_figures: Whether to create and save figures.
        :param verbose: Whether to print the progress of the match ratio calculation.
        :return: A dictionary mapping each matching scheme to a DataFrame containing the calculated match ratios.
        """
        from Analysis.Calculators.MatchRatioCalculator import MatchRatioCalculator
        start = time.time()
        event_name = cnst.EVENT if label is None else label.name.lower()
        if verbose:
            print(f"Analyzing {event_name.capitalize()} Match Ratios...")
        # ignore events that are not of the required type
        new_events_df = events_df.copy(deep=True)
        new_matches = copy.deepcopy(matches)
        if label is not None:
            new_events_df = new_events_df.map(
                lambda cell: [event for event in cell if event.event_label == label]
                if pd.notnull(cell).all() else None
            )
            for scheme, df in new_matches.items():
                df = df.map(
                    lambda cell: {gt: pred for gt, pred in cell.items() if gt.event_label == label}
                    if pd.notnull(cell) else None
                )
                new_matches[scheme] = df
        # calculate ratios & create figures
        ratios = MatchRatioCalculator.calculate(
            file_path=os.path.join(self._output_dir, f"{event_name}_match_ratios.pkl"),
            events=new_events_df,
            matches=new_matches,
            verbose=False,
        )
        if create_figures:
            if verbose:
                print(f"Creating {event_name.capitalize()} Match Ratio Figures...")
            matched_features_dir = os.path.join(self._output_dir, event_name, self._MATCHED_FEATURES_STR)
            if not os.path.exists(matched_features_dir):
                os.makedirs(matched_features_dir, exist_ok=True)
            _ = figs.create_matching_ratio_distributions(
                ratios, self.dataset_name, matched_features_dir, self._figure_columns
            )
        end = time.time()
        if verbose:
            print(f"Match Ratios Analysis for {event_name.capitalize()}:\t{end - start:.2f}s")
        return ratios

    def process_matched_features(
            self,
            matches: Dict[str, pd.DataFrame],
            label: Optional[cnfg.EVENT_LABELS] = None,
            feature_names: Set[str] = None,
            create_figures=False,
            verbose=False,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        from Analysis.Calculators.MatchedFeaturesCalculator import MatchedFeaturesCalculator
        start = time.time()
        event_name = cnst.EVENT if label is None else label.name.lower()
        if verbose:
            print(f"Analyzing {event_name.capitalize()} Matched Features...")
        # ignore events that are not of the required type
        new_matches = copy.deepcopy(matches)
        if label is not None:
            for scheme, df in new_matches.items():
                df = df.map(
                    lambda cell: {gt: pred for gt, pred in cell.items() if gt.event_label == label}
                    if pd.notnull(cell) else None
                )
                new_matches[scheme] = df
        # calculate features & create figures
        features = MatchedFeaturesCalculator.calculate(
            file_path=os.path.join(self._output_dir, f"{event_name}_matched_features.pkl"),
            matches=new_matches,
            feature_names=feature_names or self.__get_default_matched_event_features(label),
            verbose=False,
        )
        if create_figures:
            if verbose:
                print(f"Creating {event_name.capitalize()} Matched-Features Figures...")
            matched_features_dir = os.path.join(self._output_dir, event_name, self._MATCHED_FEATURES_STR)
            if not os.path.exists(matched_features_dir):
                os.makedirs(matched_features_dir, exist_ok=True)
            _ = figs.create_matched_event_feature_distributions(
                features, self.dataset_name, matched_features_dir, self._figure_columns
            )
        end = time.time()
        if verbose:
            print(f"Matched Features Analysis for {event_name.capitalize()}:\t{end - start:.2f}s")
        return features

    @staticmethod
    def __get_default_sample_metrics(label: Optional[cnfg.EVENT_LABELS]) -> Set[str]:
        # Note: Precision, Recall & F1 are weighted if `label` is None, and binary if `label` is specified.
        certainty_metrics = {"Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1 Score"}
        agreement_metrics = {"1-NLD", "Cohen's Kappa", "Mathew's Correlation"}
        transition_metrics = {"Transition Matrix l2-norm", "Transition Matrix KL-Divergence"}
        if label is None:
            return {"Count", "Confusion Matrix", *certainty_metrics, *agreement_metrics, *transition_metrics}
        return certainty_metrics


    @staticmethod
    def __get_default_event_features(label: Optional[cnfg.EVENT_LABELS]) -> Set[str]:
        if label is None:
            return {"Count", "Amplitude", "Duration", "Peak Velocity"}
        if label == cnfg.EVENT_LABELS.FIXATION:
            return {"Duration", "Peak Velocity"}
        if label == cnfg.EVENT_LABELS.SACCADE:
            return {"Micro-Saccade Ratio", "Amplitude", "Duration", "Azimuth", "Peak Velocity"}
        raise ValueError(f"No default features for {label.name} events.")

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

    @staticmethod
    def __get_default_matched_event_features(label: Optional[cnfg.EVENT_LABELS]) -> Set[str]:
        if label is None:
            return {
                "Start Time", "End Time", "Duration", "Amplitude", "Azimuth", "Peak Velocity", "L2 Timing Difference",
                "IoU", "Overlap Time",
            }
        if label == cnfg.EVENT_LABELS.FIXATION:
            return {
                "Start Time", "End Time", "Duration", "Amplitude", "Azimuth", "Peak Velocity", "L2 Timing Difference",
                "IoU", "Overlap Time", "CoM Distance",
            }
        if label == cnfg.EVENT_LABELS.SACCADE:
            return {
                "Start Time", "End Time", "Duration", "Amplitude", "Azimuth", "Peak Velocity", "L2 Timing Difference",
                "IoU", "Overlap Time",
            }
        raise ValueError(f"No default matched-features for {label.name} events.")
