import os
import time
import warnings
from abc import abstractmethod
from typing import List, Callable

import pandas as pd

import Config.experiment_config as cnfg
import Analysis.helpers as hlp
from Analysis.Pipelines.BasePipeline import BasePipeline
from DataSetLoaders.DataSetFactory import DataSetFactory
from GazeDetectors.BaseDetector import BaseDetector


class BaseComparisonPipeline(BasePipeline):

    def __init__(self, dataset_name: str, reference_rater: str):
        super().__init__(dataset_name)
        self.reference_rater = reference_rater

    @classmethod
    @abstractmethod
    def _get_default_detectors(cls) -> List[BaseDetector]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _column_mapper(colname: str) -> str:
        raise NotImplementedError

    def run(
            self,
            detectors: List[BaseDetector] = None,
            allow_cross_matching: bool = False,
            verbose=False
    ):
        samples, events, detector_results = self.load_and_detect(
            detectors=detectors or self._get_default_detectors(),
            column_mapper=self._column_mapper,
            save=True,
            verbose=verbose,
        )
        matches = self.match_events(
            events_df=events,
            matching_schemes=None,
            allow_cross_matching=allow_cross_matching,
            verbose=verbose,
        )
        sample_metrics = self.process_samples(
            samples_df=samples,
            sample_label=None,
            metric_names=None,
            create_figures=True,
            verbose=verbose,
        )
        event_features, fixation_features, saccade_features = self._process_event_features(
            events=events,
            create_figures=True,
            verbose=verbose,
        )
        event_match_ratios, fixation_match_ratios, saccade_match_ratios = self._process_match_ratios(
            events=events,
            matches=matches,
            create_figures=True,
            verbose=verbose,
        )
        matched_event_features, matched_fixation_features, matched_saccade_features = self._process_matched_features(
            matches=matches,
            create_figures=True,
            verbose=verbose,
        )
        return (
            samples, events, detector_results, matches, sample_metrics, event_features, fixation_features,
            saccade_features,
            event_match_ratios, fixation_match_ratios, saccade_match_ratios, matched_event_features,
            matched_fixation_features, matched_saccade_features
        )

    def load_and_detect(
            self,
            detectors: List[BaseDetector],
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
                samples = pd.read_pickle(os.path.join(self._output_dir, "samples.pkl"))
                events = pd.read_pickle(os.path.join(self._output_dir, "events.pkl"))
                detector_results = pd.read_pickle(os.path.join(self._output_dir, "detector_results.pkl"))
            except FileNotFoundError:
                samples, events, detector_results = DataSetFactory.load_and_detect(self.dataset_name,
                                                                                   detectors=detectors,
                                                                                   column_mapper=column_mapper)
                if save:
                    samples.to_pickle(os.path.join(self._output_dir, "samples.pkl"))
                    events.to_pickle(os.path.join(self._output_dir, "events.pkl"))
                    detector_results.to_pickle(os.path.join(self._output_dir, "detector_results.pkl"))
            self._figure_columns = [
                pair for pair in hlp.extract_rater_detector_pairs(samples) if pair[0] == self.reference_rater
            ]
            end = time.time()
            if verbose:
                print(f"Preprocessing Completed:\t{end - start:.2f}s")
        return samples, events, detector_results

    def _process_event_features(self, events, create_figures=False, verbose=False):
        event_features = self.process_event_features(
            events_df=events,
            event_label=None,
            feature_names=None,
            create_figures=create_figures,
            verbose=verbose,
        )
        fixation_features = self.process_event_features(
            events_df=events,
            event_label=cnfg.EVENT_LABELS.FIXATION,
            feature_names=None,
            create_figures=create_figures,
            verbose=verbose,
        )
        saccade_features = self.process_event_features(
            events_df=events,
            event_label=cnfg.EVENT_LABELS.SACCADE,
            feature_names=None,
            create_figures=create_figures,
            verbose=verbose,
        )
        return event_features, fixation_features, saccade_features

    def _process_match_ratios(self, events, matches, create_figures=False, verbose=False):
        event_match_ratios = self.process_match_ratios(
            events_df=events,
            matches=matches,
            event_label=None,
            create_figures=create_figures,
            verbose=verbose,
        )
        fixation_match_ratios = self.process_match_ratios(
            events_df=events,
            matches=matches,
            event_label=cnfg.EVENT_LABELS.FIXATION,
            create_figures=create_figures,
            verbose=verbose,
        )
        saccade_match_ratios = self.process_match_ratios(
            events_df=events,
            matches=matches,
            event_label=cnfg.EVENT_LABELS.SACCADE,
            create_figures=create_figures,
            verbose=verbose,
        )
        return event_match_ratios, fixation_match_ratios, saccade_match_ratios

    def _process_matched_features(self, matches, create_figures=False, verbose=False):
        matched_event_features = self.process_matched_features(
            matches=matches,
            event_label=None,
            feature_names=None,
            create_figures=create_figures,
            verbose=verbose,
        )
        matched_fixation_features = self.process_matched_features(
            matches=matches,
            event_label=cnfg.EVENT_LABELS.FIXATION,
            feature_names=None,
            create_figures=create_figures,
            verbose=verbose,
        )
        matched_saccade_features = self.process_matched_features(
            matches=matches,
            event_label=cnfg.EVENT_LABELS.SACCADE,
            feature_names=None,
            create_figures=create_figures,
            verbose=verbose,
        )
        return matched_event_features, matched_fixation_features, matched_saccade_features
