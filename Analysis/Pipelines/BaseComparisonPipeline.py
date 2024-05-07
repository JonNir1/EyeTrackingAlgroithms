import os
import time
import warnings
from abc import abstractmethod
from typing import List, Callable, Optional

import pandas as pd

import Config.experiment_config as cnfg
import Analysis.helpers as hlp
from Analysis.Pipelines.BasePipeline import BasePipeline
from DataSetLoaders.DataSetFactory import DataSetFactory
from GazeDetectors.BaseDetector import BaseDetector


class BaseComparisonPipeline(BasePipeline):

    def __init__(self, dataset_name: str, reference_rater: str, pipeline_name: Optional[str] = None):
        super().__init__(dataset_name, pipeline_name)
        self.reference_rater = reference_rater

    @classmethod
    @abstractmethod
    def _get_default_detectors(cls) -> List[BaseDetector]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _column_mapper(colname: str) -> str:
        raise NotImplementedError

    def _run_impl(
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
        event_sample_metrics, event_features, event_match_ratios, event_matched_features = self._process_events(
            samples=samples,
            events=events,
            matches=matches,
            label=None,
            create_figures=True,
            verbose=verbose,
        )
        fix_sample_metrics, fix_features, fix_match_ratios, fix_matched_features = self._process_events(
            samples=samples,
            events=events,
            matches=matches,
            label=cnfg.EVENT_LABELS.FIXATION,
            create_figures=True,
            verbose=verbose,
        )
        sac_sample_metrics, sac_features, sac_match_ratios, sac_matched_features = self._process_events(
            samples=samples,
            events=events,
            matches=matches,
            label=cnfg.EVENT_LABELS.SACCADE,
            create_figures=True,
            verbose=verbose,
        )
        return (
            samples, events, detector_results, matches,
            event_sample_metrics, event_features, event_match_ratios, event_matched_features,
            fix_sample_metrics, fix_features, fix_match_ratios, fix_matched_features,
            sac_sample_metrics, sac_features, sac_match_ratios, sac_matched_features,
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

    def _process_events(self, samples, events, matches,
                        label: Optional[cnfg.EVENT_LABELS], create_figures=False, verbose=False):
        sample_metrics = self.process_samples(samples_df=samples, label=label,
                                              metric_names=None, create_figures=True, verbose=verbose)
        features = self.process_event_features(events_df=events, label=label,
                                               feature_names=None, create_figures=create_figures, verbose=verbose)
        match_ratios = self.process_match_ratios(events_df=events, matches=matches, label=label,
                                                 create_figures=create_figures, verbose=verbose)
        matched_features = self.process_matched_features(matches=matches, label=label, feature_names=None,
                                                         create_figures=create_figures, verbose=verbose)
        return sample_metrics, features, match_ratios, matched_features
