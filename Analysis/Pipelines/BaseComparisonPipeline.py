from abc import abstractmethod

import pandas as pd

import Config.experiment_config as cnfg
from Analysis.Pipelines.BasePipeline import BasePipeline


class BaseComparisonPipeline(BasePipeline):

    @abstractmethod
    def _preprocess(self, verbose=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        raise NotImplementedError

    def run(self, verbose=False, **kwargs):
        samples, events, detector_results = self._preprocess(
            verbose=verbose,
        )
        sample_metrics = self.process_samples(
            samples_df=samples,
            metric_names=None,
            create_figures=kwargs.get("create_figures", False),
            verbose=verbose,
        )
        event_features, fixation_features, saccade_features = self._process_event_features(
            events=events,
            create_figures=kwargs.get("create_figures", False),
            verbose=verbose,
        )
        matches = self.match_events(
            events_df=events,
            matching_schemes=None,
            allow_cross_matching=kwargs.get("allow_cross_matching", False),
            verbose=verbose,
        )
        event_match_ratios, fixation_match_ratios, saccade_match_ratios = self._process_match_ratios(
            events=events,
            matches=matches,
            create_figures=kwargs.get("create_figures", False),
            verbose=verbose,
        )
        matched_event_features, matched_fixation_features, matched_saccade_features = self._process_matched_features(
            matches=matches,
            create_figures=kwargs.get("create_figures", False),
            verbose=verbose,
        )
        return (
            samples, events, detector_results, matches, sample_metrics, event_features, fixation_features,
            saccade_features,
            event_match_ratios, fixation_match_ratios, saccade_match_ratios, matched_event_features,
            matched_fixation_features, matched_saccade_features
        )

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
