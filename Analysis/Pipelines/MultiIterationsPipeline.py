import os
import time
import warnings
import copy

import numpy as np
import pandas as pd

import Config.constants as cnst
import Config.experiment_config as cnfg
from Analysis.Pipelines.BasePipeline import BasePipeline
from GazeDetectors.BaseDetector import BaseDetector
from DataSetLoaders.DataSetFactory import DataSetFactory
import Analysis.figures as figs


class MultiIterationsPipeline(BasePipeline):

    _DETECTOR_STR = "Detector"
    _ITERATION_STR = "Iteration"
    _DEFAULT_NUM_ITERATIONS = 5
    _INDEXERS = [cnst.TRIAL, cnst.SUBJECT, cnst.SUBJECT_ID, cnst.STIMULUS, f"{cnst.STIMULUS}_name"]

    def __init__(self, dataset_name: str, detector: BaseDetector):
        super().__init__(dataset_name)
        self.detector = detector
        subdir_name = detector.name[:detector.name.index(self._DETECTOR_STR)]
        self._output_dir = os.path.join(self._output_dir, subdir_name)
        os.makedirs(self._output_dir, exist_ok=True)

    def _run_impl(
            self,
            allow_cross_matching: bool = False,
            verbose=False
    ):
        samples, events, detector_results = self.load_and_detect(
            save=True,
            verbose=verbose,
        )
        event_features, fixation_features, saccade_features = self._process_event_features(
            events=events,
            create_figures=True,
            verbose=verbose,
        )
        # create scarfplots
        scarfplot_dir = os.path.join(self._output_dir, self._SCARFPLOTS_STR)
        if not os.path.exists(scarfplot_dir):
            os.makedirs(scarfplot_dir, exist_ok=True)
        _ = figs.create_comparison_scarfplots(samples, scarfplot_dir)
        return samples, events, detector_results, event_features, fixation_features, saccade_features

    def load_and_detect(
            self,
            save=True,
            verbose=False
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.time()
            if verbose:
                print(f"Multi-Iteration preprocessing for dataset `{self.dataset_name}`...")
            try:
                samples = pd.read_pickle(os.path.join(self._output_dir, "samples.pkl"))
                events = pd.read_pickle(os.path.join(self._output_dir, "events.pkl"))
                detector_results = pd.read_pickle(os.path.join(self._output_dir, "detector_results.pkl"))
            except FileNotFoundError:
                dataset = DataSetFactory.load(self.dataset_name)
                detector_results = self._detect_multiple_times(dataset, self.detector)
                samples = detector_results.map(lambda cell: cell[cnst.GAZE][cnst.EVENT])
                events = detector_results.map(lambda cell: cell[cnst.EVENTS])
                if save:
                    samples.to_pickle(os.path.join(self._output_dir, "samples.pkl"))
                    events.to_pickle(os.path.join(self._output_dir, "events.pkl"))
                    detector_results.to_pickle(os.path.join(self._output_dir, "detector_results.pkl"))
            self._figure_columns = samples.columns.to_list()
            end = time.time()
            if verbose:
                print(f"Preprocessing Completed:\t{end - start:.2f}s")
        return samples, events, detector_results

    def _process_event_features(self, events, create_figures=False, verbose=False):
        event_features = self.process_event_features(events_df=events, label=None, feature_names=None,
                                                     create_figures=create_figures, verbose=verbose)
        fixation_features = self.process_event_features(events_df=events, label=cnfg.EVENT_LABELS.FIXATION,
                                                        feature_names=None, create_figures=create_figures,
                                                        verbose=verbose)
        saccade_features = self.process_event_features(events_df=events, label=cnfg.EVENT_LABELS.SACCADE,
                                                       feature_names=None, create_figures=create_figures,
                                                       verbose=verbose)
        return event_features, fixation_features, saccade_features

    @staticmethod
    def _detect_multiple_times(
            data: pd.DataFrame,
            detector: BaseDetector,
            num_iterations: int = _DEFAULT_NUM_ITERATIONS,
    ):
        data = data.copy()  # copy the data to avoid modifying the original data
        indexers = [col for col in MultiIterationsPipeline._INDEXERS if col in data.columns]
        results = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(num_iterations):
                iter_results = {}
                for tr in data[cnst.TRIAL].unique():
                    trial_data = data[data[cnst.TRIAL] == tr]
                    res = detector.detect(
                        trial_data[cnst.T].values,
                        trial_data[cnst.X].values,
                        trial_data[cnst.Y].values
                    )
                    idx = tuple(trial_data[indexers].iloc[0].to_list())
                    iter_results[idx] = copy.deepcopy(res)  # deep copy to avoid overwriting the res object

                    # nullify detected saccades
                    detected_event_labels = res[cnst.GAZE][cnst.EVENT]
                    saccade_idxs = trial_data.index[detected_event_labels == cnfg.EVENT_LABELS.SACCADE]
                    data.loc[saccade_idxs, cnst.X] = np.nan
                    data.loc[saccade_idxs, cnst.Y] = np.nan

                results[f"{MultiIterationsPipeline._ITERATION_STR[:4]}{i + 1}"] = pd.Series(iter_results, name=i + 1)
        results = pd.DataFrame(results)
        results.index.names = indexers
        results.columns.name = MultiIterationsPipeline._ITERATION_STR
        return results
