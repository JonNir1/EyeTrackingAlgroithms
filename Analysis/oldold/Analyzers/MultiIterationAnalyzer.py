import time
import warnings
import copy

import numpy as np
import pandas as pd


import Config.constants as cnst
import Config.experiment_config as cnfg
from Analysis.oldold.Analyzers.EventFeaturesAnalyzer import EventFeaturesAnalyzer
from DataSetLoaders.DataSetFactory import DataSetFactory
from GazeDetectors.BaseDetector import BaseDetector


class MultiIterationAnalyzer(EventFeaturesAnalyzer):

    LAMBDA_STR = "λ"
    ITERATION_STR = "Iteration"
    DEFAULT_NUM_ITERATIONS = 5
    _INDEXERS = [cnst.TRIAL, cnst.SUBJECT, cnst.SUBJECT_ID, cnst.STIMULUS, f"{cnst.STIMULUS}_name"]

    @staticmethod
    def preprocess_dataset(dataset_name: str,
                           detector: BaseDetector = None,
                           num_iterations: int = DEFAULT_NUM_ITERATIONS,
                           verbose=False):
        """
        Preprocess the dataset by:
            1. Loading the dataset
            2. Detecting events using the given detectors
            3. Match events detected by each pair of detectors
            4. Extract pairs of (human-rater, detector) for future analysis

        :param dataset_name: The name of the dataset to load and preprocess.
        :param detector: A gaze-event detector to use for detecting events. If None, raise an error.
        :param num_iterations: The number of event-detection iterations.
        :param verbose: Whether to print the progress of the preprocessing.

        :return: A DataFrame containing the detected events in each iteration.
        """
        if detector is None:
            raise ValueError("A GazeDetector object must be provided to preprocess the dataset.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.time()
            if verbose:
                print(f"Preprocessing dataset `{dataset_name}` with {detector.name}...")
            dataset = DataSetFactory.load(dataset_name)
            detection_results = MultiIterationAnalyzer._detect_multiple_times(dataset,
                                                                              detector,
                                                                              num_iterations=num_iterations,
                                                                              events_only=False)
            detected_events = detection_results.map(lambda cell: cell[cnst.EVENTS])
            end = time.time()
            if verbose:
                print(f"\tPreprocessing:\t{end - start:.2f}s")
        return detected_events

    @staticmethod
    def _detect_multiple_times(data: pd.DataFrame,
                               detector: BaseDetector,
                               num_iterations: int = DEFAULT_NUM_ITERATIONS,
                               events_only: bool = True):
        data = data.copy()  # copy the data to avoid modifying the original data
        indexers = [col for col in MultiIterationAnalyzer._INDEXERS if col in data.columns]
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

                results[i + 1] = pd.Series(iter_results, name=i + 1)
        results = pd.DataFrame(results)
        results.index.names = indexers
        results.columns.name = MultiIterationAnalyzer.ITERATION_STR
        if events_only:
            results = results.map(lambda cell: cell[cnst.EVENTS])
        return pd.DataFrame(results)
