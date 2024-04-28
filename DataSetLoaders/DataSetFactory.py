import warnings
import copy
from abc import ABC
from typing import List, Callable

import pandas as pd

import Config.constants as cnst
from DataSetLoaders.BaseDataSetLoader import BaseDataSetLoader
from GazeDetectors.BaseDetector import BaseDetector
from GazeEvents.EventFactory import EventFactory


class DataSetFactory(ABC):
    _INDEXERS = [cnst.TRIAL, cnst.SUBJECT, cnst.SUBJECT_ID, cnst.STIMULUS, f"{cnst.STIMULUS}_name"]

    @staticmethod
    def load_and_detect(name: str,
                        detectors: List[BaseDetector],
                        raters: List[str] = None,
                        column_mapper: Callable[[str], str] = None) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Loads the dataset and detects events in it based on human-annotations and detection algorithms.
        Returns three dataframes:
             - The sequence of event-labels per sample for each trial and rater/detector.
             - The sequence of event objects for each trial and rater/detector.
             - The full detection results for each detector, over every trial.

        :param dataset: The dataset to detect events in.
        :param raters: The list of human raters in the dataset.
        :param detectors: The list of detectors to use for detecting events.
        :param column_mapper: A function to map the column names of the output DataFrames (default: identity).
        """
        dataset = DataSetFactory.load(name)
        raters = raters if raters else DataSetFactory.__get_default_raters(name)
        return DataSetFactory.detect(dataset, raters, detectors, column_mapper)

    @staticmethod
    def load(name: str) -> pd.DataFrame:
        """ Loads the dataset. """
        from DataSetLoaders.GazeComDataSetLoader import GazeComDataSetLoader
        from DataSetLoaders.HFCDataSetLoader import HFCDataSetLoader
        from DataSetLoaders.IRFDataSetLoader import IRFDataSetLoader
        from DataSetLoaders.Lund2013DataSetLoader import Lund2013DataSetLoader
        loader_class = [c for c in BaseDataSetLoader.__subclasses__() if c.dataset_name().lower() == name.lower()]
        if not loader_class:
            raise ValueError(f"Dataset loader for {name} not found")
        loader_class = loader_class[0]
        dataset = loader_class.load(should_save=False)
        return dataset

    @staticmethod
    def detect(dataset: pd.DataFrame,
               raters: List[str],
               detectors: List[BaseDetector],
               column_mapper: Callable[[str], str] = None) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Detects events in the dataset based on human-annotations in the dataset and the provided detectors.
        Returns three dataframes:
             - The sequence of event-labels per sample for each trial and rater/detector.
             - The sequence of event objects for each trial and rater/detector.
             - The full detection results for each detector, over every trial.

        :param dataset: The dataset to detect events in.
        :param raters: The list of human raters in the dataset.
        :param detectors: The list of detectors to use for detecting events.
        :param column_mapper: A function to map the column names of the output DataFrames (default: identity).
        """
        indexers = [col for col in DataSetFactory._INDEXERS if col in dataset.columns]
        samples_dict, event_dict, detector_results_dict = {}, {}, {}
        for trial_num in dataset[cnst.TRIAL].unique():
            trial_data = dataset[dataset[cnst.TRIAL] == trial_num]
            labels, events, detector_results = DataSetFactory._process_trial(trial_data, raters, detectors)
            idx = tuple(trial_data[indexers].iloc[0].to_list())
            samples_dict[idx] = labels
            event_dict[idx] = events
            detector_results_dict[idx] = copy.deepcopy(detector_results)

        # create output dataframes
        column_mapper = column_mapper or (lambda col: col)
        samples_df = pd.DataFrame.from_dict(samples_dict, orient="index").sort_index()
        samples_df.index.names = indexers
        samples_df.rename(columns=column_mapper, inplace=True)
        events_df = pd.DataFrame.from_dict(event_dict, orient="index").sort_index()
        events_df.index.names = indexers
        events_df.rename(columns=column_mapper, inplace=True)
        detector_results_df = pd.DataFrame.from_dict(detector_results_dict, orient="index").sort_index()
        detector_results_df.index.names = indexers
        detector_results_df.rename(columns=column_mapper, inplace=True)
        return samples_df, events_df, detector_results_df

    @staticmethod
    def _process_trial(trial_data: pd.DataFrame, raters: List[str], detectors: List[BaseDetector]):
        viewer_distance = trial_data["viewer_distance_cm"].to_numpy()[0]
        pixel_size = trial_data["pixel_size_cm"].to_numpy()[0]
        with warnings.catch_warnings(action="ignore"):
            labels = {
                rater: trial_data[rater] if rater in trial_data.columns and pd.notnull(trial_data[rater]).all()
                else [float("nan")] for rater in raters
            }
            events = {
                rater: EventFactory.make_from_gaze_data(
                    trial_data, vd=viewer_distance, ps=pixel_size, column_mapping={rater: cnst.EVENT}
                ) if rater in trial_data.columns else [float("nan")] for rater in raters
            }
        detector_results = {}
        for det in detectors:
            with warnings.catch_warnings(action="ignore"):
                res = det.detect(
                    t=trial_data[cnst.T].to_numpy(),
                    x=trial_data[cnst.X].to_numpy(),
                    y=trial_data[cnst.Y].to_numpy()
                )
            detector_results[det.name] = res
            labels[det.name] = res[cnst.GAZE][cnst.EVENT]
            events[det.name] = res[cnst.EVENTS]
        return labels, events, detector_results

    @staticmethod
    def __get_default_raters(dataset_name: str) -> List[str]:
        if dataset_name == "IRF":
            return ["RZ"]
        if dataset_name == "Lund2013":
            return ["MN", "RA"]
        raise ValueError(f"Unknown dataset name: {dataset_name}")
