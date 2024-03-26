import warnings
import pandas as pd
from abc import ABC
from typing import List

import Config.constants as cnst
from DataSetLoaders.BaseDataSetLoader import BaseDataSetLoader
from GazeDetectors.BaseDetector import BaseDetector
from GazeEvents.EventFactory import EventFactory


class DataSetFactory(ABC):
    _INDEXERS = [cnst.TRIAL, cnst.SUBJECT_ID, cnst.STIMULUS, f"{cnst.STIMULUS}_name"]

    @staticmethod
    def load_and_process(name: str, raters: List[str], detectors: List[BaseDetector]) -> (pd.DataFrame, pd.DataFrame):
        """
        Loads the dataset and detects events in it based on human-annotations and detection algorithms.
        Returns two dataframes:
            - The sequence of event-labels per sample for each trial and rater/detector.
            - The sequence of event objects for each trial and rater/detector.
        """
        loader_class = [c for c in BaseDataSetLoader.__subclasses__() if c.dataset_name().lower() == name.lower()]
        if not loader_class:
            raise ValueError(f"Dataset loader for {name} not found")
        dataset = loader_class[0].load(should_save=False)
        return DataSetFactory.process(dataset, raters, detectors)

    @staticmethod
    def process(dataset: pd.DataFrame,
                raters: List[str],
                detectors: List[BaseDetector]) -> (pd.DataFrame, pd.DataFrame):
        """
        Detects events in the dataset based on human-annotations in the dataset and the provided detectors.
        Returns two dataframes:
             - The sequence of event-labels per sample for each trial and rater/detector.
             - The sequence of event objects for each trial and rater/detector.
        """
        indexers = [col for col in DataSetFactory._INDEXERS if col in dataset.columns]
        samples_dict, event_dict = {}, {}
        for trial_num in dataset[cnst.TRIAL].unique():
            trial_data = dataset[dataset[cnst.TRIAL] == trial_num]
            labels, events = DataSetFactory._process_trial(trial_data, raters, detectors)
            _, subject_id, stimulus, stimulus_name = trial_data[indexers].iloc[0]
            samples_dict[(trial_num, subject_id, stimulus, stimulus_name)] = labels
            event_dict[(trial_num, subject_id, stimulus, stimulus_name)] = events
        # create output dataframes
        samples_df = pd.DataFrame.from_dict(samples_dict, orient="index").sort_index()
        samples_df.index.names = indexers
        events_df = pd.DataFrame.from_dict(event_dict, orient="index").sort_index()
        events_df.index.names = indexers
        return samples_df, events_df

    @staticmethod
    def _process_trial(trial_data: pd.DataFrame, raters: List[str], detectors: List[BaseDetector]):
        viewer_distance = trial_data["viewer_distance_cm"].to_numpy()[0]
        pixel_size = trial_data["pixel_size_cm"].to_numpy()[0]
        with warnings.catch_warnings(action="ignore"):
            labels = {rater: trial_data[rater] for rater in raters}
            events = {
                rater: EventFactory.make_from_gaze_data(
                    trial_data, vd=viewer_distance, ps=pixel_size, column_mapping={rater: cnst.EVENT}
                ) for rater in raters}
        for det in detectors:
            with warnings.catch_warnings(action="ignore"):
                res = det.detect(
                    t=trial_data[cnst.T].to_numpy(),
                    x=trial_data[cnst.X].to_numpy(),
                    y=trial_data[cnst.Y].to_numpy()
                )
            labels[det.name] = res[cnst.GAZE][cnst.EVENT]
            events[det.name] = res[cnst.EVENTS]
        return labels, events
