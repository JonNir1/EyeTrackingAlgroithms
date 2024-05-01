from typing import Set, Dict

import numpy as np
import pandas as pd

import Config.experiment_config as cnfg
from Analysis.Calculators.BaseCalculator import BaseCalculator


class EventFeaturesCalculator(BaseCalculator):

    @classmethod
    def _calculate_impl(
            cls,
            events: pd.DataFrame,
            feature_names: Set[str],
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates event-level features for each column in the given DataFrame.
        :param events_df: A DataFrame containing the detected events of each rater/detector.
        :param feature_names: A set of feature names to calculate. If None, the default set of features will be calculated.
        :param verbose: Whether to print the progress of the feature calculation.
        :return: A dictionary mapping each feature to a DataFrame containing the calculated feature values.
        """
        event_features = {}
        for feature in feature_names:
            feat = feature.lower()
            if feat in {"count", "counts", "event count", "event counts"}:
                computed = cls._event_counts_impl(events)
            elif feat in {"micro-saccade ratio", "microsaccade ratio"}:
                computed = cls._microsaccade_ratio_impl(events)
            else:
                attr = feat.lower().replace(" ", "_")
                computed = events.map(lambda cell: [getattr(e, attr) for e in cell if hasattr(e, attr)])
            event_features[feature] = computed
        return event_features

    @staticmethod
    def _event_counts_impl(events: pd.DataFrame) -> pd.DataFrame:
        """
        Counts the number of detected events for each detector by type of event.
        :param events: A DataFrame containing the detected events of each rater/detector.
        :return: A DataFrame containing the count of events detected by each rater/detector (cols), grouped by the given
            criteria (rows).
        """
        from typing import List, Union
        from GazeEvents.BaseEvent import BaseEvent

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
    def _microsaccade_ratio_impl(events: pd.DataFrame,
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
