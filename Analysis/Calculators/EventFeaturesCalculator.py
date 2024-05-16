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
        :param events: A DataFrame containing the detected events of each rater/detector.
        :param feature_names: A set of feature names to calculate. If None, the default set of features will be calculated.
        :return: A dictionary mapping each feature to a DataFrame containing the calculated feature values.
        """
        event_features = {}
        for feature in feature_names:
            feat = feature.lower()
            if feat in {"count", "counts", "event count", "event counts"}:
                from GazeEvents.helpers import count_labels_or_events
                computed = events.map(count_labels_or_events)
            elif feat in {"micro-saccade ratio", "microsaccade ratio"}:
                computed = cls._microsaccade_ratio_impl(events)
            else:
                attr = feat.lower().replace(" ", "_")
                computed = events.map(
                    lambda cell: [getattr(e, attr) for e in cell if hasattr(e, attr)] if cell is not None else None
                )
            event_features[feature] = computed
        return event_features

    @staticmethod
    def _microsaccade_ratio_impl(events: pd.DataFrame,
                                 threshold_amplitude: float = cnfg.MICROSACCADE_AMPLITUDE_THRESHOLD) -> pd.DataFrame:
        saccades = events.map(
            lambda cell: [e for e in cell if e.event_label == cnfg.EVENT_LABELS.SACCADE]
            if cell is not None else None
        )
        saccades_count = saccades.map(
            lambda cell: len(cell) if cell is not None else 0
        ).to_numpy()
        microsaccades = saccades.map(
            lambda cell: [e for e in cell if e.amplitude < threshold_amplitude]
            if cell is not None else None
        )
        microsaccades_count = microsaccades.map(
            lambda cell: len(cell) if cell is not None else 0
        ).to_numpy()

        ratios = np.divide(microsaccades_count, saccades_count,
                           out=np.full_like(saccades_count, fill_value=np.nan, dtype=float),  # fill NaN if denom is 0
                           where=saccades_count != 0)
        ratios = pd.DataFrame(ratios, index=events.index, columns=events.columns)
        return ratios
