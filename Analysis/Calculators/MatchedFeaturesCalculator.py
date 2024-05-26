import time
import warnings
from typing import Set, Dict

import numpy as np
import pandas as pd
import pickle as pkl

import Config.experiment_config as cnfg
from Analysis.Calculators.BaseCalculator import BaseCalculator


class MatchedFeaturesCalculator(BaseCalculator):
    MATCHED_EVENT_FEATURES_WITHIN = {
        "Start Time", "End Time", "Duration", "Amplitude", "Azimuth", "Peak Velocity",
    }
    MATCHED_EVENT_FEATURES_BETWEEN = {
        "L2 Timing Difference", "IoU", "Overlap Time", "CoM Distance"
        # uninteresting:
        # "Duration Difference", "Amplitude Difference", "Azimuth Difference",
        # "Peak Velocity Difference", "Onset Jitter", "Offset Jitter"
    }

    @classmethod
    def calculate(
            cls,
            file_path: str,
            matches: Dict[str, pd.DataFrame],
            feature_names: Set[str],
            verbose=False,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.time()
            if verbose:
                print(f"Calculating {cls._name()}...")
            try:
                matches_schemes = set(matches.keys())
                with open(file_path, 'rb') as f:
                    results = pkl.load(f)
                results_features = set(results.keys())
                results_schemes = set(results[list(results_features)[0]].keys())

                # calculate missing schemes for existing features
                missing_schemes = matches_schemes - results_schemes
                if missing_schemes:
                    missing_matches = {scheme: match for scheme, match in matches.items() if scheme in missing_schemes}
                    new_results = cls._calculate_impl(missing_matches, results_features)
                    for feature in results_features:
                        results[feature].update(new_results[feature])

                # calculate for missing features
                missing_features = feature_names - results_features
                if missing_features:
                    new_results = cls._calculate_impl(matches, missing_features)
                    results.update(new_results)

                if missing_schemes or missing_features:
                    cls._save(results, file_path)
            except FileNotFoundError:
                results = cls._calculate_impl(matches, feature_names)
                cls._save(results, file_path)
            end = time.time()
            if verbose:
                print(f"\tCompleted:\t{end - start:.2f}s")
        return results

    @classmethod
    def _calculate_impl(
            cls,
            matches: Dict[str, pd.DataFrame],
            feature_names: Set[str],
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculates features for matched events for each matching scheme.
        :param matches: A dictionary mapping each matching scheme to a DataFrame containing the matched events.
        :param feature_names: A set of feature names to calculate.
        :return: A dictionary mapping each feature to a dictionary of matching schemes to a DataFrame containing the
            calculated feature values.
        """
        matched_results = {}
        for feature in feature_names:
            matched_results[feature] = {}
            for scheme, matches_df in matches.items():
                if feature in cls.MATCHED_EVENT_FEATURES_WITHIN:
                    attr = feature.lower().replace(" ", "_")
                    computed = matches_df.map(
                        lambda cell: [(getattr(k, attr), getattr(v, attr)) for k, v in cell.items()
                                      if hasattr(k, attr) and hasattr(v, attr)]
                        if pd.notnull(cell) else np.nan
                    )
                elif feature in cls.MATCHED_EVENT_FEATURES_BETWEEN:
                    computed = cls.__calculate_dual_feature_impl(matches_df, feature)
                else:
                    raise NotImplementedError(f"Unknown feature for matched events:\t{feature}")
                matched_results[feature][scheme] = computed
        return matched_results

    @staticmethod
    def __calculate_dual_feature_impl(matches_df: pd.DataFrame, feature: str) -> pd.DataFrame:
        if feature == "Onset Jitter":
            feature_df = matches_df.map(
                lambda cell: [k.start_time - v.start_time for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "Offset Jitter":
            feature_df = matches_df.map(
                lambda cell: [k.end_time - v.end_time for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "L2 Timing Difference":
            feature_df = matches_df.map(
                lambda cell: [k.l2_timing_offset(v) for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "IoU":
            feature_df = matches_df.map(
                lambda cell: [k.intersection_over_union(v) for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "Overlap Time":
            feature_df = matches_df.map(
                lambda cell: [k.overlap_time(v) for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "CoM Distance":
            feature_df = matches_df.map(
                lambda cell: [
                    k.center_distance(v) for k, v in cell.items()
                    if k.event_label == v.event_label == cnfg.EVENT_LABELS.FIXATION
                ] if pd.notnull(cell) else np.nan
            )
        elif feature == "Duration Difference":
            feature_df = matches_df.map(
                lambda cell: [k.duration - v.duration for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "Amplitude Difference":
            feature_df = matches_df.map(
                lambda cell: [k.amplitude - v.amplitude for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "Azimuth Difference":
            feature_df = matches_df.map(
                lambda cell: [k.azimuth - v.azimuth for k, v in cell.items()] if pd.notnull(cell) else np.nan
            )
        elif feature == "Peak Velocity Difference":
            feature_df = matches_df.map(
                lambda cell: [k.peak_velocity_px - v.peak_velocity_px for k, v in cell.items()] if pd.notnull(
                    cell) else np.nan
            )
        elif feature == "Match Ratio":
            raise ValueError("Match Ratio feature should be calculated separately.")
        else:
            raise ValueError(f"Unknown feature: {feature}")
        return feature_df
