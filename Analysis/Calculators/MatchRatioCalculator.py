import time
import warnings
from typing import Dict

import numpy as np
import pandas as pd
import pickle as pkl

from Analysis.Calculators.BaseCalculator import BaseCalculator


class MatchRatioCalculator(BaseCalculator):

    @classmethod
    def calculate(
            cls,
            file_path: str,
            events: pd.DataFrame,
            matches: Dict[str, pd.DataFrame],
            verbose=False,
    ) -> Dict[str, pd.DataFrame]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.time()
            if verbose:
                print(f"Calculating {cls._name()}...")
            try:
                matches_schemes = set(matches.keys())
                with open(file_path, 'rb') as f:
                    results = pkl.load(f)
                results_schemes = set(results.keys())
                missing_schemes = matches_schemes - results_schemes
                if missing_schemes:
                    missing_matches = {scheme: match for scheme, match in matches.items() if scheme in missing_schemes}
                    new_results = cls._calculate_impl(events, missing_matches)
                    results.update(new_results)
                    cls._save(results, file_path)
            except FileNotFoundError:
                results = cls._calculate_impl(events, matches)
                cls._save(results, file_path)
            end = time.time()
            if verbose:
                print(f"\tCompleted:\t{end - start:.2f}s")
        return results

    @classmethod
    def _calculate_impl(
            cls,
            events: pd.DataFrame,
            matches: Dict[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates the ratio of matched events to the total number of detected events for each detector.
        :param events: A DataFrame containing the detected events of each rater/detector.
        :param matches: A dictionary mapping each matching scheme to a DataFrame containing the matched events.
        :return: A dictionary mapping each matching scheme to a DataFrame containing the calculated match ratios.
        """
        event_counts = events.map(lambda cell: len(cell) if cell is not None else None)
        match_ratios = {}
        for scheme, matches_df in matches.items():
            match_counts = matches_df.map(lambda cell: len(cell) if pd.notnull(cell) else None)
            ratios = np.zeros_like(match_counts, dtype=float)
            for i in range(match_counts.index.size):
                for j in range(match_counts.columns.size):
                    gt_col, _pred_col = match_counts.columns[j]
                    ratios[i, j] = match_counts.iloc[i, j] / event_counts.iloc[i][gt_col]
            ratios = pd.DataFrame(ratios, index=match_counts.index, columns=match_counts.columns)
            match_ratios[scheme] = ratios
        return match_ratios

