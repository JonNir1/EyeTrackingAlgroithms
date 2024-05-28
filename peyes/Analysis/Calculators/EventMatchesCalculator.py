import time
import warnings
from typing import Dict

import pandas as pd
import pickle as pkl

from peyes.Analysis.Calculators.BaseCalculator import BaseCalculator
from peyes.Analysis.EventMatcher import EventMatcher as Matcher


class EventMatchesCalculator(BaseCalculator):

    @classmethod
    def calculate(
            cls,
            file_path: str,
            events_df: pd.DataFrame,
            matching_schemes: Dict[str, Dict[str, float]],
            verbose=False,
    ) -> Dict[str, pd.DataFrame]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.time()
            if verbose:
                print(f"Calculating {cls._name()}...")
            try:
                with open(file_path, 'rb') as f:
                    results = pkl.load(f)
                missing_schemes = {
                    scheme_name: scheme_params for scheme_name, scheme_params in matching_schemes.items()
                    if scheme_name not in results.keys()
                }
                if missing_schemes:
                    new_results = cls._calculate_impl(events_df, missing_schemes)
                    results.update(new_results)
                    cls._save(results, file_path)
            except FileNotFoundError:
                results = cls._calculate_impl(events_df, matching_schemes)
                cls._save(results, file_path)
            end = time.time()
            if verbose:
                print(f"\tCompleted:\t{end - start:.2f}s")
        return results

    @classmethod
    def _calculate_impl(
            cls,
            events: pd.DataFrame,
            matching_schemes: Dict[str, Dict[str, float]],
    ) -> Dict[str, pd.DataFrame]:
        matches = {}
        for scheme_name, scheme_kwargs in matching_schemes.items():
            scheme_name = scheme_name.lower()
            matched_events = Matcher.match_events(events,
                                                  match_by=scheme_kwargs.pop("match_by", scheme_name),
                                                  is_symmetric=False,
                                                  **scheme_kwargs)
            matches[scheme_name] = matched_events
        return matches
