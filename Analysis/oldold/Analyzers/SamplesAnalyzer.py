import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

import Analysis.helpers as hlp
import Utils.metrics as metrics
from Analysis.old.Analyzers.BaseAnalyzer import BaseAnalyzer
from GazeDetectors.BaseDetector import BaseDetector


class SamplesAnalyzer(BaseAnalyzer):

    SAMPLE_METRICS = {
        "Accuracy": "acc",
        "Levenshtein Ratio": "lev",
        "Cohen's Kappa": "kappa",
        "Mathew's Correlation": "mcc",
        "Transition Matrix l2-norm": "frobenius",
        "Transition Matrix KL-Divergence": "kl"
    }

    _DEFAULT_STATISTICAL_TEST = "Wilcoxon"

    @staticmethod
    def preprocess_dataset(dataset_name: str,
                           detectors: List[BaseDetector] = None,
                           verbose=False,
                           **kwargs) -> (pd.DataFrame, List[str]):
        """
        Loads the dataset and preprocesses it to extract the detected events by each rater/detector.
        """
        if verbose:
            print(f"Preprocessing dataset `{dataset_name}`...")
        start = time.time()
        samples_df, _, _ = super(SamplesAnalyzer, SamplesAnalyzer).preprocess_dataset(
            dataset_name, detectors, False, **kwargs
        )
        comparison_columns = SamplesAnalyzer._extract_rater_detector_pairs(samples_df)
        end = time.time()
        if verbose:
            print(f"\tPreprocessing:\t{end - start:.2f}s")
        return samples_df, comparison_columns

    @classmethod
    def calculate_observed_data(cls,
                                samples_df: pd.DataFrame,
                                verbose: bool = False,
                                **kwargs) -> Dict[str, pd.DataFrame]:
        with warnings.catch_warnings():
            if verbose:
                print("Extracting sample metrics...")
            warnings.simplefilter("ignore")
            results = {}
            for metric_name, metric_short in cls.SAMPLE_METRICS.items():
                start = time.time()
                computed = cls.__calc_sample_metric_impl(samples_df, metric_short)
                results[metric_name] = computed
                end = time.time()
                if verbose:
                    print(f"\tCalculating `{metric_name}`:\t{end - start:.2f}s")
        return results

    @classmethod
    def statistical_analysis(cls,
                             metrics_dict: Dict[str, pd.DataFrame],
                             test_name: str = _DEFAULT_STATISTICAL_TEST) -> Dict[str, pd.DataFrame]:
        """
        Performs a one-sample statistical test on the set calculated sample-metrics between two raters/detectors.
        The calculated metrics are compared to a null hypothesis value, which is determined by the metric being tested:
            - Accuracy, Levenshtein Ratio, Cohen's Kappa, Matthew's Correlation: 1
            - Transition Matrix l2-norm, Transition Matrix KL-Divergence: 0

        :param metrics_dict: A dictionary mapping a metric name to a DataFrame containing the calculated sample-metrics
            for each pair of raters/detectors. Each column represents a different pair of raters/detectors, and each
            cell contains a list of the calculated sample-metrics for each trial.
        :param test_name: The name of the statistical test to perform.
        :return: A DataFrame containing the results of the statistical test between each pair of raters/detectors.
        """
        stat_test = cls._get_statistical_test_func(test_name)
        results = {}
        for metric_name, metric_df in metrics_dict.items():
            metric_df = metric_df.map(lambda cell: [v for v in cell if not np.isnan(v)])
            null_hypothesis_value = 0 if "Transition Matrix" in metric_name else 1
            stat_res = metric_df.map(lambda cell: stat_test(cell, np.full_like(cell, null_hypothesis_value)))
            results[metric_name] = cls._rearrange_statistical_results(stat_res)
        return results

    @staticmethod
    def __calc_sample_metric_impl(samples: pd.DataFrame,
                                  metric_name: str) -> pd.DataFrame:
        # extract the function to calculate the metric
        if metric_name == "acc" or metric_name == "accuracy":
            metric_func = metrics.accuracy
        elif metric_name == "balanced accuracy":
            metric_func = metrics.balanced_accuracy
        elif metric_name == "lev" or metric_name == "levenshtein":
            metric_func = metrics.complement_nld
        elif metric_name == "kappa" or metric_name == "cohen kappa":
            metric_func = metrics.cohen_kappa
        elif metric_name == "mcc" or metric_name == "matthews correlation":
            metric_func = metrics.matthews_correlation
        elif metric_name == "fro" or metric_name == "frobenius" or metric_name == "l2":
            metric_func = lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="fro")
        elif metric_name == "kl" or metric_name == "kl divergence" or metric_name == "kullback leibler":
            metric_func = lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="kl")
        else:
            raise NotImplementedError(f"Unknown metric for samples:\t{metric_name}")

        # perform the calculation and aggregate over stimuli
        metric_values = hlp.apply_on_column_pairs(samples, metric_func, is_symmetric=True)
        return SamplesAnalyzer.group_and_aggregate(metric_values)
