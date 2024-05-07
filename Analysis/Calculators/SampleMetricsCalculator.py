from typing import Set, Dict

import pandas as pd

from Analysis.Calculators.BaseCalculator import BaseCalculator
import Utils.metrics as metrics
import Analysis.helpers as hlp


class SampleMetricsCalculator(BaseCalculator):

    @classmethod
    def _calculate_impl(
            cls,
            samples: pd.DataFrame,
            metric_names: Set[str],
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculates sample-level metrics for every pair of columns in the given DataFrame, and groups the results by the
            stimulus.
        :param samples: A DataFrame containing the label-per-sample of each rater/detector.
        :param metric_names: A set of metric names to calculate. If None, the default set of metrics will be calculated.
        :return: A dictionary mapping each metric to a DataFrame containing the calculated metric values.
        """
        sample_metrics = {}
        for metric in metric_names:
            met = metric.lower()
            if met in {"acc", "accuracy", "balanced accuracy"}:
                metric_func = metrics.balanced_accuracy
            elif met in {"lev", "levenshtein", "levenshtein ratio"}:
                metric_func = metrics.levenshtein_ratio
            elif met in {"kappa", "cohen kappa", "cohen's kappa"}:
                metric_func = metrics.cohen_kappa
            elif met in {"mcc", "mathew's correlation", "mathews correlation"}:
                metric_func = metrics.matthews_correlation
            elif met in {"confusion matrix", "confusion"}:
                metric_func = metrics.confusion_matrix
            elif met in {"fro", "frobenius", "l2", "transition matrix l2-norm"}:
                metric_func = lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="fro")
            elif met in {"kl", "kl divergence", "kullback leibler", "transition matrix kl-divergence"}:
                metric_func = lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="kl")
            else:
                raise NotImplementedError(f"Unknown metric for samples:\t{metric}")
            computed = hlp.apply_on_column_pairs(samples, metric_func, is_symmetric=False)
            sample_metrics[metric] = computed
        return sample_metrics
