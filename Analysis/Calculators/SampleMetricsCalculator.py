from typing import Set, Dict

import pandas as pd

from Analysis.Calculators.BaseCalculator import BaseCalculator
from GazeEvents.helpers import count_labels_or_events
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
            if metric.lower() in {"count", "counts", "label counts"}:
                computed = samples.map(count_labels_or_events)
            else:
                computed = cls.__calculate_on_column_pairs(samples, metric)
            sample_metrics[metric] = computed
        return sample_metrics

    @staticmethod
    def __calculate_on_column_pairs(samples: pd.DataFrame, metric_name: str) -> pd.DataFrame:
        met = metric_name.lower()
        if met in {"acc", "accuracy"}:
            metric_func = metrics.accuracy
        elif met in {"balanced accuracy", "balanced acc", "weighted accuracy", "weighted acc"}:
            metric_func = metrics.balanced_accuracy
        elif met in {"precision", "prec"}:
            metric_func = lambda s1, s2: metrics.precision(s1, s2)
        elif met in {"recall", "sensitivity"}:
            metric_func = lambda s1, s2: metrics.recall(s1, s2)
        elif met in {"f1", "f1-score", "f1 score"}:
            metric_func = lambda s1, s2: metrics.f1_score(s1, s2)
        elif met in {"lev", "levenshtein", "nld", "1-nld", "complement nld"}:
            metric_func = metrics.complement_nld
        elif met in {"kappa", "cohen kappa", "cohen's kappa"}:
            metric_func = metrics.cohen_kappa
        elif met in {"mcc", "mathew's correlation", "mathews correlation"}:
            metric_func = metrics.matthews_correlation
        elif met in {"confusion matrix", "confusion"}:
            metric_func = metrics.confusion_matrix
        elif met in {"dkl", "kl", "kl divergence", "kullback leibler", "transition matrix kl-divergence"}:
            metric_func = lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="kl")
        elif met in {"fro", "frobenius", "l2", "transition matrix l2-norm"}:
            metric_func = lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="fro")
        elif met in {"l1", "manhattan", "transition matrix l1-norm"}:
            metric_func = lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="l1")
        elif met in {"linf", "infinity", "max", "inf", "transition matrix l-infinity norm"}:
            metric_func = lambda s1, s2: metrics.transition_matrix_distance(s1, s2, norm="linf")
        else:
            raise NotImplementedError(f"Unknown metric for samples:\t{metric_name}")
        computed = hlp.apply_on_column_pairs(samples, metric_func, is_symmetric=False)
        return computed
