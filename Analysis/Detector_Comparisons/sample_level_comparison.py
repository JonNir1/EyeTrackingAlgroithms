import time
import warnings
import itertools
from typing import List

import pandas as pd
import plotly.io as pio

import Config.constants as cnst
from DataSetLoaders.DataSetFactory import DataSetFactory
import Analysis.comparisons as cmps
import Analysis.figures as figs

pio.renderers.default = "browser"

###############################

SAMPLE_METRICS = {
    "Accuracy": "acc",
    "Levenshtein Distance": "lev",
    "Cohen's Kappa": "kappa",
    "Mathew's Correlation": "mcc",
    "Transition Matrix l2-norm": "frobenius",
    "Transition Matrix KL-Divergence": "kl"
}


def preprocess_dataset(dataset_name: str, verbose=False) -> (pd.DataFrame, List[str]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if verbose:
            print(f"Preprocessing dataset `{dataset_name}`...")
        start = time.time()
        samples_df, _, _ = DataSetFactory.load_and_process(dataset_name)
        samples_df.rename(columns=lambda col: col[:col.index("ector")] if "ector" in col else col, inplace=True)

        # extract column-pairs to compare
        rater_names = [col.upper() for col in samples_df.columns if len(col) == 2]
        detector_names = [col for col in samples_df.columns if "det" in col.lower()]
        rater_rater_pairs = list(itertools.combinations(sorted(rater_names), 2))
        rater_detector_pairs = [(rater, detector) for rater in rater_names for detector in detector_names]
        comparison_columns = rater_rater_pairs + rater_detector_pairs
        end = time.time()
        if verbose:
            print(f"\tPreprocessing:\t{end - start:.2f}s")
    return samples_df, comparison_columns


def calculate_sample_metrics(samples_df: pd.DataFrame,
                             comparison_columns: List[str],
                             show_distributions=False,
                             verbose=False):
    global_start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = {}
        for metric_name, metric_short in SAMPLE_METRICS.items():
            start = time.time()
            computed_metric = cmps.compare_samples(samples=samples_df, metric=metric_short, group_by=cnst.STIMULUS)
            results[metric_name] = computed_metric
            if show_distributions:
                distribution_fig = figs.distributions_grid(
                    computed_metric[comparison_columns],
                    plot_type="violin",
                    title=f"Sample-Level `{metric_name.title()}` Distribution",
                    column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
                )
                distribution_fig.show()
            end = time.time()
            if verbose:
                print(f"\tCalculating `{metric_name}`:\t{end - start:.2f}s")
        global_end = time.time()
        if verbose:
            print(f"Total time:\t{global_end - global_start:.2f}s\n")
    return results


# %%
# Lund2013 Dataset
lund_samples, lund_comparison_columns = preprocess_dataset("Lund2013", verbose=True)
lund_sample_metrics = calculate_sample_metrics(lund_samples, lund_comparison_columns, show_distributions=True, verbose=True)

# %%
# IRF Dataset
irf_samples, irf_comparison_columns = preprocess_dataset("IRF", verbose=True)
irf_sample_metrics = calculate_sample_metrics(irf_samples, irf_comparison_columns, show_distributions=True, verbose=True)
