import time
import warnings
import itertools

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
    "Transition Matrix KL-Divergence": "kl"}


def calculate_all_metrics(dataset_name: str, show_distributions=False, verbose=False):
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples_df, _, _ = DataSetFactory.load_and_process(dataset_name)
        samples_df.rename(columns=lambda col: col[:col.index("ector")] if "ector" in col else col, inplace=True)
    end = time.time()
    if verbose:
        print(f"Finished preprocessing dataset `{dataset_name}` in {end - start:.2f} seconds.")

    # extract column-pairs to compare
    rater_names = [col.capitalize() for col in samples_df.columns if len(col) == 2]
    detector_names = [col for col in samples_df.columns if "detector" in col.lower()]
    rater_rater_pairs = list(itertools.combinations(sorted(rater_names), 2))
    rater_detector_pairs = [(rater, detector) for rater in rater_names for detector in detector_names]
    comparison_columns = rater_rater_pairs + rater_detector_pairs

    results = {}
    for metric_name, metric_short in SAMPLE_METRICS.items():
        start = time.time()
        computed_metric = cmps.compare_samples(samples=samples_df, metric=metric_short, group_by=cnst.STIMULUS)
        results[metric_name] = computed_metric
        if show_distributions:
            distribution_fig = figs.distributions_grid(
                computed_metric[comparison_columns],
                plot_type="violin",
                title=f"{dataset_name.upper()} Dataset:\t\tSample-Level `{metric_name.capitalize()}` Distribution",
                column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
            )
            distribution_fig.show()
        end = time.time()
        if verbose:
            print(f"\tFinished calculating `{metric_name}` in {end - start:.2f} seconds.")
    return results

###############################


lund_results = calculate_all_metrics("Lund2013", show_distributions=True)
irf_results = calculate_all_metrics("IRF", show_distributions=True)
