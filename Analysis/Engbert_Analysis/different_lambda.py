import numpy as np
import plotly.io as pio

from GazeDetectors.EngbertDetector import EngbertDetector

import Analysis.figures as figs
import Analysis.scarfplot as scarf
from Analysis.Analyzer import Analyzer

pio.renderers.default = "browser"

DATASET = "Lund2013"
LAMBDA_STR = "λ"
COL_MAPPER = lambda col: col[col.index(LAMBDA_STR):col.index(",")].replace("'", "") if LAMBDA_STR in col else col
DETECTORS = [EngbertDetector(lambdaa=lmda) for lmda in np.arange(1, 3)]

# %%
# Pre-Process
samples, events, detector_results_df, event_matches, comparison_columns = Analyzer.preprocess_dataset(DATASET,
                                                                                                      detectors=DETECTORS,
                                                                                                      column_mapper=COL_MAPPER,
                                                                                                      verbose=True)

# %%
# Compare scarfplots
scarfplot_figures = {}
for i, idx in enumerate(samples.index):
    num_samples = samples.loc[idx].map(len).max()  # Number of samples in the longest detected sequence
    t = np.arange(num_samples)
    detected_events = samples.loc[idx]
    fig = scarf.compare_scarfplots(t, *detected_events.to_list(), names=detected_events.index)
    scarfplot_figures[idx] = fig
    fig.show()
    break

# %%
# TODO: repeat the following analysis only for fixations/saccades

# %%
all_event_metrics = Analyzer.analyze(events, event_matches, samples, verbose=True)
sample_metrics = all_event_metrics["Sample Metrics"]
event_features = all_event_metrics["Event Features"]
event_matching_ratios = all_event_metrics["Event Matching Ratios"]
event_matching_feature_diffs = all_event_metrics["Event Matching Feature Diffs"]

# %%
# Compare to Ground Truth - Sample-by-Sample
sample_metric_figures = {}
for metric, metric_df in sample_metrics.items():
    fig = figs.distributions_grid(
        metric_df[comparison_columns],
        title=f"{DATASET.upper()}:\t\tSample-Level {metric.title()}",
        pdf_min_val=0 if "Transition Matrix" not in metric else None,
        pdf_max_val=1 if "Transition Matrix" not in metric else None,
        column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
    )
    sample_metric_figures[metric] = fig
    fig.show()

# %%
# Compare to Ground Truth - Event features
event_feature_figures = {}
for feature, feature_df in event_features.items():
    if feature == "Counts":
        title = f"{DATASET.upper()}:\t\tEvent {feature.title()}"
    else:
        title = f"{DATASET.upper()}:\t\tEvents' {feature.title()} Distribution"
    feature_figure = figs.distributions_grid(
        feature_df[comparison_columns],
        title=title,
        pdf_min_val=0,
        pdf_max_val=1,
        show_counts=feature == "Counts",
    )
    feature_figure.show()
    event_feature_figures[feature] = feature_figure

# %%
# Compare to Ground Truth - Event Matching
event_matching_figures = {}

event_matching_ratios_figure = figs.distributions_grid(
    event_matching_ratios["Match Ratio"][comparison_columns],
    title=f"{DATASET.upper()}:\t\tEvent-Matching Ratios",
    pdf_min_val=0,
    pdf_max_val=100,
    column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
)
event_matching_ratios_figure.show()
event_feature_figures["Match Ratio"] = event_matching_ratios_figure

for feature, feature_df in event_matching_feature_diffs.items():
    feature_figure = figs.distributions_grid(
        feature_df[comparison_columns],
        title=f"{DATASET.upper()}:\t\tMatched-Events' {feature.title()} Distribution",
        column_title_mapper=lambda col: f"{col[0]}→{col[1]}",
        pdf_min_val=0 if feature == "IoU" else None,
        pdf_max_val=1 if feature == "IoU" else None,
    )
    feature_figure.show()
    event_matching_figures[feature] = feature_figure
