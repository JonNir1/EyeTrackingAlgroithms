import numpy as np
import plotly.io as pio

from GazeDetectors.EngbertDetector import EngbertDetector
from Visualization.distributions_grid import *
import Visualization.scarfplot as scarf

from Analysis.Analyzers.SamplesAnalyzer import SamplesAnalyzer
from Analysis.Analyzers.EventFeaturesAnalyzer import EventFeaturesAnalyzer
from Analysis.Analyzers.MatchedEventsAnalyzer import MatchedEventsAnalyzer

pio.renderers.default = "browser"

DATASET = "Lund2013"
LAMBDA_STR = "λ"
COL_MAPPER = lambda col: col[col.index(LAMBDA_STR):col.index(",")].replace("'", "") if LAMBDA_STR in col else col
DETECTORS = [EngbertDetector(lambdaa=lmda) for lmda in np.arange(1, 7)]

# %%
###########################################
# Pre-Process & Analyze Samples
SAMPLES_STAT_TEST = "Wilcoxon"
samples, samples_comp_cols = SamplesAnalyzer.preprocess_dataset(DATASET,
                                                                detectors=DETECTORS,
                                                                column_mapper=COL_MAPPER,
                                                                verbose=True)
sample_metrics, sample_metric_stats = SamplesAnalyzer.analyze(samples, test_name=SAMPLES_STAT_TEST, verbose=True)
print(f"Available sample metrics: {list(sample_metrics.keys())}")

# %%
###########################################
# Pre-Process & Analyze Event Features
FEATURES_STAT_TEST = "Mann-Whitney"
events = EventFeaturesAnalyzer.preprocess_dataset(DATASET, column_mapper=COL_MAPPER, verbose=True)
event_features, event_feature_stats = EventFeaturesAnalyzer.analyze(events, None, test_name=FEATURES_STAT_TEST, verbose=True)
print(f"Available event features: {list(event_features.keys())}")

# %%
###########################################
# Pre-Process & Analyze Event Features
MATCHED_EVENTS_STAT_TEST = "Wilcoxon"
_, event_matches, matches_comp_cols = MatchedEventsAnalyzer.preprocess_dataset(DATASET,
                                                                               column_mapper=COL_MAPPER,
                                                                               verbose=True)
events_matched_features, events_matched_feature_stats = MatchedEventsAnalyzer.analyze(events,
                                                                                      ignore_events=None,
                                                                                      matches_df=event_matches,
                                                                                      paired_sample_test=MATCHED_EVENTS_STAT_TEST,
                                                                                      single_sample_test=MATCHED_EVENTS_STAT_TEST,
                                                                                      verbose=True)
print(f"Available matched-event feature differences: {list(events_matched_features.keys())}")

# %%
# Compare scarfplots
scarfplot_figures = {}
for i, idx in enumerate(samples.index):
    num_samples = samples.loc[idx].map(len).max()  # Number of samples in the longest detected sequence
    t = np.arange(num_samples)
    detected_events = samples.loc[idx]
    fig = scarf.scarfplots_comparison_figure(t, *detected_events.to_list(), names=detected_events.index)
    scarfplot_figures[idx] = fig
    fig.show()

# %%
# TODO: repeat the following analysis only for fixations/saccades

# %%
# Compare to Ground Truth - Sample-by-Sample

# show feature distributions
sample_metric_figures = {}
for metric in sample_metrics.keys():
    data = sample_metrics[metric]
    fig = distributions_grid(
        data=data[samples_comp_cols],
        title=f"{DATASET.upper()}:\t\tSample-Level {metric.title()}",
        pdf_min_val=0 if "Transition Matrix" not in metric else None,
        pdf_max_val=1 if "Transition Matrix" not in metric else None,
        column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
    )
    sample_metric_figures[metric] = fig
    fig.show()

# %%
# Compare to Ground Truth - Event features

# show feature distributions
all_events_distribution_figures = dict()
for feature in event_features.keys():
    if feature == "Counts":
        title = f"{DATASET.upper()}:\t\tEvent {feature.title()}"
    else:
        title = f"{DATASET.upper()}:\t\tEvents' {feature.title()} Distribution"
    fig = distributions_grid(
        data=event_features[feature],
        title=title,
        show_counts=feature == "Count",
        pdf_min_val=0, pdf_max_val=1,  # only applies if show_counts is False
    )
    all_events_distribution_figures[feature] = fig
    fig.show()

# %%
# Compare to Ground Truth - Event Matching

# show distributions
events_matched_features_distribution_figures = {}
for feature in events_matched_features.keys():
    data = events_matched_features[feature]
    fig = distributions_grid(
        data=data[events_matched_feature_stats],
        title=f"{DATASET.upper()}:\t\tMatched-Events' {feature.title()} Distribution",
        column_title_mapper=lambda col: f"{col[0]}→{col[1]}",
        pdf_min_val=0 if feature == "IoU" else None,
        pdf_max_val=1 if feature == "IoU" else None,
    )
    events_matched_features_distribution_figures[feature] = fig
    fig.show()
