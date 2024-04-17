import numpy as np
import plotly.io as pio

import Config.constants as cnst
import Config.experiment_config as cnfg
from Analysis.Analyzers.EventFeaturesAnalyzer import EventFeaturesAnalyzer
from Visualization.distributions_grid import distributions_grid
from Visualization.p_value_heatmap import heatmap_grid

pio.renderers.default = "browser"

DATASET = "Lund2013"
COL_MAPPER = lambda col: col[:col.index("ector")] if "ector" in col else col

STAT_TEST_NAME = "Mann-Whitney"
CRITICAL_VALUE = 0.05
CORRECTION = "Bonferroni"

events = EventFeaturesAnalyzer.preprocess_dataset(DATASET, column_mapper=COL_MAPPER, verbose=True)

# %%
#############################################
# All Events Features
event_features, event_feature_stats = EventFeaturesAnalyzer.analyze(events, STAT_TEST_NAME, None, verbose=True)
print(f"Available event features: {list(event_features.keys())}")

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
    )
    all_events_distribution_figures[feature] = fig
    fig.show()

# show p-value heatmaps
all_events_p_value_heatmaps = dict()
for feature in event_feature_stats.keys():
    p_values = event_feature_stats[feature].xs(cnst.P_VALUE, axis=1, level=2)
    fig = heatmap_grid(
        p_values,
        title=f"{DATASET.upper()}:\t\tStatistical Comparison of Events' {feature.title()}",
        critical_value=CRITICAL_VALUE,
        correction=CORRECTION,
        add_annotations=True,
        ignore_above_critical=True
    )
    all_events_p_value_heatmaps[feature] = fig
    fig.show()

del feature, title, p_values


# %%
#############################################
# Fixation Features
fixation_features, fixation_feature_stats = EventFeaturesAnalyzer.analyze(events,
                                                                          STAT_TEST_NAME,
                                                                          ignore_events={v for v in cnfg.EVENT_LABELS if
                                                                                         v != cnfg.EVENT_LABELS.FIXATION},
                                                                          verbose=True)

# show feature distributions
fixation_features_distribution_figures = dict()
for feature in fixation_features.keys():
    if feature == "Counts":
        title = f"{DATASET.upper()}:\t\tEvent {feature.title()}"
    else:
        title = f"{DATASET.upper()}:\t\tEvents' {feature.title()} Distribution"
    fig = distributions_grid(
        data=fixation_features[feature],
        title=title,
        show_counts=feature == "Counts",
    )
    fixation_features_distribution_figures[feature] = fig
    fig.show()

# show p-value heatmaps
fixation_p_value_heatmaps = dict()
for feature in fixation_feature_stats.keys():
    p_values = fixation_feature_stats[feature].xs(cnst.P_VALUE, axis=1, level=2)
    fig = heatmap_grid(
        p_values,
        title=f"{DATASET.upper()}:\t\tStatistical Comparison of Fixations' {feature.title()}",
        critical_value=CRITICAL_VALUE,
        correction=CORRECTION,
        add_annotations=True,
        ignore_above_critical=True
    )
    fixation_p_value_heatmaps[feature] = fig
    fig.show()

del feature, title, p_values

# %%
#############################################
# Saccade Features
saccade_features, saccade_feature_stats = EventFeaturesAnalyzer.analyze(events,
                                                                        STAT_TEST_NAME,
                                                                        ignore_events={v for v in cnfg.EVENT_LABELS if
                                                                                       v != cnfg.EVENT_LABELS.SACCADE},
                                                                        verbose=True)

# show feature distributions
saccade_features_distribution_figures = dict()
for feature in saccade_features.keys():
    if feature == "Counts":
        title = f"{DATASET.upper()}:\t\tEvent {feature.title()}"
    else:
        title = f"{DATASET.upper()}:\t\tEvents' {feature.title()} Distribution"
    fig = distributions_grid(
        data=saccade_features[feature],
        title=title,
        show_counts=feature == "Counts",
    )
    saccade_features_distribution_figures[feature] = fig
    fig.show()

# show p-value heatmaps
saccade_p_value_heatmaps = dict()
for feature in saccade_feature_stats.keys():
    p_values = saccade_feature_stats[feature].xs(cnst.P_VALUE, axis=1, level=2)
    fig = heatmap_grid(
        p_values,
        title=f"{DATASET.upper()}:\t\tStatistical Comparison of Saccades' {feature.title()}",
        critical_value=CRITICAL_VALUE,
        correction=CORRECTION,
        add_annotations=True,
        ignore_above_critical=True
    )
    saccade_p_value_heatmaps[feature] = fig
    fig.show()

del feature, title, p_values
