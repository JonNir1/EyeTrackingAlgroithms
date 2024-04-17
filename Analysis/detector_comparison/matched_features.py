import numpy as np
import plotly.io as pio

import Config.constants as cnst
import Config.experiment_config as cnfg
from Analysis.detector_comparison.DetectorComparisonAnalyzer import DetectorComparisonAnalyzer
from Visualization.distributions_grid import distributions_grid
from Visualization.p_value_heatmap import heatmap_grid

pio.renderers.default = "browser"

DATASET = "Lund2013"
COL_MAPPER = lambda col: col[:col.index("ector")] if "ector" in col else col

_, events, _, event_matches, comparison_columns = DetectorComparisonAnalyzer.preprocess_dataset(DATASET,
                                                                                                column_mapper=COL_MAPPER,
                                                                                                verbose=True)

# %%
#############################################
# All Events' Matched-Features
all_event_metrics = DetectorComparisonAnalyzer.analyze(events,
                                                       event_matches,
                                                       None,
                                                       verbose=True)
event_matching_feature_diffs = all_event_metrics[DetectorComparisonAnalyzer.MATCH_FEATURES_STR]
print(f"Available matched-event feature differences: {list(event_matching_feature_diffs.keys())}")

# show feature distributions
matched_events_feature_diff_distributions_figures = {}
for feature in event_matching_feature_diffs.keys():
    data = event_matching_feature_diffs[feature]
    fig = distributions_grid(
        data=data[comparison_columns],
        title=f"{DATASET.upper()}:\t\tMatched-Events' {feature.title()} Distribution",
        column_title_mapper=lambda col: f"{col[0]}→{col[1]}",
        pdf_min_val=0 if feature == "IoU" else None,
        pdf_max_val=1 if feature == "IoU" else None,
    )
    matched_events_feature_diff_distributions_figures[feature] = fig
    fig.show()

# show p-value heatmaps
# TODO

# %%
#############################################
# Fixations' Matched-Features
fixation_metrics = DetectorComparisonAnalyzer.analyze(events,
                                                      event_matches,
                                                      None,
                                                      ignore_events={v for v in cnfg.EVENT_LABELS if
                                                                     v != cnfg.EVENT_LABELS.FIXATION},
                                                      verbose=True)
fixation_matching_feature_diffs = fixation_metrics[DetectorComparisonAnalyzer.MATCH_FEATURES_STR]

# show feature distributions
matched_fixations_feature_diff_distributions_figures = {}
for feature in fixation_matching_feature_diffs.keys():
    data = fixation_matching_feature_diffs[feature]
    fig = distributions_grid(
        data=data[comparison_columns],
        title=f"{DATASET.upper()}:\t\tMatched-Events' {feature.title()} Distribution",
        column_title_mapper=lambda col: f"{col[0]}→{col[1]}",
        pdf_min_val=0 if feature == "IoU" else None,
        pdf_max_val=1 if feature == "IoU" else None,
    )
    matched_fixations_feature_diff_distributions_figures[feature] = fig
    fig.show()

# show p-value heatmaps
# TODO

# %%
#############################################
# Saccades' Matched-Features
saccade_metrics = DetectorComparisonAnalyzer.analyze(events,
                                                     event_matches,
                                                     None,
                                                     ignore_events={v for v in cnfg.EVENT_LABELS if
                                                                    v != cnfg.EVENT_LABELS.SACCADE},
                                                     verbose=True)
saccade_matching_feature_diffs = saccade_metrics[DetectorComparisonAnalyzer.MATCH_FEATURES_STR]

# show feature distributions
matched_saccades_feature_diff_distributions_figures = {}
for feature in saccade_matching_feature_diffs.keys():
    data = saccade_matching_feature_diffs[feature]
    fig = distributions_grid(
        data=data[comparison_columns],
        title=f"{DATASET.upper()}:\t\tMatched-Events' {feature.title()} Distribution",
        column_title_mapper=lambda col: f"{col[0]}→{col[1]}",
        pdf_min_val=0 if feature == "IoU" else None,
        pdf_max_val=1 if feature == "IoU" else None,
    )
    matched_saccades_feature_diff_distributions_figures[feature] = fig
    fig.show()

# show p-value heatmaps
# TODO
