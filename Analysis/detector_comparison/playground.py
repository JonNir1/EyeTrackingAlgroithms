import numpy as np
import pandas as pd
import scipy.stats as stat

import Config.constants as cnst
import Config.experiment_config as cnfg
from Analysis.detector_comparison.DetectorComparisonAnalyzer import DetectorComparisonAnalyzer

DATASET = "Lund2013"
COL_MAPPER = lambda col: col[:col.index("ector")] if "ector" in col else col

samples, events, _, event_matches, comparison_columns = DetectorComparisonAnalyzer.preprocess_dataset(DATASET,
                                                                                                      column_mapper=COL_MAPPER,
                                                                                                      verbose=True)

# %%
#############################################
# All-Event Metrics
all_event_metrics = DetectorComparisonAnalyzer.analyze(events, event_matches, samples, verbose=True)
sample_metrics = all_event_metrics[DetectorComparisonAnalyzer.SAMPLE_METRICS_STR]
event_features = all_event_metrics[DetectorComparisonAnalyzer.EVENT_FEATURES_STR]
event_matching_ratios = all_event_metrics[DetectorComparisonAnalyzer.MATCH_RATIO_STR]
event_matching_feature_diffs = all_event_metrics[DetectorComparisonAnalyzer.MATCH_FEATURES_STR]

# %%
#############################################
event_amplitudes = event_features[cnst.AMPLITUDE.capitalize()].map(lambda cell: [v for v in cell if not np.isnan(v)])
amplitudes_test_res = DetectorComparisonAnalyzer.event_feature_statistical_comparison(event_amplitudes,
                                                                                      "u")

# TODO: start from here, implement this as a function in BaseAnalyzer?
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import Visualization.p_value_heatmap as pvh

pio.renderers.default = "browser"

z = amplitudes_test_res.xs(cnst.P_VALUE, axis=1, level=2)

fig = pvh.heatmap_grid(z, alpha=0.05, correction="bonferroni", add_annotations=True, ignore_above_critical=True)
fig.show()

# %%
#############################################
# Fixation Metrics
fixation_metrics = DetectorComparisonAnalyzer.analyze(events, event_matches,
                                                      ignore_events={v for v in cnfg.EVENT_LABELS if v != cnfg.EVENT_LABELS.FIXATION},
                                                      verbose=True)
fixation_features = fixation_metrics[DetectorComparisonAnalyzer.EVENT_FEATURES_STR]
fixation_matching_ratios = fixation_metrics[DetectorComparisonAnalyzer.MATCH_RATIO_STR]
fixation_matching_feature_diffs = fixation_metrics["Event Matching Feature Diffs"]

# %%
#############################################
# Saccade Metrics
saccade_metrics = DetectorComparisonAnalyzer.analyze(events, event_matches,
                                                     ignore_events={v for v in cnfg.EVENT_LABELS if v != cnfg.EVENT_LABELS.SACCADE},
                                                     verbose=True)
saccade_features = saccade_metrics[DetectorComparisonAnalyzer.EVENT_FEATURES_STR]
saccade_matching_ratios = saccade_metrics[DetectorComparisonAnalyzer.MATCH_RATIO_STR]
saccade_matching_feature_diffs = saccade_metrics[DetectorComparisonAnalyzer.MATCH_FEATURES_STR]
