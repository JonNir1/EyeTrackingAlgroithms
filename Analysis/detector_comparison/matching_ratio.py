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

STAT_TEST_NAME = "Mann-Whitney"
CRITICAL_VALUE = 0.05
CORRECTION = "bonferroni"

_, events, _, event_matches, comparison_columns = DetectorComparisonAnalyzer.preprocess_dataset(DATASET,
                                                                                                column_mapper=COL_MAPPER,
                                                                                                verbose=True)

# %%
#############################################
# All Events Matching Ratios
all_event_metrics = DetectorComparisonAnalyzer.analyze(events, event_matches, None, verbose=True)
event_matching_ratios = all_event_metrics[DetectorComparisonAnalyzer.MATCH_RATIO_STR][
    DetectorComparisonAnalyzer.MATCH_RATIO_STR]

# show matching-ratio distributions
events_matching_ratio_fig = distributions_grid(
    data=event_matching_ratios[comparison_columns],
    title=f"{DATASET.upper()}:\t\tEvent-Matching Ratios",
    pdf_min_val=0,
    pdf_max_val=100,
    column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
)
events_matching_ratio_fig.show()

# show p-value heatmaps
ratios = event_matching_ratios.map(lambda cell: [v for v in cell if not np.isnan(v)])
stat_test_res = DetectorComparisonAnalyzer.event_feature_statistical_comparison(ratios, STAT_TEST_NAME)
p_values = stat_test_res.xs(cnst.P_VALUE, axis=1, level=2)
event_matching_pvalue_fig = heatmap_grid(
    p_values,
    critical_value=CRITICAL_VALUE,
    correction=CORRECTION,
    add_annotations=True,
    ignore_above_critical=True
)
event_matching_pvalue_fig.show()

del ratios, stat_test_res, p_values

# %%
#############################################
# Fixation Matching Ratios
fixation_metrics = DetectorComparisonAnalyzer.analyze(events,
                                                      event_matches,
                                                      None,
                                                      ignore_events={v for v in cnfg.EVENT_LABELS if
                                                                     v != cnfg.EVENT_LABELS.FIXATION},
                                                      verbose=True)
fixation_matching_ratios = fixation_metrics[DetectorComparisonAnalyzer.MATCH_RATIO_STR][
    DetectorComparisonAnalyzer.MATCH_RATIO_STR]

# show matching-ratio distributions
fixation_matching_ratio_fig = distributions_grid(
    data=fixation_matching_ratios[comparison_columns],
    title=f"{DATASET.upper()}:\t\tEvent-Matching Ratios",
    pdf_min_val=0,
    pdf_max_val=100,
    column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
)
fixation_matching_ratio_fig.show()

# show p-value heatmaps
ratios = fixation_matching_ratios.map(lambda cell: [v for v in cell if not np.isnan(v)])
stat_test_res = DetectorComparisonAnalyzer.event_feature_statistical_comparison(ratios, STAT_TEST_NAME)
p_values = stat_test_res.xs(cnst.P_VALUE, axis=1, level=2)
fixation_matching_pvalue_fig = heatmap_grid(
    p_values,
    critical_value=CRITICAL_VALUE,
    correction=CORRECTION,
    add_annotations=True,
    ignore_above_critical=True
)
fixation_matching_pvalue_fig.show()

del ratios, stat_test_res, p_values

# %%
#############################################
# Saccade Matching Ratios
Saccade_metrics = DetectorComparisonAnalyzer.analyze(events,
                                                     event_matches,
                                                     None,
                                                     ignore_events={v for v in cnfg.EVENT_LABELS if
                                                                    v != cnfg.EVENT_LABELS.SACCADE},
                                                     verbose=True)
saccade_matching_ratios = Saccade_metrics[DetectorComparisonAnalyzer.MATCH_RATIO_STR][
    DetectorComparisonAnalyzer.MATCH_RATIO_STR]

# show matching-ratio distributions
saccade_matching_ratio_fig = distributions_grid(
    data=saccade_matching_ratios[comparison_columns],
    title=f"{DATASET.upper()}:\t\tEvent-Matching Ratios",
    pdf_min_val=0,
    pdf_max_val=100,
    column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
)
saccade_matching_ratio_fig.show()

# show p-value heatmaps
ratios = saccade_matching_ratios.map(lambda cell: [v for v in cell if not np.isnan(v)])
stat_test_res = DetectorComparisonAnalyzer.event_feature_statistical_comparison(ratios, STAT_TEST_NAME)
p_values = stat_test_res.xs(cnst.P_VALUE, axis=1, level=2)
saccade_matching_pvalue_fig = heatmap_grid(
    p_values,
    critical_value=CRITICAL_VALUE,
    correction=CORRECTION,
    add_annotations=True,
    ignore_above_critical=True
)
saccade_matching_pvalue_fig.show()

del ratios, stat_test_res, p_values
