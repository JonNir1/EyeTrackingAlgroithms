import plotly.io as pio

import Config.experiment_config as cnfg
from Analysis.old.Analyzers.MatchedEventsAnalyzer import MatchedEventsAnalyzer
from Visualization.distributions_grid import distributions_grid
from Visualization.p_value_heatmap import heatmap_grid

pio.renderers.default = "browser"

DATASET = "Lund2013"
COL_MAPPER = lambda col: col[:col.index("ector")] if "ector" in col else col

SINGLE_TEST_NAME = "Wilcoxon"
PAIRED_TEST_NAME = "Wilcoxon"
CRITICAL_VALUE = 0.05
CORRECTION = "Bonferroni"

events, event_matches, comparison_columns = MatchedEventsAnalyzer.preprocess_dataset(DATASET,
                                                                                     column_mapper=COL_MAPPER,
                                                                                     verbose=True)

# %%
#############################################
# All Events' Matched-Features
events_matched_features, events_matched_feature_stats = MatchedEventsAnalyzer.analyze(events,
                                                                                      ignore_events=None,
                                                                                      matches_df=event_matches,
                                                                                      paired_sample_test=PAIRED_TEST_NAME,
                                                                                      single_sample_test=SINGLE_TEST_NAME,
                                                                                      verbose=True)
print(f"Available matched-event feature differences: {list(events_matched_features.keys())}")

# show feature distributions
events_matched_features_distribution_figures = {}
for feature in events_matched_features.keys():
    data = events_matched_features[feature]
    fig = distributions_grid(
        data=data[comparison_columns],
        title=f"{DATASET.upper()}:\t\tMatched-Events' {feature.title()} Distribution",
        column_title_mapper=lambda col: f"{col[0]}→{col[1]}",
        pdf_min_val=0 if feature in {"IoU", "Overlap Time"} else None,
        pdf_max_val=1 if feature in {"IoU", "Overlap Time"} else None,
    )
    events_matched_features_distribution_figures[feature] = fig
    fig.show()

# show p-value heatmaps
events_matched_features_pvalue_figures = {}
for feature in events_matched_feature_stats.keys():
    p_values = events_matched_feature_stats[feature].xs("p-value", axis=1, level=2)
    fig = heatmap_grid(
        p_values,
        title=f"{DATASET.upper()}:\t\tStatistical Comparison of Matched-Events' {feature.title()}",
        critical_value=CRITICAL_VALUE,
        correction=CORRECTION,
        add_annotations=True,
        ignore_above_critical=True
    )
    events_matched_features_pvalue_figures[feature] = fig
    fig.show()

del data, feature, p_values

# %%
#############################################
# Fixations' Matched-Features
fixations_matched_features, fixations_matched_feature_stats = MatchedEventsAnalyzer.analyze(events,
                                                                                            ignore_events={v for v in
                                                                                                           cnfg.EVENT_LABELS
                                                                                                           if
                                                                                                           v != cnfg.EVENT_LABELS.FIXATION},
                                                                                            matches_df=event_matches,
                                                                                            paired_sample_test=PAIRED_TEST_NAME,
                                                                                            single_sample_test=SINGLE_TEST_NAME,
                                                                                            verbose=True)

# show feature distributions
fixations_matched_features_distribution_figures = {}
for feature in fixations_matched_features.keys():
    data = fixations_matched_features[feature]
    fig = distributions_grid(
        data=data[comparison_columns],
        title=f"{DATASET.upper()}:\t\tMatched-Fixations' {feature.title()} Distribution",
        column_title_mapper=lambda col: f"{col[0]}→{col[1]}",
        pdf_min_val=0 if feature in {"IoU", "Overlap Time"} else None,
        pdf_max_val=1 if feature in {"IoU", "Overlap Time"} else None,
    )
    fixations_matched_features_distribution_figures[feature] = fig
    fig.show()

# show p-value heatmaps
fixations_matched_features_pvalue_figures = {}
for feature in fixations_matched_feature_stats.keys():
    p_values = fixations_matched_feature_stats[feature].xs("p-value", axis=1, level=2)
    fig = heatmap_grid(
        p_values,
        title=f"{DATASET.upper()}:\t\tStatistical Comparison of Matched-Fixations' {feature.title()}",
        critical_value=CRITICAL_VALUE,
        correction=CORRECTION,
        add_annotations=True,
        ignore_above_critical=True
    )
    fixations_matched_features_pvalue_figures[feature] = fig
    fig.show()

del data, feature, p_values

# %%
#############################################
# Saccades' Matched-Features
saccades_matched_features, saccades_matched_feature_stats = MatchedEventsAnalyzer.analyze(events,
                                                                                          ignore_events={v for v in
                                                                                                         cnfg.EVENT_LABELS
                                                                                                         if
                                                                                                         v != cnfg.EVENT_LABELS.SACCADE},
                                                                                          matches_df=event_matches,
                                                                                          paired_sample_test=PAIRED_TEST_NAME,
                                                                                          single_sample_test=SINGLE_TEST_NAME,
                                                                                          verbose=True)

# show feature distributions
saccades_matched_features_distribution_figures = {}
for feature in saccades_matched_features.keys():
    data = saccades_matched_features[feature]
    fig = distributions_grid(
        data=data[comparison_columns],
        title=f"{DATASET.upper()}:\t\tMatched-Saccades' {feature.title()} Distribution",
        column_title_mapper=lambda col: f"{col[0]}→{col[1]}",
        pdf_min_val=0 if feature in {"IoU", "Overlap Time"} else None,
        pdf_max_val=1 if feature in {"IoU", "Overlap Time"} else None,
    )
    saccades_matched_features_distribution_figures[feature] = fig
    fig.show()

# show p-value heatmaps
saccades_matched_features_pvalue_figures = {}
for feature in saccades_matched_feature_stats.keys():
    p_values = saccades_matched_feature_stats[feature].xs("p-value", axis=1, level=2)
    fig = heatmap_grid(
        p_values,
        title=f"{DATASET.upper()}:\t\tStatistical Comparison of Matched-Saccades' {feature.title()}",
        critical_value=CRITICAL_VALUE,
        correction=CORRECTION,
        add_annotations=True,
        ignore_above_critical=True
    )
    saccades_matched_features_pvalue_figures[feature] = fig
    fig.show()

del data, feature, p_values
