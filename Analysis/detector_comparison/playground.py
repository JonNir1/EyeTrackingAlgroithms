import Config.constants as cnst
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
sample_metrics = all_event_metrics["Sample Metrics"]
event_features = all_event_metrics["Event Features"]
event_matching_ratios = all_event_metrics["Event Matching Ratios"]
event_matching_feature_diffs = all_event_metrics["Event Matching Feature Diffs"]

# %%
#############################################
# Fixation Metrics
fixation_metrics = DetectorComparisonAnalyzer.analyze(events, event_matches,
                                                      ignore_events={v for v in cnst.EVENT_LABELS if v != cnst.EVENT_LABELS.FIXATION},
                                                      verbose=True)
fixation_features = fixation_metrics["Event Features"]
fixation_matching_ratios = fixation_metrics["Event Matching Ratios"]
fixation_matching_feature_diffs = fixation_metrics["Event Matching Feature Diffs"]

# %%
#############################################
# Saccade Metrics
saccade_metrics = DetectorComparisonAnalyzer.analyze(events, event_matches,
                                                     ignore_events={v for v in cnst.EVENT_LABELS if v != cnst.EVENT_LABELS.SACCADE},
                                                     verbose=True)
saccade_features = saccade_metrics["Event Features"]
saccade_matching_ratios = saccade_metrics["Event Matching Ratios"]
saccade_matching_feature_diffs = saccade_metrics["Event Matching Feature Diffs"]
