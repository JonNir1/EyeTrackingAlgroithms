import os

import Config.experiment_config as cnfg
import Analysis.helpers as hlp
from Analysis.old.PreProcessor import PreProcessor
import Analysis.figures as figs

DATASET_NAME = "Lund2013"
PIPELINE_NAME = "Detector_Comparison"

REFERENCE_RATER = "RA"
DATASET_DIR = os.path.join(cnfg.OUTPUT_DIR, DATASET_NAME)
FIGURES_DIR = os.path.join(cnfg.OUTPUT_DIR, DATASET_NAME, PIPELINE_NAME, "All_Events")
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)


# %%
# Load the Data
results = PreProcessor.load_or_run(
    DATASET_NAME,
    PIPELINE_NAME,
    verbose=True,
    column_mapper=lambda col: col[:col.index("ector")] if "ector" in col else col
)
samples, events, _, matches, sample_metrics, event_features, match_ratios, matched_features = results
del results

# %%
# Per-Trial Scarfplots
SCARFPLOT_DIR = os.path.join(FIGURES_DIR, "scarfplots")
if not os.path.exists(SCARFPLOT_DIR):
    os.makedirs(SCARFPLOT_DIR)
_scarfplots = figs.create_comparison_scarfplots(samples, SCARFPLOT_DIR)

# %%
# Sample Metric Distributions Per-Stimulus Type
SAMPLE_METRIC_DIR = os.path.join(FIGURES_DIR, "sample_metrics")
if not os.path.exists(SAMPLE_METRIC_DIR):
    os.makedirs(SAMPLE_METRIC_DIR)

rater_detector_pairs = [pair for pair in hlp.extract_rater_detector_pairs(samples) if pair[0] == REFERENCE_RATER]
_sample_metric_figures = figs.create_sample_metric_distributions(
    sample_metrics, DATASET_NAME, SAMPLE_METRIC_DIR, rater_detector_pairs
)

# %%
# Event Feature Distributions Per-Stimulus Type
EVENT_FEATURE_DIR = os.path.join(FIGURES_DIR, "event_features")
if not os.path.exists(EVENT_FEATURE_DIR):
    os.makedirs(EVENT_FEATURE_DIR)
_event_feature_figures = figs.create_event_feature_distributions(
    event_features, DATASET_NAME, EVENT_FEATURE_DIR, columns=None
)

# %%
# Event Matching Feature Distributions Per-Stimulus Type
MATCHED_EVENT_FEATURE_DIR = os.path.join(FIGURES_DIR, "matched_event_features")
if not os.path.exists(MATCHED_EVENT_FEATURE_DIR):
    os.makedirs(MATCHED_EVENT_FEATURE_DIR)
rater_detector_pairs = [pair for pair in hlp.extract_rater_detector_pairs(events) if pair[0] == REFERENCE_RATER]
_matched_event_feature_figures = figs.create_matched_event_feature_distributions(
    matched_features, DATASET_NAME, MATCHED_EVENT_FEATURE_DIR, rater_detector_pairs
)
_matching_ratio_fig = figs.create_matching_ratio_distributions(
    match_ratios, DATASET_NAME, MATCHED_EVENT_FEATURE_DIR, rater_detector_pairs
)
