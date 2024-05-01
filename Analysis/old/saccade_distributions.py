import os

import Config.experiment_config as cnfg
import Analysis.helpers as hlp
from Analysis.old.PreProcessor import PreProcessor
import Analysis.figures as figs

DATASET_NAME = "Lund2013"
PIPELINE_NAME = "Detector_Comparison"

REFERENCE_RATER = "RA"
DATASET_DIR = os.path.join(cnfg.OUTPUT_DIR, DATASET_NAME)
FIGURES_DIR = os.path.join(cnfg.OUTPUT_DIR, DATASET_NAME, PIPELINE_NAME, "Saccades")
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
_, events, _, matches, _, _, _, _ = results

# Filter Out Non-Saccade Events
saccades = events.map(lambda cell: [event for event in cell if event.event_label == cnfg.EVENT_LABELS.SACCADE])
saccade_matches = {scheme: df.map(
    lambda cell: {k: v for k, v in cell.items() if k.event_label == cnfg.EVENT_LABELS.SACCADE} if cell is not None else None
) for scheme, df in matches.items()}

del results, events, matches

# %%
# Calculate Saccade-Specific Features
saccade_features = PreProcessor.calculate_event_features(
    saccades,
    feature_names=PreProcessor.EVENT_FEATURES - {"Count", "Micro-Saccade Ratio"},
    verbose=True,
)
saccade_match_ratios = PreProcessor.calculate_match_ratios(
    saccades,
    saccade_matches,
    verbose=True,
)
saccade_matched_features = PreProcessor.calculate_matched_event_features(
    saccade_matches,
    verbose=True,
)

# %%
# Saccade Feature Distributions Per-Stimulus Type
SACCADE_FEATURE_DIR = os.path.join(FIGURES_DIR, "saccade_features")
if not os.path.exists(SACCADE_FEATURE_DIR):
    os.makedirs(SACCADE_FEATURE_DIR)
_event_feature_figures = figs.create_event_feature_distributions(
    saccade_features, DATASET_NAME, SACCADE_FEATURE_DIR, columns=None
)

# %%
# Event Matching Feature Distributions Per-Stimulus Type
MATCHED_SACCADE_FEATURE_DIR = os.path.join(FIGURES_DIR, "matched_saccade_features")
if not os.path.exists(MATCHED_SACCADE_FEATURE_DIR):
    os.makedirs(MATCHED_SACCADE_FEATURE_DIR)
rater_detector_pairs = [pair for pair in hlp.extract_rater_detector_pairs(saccades) if pair[0] == REFERENCE_RATER]
_matched_saccade_feature_figures = figs.create_matched_event_feature_distributions(
    saccade_matched_features, DATASET_NAME, MATCHED_SACCADE_FEATURE_DIR, rater_detector_pairs
)
_matching_ratio_fig = figs.create_matching_ratio_distributions(
    saccade_match_ratios, DATASET_NAME, MATCHED_SACCADE_FEATURE_DIR, rater_detector_pairs
)
