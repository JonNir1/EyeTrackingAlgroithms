import os

import Config.experiment_config as cnfg
import Analysis.helpers as hlp
from Analysis.old.PreProcessor import PreProcessor
import Analysis.figures as figs

DATASET_NAME = "Lund2013"
PIPELINE_NAME = "Detector_Comparison"

REFERENCE_RATER = "RA"
DATASET_DIR = os.path.join(cnfg.OUTPUT_DIR, DATASET_NAME)
FIGURES_DIR = os.path.join(cnfg.OUTPUT_DIR, DATASET_NAME, PIPELINE_NAME, "Fixations")
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

# Filter Out Non-Fixation Events
fixations = events.map(lambda cell: [event for event in cell if event.event_label == cnfg.EVENT_LABELS.FIXATION])
fixation_matches = {scheme: df.map(
    lambda cell: {k: v for k, v in cell.items() if k.event_label == cnfg.EVENT_LABELS.FIXATION} if cell is not None else None
) for scheme, df in matches.items()}

del results, events, matches

# %%
# Calculate Fixation-Specific Features
fixation_features = PreProcessor.calculate_event_features(
    fixations,
    feature_names={"Duration", "Peak Velocity"},
    verbose=True,
)
fixation_match_ratios = PreProcessor.calculate_match_ratios(
    fixations,
    fixation_matches,
    verbose=True,
)
fixation_matched_features = PreProcessor.calculate_matched_event_features(
    fixation_matches,
    verbose=True,
)
fixation_matched_features["CoM Distance"] = {scheme: fixation_matches[scheme].map(
    lambda cell: [k.center_distance(v) for k, v in cell.items()] if cell is not None else None
) for scheme in fixation_matches.keys()}
fixation_matched_features["Dispersion Ratio"] = {scheme: fixation_matches[scheme].map(
    lambda cell: [k.dispersion / v.dispersion for k, v in cell.items()] if cell is not None else None
) for scheme in fixation_matches.keys()}

# %%
# Fixation Feature Distributions Per-Stimulus Type
FIXATION_FEATURE_DIR = os.path.join(FIGURES_DIR, "fixation_features")
if not os.path.exists(FIXATION_FEATURE_DIR):
    os.makedirs(FIXATION_FEATURE_DIR)
_fixation_feature_figures = figs.create_event_feature_distributions(
    fixation_features, DATASET_NAME, FIXATION_FEATURE_DIR, columns=None
)

# %%
# Fixation Matching Feature Distributions Per-Stimulus Type
MATCHED_FIXATION_FEATURE_DIR = os.path.join(FIGURES_DIR, "matched_fixation_features")
if not os.path.exists(MATCHED_FIXATION_FEATURE_DIR):
    os.makedirs(MATCHED_FIXATION_FEATURE_DIR)
rater_detector_pairs = [pair for pair in hlp.extract_rater_detector_pairs(fixations) if pair[0] == REFERENCE_RATER]
_matched_fixation_feature_figures = figs.create_matched_event_feature_distributions(
    fixation_matched_features, DATASET_NAME, MATCHED_FIXATION_FEATURE_DIR, rater_detector_pairs
)
_matching_ratio_fig = figs.create_matching_ratio_distributions(
    fixation_match_ratios, DATASET_NAME, MATCHED_FIXATION_FEATURE_DIR, rater_detector_pairs
)
