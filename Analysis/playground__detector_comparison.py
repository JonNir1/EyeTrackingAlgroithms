import numpy as np
import pandas as pd
import plotly.io as pio

import Config.constants as cnst

import GazeEvents.helpers as hlp
from GazeDetectors.EngbertDetector import EngbertDetector
from GazeDetectors.NHDetector import NHDetector

from Analysis.DetectorComparisonCalculator import DetectorComparisonCalculator
import Analysis.metrics as metrics
import Analysis.figures as figs

pio.renderers.default = "browser"

# DATASET_NAME = "IRF"
# RATERS = ["RZ"]
# DETECTORS = [EngbertDetector(), NHDetector()]
# COMPARISON_COLUMNS = [(r, d.name) for r in RATERS for d in DETECTORS]

DATASET_NAME = "Lund2013"
RATERS = ["MN", "RA"]
DETECTORS = [EngbertDetector(), NHDetector()]
COMPARISON_COLUMNS = [tuple(RATERS)] + [(r, d.name) for r in RATERS for d in DETECTORS]
EVENT_MATCHING_PARAMS = {"match_by": "onset", "max_onset_latency": 15, "allow_cross_matching": False}

comp_calc = DetectorComparisonCalculator(DATASET_NAME, RATERS, DETECTORS)

############################
## Sample-Level Contrasts ##
############################

samples_levenshtein = comp_calc.compare_samples(compare_by="lev", group_by=None)
samples_frobenius = comp_calc.compare_samples(compare_by="frobenius", group_by=None)
samples_kl = comp_calc.compare_samples(compare_by="kl", group_by=None)

# lev_heatmap = figs.similarity_heatmap(samples_levenshtein, "Levenshtein Distance", "Levenshtein Distance")
# frob_heatmap = figs.similarity_heatmap(samples_frobenius, "Frobenius Norm", r"$L_2$")
# kl_heatmap = figs.similarity_heatmap(samples_kl, "Kullback-Leibler Divergence", r"$D_{KL}$")

samples_levenshtein_grouped = comp_calc.compare_samples(compare_by="lev", group_by=cnst.STIMULUS)
samples_frobenius_grouped = comp_calc.compare_samples(compare_by="frobenius", group_by=cnst.STIMULUS)
samples_kl_grouped = comp_calc.compare_samples(compare_by="kl", group_by=cnst.STIMULUS)

lev_dist = figs.distributions_grid(
    samples_levenshtein_grouped[COMPARISON_COLUMNS],
    plot_type="violin",
    title="Levenshtein Distance Distribution",
    column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
)
frob_norm = figs.distributions_grid(
    samples_frobenius_grouped[COMPARISON_COLUMNS],
    plot_type="violin",
    title="Frobenius Norm Distribution",
    column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
)
kl_div = figs.distributions_grid(
    samples_kl_grouped[COMPARISON_COLUMNS],
    plot_type="violin",
    title="Kullback-Leibler Divergence Distribution",
    column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
)

lev_dist.show()
frob_norm.show()
kl_div.show()

##############################
## Matched-Events Contrasts ##
##############################

match_ratios = comp_calc.event_matching_ratio(group_by=None, **EVENT_MATCHING_PARAMS)
# match_ratio_heatmap = figs.similarity_heatmap(match_ratios, "Event Matching Ratio", "Percent Unmatched")
# match_ratio_heatmap.show()

match_ratios_grouped = comp_calc.event_matching_ratio(group_by=cnst.STIMULUS, **EVENT_MATCHING_PARAMS)
match_ratio_distributions = figs.distributions_grid(
    match_ratios_grouped[COMPARISON_COLUMNS],
    plot_type="violin",
    title="Matched-Events Ratio",
    column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
)
match_ratio_distributions.show()

all_types_onset_jitter = comp_calc.compare_matched_events(compare_by="onset",
                                                          group_by=cnst.STIMULUS,
                                                          **EVENT_MATCHING_PARAMS)
onset_jitter_distributions = figs.distributions_grid(
    all_types_onset_jitter[COMPARISON_COLUMNS],
    plot_type="violin",
    title="Matched-Events Onset Jitter",
    column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
)
onset_jitter_distributions.show()

all_types_duration_diffs = comp_calc.compare_matched_events(compare_by="duration",
                                                            group_by=cnst.STIMULUS,
                                                            **EVENT_MATCHING_PARAMS)
duration_diff_distributions = figs.distributions_grid(
    all_types_duration_diffs[COMPARISON_COLUMNS],
    plot_type="violin",
    title="Matched-Events Duration Difference",
    column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
)
duration_diff_distributions.show()

#################################
## Matched-Fixations Contrasts ##
#################################

fixation_onset_jitter = comp_calc.compare_matched_events(compare_by="onset", group_by=cnst.STIMULUS,
                                                         ignore_events=[v for v in cnst.EVENT_LABELS
                                                                            if v != cnst.EVENT_LABELS.FIXATION],
                                                         **EVENT_MATCHING_PARAMS)
fixation_onset_jitter_distributions = figs.distributions_grid(
    fixation_onset_jitter[COMPARISON_COLUMNS],
    plot_type="violin",
    title="Matched-Fixations Onset Jitter",
    column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
)
fixation_onset_jitter_distributions.show()

fixation_duration_diffs = comp_calc.compare_matched_events(compare_by="duration", group_by=cnst.STIMULUS,
                                                           ignore_events=[v for v in cnst.EVENT_LABELS
                                                                              if v != cnst.EVENT_LABELS.FIXATION],
                                                           **EVENT_MATCHING_PARAMS)

fixation_duration_diff_distributions = figs.distributions_grid(
    fixation_duration_diffs[COMPARISON_COLUMNS],
    plot_type="violin",
    title="Matched-Fixations Duration Difference",
    column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
)

fixation_duration_diff_distributions.show()

################################
## Matched-Saccades Contrasts ##
################################

saccades_onset_jitter = comp_calc.compare_matched_events(compare_by="onset", group_by=cnst.STIMULUS,
                                                         ignore_events=[v for v in cnst.EVENT_LABELS
                                                                            if v != cnst.EVENT_LABELS.SACCADE],
                                                         **EVENT_MATCHING_PARAMS)
saccade_onset_jitter_distributions = figs.distributions_grid(
    saccades_onset_jitter[COMPARISON_COLUMNS],
    plot_type="violin",
    title="Matched-Saccades Onset Jitter",
    column_title_mapper=lambda col: f"{col[0]}→{col[1]}", )
saccade_onset_jitter_distributions.show()

saccade_duration_diffs = comp_calc.compare_matched_events(compare_by="duration", group_by=cnst.STIMULUS,
                                                          ignore_events=[v for v in cnst.EVENT_LABELS
                                                                             if v != cnst.EVENT_LABELS.SACCADE],
                                                          **EVENT_MATCHING_PARAMS)
saccade_duration_diff_distributions = figs.distributions_grid(
    saccade_duration_diffs[COMPARISON_COLUMNS],
    plot_type="violin",
    title="Matched-Saccades Duration Difference",
    column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
)
saccade_duration_diff_distributions.show()
