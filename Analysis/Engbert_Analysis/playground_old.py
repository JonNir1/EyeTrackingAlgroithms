import time
import copy
import warnings

import numpy as np
import pandas as pd
import plotly.io as pio

import Config.constants as cnst
from GazeDetectors.EngbertDetector import EngbertDetector
from DataSetLoaders.DataSetFactory import DataSetFactory

import Analysis.Engbert_Analysis.comparisons as cmps
import Analysis.figures as figs

pio.renderers.default = "browser"

###################################

LAMBDA = "λ"

DATASET_NAME = "Lund2013"
RATERS = ["MN", "RA"]
DETECTORS = [EngbertDetector(lambdaa=lmda) for lmda in np.arange(0.5, 6.1, 0.5)]

rename_columns = lambda col: col[col.index(LAMBDA):col.index(",")].replace("'", "") if LAMBDA in col else col
COMPARISON_COLUMNS = [(RATERS[1], rename_columns(d.name)) for d in DETECTORS]
EVENT_MATCHING_PARAMS = {"match_by": "onset", "max_onset_latency": 15, "allow_cross_matching": False}

start = time.time()

lund_dataset = DataSetFactory.load(DATASET_NAME)
lund_samples, lund_events, lund_detector_res = DataSetFactory.detect(lund_dataset, RATERS, DETECTORS)

lund_samples.rename(columns=rename_columns, inplace=True)
lund_events.rename(columns=rename_columns, inplace=True)
lund_detector_res.rename(columns=rename_columns, inplace=True)

end = time.time()
print(f"Finished Detecting Events:\t{end - start:.2f} seconds")
del start, end

###################################
# Compare to Ground Truth
###################################

cohen_kappa = cmps.compare_samples(lund_samples, metric='kappa', group_by=cnst.STIMULUS)
cohen_kappa_fig = figs.distributions_grid(cohen_kappa[COMPARISON_COLUMNS],
                                          plot_type="violin",
                                          title="Cohen's Kappa",
                                          column_title_mapper=lambda col: f"{col[0]}→{col[1]}")
cohen_kappa_fig.show()

matching_ratio = cmps.event_matching_ratio(lund_events, group_by=cnst.STIMULUS, **EVENT_MATCHING_PARAMS)
matching_ratio_fig = figs.distributions_grid(matching_ratio[COMPARISON_COLUMNS],
                                             plot_type="violin",
                                             title="Event Matching Ratio",
                                             column_title_mapper=lambda col: f"{col[0]}→{col[1]}")
matching_ratio_fig.show()

###################################
# Threshold Distribution
###################################

thresholds = pd.concat([lund_detector_res[f"{LAMBDA}:1.0"].map(lambda cell: cell['thresh_Vx']),
                        lund_detector_res[f"{LAMBDA}:1.0"].map(lambda cell: cell['thresh_Vy'])],
                       axis=1, keys=["Vx", "Vy"])
agg_thresholds = cmps.group_and_aggregate(thresholds, group_by=cnst.STIMULUS)
threshold_distribution_fig = figs.distributions_grid(agg_thresholds,
                                                     plot_type="violin",
                                                     title="Thresholds Distribution")
threshold_distribution_fig.show()

###################################
# Multi-Iteration Detection
###################################

start = time.time()

ITERATION = "Iteration"
NUM_ITERATIONS = 4

engbert = EngbertDetector()
prev_gaze_data = lund_dataset.copy()
indexers = [col for col in DataSetFactory.INDEXERS if col in prev_gaze_data.columns]

results = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for i in range(1, NUM_ITERATIONS + 1):
        for trial_num in prev_gaze_data[cnst.TRIAL].unique():
            trial_data = prev_gaze_data[prev_gaze_data[cnst.TRIAL] == trial_num]
            key = tuple(trial_data[indexers].iloc[0].to_list() + [i])
            res = engbert.detect(
                t=trial_data[cnst.T].to_numpy(),
                x=trial_data[cnst.X].to_numpy(),
                y=trial_data[cnst.Y].to_numpy()
            )
            results[key] = copy.deepcopy(res)

            # nullify detected saccades
            detected_event_labels = res[cnst.GAZE][cnst.EVENT]
            saccade_idxs = detected_event_labels[detected_event_labels == cnst.EVENT_LABELS.SACCADE].index
            prev_gaze_data.loc[saccade_idxs, cnst.X] = np.nan
            prev_gaze_data.loc[saccade_idxs, cnst.Y] = np.nan

iteration_events = pd.Series({k: [e for e in v[cnst.EVENTS] if e.event_label != cnst.EVENT_LABELS.BLINK]
                              for k, v in results.items()}).sort_index()
iteration_events.index.names = indexers + ["Iteration"]
# grouped = iteration_events.groupby(level=[cnst.STIMULUS, "Iteration"]).agg(pd.Series.explode).map(lambda cell: [e for e in cell if pd.notnull(e)])
# group_all = grouped.groupby(level="Iteration").agg(pd.Series.explode)
# group_all.index = pd.MultiIndex.from_tuples([("all", i) for i in range(1, NUM_ITERATIONS + 1)],
#                                             names=[cnst.STIMULUS, "Iteration"])
# grouped_events = pd.concat([grouped.T, group_all]).T

end = time.time()
print(f"Finished Multi-Iteration Detection:\t{end - start:.2f} seconds")
del start, end

# Count number of events per iteration
label_counts = cmps.label_counts(iteration_events.to_frame(), group_by=[cnst.STIMULUS, ITERATION]).drop(index="all")
label_counts.index = pd.MultiIndex.from_tuples(label_counts.index, names=[cnst.STIMULUS, ITERATION])
label_counts_all_stim = label_counts.groupby(level=ITERATION).agg("sum")
label_counts_all_stim.index = pd.MultiIndex.from_tuples([("all", idx) for idx in label_counts_all_stim.index],
                                                        names=[cnst.STIMULUS, ITERATION])
label_counts = pd.concat([label_counts, label_counts_all_stim])[0].unstack(level=ITERATION)
label_counts_fig = figs.distributions_grid(label_counts,
                                           title="Event Counts per Iteration",
                                           show_counts=True)
label_counts_fig.show()

# Compare saccade amplitudes between iterations
saccade_amplitudes = cmps.event_features(iteration_events.to_frame(),
                                         feature="amplitude",
                                         group_by=[cnst.STIMULUS, ITERATION],
                                         ignore_events=[v for v in cnst.EVENT_LABELS if
                                                        v != cnst.EVENT_LABELS.SACCADE]).drop(index="all")
saccade_amplitudes.index = pd.MultiIndex.from_tuples(saccade_amplitudes.index, names=[cnst.STIMULUS, ITERATION])
saccade_amplitudes_all_stim = saccade_amplitudes.groupby(level=ITERATION).agg("sum")
saccade_amplitudes_all_stim.index = pd.MultiIndex.from_tuples(
    [("all", idx) for idx in saccade_amplitudes_all_stim.index],
    names=[cnst.STIMULUS, ITERATION])
saccade_amplitudes = pd.concat([saccade_amplitudes, saccade_amplitudes_all_stim])[0].unstack(level=ITERATION)
saccade_amplitudes_fig = figs.distributions_grid(saccade_amplitudes,
                                                 plot_type="violin",
                                                 title="Saccade Amplitudes per Iteration")
saccade_amplitudes_fig.show()

# percent of micro-saccades per iteration
MICROSACCADE_MAX_AMPLITUDE = 1.0
saccade_counts = iteration_events.map(lambda cell: len([e for e in cell if e.event_label == cnst.EVENT_LABELS.SACCADE]))
microsaccade_counts = iteration_events.map(lambda cell: len([e for e in cell if
                                                             e.event_label == cnst.EVENT_LABELS.SACCADE and
                                                             e.amplitude <= MICROSACCADE_MAX_AMPLITUDE]))
microsaccade_ratio = microsaccade_counts / saccade_counts
microsaccade_ratio = microsaccade_ratio.groupby(level=[cnst.STIMULUS, ITERATION]).agg(list).map(
    lambda cell: [e for e in cell if pd.notnull(e)])
microsaccade_ratio_all_stim = microsaccade_ratio.groupby(level=ITERATION).agg("sum")
microsaccade_ratio_all_stim.index = pd.MultiIndex.from_tuples(
    [("all", idx) for idx in microsaccade_ratio_all_stim.index],
    names=[cnst.STIMULUS, ITERATION])
microsaccade_ratio = pd.concat([microsaccade_ratio, microsaccade_ratio_all_stim]).unstack(level=ITERATION)
microsaccade_ratio_fig = figs.distributions_grid(microsaccade_ratio,
                                                 plot_type="violin",
                                                 limit_pdf=True,
                                                 title="Microsaccade Ratio per Iteration")
microsaccade_ratio_fig.show()
