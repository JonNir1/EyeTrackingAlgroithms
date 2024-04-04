import time
import copy
import warnings

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

import Config.constants as cnst
from GazeDetectors.EngbertDetector import EngbertDetector
from DataSetLoaders.DataSetFactory import DataSetFactory

import Analysis.comparisons as cmps
import Analysis.figures as figs

pio.renderers.default = "browser"

###################################

LMDA = "λ"

DATASET_NAME = "Lund2013"
RATERS = ["MN", "RA"]
DETECTORS = [EngbertDetector(lambdaa=lmda) for lmda in np.arange(0.5, 6.1, 0.5)]

start = time.time()

lund_dataset = DataSetFactory.load(DATASET_NAME)
lund_samples, lund_events, lund_detector_res = DataSetFactory.process(lund_dataset, RATERS, DETECTORS)
lund_detector_res.rename(columns=lambda col: col[col.index("λ"):col.index(",")].replace("'", ""), inplace=True)

end = time.time()
print(f"Elapsed time: {end - start:.2f} seconds")
del start, end


###################################
# Threshold Distribution
###################################

thresholds = pd.concat([lund_detector_res[f"{LMDA}:1.0"].map(lambda cell: cell['thresh_Vx']),
                        lund_detector_res[f"{LMDA}:1.0"].map(lambda cell: cell['thresh_Vy'])],
                       axis=1, keys=["Vx", "Vy"])
agg_thresholds = cmps.group_and_aggregate(thresholds, group_by=cnst.STIMULUS)
threshold_distribution_fig = figs.distributions_grid(agg_thresholds,
                                                     plot_type="violin",
                                                     title="Thresholds Distribution")
threshold_distribution_fig.show()

###################################
# Multi-Iteration Detection
###################################

NUM_ITERATIONS = 2

engbert = EngbertDetector()
prev_gaze_data = lund_dataset.copy()
indexers = [col for col in DataSetFactory._INDEXERS if col in prev_gaze_data.columns]

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

iteration_events = pd.DataFrame.from_dict({k: [e for e in v[cnst.EVENTS] if e.event_label != cnst.EVENT_LABELS.BLINK]
                                           for k, v in results.items()}, orient="index").sort_index()
iteration_events.index = pd.MultiIndex.from_tuples(iteration_events.index, names=indexers + ["Iteration"])
grouped = iteration_events.groupby(level=[cnst.STIMULUS, "Iteration"]).agg(list)
