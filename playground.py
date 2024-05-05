import os
import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import Config.constants as cnst
import Config.experiment_config as cnfg

from GazeDetectors.EngbertDetector import EngbertDetector
from DataSetLoaders.DataSetFactory import DataSetFactory

pio.renderers.default = "browser"

DATASET_NAME = "Lund2013"

# %%

loaded_detector_results = pd.read_pickle(os.path.join(cnfg.OUTPUT_DIR, "MultiIterations", DATASET_NAME, "detector_results.pkl"))

# %%
engbert = EngbertDetector()

dataset = DataSetFactory.load(DATASET_NAME)
trial1 = dataset[dataset[cnst.TRIAL] == 1]
viewer_distance = trial1[f"{cnst.VIEWER_DISTANCE}_cm"].iloc[0]
pixel_size = trial1[f"{cnst.PIXEL_SIZE}_cm"].iloc[0]

NUM_ITERS = 5
trial1_data = {0: trial1.copy(deep=True)}
detector_results = {}
for i in range(NUM_ITERS):
    data = trial1_data[i]
    res = engbert.detect(
        t=data[cnst.T].values,
        x=data[cnst.X].values,
        y=data[cnst.Y].values,
        vd=viewer_distance,
        ps=pixel_size
    )
    detector_results[i+1] = copy.deepcopy(res)

    # nullify detected saccades
    new_data = data.copy(deep=True)
    detected_event_labels = detector_results[i+1][cnst.GAZE][cnst.EVENT]

    print(i+1)
    print(sum(detected_event_labels == cnfg.EVENT_LABELS.SACCADE))

    saccade_idxs = new_data.index[detected_event_labels == cnfg.EVENT_LABELS.SACCADE]
    new_data.loc[saccade_idxs, cnst.X] = np.nan
    new_data.loc[saccade_idxs, cnst.Y] = np.nan
    trial1_data[i + 1] = new_data

del i, data, res, new_data, detected_event_labels, saccade_idxs


trial1_data[3].equals(trial1_data[2])
detector_results[3][cnst.GAZE].equals(detector_results[2][cnst.GAZE])
