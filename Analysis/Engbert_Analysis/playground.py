import time
from typing import List, Union, Optional

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

import Config.constants as cnst
from GazeEvents.BaseEvent import BaseEvent
from GazeDetectors.EngbertDetector import EngbertDetector
from DataSetLoaders.DataSetFactory import DataSetFactory

pio.renderers.default = "browser"

###################################

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

vx = lund_detector_res.map(lambda cell: cell['thresh_Vx'])
vy = lund_detector_res.map(lambda cell: cell['thresh_Vy'])

