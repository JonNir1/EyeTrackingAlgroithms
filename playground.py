import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import Config.constants as cnst
import Config.experiment_config as cnfg
from Visualization import scarfplot

pio.renderers.default = "browser"

irf_samples = pd.read_pickle(os.path.join(cnfg.OUTPUT_DIR, "DetectorComparison", "IRF", "samples.pkl"))
