import os
import copy

import numpy as np
import pandas as pd
import pickle as pkl
import sklearn.metrics as met
import plotly.graph_objects as go
import plotly.io as pio

import Config.constants as cnst
import Config.experiment_config as cnfg

from GazeDetectors.EngbertDetector import EngbertDetector
from DataSetLoaders.DataSetFactory import DataSetFactory

pio.renderers.default = "browser"

DATASET_NAME = "Lund2013"

###################

PATH = r'C:\Users\nirjo\Documents\University\Masters\Projects\EyeTrackingAlgroithms\Results\DetectorComparison\Lund2013'
samples = pd.read_pickle(os.path.join(PATH, 'samples.pkl'))
events = pd.read_pickle(os.path.join(PATH, 'events.pkl'))
with open(os.path.join(PATH, 'matches.pkl'), 'rb') as f:
    matches = pkl.load(f)
with open(os.path.join(PATH, 'event_features.pkl'), 'rb') as f:
    event_features = pkl.load(f)
del f

