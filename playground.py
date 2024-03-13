import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import Config.constants as cnst
import Config.experiment_config as cnfg

# from DataSetLoaders.HFCDataSetLoader import HFCDataSetLoader as HFC
# from DataSetLoaders.IRFDataSetLoader import IRFDataSetLoader as IRF
from DataSetLoaders.Lund2013DataSetLoader import Lund2013DataSetLoader as Lund2013
# from DataSetLoaders.GazeComDataSetLoader import GazeComDataSetLoader as GazeCom

from GazeDetectors.EngbertDetector import EngbertDetector
# from GazeDetectors.IVTDetector import IVTDetector
# from GazeDetectors.IDTDetector import IDTDetector
from GazeDetectors.NHDetector import NHDetector
from GazeDetectors.REMoDNaVDetector import REMoDNaVDetector

import GazeEvents.EventFactory as EF
import GazeEvents.EventMatching as EM

pio.renderers.default = "browser"

######################################
# Load Data

start = time.time()

# hfc = HFC().load(should_save=True)
# irf = IRF().load(should_save=True)
lund = Lund2013().load(should_save=True)
# gazecom = GazeCom().load(should_save=True)

end = time.time()
print(f"Time to load :\t{(end - start):.2f} seconds")

######################################
# Gaze Event Detection

start = time.time()

trial2 = lund[lund[cnst.TRIAL] == 2]
pixel_size = trial2["pixel_size_cm"].to_numpy()[0]
viewer_distance = trial2["viewer_distance_cm"].to_numpy()[0]

gt = trial2["MN"].to_numpy()
# gt_events = EF.EventFactory.make_from_gaze_data(trial2, viewer_distance, pixel_size)  # todo: fixme

engbert = EngbertDetector(viewer_distance=viewer_distance, pixel_size=pixel_size)
engbert_res = engbert.detect(t=trial2[cnst.MILLISECONDS].to_numpy(),
                             x=trial2[cnst.X].to_numpy(),
                             y=trial2[cnst.Y].to_numpy())

nh = NHDetector(viewer_distance=viewer_distance, pixel_size=pixel_size)
nh_res = nh.detect(t=trial2[cnst.MILLISECONDS].to_numpy(),
                   x=trial2[cnst.X].to_numpy(),
                   y=trial2[cnst.Y].to_numpy())

rmdnv = REMoDNaVDetector(viewer_distance=viewer_distance, pixel_size=pixel_size)
rmdnv_res = rmdnv.detect(t=trial2[cnst.MILLISECONDS].to_numpy(),
                         x=trial2[cnst.X].to_numpy(),
                         y=trial2[cnst.Y].to_numpy())

end = time.time()
print(f"Time to detect:\t{(end - start):.2f} seconds")
del start, end

######################################
# Event Matching

iou_match__eng_nh = EM.iou_matching(engbert_res[cnst.EVENTS], nh_res[cnst.EVENTS])
iou_match__nh_eng = EM.iou_matching(nh_res[cnst.EVENTS], engbert_res[cnst.EVENTS])
