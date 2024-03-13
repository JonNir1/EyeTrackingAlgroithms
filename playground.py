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
from GazeDetectors.IVTDetector import IVTDetector
from GazeDetectors.IDTDetector import IDTDetector
from GazeDetectors.NHDetector import NHDetector
from GazeDetectors.REMoDNaVDetector import REMoDNaVDetector

from GazeEvents.EventFactory import EventFactory

pio.renderers.default = "browser"

######################################

start = time.time()

# hfc = HFC().load(should_save=True)
# irf = IRF().load(should_save=True)
lund = Lund2013().load(should_save=True)
# gazecom = GazeCom().load(should_save=True)

end = time.time()
print(f"Time to load :\t{(end - start):.2f} seconds")

######################################

start = time.time()

trial2 = lund[lund[cnst.TRIAL] == 2]
gt = trial2["MN"].to_numpy()
pixel_size = trial2["pixel_size_cm"].to_numpy()[0]
viewer_distance = trial2["viewer_distance_cm"].to_numpy()[0]

engbert = EngbertDetector(viewer_distance=viewer_distance, pixel_size=pixel_size)
engbert_res = engbert.detect(t=trial2[cnst.MILLISECONDS].to_numpy(),
                             x=trial2[cnst.X].to_numpy(),
                             y=trial2[cnst.Y].to_numpy())
engbert_events = EventFactory.make_from_gaze_data(engbert_res[cnst.GAZE],
                                                  engbert_res[cnst.VIEWER_DISTANCE],
                                                  engbert_res[cnst.PIXEL_SIZE])

nh = NHDetector(viewer_distance=viewer_distance, pixel_size=pixel_size)
nh_res = nh.detect(t=trial2[cnst.MILLISECONDS].to_numpy(),
                   x=trial2[cnst.X].to_numpy(),
                   y=trial2[cnst.Y].to_numpy())
nh_events = EventFactory.make_from_gaze_data(nh_res[cnst.GAZE],
                                             nh_res[cnst.VIEWER_DISTANCE],
                                             nh_res[cnst.PIXEL_SIZE])

rmdnv = REMoDNaVDetector(viewer_distance=viewer_distance, pixel_size=pixel_size)
rmdnv_res = rmdnv.detect(t=trial2[cnst.MILLISECONDS].to_numpy(),
                         x=trial2[cnst.X].to_numpy(),
                         y=trial2[cnst.Y].to_numpy())
rmdnv_events = EventFactory.make_from_gaze_data(rmdnv_res[cnst.GAZE],
                                                rmdnv_res[cnst.VIEWER_DISTANCE],
                                                rmdnv_res[cnst.PIXEL_SIZE])

end = time.time()
print(f"Time to detect:\t{(end - start):.2f} seconds")
del start, end
