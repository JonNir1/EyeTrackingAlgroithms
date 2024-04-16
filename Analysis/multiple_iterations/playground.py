import plotly.io as pio

import Config.constants as cnst
from GazeDetectors.EngbertDetector import EngbertDetector
from Analysis.multiple_iterations.MultiIterationAnalyzer import MultiIterationAnalyzer
from Visualization.distributions_grid import *

pio.renderers.default = "browser"

DATASET_NAME = "Lund2013"

#######################################

detector = EngbertDetector()
multi_detect_events = MultiIterationAnalyzer.preprocess_dataset(DATASET_NAME, detector=detector, verbose=True)

all_event_features = MultiIterationAnalyzer.analyze(multi_detect_events)[MultiIterationAnalyzer.EVENT_FEATURES_STR]
saccade_features = MultiIterationAnalyzer.analyze(multi_detect_events,
                                                  ignore_events={v for v in cnst.EVENT_LABELS if v != cnst.EVENT_LABELS.SACCADE})[MultiIterationAnalyzer.EVENT_FEATURES_STR]
fixation_features = MultiIterationAnalyzer.analyze(multi_detect_events,
                                                   ignore_events={v for v in cnst.EVENT_LABELS if v != cnst.EVENT_LABELS.FIXATION})[MultiIterationAnalyzer.EVENT_FEATURES_STR]

#######################################
