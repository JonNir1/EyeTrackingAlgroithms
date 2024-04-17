import numpy as np
import scipy.stats as stat
import plotly.io as pio

import Config.constants as cnst
import Config.experiment_config as cnfg
from GazeDetectors.EngbertDetector import EngbertDetector
from Analysis.multiple_iterations.MultiIterationAnalyzer import MultiIterationAnalyzer

pio.renderers.default = "browser"

DATASET_NAME = "Lund2013"

#######################################

detector = EngbertDetector()
multi_detect_events = MultiIterationAnalyzer.preprocess_dataset(DATASET_NAME, detector=detector, verbose=True)

all_event_features = MultiIterationAnalyzer.analyze_impl(multi_detect_events)[MultiIterationAnalyzer.EVENT_FEATURES_STR]
saccade_features = MultiIterationAnalyzer.analyze_impl(multi_detect_events,
                                                       ignore_events={v for v in cnfg.EVENT_LABELS if
                                                                      v != cnfg.EVENT_LABELS.SACCADE})[MultiIterationAnalyzer.EVENT_FEATURES_STR]
fixation_features = MultiIterationAnalyzer.analyze_impl(multi_detect_events,
                                                        ignore_events={v for v in cnfg.EVENT_LABELS if
                                                                       v != cnfg.EVENT_LABELS.FIXATION})[MultiIterationAnalyzer.EVENT_FEATURES_STR]

#######################################

saccade_amplitudes = saccade_features[cnst.AMPLITUDE.capitalize()].map(lambda cell: [e for e in cell if not np.isnan(e)])
mw = stat.mannwhitneyu(saccade_amplitudes.iloc[0, 0], saccade_amplitudes.iloc[1, 0])
rnk = stat.ranksums(saccade_amplitudes.iloc[0, 0], saccade_amplitudes.iloc[1, 0])