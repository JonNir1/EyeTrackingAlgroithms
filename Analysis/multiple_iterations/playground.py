import numpy as np
import scipy.stats as stat
import plotly.io as pio

import Config.constants as cnst
import Config.experiment_config as cnfg
from GazeDetectors.EngbertDetector import EngbertDetector
from Analysis.Analyzers.MultiIterationAnalyzer import MultiIterationAnalyzer

pio.renderers.default = "browser"

DATASET_NAME = "Lund2013"

STAT_TEST_NAME = "Mann-Whitney"
CRITICAL_VALUE = 0.05
CORRECTION = "Bonferroni"

#######################################
detector = EngbertDetector()
multi_detect_events = MultiIterationAnalyzer.preprocess_dataset(DATASET_NAME, detector=detector, verbose=True)

event_features, event_feature_stats = MultiIterationAnalyzer.analyze(multi_detect_events, None,
                                                                     test_name=STAT_TEST_NAME, verbose=True)
saccade_features, saccade_feature_stats = MultiIterationAnalyzer.analyze(multi_detect_events,
                                                                         ignore_events={v for v in cnfg.EVENT_LABELS if
                                                                                        v != cnfg.EVENT_LABELS.SACCADE},
                                                                         test_name=STAT_TEST_NAME,
                                                                         verbose=True)
fixation_features, fixation_feature_stats = MultiIterationAnalyzer.analyze(multi_detect_events,
                                                                           ignore_events={v for v in cnfg.EVENT_LABELS if
                                                                                          v != cnfg.EVENT_LABELS.FIXATION},
                                                                           test_name=STAT_TEST_NAME,
                                                                           verbose=True)

#######################################

# TODO: add stat comparisons for event features
# TODO: add scarfplot comparisons (requires pre-processing to return samples)
