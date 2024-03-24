"""
This file contains the configuration for each specific experiment.
"""
import os
import numpy as np
import posixpath as psx

import Config.constants as cnst
from Config.ScreenMonitor import ScreenMonitor
from Config.ExperimentTriggerEnum import ExperimentTriggerEnum

# GLOBAL VARIABLES
SCREEN_MONITOR: ScreenMonitor = ScreenMonitor.from_tobii_default()  # global variable: screen monitor object
TRIGGERS = ExperimentTriggerEnum  # global variable: triggers enum
EVENT_MAPPING = {
    cnst.EVENT_LABELS.UNDEFINED: {cnst.LABEL: cnst.EVENT_LABELS.UNDEFINED.name,
                                  cnst.COLOR: "#dddddd", cnst.MIN_DURATION: 0, cnst.MAX_DURATION: 1e6},
    cnst.EVENT_LABELS.FIXATION: {cnst.LABEL: cnst.EVENT_LABELS.FIXATION.name,
                                 cnst.COLOR: "#1f78b4", cnst.MIN_DURATION: 50, cnst.MAX_DURATION: 2000},
    cnst.EVENT_LABELS.SACCADE: {cnst.LABEL: cnst.EVENT_LABELS.SACCADE.name,
                                cnst.COLOR: "#33a02c", cnst.MIN_DURATION: 10, cnst.MAX_DURATION: 250},
    cnst.EVENT_LABELS.PSO: {cnst.LABEL: cnst.EVENT_LABELS.PSO.name,
                            cnst.COLOR: "#b2df8a", cnst.MIN_DURATION: 10, cnst.MAX_DURATION: 80},
    cnst.EVENT_LABELS.SMOOTH_PURSUIT: {cnst.LABEL: cnst.EVENT_LABELS.SMOOTH_PURSUIT.name,
                                       cnst.COLOR: "#fb9a99", cnst.MIN_DURATION: 40, cnst.MAX_DURATION: 2000},
    cnst.EVENT_LABELS.BLINK: {cnst.LABEL: cnst.EVENT_LABELS.BLINK.name,
                              cnst.COLOR: "#222222", cnst.MIN_DURATION: 20, cnst.MAX_DURATION: 2000}
}

DEFAULT_MISSING_VALUE = np.nan  # default value for missing data

DEFAULT_VIEWER_DISTANCE = 60  # cm

DEFAULT_NAN_PADDING = 0  # amount (ms) by which to extend before and after periods of data loss

# DIRECTORIES
BASE_DIR = os.getcwd()  # TODO: set the base directory for the experiment
STIMULI_DIR = psx.join(BASE_DIR, "Stimuli")
RAW_DATA_DIR = psx.join(BASE_DIR, "RawData")
OUTPUT_DIR = psx.join(BASE_DIR, "Results")
DATASETS_DIR = psx.join(BASE_DIR, "Datasets")
LOGS_DIR = psx.join(BASE_DIR, "Logs")

EXPERIMENT_SPECIFIC_VARIABLES = []  # additional variable recorded in the experiment and extracted from the raw data
