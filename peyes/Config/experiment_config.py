"""
This file contains the configuration for each specific experiment.
"""
import os
import numpy as np
import posixpath as psx

import peyes.Config.constants as cnst
from peyes.Config.ScreenMonitor import ScreenMonitor
from peyes.Config.ExperimentTriggerEnum import ExperimentTriggerEnum
from peyes.Config.GazeEventTypeEnum import GazeEventTypeEnum

# GLOBAL VARIABLES
SCREEN_MONITOR: ScreenMonitor = ScreenMonitor.from_tobii_default()  # global variable: screen monitor object
EVENT_LABELS = GazeEventTypeEnum  # global variable: event labels enum
TRIGGERS = ExperimentTriggerEnum  # global variable: triggers enum

# Event Detection Parameters
MINIMUM_SAMPLES_IN_EVENT: int = 2  # minimum number of samples in an event
DEFAULT_MISSING_VALUE = np.nan  # default value for missing data
DEFAULT_NAN_PADDING = 0  # amount (ms) by which to extend before and after periods of data loss
DEFAULT_VIEWER_DISTANCE = 60  # cm
MICROSACCADE_AMPLITUDE_THRESHOLD = 1.0  # degrees

EVENT_MAPPING = {
    EVENT_LABELS.UNDEFINED: {cnst.LABEL: EVENT_LABELS.UNDEFINED.name,
                             cnst.COLOR: "#dddddd", cnst.MIN_DURATION: 0, cnst.MAX_DURATION: 1e6},
    EVENT_LABELS.FIXATION: {cnst.LABEL: EVENT_LABELS.FIXATION.name,
                            cnst.COLOR: "#1f78b4", cnst.MIN_DURATION: 50, cnst.MAX_DURATION: 2000},
    EVENT_LABELS.SACCADE: {cnst.LABEL: EVENT_LABELS.SACCADE.name,
                           cnst.COLOR: "#33a02c", cnst.MIN_DURATION: 10, cnst.MAX_DURATION: 250},
    EVENT_LABELS.PSO: {cnst.LABEL: EVENT_LABELS.PSO.name,
                       cnst.COLOR: "#b2df8a", cnst.MIN_DURATION: 10, cnst.MAX_DURATION: 80},
    EVENT_LABELS.SMOOTH_PURSUIT: {cnst.LABEL: EVENT_LABELS.SMOOTH_PURSUIT.name,
                                  cnst.COLOR: "#fb9a99", cnst.MIN_DURATION: 40, cnst.MAX_DURATION: 2000},
    EVENT_LABELS.BLINK: {cnst.LABEL: EVENT_LABELS.BLINK.name,
                         cnst.COLOR: "#222222", cnst.MIN_DURATION: 20, cnst.MAX_DURATION: 2000}
}

# DIRECTORIES
BASE_DIR = os.getcwd()  # TODO: set the base directory for the experiment
STIMULI_DIR = psx.join(BASE_DIR, "Stimuli")
RAW_DATA_DIR = psx.join(BASE_DIR, "RawData")
OUTPUT_DIR = psx.join(BASE_DIR, "Results")
DATASETS_DIR = psx.join(BASE_DIR, "Datasets")
LOGS_DIR = psx.join(BASE_DIR, "Logs")

EXPERIMENT_SPECIFIC_VARIABLES = []  # additional variable recorded in the experiment and extracted from the raw data
