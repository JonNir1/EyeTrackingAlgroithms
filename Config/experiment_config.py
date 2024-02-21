"""
This file contains the configuration for each specific experiment.
"""

import os

import Config.constants as cnst
from Config.ScreenMonitor import ScreenMonitor
from Config.ExperimentTriggerEnum import ExperimentTriggerEnum

# GLOBAL VARIABLES
SCREEN_MONITOR: ScreenMonitor = ScreenMonitor.from_tobii_default()  # global variable: screen monitor object
TRIGGERS = ExperimentTriggerEnum  # global variable: triggers enum
EVENT_DURATIONS = {
    # global variable: duration range for each event type (in ms)
    cnst.EVENTS.BLINK: (10, 250),
    cnst.EVENTS.FIXATION: (50, 2000),
    cnst.EVENTS.SACCADE: (10, 250)
}

# DIRECTORIES
BASE_DIR = ""
STIMULI_DIR = os.path.join(BASE_DIR, "Stimuli", "generated_stim1")
RAW_DATA_DIR = os.path.join(BASE_DIR, "RawData")
OUTPUT_DIR = os.path.join(BASE_DIR, "Results")

EXPERIMENT_SPECIFIC_VARIABLES = []  # additional variable recorded in the experiment and extracted from the raw data
