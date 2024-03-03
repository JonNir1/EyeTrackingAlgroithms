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
EVENT_MAPPING = {
    cnst.EVENTS.UNDEFINED: {"label": "Undefined", "color": "#dddddd", "min_duration": 0, "max_duration": 1e6},
    cnst.EVENTS.FIXATION: {"label": "Fixation", "color": "#1f78b4", "min_duration": 50, "max_duration": 2000},
    cnst.EVENTS.SACCADE: {"label": "Saccade", "color": "#33a02c", "min_duration": 10, "max_duration": 250},
    cnst.EVENTS.PSO: {"label": "PSO", "color": "#b2df8a", "min_duration": 10, "max_duration": 80},
    cnst.EVENTS.SMOOTH_PURSUIT: {"label": "Smooth Pursuit", "color": "#fb9a99", "min_duration": 10, "max_duration": 2000},
    cnst.EVENTS.BLINK: {"label": "Blink", "color": "#222222", "min_duration": 10, "max_duration": 2000}
}

# DIRECTORIES
BASE_DIR = ""
STIMULI_DIR = os.path.join(BASE_DIR, "Stimuli", "generated_stim1")
RAW_DATA_DIR = os.path.join(BASE_DIR, "RawData")
OUTPUT_DIR = os.path.join(BASE_DIR, "Results")

EXPERIMENT_SPECIFIC_VARIABLES = []  # additional variable recorded in the experiment and extracted from the raw data
