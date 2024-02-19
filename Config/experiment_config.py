"""
This file contains the configuration for each specific experiment.
"""

import os
from Config.ScreenMonitor import ScreenMonitor
from Config.ExperimentTriggerEnum import ExperimentTriggerEnum
from Config.GazeEventTypeEnum import GazeEventTypeEnum

# GLOBAL VARIABLES
SCREEN_MONITOR: ScreenMonitor = ScreenMonitor.from_tobii_default()  # global variable: screen monitor object
TRIGGERS = ExperimentTriggerEnum  # global variable: triggers enum
EVENTS = GazeEventTypeEnum  # global variable: events enum

# DIRECTORIES
BASE_DIR = ""
STIMULI_DIR = os.path.join(BASE_DIR, "Stimuli", "generated_stim1")
RAW_DATA_DIR = os.path.join(BASE_DIR, "RawData")
OUTPUT_DIR = os.path.join(BASE_DIR, "Results")

# GAZE-EVENT VALUES
EVENT_DURATIONS = {
    EVENTS.BLINK: (10, 250),
    EVENTS.FIXATION: (50, 2000),
    EVENTS.SACCADE: (10, 250)
}  # duration range for each event type

EXPERIMENT_SPECIFIC_VARIABLES = []  # additional variable recorded in the experiment and extracted from the raw data
