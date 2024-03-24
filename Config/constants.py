"""
List of all the constants used as field names, column names, etc.
"""

from Config.GazeEventTypeEnum import GazeEventTypeEnum

EPSILON = 1e-8

MILLISECONDS_PER_SECOND = 1000
MICROSECONDS_PER_MILLISECOND = 1000
MICROSECONDS_PER_SECOND = MICROSECONDS_PER_MILLISECOND * MILLISECONDS_PER_SECOND  # 1,000,000

SUBJECT = "subject"
SUBJECT_ID = f"{SUBJECT}_id"
TRIAL = "trial"
GAZE = "gaze"
TIME = "time"
TRIGGER = "trigger"
TARGET = "target"
DURATION = "duration"
DISTANCE = "distance"
ANGLE = "angle"
STIMULUS = "stimulus"
MILLISECONDS = "milliseconds"
MICROSECONDS = "microseconds"
LABEL = "label"
EVENT, EVENTS = "event", "events"
SAMPLING_RATE = "sampling_rate"
VELOCITY = "velocity"
ACCELERATION = "acceleration"

T, X, Y = 't', 'x', 'y'
LEFT = "left"
RIGHT = "right"
PUPIL = "pupil"
LEFT_X, RIGHT_X = f"{LEFT}_{X}", f"{RIGHT}_{X}"
LEFT_Y, RIGHT_Y = f"{LEFT}_{Y}", f"{RIGHT}_{Y}"
LEFT_PUPIL, RIGHT_PUPIL = f"{LEFT}_{PUPIL}", f"{RIGHT}_{PUPIL}"
VIEWER_DISTANCE = "viewer_distance"
PIXEL_SIZE = "pixel_size"
COLOR = "color"
MIN_DURATION = "min_duration"
MAX_DURATION = "max_duration"

EVENT_LABELS = GazeEventTypeEnum
MINIMUM_SAMPLES_IN_EVENT: int = 2  # minimum number of samples in an event
