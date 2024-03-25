import Config.constants as cnst
from Config.GazeEventTypeEnum import GazeEventTypeEnum


def to_event_label(val) -> cnst.EVENT_LABELS:
    """ Converts a value to a GazeEventTypeEnum value. """
    if isinstance(val, GazeEventTypeEnum):
        return val
    if isinstance(val, cnst.EVENT_LABELS):
        return val
    if isinstance(val, int):
        return cnst.EVENT_LABELS(val)
    if isinstance(val, str):
        return cnst.EVENT_LABELS[val]
    if isinstance(val, float):
        return cnst.EVENT_LABELS(int(val))
    raise ValueError(f"Unknown type: {type(val)}")
