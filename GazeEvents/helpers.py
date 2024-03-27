from typing import Union, Sequence

import pandas as pd

import Config.constants as cnst
from GazeEvents.BaseEvent import BaseEvent


def parse_event_label(label: Union[cnst.EVENT_LABELS, int, str, float],
                      safe: bool = True) -> cnst.EVENT_LABELS:
    """
    Parses a gaze label from the original dataset's type to type GazeEventTypeEnum
    :param label: the gaze label to parse
    :param safe: if True, returns GazeEventTypeEnum.UNDEFINED when the parsing fails
    :return: the parsed gaze label
    """
    try:
        if isinstance(label, cnst.EVENT_LABELS):
            return label
        if isinstance(label, int):
            return cnst.EVENT_LABELS(label)
        if isinstance(label, str):
            return cnst.EVENT_LABELS[label.upper()]
        if isinstance(label, float):
            if not label.is_integer():
                raise ValueError(f"Invalid value: {label}")
            return cnst.EVENT_LABELS(int(label))
        raise TypeError(f"Incompatible type: {type(label)}")
    except Exception as err:
        if safe and (isinstance(err, ValueError) or isinstance(err, TypeError)):
            return cnst.EVENT_LABELS.UNDEFINED
        raise err


def drop_events(seq: Sequence, to_drop: Sequence[cnst.EVENT_LABELS] = None) -> Sequence:
    """ Drops events from the given sequence if they are in the set of event-labels to drop. """
    if len(seq) == 0 or pd.isnull(seq).all():
        return seq
    if to_drop is None or len(to_drop) == 0:
        return seq
    to_drop = set(to_drop)
    out = []
    for e in seq:
        if isinstance(e, BaseEvent):
            if e.event_label not in to_drop:
                out.append(e)
        else:
            if parse_event_label(e) not in to_drop:
                out.append(e)
    return out
