from typing import Union, Sequence

import pandas as pd

import Config.experiment_config as cnfg
from GazeEvents.BaseEvent import BaseEvent


def parse_event_label(val: Union[cnfg.EVENT_LABELS, BaseEvent, int, str, float],
                      safe: bool = True) -> cnfg.EVENT_LABELS:
    """
    Parses a gaze label from the original dataset's type to type GazeEventTypeEnum
    :param val: the value to parse
    :param safe: if True, returns GazeEventTypeEnum.UNDEFINED when the parsing fails
    :return: the parsed gaze label
    """
    try:
        if isinstance(val, cnfg.EVENT_LABELS):
            return val
        if isinstance(val, BaseEvent):
            return val.event_label
        if isinstance(val, int):
            return cnfg.EVENT_LABELS(val)
        if isinstance(val, str):
            return cnfg.EVENT_LABELS[val.upper()]
        if isinstance(val, float):
            if not val.is_integer():
                raise ValueError(f"Invalid value: {val}")
            return cnfg.EVENT_LABELS(int(val))
        raise TypeError(f"Incompatible type: {type(val)}")
    except Exception as err:
        if safe and (isinstance(err, ValueError) or isinstance(err, TypeError)):
            return cnfg.EVENT_LABELS.UNDEFINED
        raise err


def count_labels_or_events(data: Sequence[Union[BaseEvent, cnfg.EVENT_LABELS]]) -> pd.Series:
    """
    Counts the number of same-type labels/events in the given data, and fills in missing labels with 0 counts.
    Returns a Series with the counts of each GazEventTypeEnum label.
    """
    labels = pd.Series([e.event_label if isinstance(e, BaseEvent) else e for e in data])
    counts = labels.value_counts()
    if counts.empty:
        return pd.Series({l: 0 for l in cnfg.EVENT_LABELS})
    if len(counts) == len(cnfg.EVENT_LABELS):
        return counts
    missing_labels = pd.Series({l: 0 for l in cnfg.EVENT_LABELS if l not in counts.index})
    return pd.concat([counts, missing_labels]).sort_index()


def drop_events(seq: Sequence, to_drop: Sequence[cnfg.EVENT_LABELS] = None) -> Sequence:
    """ Drops events from the given sequence if they are in the set of event-labels to drop. """
    if len(seq) == 0 or pd.isnull(seq).all():
        return seq
    if to_drop is None or len(to_drop) == 0:
        return seq
    to_drop = set(to_drop)
    out = [e for e in seq if parse_event_label(e) not in to_drop]
    return out
