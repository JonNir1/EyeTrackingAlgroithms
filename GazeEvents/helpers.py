from typing import Union

import Config.constants as cnst


def parse_gaze_event(ev: Union[cnst.EVENT_LABELS, int, str, float],
                     safe: bool = True) -> cnst.EVENT_LABELS:
    """
    Parses a gaze label from the original dataset's type to type GazeEventTypeEnum
    :param ev: the gaze label to parse
    :param safe: if True, returns GazeEventTypeEnum.UNDEFINED when the parsing fails
    :return: the parsed gaze label
    """
    try:
        if isinstance(ev, cnst.EVENT_LABELS):
            return ev
        if isinstance(ev, int):
            return cnst.EVENT_LABELS(ev)
        if isinstance(ev, str):
            return cnst.EVENT_LABELS[ev.upper()]
        if isinstance(ev, float):
            if not ev.is_integer():
                raise ValueError(f"Invalid value: {ev}")
            return cnst.EVENT_LABELS(int(ev))
        raise TypeError(f"Incompatible type: {type(ev)}")
    except Exception as err:
        if safe and (isinstance(err, ValueError) or isinstance(err, TypeError)):
            return cnst.EVENT_LABELS.UNDEFINED
        raise err
