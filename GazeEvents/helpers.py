import Config.constants as cnst
from Config.GazeEventTypeEnum import GazeEventTypeEnum
from typing import Union


def parse_gaze_event(ev: Union[GazeEventTypeEnum, cnst.EVENT_LABELS, int, str, float],
                     safe: bool = True) -> cnst.EVENT_LABELS:
    """
    Parses a gaze label from the original dataset's type to type GazeEventTypeEnum
    :param ev: the gaze label to parse
    :param safe: if True, returns GazeEventTypeEnum.UNDEFINED when the parsing fails
    :return: the parsed gaze label
    """
    try:
        if isinstance(ev, GazeEventTypeEnum):
            return ev
        if isinstance(ev, cnst.EVENT_LABELS):
            return ev
        if isinstance(ev, int):
            return GazeEventTypeEnum(ev)
        if isinstance(ev, str):
            return GazeEventTypeEnum[ev.upper()]
        if isinstance(ev, float):
            if not ev.is_integer():
                raise ValueError(f"Invalid value: {ev}")
            return GazeEventTypeEnum(int(ev))
        raise TypeError(f"Incompatible type: {type(ev)}")
    except Exception as err:
        if safe and (isinstance(err, ValueError) or isinstance(err, TypeError)):
            return GazeEventTypeEnum.UNDEFINED
        raise err
