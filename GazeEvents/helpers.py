from typing import Union

import Config.constants as cnst


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
