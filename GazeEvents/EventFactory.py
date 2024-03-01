import numpy as np
from abc import ABC
from typing import List

import Config.constants as cnst
from GazeEvents.BaseEvent import BaseEvent
from GazeEvents.BlinkEvent import BlinkEvent
from GazeEvents.FixationEvent import FixationEvent
from GazeEvents.SaccadeEvent import SaccadeEvent
import Utils.array_utils as arr_utils


class EventFactory(ABC):

    @staticmethod
    def make(et: cnst.EVENTS, t: np.ndarray, **event_data) -> BaseEvent:
        """
        Creates a single GazeEvent from the given data.

        :param et: The type of event to create
        :param t: The timestamps of the event
        :param event_data: The data to use for the event (e.g. x, y, pupil, etc.)

        :return: a GazeEvent object
        :raise: ValueError if the given event type is not valid
        """
        if et == cnst.EVENTS.BLINK:
            return BlinkEvent(timestamps=t)
        if et == cnst.EVENTS.SACCADE:
            return SaccadeEvent(timestamps=t,
                                x=event_data.get("x", np.array([])),
                                y=event_data.get("y", np.array([])),
                                viewer_distance=event_data.get("viewer_distance", np.nan))
        if et == cnst.EVENTS.FIXATION:
            return FixationEvent(timestamps=t,
                                 x=event_data.get("x", np.array([])),
                                 y=event_data.get("y", np.array([])),
                                 pupil=event_data.get("pupil", np.array([])),
                                 viewer_distance=event_data.get("viewer_distance", np.nan))
        raise ValueError(f"Invalid event type: {et}")

    @staticmethod
    def make_multiple(ets: np.ndarray, t: np.ndarray, **kwargs) -> List[BaseEvent]:
        """
        Creates a list of GazeEvents from the given data.

        :param ets: array of event types
        :param t: array of timestamps (must be same length as ets)
        :param kwargs: dictionary of arrays of event data (e.g. x, y, pupil, etc.)

        :return: a list of GazeEvent objects
        :raise: ValueError if the length of `ets` and `t` are not the same
        """
        if len(ets) != len(t):
            raise ValueError("Length of `ets` and `t` must be the same")
        chunk_idxs = arr_utils.get_chunk_indices(ets)
        event_list = []
        for idxs in chunk_idxs:
            et: cnst.EVENTS = ets[idxs[0]]
            event_data = {k: v[idxs] for k, v in kwargs.items()}
            event = EventFactory.make(et, t[idxs], **event_data)
            event_list.append(event)
        return event_list


