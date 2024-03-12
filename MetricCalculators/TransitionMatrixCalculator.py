import numpy as np
from typing import Sequence

import Config.constants as cnst
from GazeEvents.BaseEvent import BaseEvent

_NUM_EVENTS = len(cnst.EVENTS)


def calculate_transition_matrix(seq: Sequence) -> np.ndarray:
    if all(isinstance(e, BaseEvent) for e in seq):
        seq = [e.event_type for e in seq]
    if not all(isinstance(e, cnst.EVENTS) for e in seq):
        raise ValueError("Sequence must be of the same type (`GazeEventTypeEnum` or `BaseEvent`)")
    matrix = np.zeros((_NUM_EVENTS, _NUM_EVENTS), dtype=int)
    for i in range(len(seq) - 1):
        from_event = seq[i]
        to_event = seq[i + 1]
        matrix[from_event][to_event] += 1
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix
