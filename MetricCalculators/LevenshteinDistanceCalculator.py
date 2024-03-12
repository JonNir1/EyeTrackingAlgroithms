import numpy as np
import Levenshtein
from typing import Sequence

import Config.constants as cnst
from GazeEvents.BaseEvent import BaseEvent

_NUM_EVENTS = len(cnst.EVENTS)


def calculate_distance(seq1: Sequence, seq2: Sequence) -> int:
    """ Calculates the Levenshtein distance between two sequences of samples or events. """
    if all(isinstance(e, cnst.EVENTS) for e in seq1) and all(isinstance(e, cnst.EVENTS) for e in seq2):
        # both sequences are sample based
        return Levenshtein.distance(seq1, seq2)
    if all(isinstance(e, BaseEvent) for e in seq1) and all(isinstance(e, BaseEvent) for e in seq2):
        # both sequences are event based
        return Levenshtein.distance(seq1, seq2, processor=lambda x: x.event_type)
    raise ValueError("Sequences must be of the same type (`GazeEventTypeEnum` or `BaseEvent`)")


def calculate_ratio(seq1: Sequence, seq2: Sequence) -> float:
    """ Calculates the Levenshtein ratio between two sequences of samples or events. """
    distance = calculate_distance(seq1, seq2)
    return 1 - distance / sum([len(seq1), len(seq2)])
