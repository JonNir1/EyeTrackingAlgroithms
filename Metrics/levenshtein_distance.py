import Levenshtein
from typing import Sequence

import Config.constants as cnst
import GazeEvents.helpers as hlp
from GazeEvents.BaseEvent import BaseEvent

_NUM_EVENTS = len(cnst.EVENT_LABELS)


def calculate_distance(seq1: Sequence, seq2: Sequence) -> int:
    """ Calculates the Levenshtein distance between two sequences of samples or events. """
    seq1 = [hlp.parse_event_label(e, safe=False) for e in seq1]
    seq2 = [hlp.parse_event_label(e, safe=False) for e in seq2]
    return Levenshtein.distance(seq1, seq2)


def calculate_ratio(seq1: Sequence, seq2: Sequence) -> float:
    """ Calculates the Levenshtein ratio between two sequences of samples or events. """
    distance = calculate_distance(seq1, seq2)
    return 1 - distance / sum([len(seq1), len(seq2)])
