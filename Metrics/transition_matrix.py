import numpy as np
from typing import Sequence

import Config.constants as cnst
import GazeEvents.helpers as hlp
from GazeEvents.BaseEvent import BaseEvent

_NUM_EVENTS = len(cnst.EVENT_LABELS)


def transition_matrix(seq: Sequence) -> np.ndarray:
    """
    Calculate the transition probabilities between events in the given sequence.
    Returns a matrix where each row represents the current event and each column represents the next event, so cells
    contain the probability P(to_column | from_row).
    """
    counts = _transition_counts(seq)
    row_sum = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sum, out=np.zeros_like(counts, dtype=float), where=row_sum != 0)
    return probs


def matrix_distance(m1: np.ndarray, m2: np.ndarray, norm: str = "fro") -> float:
    """ Calculate the distance between two matrices. """
    norm = norm.lower()
    if norm == "fro" or norm == "frobenius" or norm == "euclidean" or norm == "l2":
        return np.linalg.norm(m1 - m2, ord="fro")
    if norm == "l1" or norm == "manhattan":
        return np.linalg.norm(m1 - m2, ord=1)
    if norm == "linf" or norm == "infinity" or norm == "max" or norm == np.inf:
        return np.linalg.norm(m1 - m2, ord=np.inf)
    if norm == "kl" or norm == "kullback-leibler":
        stationary1 = _calculate_stationary_distribution(m1)
        stationary2 = _calculate_stationary_distribution(m2)
        return np.sum(stationary1 * np.log(stationary1 / stationary2))
    raise ValueError(f"Invalid norm: {norm}")


def _transition_counts(seq: Sequence) -> np.ndarray:
    """ Count the number of transitions between events in the given sequence. """
    seq = [e.event_label if isinstance(e, BaseEvent) else hlp.parse_event_label(e, safe=False) for e in seq]
    counts = np.zeros((_NUM_EVENTS, _NUM_EVENTS), dtype=int)
    for i in range(len(seq) - 1):
        from_event = seq[i]
        to_event = seq[i + 1]
        counts[from_event][to_event] += 1
    return counts


def _calculate_stationary_distribution(probs: np.ndarray, allow_zero: bool = True) -> np.ndarray:
    """
    Calculate the stationary distribution of the given transition matrix.
    The stationary distribution is the left-eigenvector (π) of the transition matrix (P), so πP = π.
    See detailed explanation: https://stackoverflow.com/a/58334399/8543025
    """
    eigenvalues, eigenvectors = np.linalg.eig(probs.T)
    stationary = np.array(eigenvectors[:, np.where(np.abs(eigenvalues - 1.) < cnst.EPSILON)[0][0]].flat)
    stationary = (stationary / np.sum(stationary)).real
    stationary[stationary < 0] = 0
    if not allow_zero:
        stationary[stationary == 0] = 0.01 * cnst.EPSILON
    stationary /= np.sum(stationary)
    return stationary
