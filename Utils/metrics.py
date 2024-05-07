from typing import Sequence, Optional
from collections import Counter

import numpy as np
import sklearn.metrics as met
import Levenshtein

import Config.constants as cnst
import Config.experiment_config as cnfg
import GazeEvents.helpers as hlp


_NUM_EVENTS = len(cnfg.EVENT_LABELS)


# TODO: calculate confusion matrix, precision, recall, F1-score, etc.


def accuracy(gt: Sequence, pred: Sequence) -> float:
    """ Calculates the accuracy between two sequences of samples or events. """
    gt = [hlp.parse_event_label(e, safe=False) for e in gt]
    pred = [hlp.parse_event_label(e, safe=False) for e in pred]
    return met.accuracy_score(gt, pred)


def balanced_accuracy(gt: Sequence, pred: Sequence) -> float:
    """
    Calculates the weighted accuracy between two sequences of samples or events.
    Use complementary support (1-P[class | GT]) as weight.
    """
    gt = [hlp.parse_event_label(e, safe=False) for e in gt]
    pred = [hlp.parse_event_label(e, safe=False) for e in pred]
    return met.balanced_accuracy_score(gt, pred)


def precision(gt: Sequence, pred: Sequence) -> float:
    """
    Calculates the precision between two sequences of samples or events.
    If the sequences are binary, calculates the binary precision; otherwise, calculates the weighted precision.
    """
    prec, _, _ = _calc_precision_recall_f1(gt, pred)
    return prec


def recall(gt: Sequence, pred: Sequence) -> float:
    """
    Calculates the recall (sensitivity) between two sequences of samples or events.
    If the sequences are binary, calculates the binary recall; otherwise, calculates the weighted recall.
    """
    _, rec, _ = _calc_precision_recall_f1(gt, pred)
    return rec


def f1_score(gt: Sequence, pred: Sequence) -> float:
    """
    Calculates the F1-score between two sequences of samples or events.
    If the sequences are binary, calculates the binary F1-score; otherwise, calculates the weighted F1-score.
    """
    _, _, f1 = _calc_precision_recall_f1(gt, pred)
    return f1


def confusion_matrix(
        gt: Sequence,
        pred: Sequence,
) -> np.ndarray:
    """
    Calculates the confusion matrix between two sequences of samples or events, where rows are the ground-truth labels
    and columns are the predicted labels.
    :param gt: The ground-truth sequence of samples or events.
    :param pred: The predicted sequence of samples or events.
    :return: the square confusion matrix.
    """
    gt = [hlp.parse_event_label(e, safe=False) for e in gt]
    pred = [hlp.parse_event_label(e, safe=False) for e in pred]
    return met.confusion_matrix(gt, pred, labels=cnfg.EVENT_LABELS)


def complement_nld(gt: Sequence, pred: Sequence) -> float:
    """
    Calculates the complement to Normalized Levenshtein Distance (1-NLD) between two sequences of samples or events.
    """
    gt = [hlp.parse_event_label(e, safe=False) for e in gt]
    pred = [hlp.parse_event_label(e, safe=False) for e in pred]
    d = Levenshtein.distance(gt, pred)
    normalized_d = d / max(len(gt), len(pred))
    return 1 - normalized_d


def cohen_kappa(gt: Sequence, pred: Sequence) -> float:
    """ Calculates the Cohen's Kappa coefficient between two sequences of samples or events. """
    gt = [hlp.parse_event_label(e, safe=False) for e in gt]
    pred = [hlp.parse_event_label(e, safe=False) for e in pred]
    return met.cohen_kappa_score(gt, pred)


def matthews_correlation(gt: Sequence, pred: Sequence) -> float:
    """ Calculates the Matthews correlation coefficient between two sequences of samples or events. """
    gt = [hlp.parse_event_label(e, safe=False) for e in gt]
    pred = [hlp.parse_event_label(e, safe=False) for e in pred]
    return met.matthews_corrcoef(gt, pred)


def normalized_kl_divergence(gt: Sequence, pred: Sequence) -> float:
    """
    Calculates the normalized Kullback-Leibler divergence between two sequences of samples or events.
    Normalization is done by taking the exponential of the negative KL-divergence, making the result in the range [0, 1]
    where 1 means the sequences are identical and 0 means they are completely different.
    """
    kld = transition_matrix_distance(gt, pred, norm="kl")
    return np.exp(-kld)


def transition_matrix_distance(gt: Sequence, pred: Sequence, norm: str) -> float:
    """ Calculate the distance between the transition matrices of two sequences. """
    tm1 = _transition_matrix(gt)
    tm2 = _transition_matrix(pred)
    norm = norm.lower()
    if norm == "fro" or norm == "frobenius" or norm == "euclidean" or norm == "l2":
        return np.linalg.norm(tm1 - tm2, ord="fro")
    if norm == "l1" or norm == "manhattan":
        return np.linalg.norm(tm1 - tm2, ord=1)
    if norm == "linf" or norm == "infinity" or norm == "max" or norm == np.inf:
        return np.linalg.norm(tm1 - tm2, ord=np.inf)
    if norm == "kl" or norm == "kullback-leibler":
        stationary1 = _calculate_stationary_distribution(tm1)
        stationary2 = _calculate_stationary_distribution(tm2)
        return np.sum(stationary1 * np.log(stationary1 / stationary2))
    raise ValueError(f"Invalid norm: {norm}")


def levenshtein_distance(gt: Sequence, pred: Sequence, do_normalization: bool = True) -> float:
    """ Calculates the Levenshtein distance between two sequences of samples or events. """
    gt = [hlp.parse_event_label(e, safe=False) for e in gt]
    pred = [hlp.parse_event_label(e, safe=False) for e in pred]
    d = Levenshtein.distance(gt, pred)
    if do_normalization:
        return d / max(len(gt), len(pred))
    return d


def levenshtein_ratio(gt: Sequence, pred: Sequence) -> float:
    """ Calculates the Levenshtein ratio between two sequences of samples or events. """
    gt = [hlp.parse_event_label(e, safe=False) for e in gt]
    pred = [hlp.parse_event_label(e, safe=False) for e in pred]
    return Levenshtein.ratio(gt, pred)


def _transition_matrix(seq: Sequence) -> np.ndarray:
    """
    Calculate the transition probabilities between events in the given sequence.
    Returns a matrix where each row represents the current event and each column represents the next event, so cells
    contain the probability P(to_column | from_row).
    """
    counts = _transition_counts(seq)
    row_sum = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sum, out=np.zeros_like(counts, dtype=float), where=row_sum != 0)
    return probs


def _transition_counts(seq: Sequence) -> np.ndarray:
    """ Count the number of transitions between events in the given sequence. """
    seq = [hlp.parse_event_label(e, safe=False) for e in seq]
    counts = np.zeros((_NUM_EVENTS, _NUM_EVENTS), dtype=int)
    for i in range(len(seq) - 1):
        from_event = seq[i]
        to_event = seq[i + 1]
        counts[from_event][to_event] += 1
    return counts


def _calculate_stationary_distribution(probs: np.ndarray, allow_zero: bool = False) -> np.ndarray:
    """
    Calculate the stationary distribution of the given transition matrix.
    The stationary distribution is the left-eigenvector (π) of the transition matrix (P), so πP = π.
    See detailed explanation: https://stackoverflow.com/a/58334399/8543025
    """
    eigenvalues, eigenvectors = np.linalg.eig(probs.T)
    try:
        stationary = np.array(eigenvectors[:, np.where(np.abs(eigenvalues - 1.) < cnst.EPSILON)[0][0]].flat)
        stationary = (stationary / np.sum(stationary)).real
        stationary[stationary < 0] = 0
        if not allow_zero:
            stationary[stationary == 0] = 0.01 * cnst.EPSILON
        stationary /= np.sum(stationary)
    except IndexError:
        # No eigenvalue is close enough to 1, so the matrix does not have a stationary distribution
        stationary = np.full_like(eigenvalues.real, dtype=float, fill_value=np.nan)
    return stationary


def _calc_precision_recall_f1(gt: Sequence, pred: Sequence) -> (float, float, float):
    """
    Calculates the precision, recall, and F1-score between two sequences of samples or events.
    If the sequences are binary, calculates the binary metrics. Otherwise, calculates the weighted metrics.
    """
    gt = [hlp.parse_event_label(e, safe=False) for e in gt]
    pred = [hlp.parse_event_label(e, safe=False) for e in pred]
    is_binary = set(gt) == set(pred) and len(set(gt)) == 2
    labels = [_find_positive_label(gt)] if is_binary else cnfg.EVENT_LABELS
    prec, rec, f1, _ = met.precision_recall_fscore_support(
        gt, pred, labels=labels, zero_division=np.nan, average="weighted"
    )
    return prec, rec, f1


def _find_positive_label(seq: Sequence):
    """ Find the least common label in the given sequence, that is not the undefined label."""
    labels = set(seq)
    assert len(labels) == 2, "The sequence must be binary"
    if cnfg.EVENT_LABELS.UNDEFINED in labels:
        # return the other label
        labels -= {cnfg.EVENT_LABELS.UNDEFINED}
        return labels.pop()
    # return the less common label
    return Counter(seq).most_common()[-1][0]

