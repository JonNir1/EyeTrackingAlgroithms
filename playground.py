import os
import copy
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import pickle as pkl
from scipy.stats import norm
import sklearn.metrics as met
import plotly.graph_objects as go
import plotly.io as pio

import Config.constants as cnst
import Config.experiment_config as cnfg

from GazeDetectors.EngbertDetector import EngbertDetector
from DataSetLoaders.DataSetFactory import DataSetFactory

pio.renderers.default = "browser"

DATASET_NAME = "Lund2013"

###################

PATH = r'C:\Users\nirjo\Documents\University\Masters\Projects\EyeTrackingAlgroithms\Results\DetectorComparison\Lund2013'
samples = pd.read_pickle(os.path.join(PATH, 'samples.pkl'))
events = pd.read_pickle(os.path.join(PATH, 'events.pkl'))
with open(os.path.join(PATH, 'matches.pkl'), 'rb') as f:
    matches = pkl.load(f)
with open(os.path.join(PATH, 'event_features.pkl'), 'rb') as f:
    event_features = pkl.load(f)
del f

del samples, event_features


label = cnfg.EVENT_LABELS.FIXATION
# P = events.map(lambda cell: len([e for e in cell if e.event_label == label]) if cell is not None else None)
# N = events.map(lambda cell: len([e for e in cell if e.event_label != label]) if cell is not None else None)

# populate detection scores:
matches_iou = matches["iou"]
TP, FP, FN, TN = (np.full_like(matches_iou, None, dtype=float) for _ in range(4))
for i, idx in enumerate(matches_iou.index):
    for j, col_pair in enumerate(matches_iou.columns):
        match_map = matches_iou.loc[idx, col_pair]
        if match_map is None:
            continue
        gt_col, pred_col = col_pair
        # TP is the number of GT/Pred events with the required label, that were successfully matched:
        TP[i, j] = sum([e in match_map.keys() for e in events.loc[idx, gt_col] if e.event_label == label])
        # TN is the number of GT/Pred events with a different label, that were successfully matched:
        TN[i, j] = sum([e in match_map.keys() for e in events.loc[idx, gt_col] if e.event_label != label])
        # FP is the number of Pred events with the required label, that were not matched:
        FP[i, j] = sum([e not in match_map.values() for e in events.loc[idx, pred_col] if e.event_label == label])
        # FN is the number of GT events with the required label, that were not matched:
        FN[i, j] = sum([e not in match_map.keys() for e in events.loc[idx, gt_col] if e.event_label == label])

        # sanity check:
        TP_from_pred = sum([e in match_map.values() for e in events.loc[idx, pred_col] if e.event_label == label])
        assert TP[i, j] == TP_from_pred, f"TP[{i}, {j}] = {TP[i, j]} != {TP_from_pred} = TP_from_pred"
        TN_from_pred = sum([e in match_map.values() for e in events.loc[idx, pred_col] if e.event_label != label])
        assert TN[i, j] == TN_from_pred, f"TN[{i}, {j}] = {TN[i, j]} != {TN_from_pred} = TN_from_pred"


del i, idx, j, col_pair, match_map, gt_col, pred_col, TP_from_pred, TN_from_pred

############


def _calculate_sdt_measures(
        tp: int, fp: int, fn: int, tn: int, correction: Optional[str] = "loglinear"
) -> (float, float, float):
    """
    Calculates Signal Detection Theory measures: d-prime, beta and criterion.
    Optionally, adjusts for floor/ceiling effects using the specified correction method.
    See information on correction methods at https://stats.stackexchange.com/a/134802/288290.
    See implementation details at https://lindeloev.net/calculating-d-in-python-and-php/.
    :return: d-prime, beta, criterion
    """
    assert ((0 <= tp) | (np.isnan(tp))).all(), "True Positive count must be non-negative"
    assert ((0 <= fp) | (np.isnan(fp))).all(), "False Positive count must be non-negative"
    assert ((0 <= fn) | (np.isnan(fn))).all(), "False Negative count must be non-negative"
    assert ((0 <= tn) | (np.isnan(tn))).all(), "True Negative count must be non-negative"
    Z = norm.ppf
    hr, far = __calculate_rates_for_sdt(tp, fp, fn, tn, correction)
    d_prime = Z(hr) - Z(far)
    beta = np.exp((Z(far) ** 2 - Z(hr) ** 2) / 2)
    criterion = -0.5 * (Z(hr) + Z(far))
    return d_prime, beta, criterion


def __calculate_rates_for_sdt(
        tp, fp, fn, tn, correction: Optional[str] = None
) -> (float, float):
    if correction is None or not correction:
        # correction not specified, return as is
        hr = tp / (tp + fn) if tp + fn > 0 else np.nan
        far = fp / (fp + tn) if fp + tn > 0 else np.nan
        return hr, far
    if correction in {"mk", "m&k", "macmillan-kaplan", "macmillan"}:
        # Macmillan & Kaplan (1985) correction
        hr = __macmillan_kaplan_correction(tp, fn)
        far = __macmillan_kaplan_correction(fp, tn)
        return hr, far
    if correction in {"ll", "loglinear", "log-linear", "hautus"}:
        # Hautus (1995) correction
        hr, far = __loglinear_rate_correction(tp, fp, fn, tn)
        return hr, far
    raise ValueError(f"Invalid correction: {correction}")


def __macmillan_kaplan_correction(positive_count: int, negative_count: int) -> float:
    """
    Calculates the ratio of p/(p+n) while adjusting for floor/ceiling effects by replacing 0 and 1 rates with 0.5/(p+n)
    and 1-0.5/(p+n), respectively - as suggested by Macmillan & Kaplan (1985).
    See more details at https://stats.stackexchange.com/a/134802/288290.
    Implementation from https://lindeloev.net/calculating-d-in-python-and-php/.
    """
    total_count = positive_count + negative_count
    rate = positive_count / total_count if total_count > 0 else np.nan
    if rate == 0:
        rate = 0.5 / total_count
    if rate == 1:
        rate = 1 - 0.5 / total_count
    return rate


def __loglinear_rate_correction(tp, fp, fn, tn) -> (float, float):
    """
    Calculates the Hit Rate & False Alarm Rate while adjusting for floor/ceiling effects by adding the proportion of
    positive and negative events to the counts, as suggested by Hautus (1995).
    See https://stats.stackexchange.com/a/134802/288290 for more details.
    """
    p, n = tp + fn, fp + tn
    hr = tp / p if p > 0 else np.nan
    far = fp / n if n > 0 else np.nan
    if hr != 0 and hr != 1 and far != 0 and far != 1:
        # no correction needed
        return hr, far
    prevalence = p / (p + n)
    tp, fp = tp + prevalence, fp + 1 - prevalence
    p, n = p + 2 * prevalence, n + 2 * (1 - prevalence)
    hr = tp / p if p > 0 else np.nan
    far = fp / n if n > 0 else np.nan
    return hr, far

############

tpr = TP / (TP + FN)
fpr = FP / (FP + TN)
ppv = TP / (TP + FP)
f1 = 2 * (tpr * ppv) / (tpr + ppv)
d_prime, beta, criterion = np.vectorize(_calculate_sdt_measures)(TP, FP, FN, TN, "loglinear")

############
