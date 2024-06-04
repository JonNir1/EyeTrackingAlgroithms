import time
import warnings
import pickle as pkl
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

import Config.constants as cnst
import Config.experiment_config as cnfg
from Analysis.Calculators.BaseCalculator import BaseCalculator


class SDTCalculator(BaseCalculator):

    @classmethod
    def _calculate_impl(
            cls,
            events: pd.DataFrame,
            matches: Dict[str, pd.DataFrame],
            label: cnfg.EVENT_LABELS,
            correction: Optional[str] = "loglinear",
    ) -> Dict[str, pd.DataFrame]:
        _calculate_sdt_measures = np.vectorize(cls._calculate_sdt_measures__scalar)
        results = {}
        for scheme, matches_df in matches.items():
            TP, FP, FN, TN = cls._calculate_contingency_table(events, matches_df, label)
            recall = TP / (TP + FN)             # hit-rate, TPR, sensitivity
            precision = TP / (TP + FP)          # PPV
            false_alarm_rate = FP / (FP + TN)   # FPR, 1 - specificity
            f1_score = 2 * (precision * recall) / (precision + recall)
            d_prime, beta, criterion = _calculate_sdt_measures(TP, FP, FN, TN, correction)

            scheme_results = np.full_like(TP, None, dtype=object)
            for i in range(scheme_results.shape[0]):
                for j in range(scheme_results.shape[1]):
                    scheme_results[i, j] = pd.Series({
                        "TP": TP[i, j], "FP": FP[i, j], "FN": FN[i, j], "TN": TN[i, j],
                        "Recall": recall[i, j], "Precision": precision[i, j], "FalseAlarmRate": false_alarm_rate[i, j],
                        "F1": f1_score[i, j], "DPrime": d_prime[i, j], "Beta": beta[i, j], "Criterion": criterion[i, j],
                    })
            scheme_results = pd.DataFrame(scheme_results, index=matches_df.index, columns=matches_df.columns)
            results[scheme] = scheme_results
        return results


    @classmethod
    def _calculate_contingency_table(
            cls,
            events: pd.DataFrame,
            scheme_matches: pd.DataFrame,
            label: cnfg.EVENT_LABELS,
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Calculates the contingency table of each trial for a specific event type (fixation, saccade, etc.).
        The contingency table is defined by the existence of matches between GT and Pred events:
            - TP is the number of GT/Pred events with the required label, that were successfully matched.
            - TN is the number of GT/Pred events with a different label, that were successfully matched.
            - FP is the number of Pred events with the required label, that were not matched.
            - FN is the number of GT events with the required label, that were not matched.
        Returns 4 numpy arrays with the same shape as the input `scheme_matches` DataFrame, where cell (i, j) contains
            the corresponding value (tp/fp/etc.) for the i-th trial and j-th detector-pair.
        """
        TP, FP, FN, TN = (np.full_like(scheme_matches, None, dtype=float) for _ in range(4))
        for i, idx in enumerate(scheme_matches.index):
            for j, col_pair in enumerate(scheme_matches.columns):
                trial_matches: dict = scheme_matches.loc[idx, col_pair]
                if trial_matches is None:
                    continue
                gt_col, pred_col = col_pair
                # TP is the number of GT/Pred events with the required label, that were successfully matched:
                TP[i, j] = sum(
                    [e in trial_matches.keys() for e in events.loc[idx, gt_col] if e.event_label == label]
                )
                # TN is the number of GT/Pred events with a different label, that were successfully matched:
                TN[i, j] = sum(
                    [e in trial_matches.keys() for e in events.loc[idx, gt_col] if e.event_label != label]
                )
                # FP is the number of Pred events with the required label, that were not matched:
                FP[i, j] = sum(
                    [e not in trial_matches.values() for e in events.loc[idx, pred_col] if e.event_label == label]
                )
                # FN is the number of GT events with the required label, that were not matched:
                FN[i, j] = sum(
                    [e not in trial_matches.keys() for e in events.loc[idx, gt_col] if e.event_label == label]
                )
                # sanity check:
                TP_from_pred = sum(
                    [e in trial_matches.values() for e in events.loc[idx, pred_col] if e.event_label == label]
                )
                assert TP[i, j] == TP_from_pred, f"TP[{i}, {j}] = {TP[i, j]} != {TP_from_pred} = TP_from_pred"
                TN_from_pred = sum(
                    [e in trial_matches.values() for e in events.loc[idx, pred_col] if e.event_label != label]
                )
                assert TN[i, j] == TN_from_pred, f"TN[{i}, {j}] = {TN[i, j]} != {TN_from_pred} = TN_from_pred"
        return TP, FP, FN, TN

    @staticmethod
    def _calculate_sdt_measures__scalar(
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
        hr, far = SDTCalculator.__calculate_rates_for_sdt__scalar(tp, fp, fn, tn, correction)
        d_prime = Z(hr) - Z(far)
        beta = np.exp((Z(far) ** 2 - Z(hr) ** 2) / 2)
        criterion = -0.5 * (Z(hr) + Z(far))
        return d_prime, beta, criterion

    @staticmethod
    def __calculate_rates_for_sdt__scalar(
            tp, fp, fn, tn, correction: Optional[str] = None
    ) -> (float, float):
        """
        Calculates Hit-Rate and False-Alarm Rate for computing Signal Detection Theory measures, while optionally
        applying a correction for floor/ceiling effects.
        See information on correction methods at https://stats.stackexchange.com/a/134802/288290.
        """
        p, n = tp + fn, fp + tn
        hr = tp / p if p > 0 else np.nan
        far = fp / n if n > 0 else np.nan
        if hr != 0 and hr != 1 and far != 0 and far != 1:
            # no correction needed
            return hr, far
        if correction is None or not correction:
            # correction not specified, return as is
            return hr, far
        if correction in {"mk", "m&k", "macmillan-kaplan", "macmillan"}:
            # apply Macmillan & Kaplan (1985) correction
            if hr == 0:
                hr = 0.5 / p
            if hr == 1:
                hr = 1 - 0.5 / p
            if far == 0:
                far = 0.5 / n
            if far == 1:
                far = 1 - 0.5 / n
            return hr, far
        if correction in {"ll", "loglinear", "log-linear", "hautus"}:
            # apply Hautus (1995) correction
            prevalence = p / (p + n)
            new_tp, new_fp = tp + prevalence, fp + 1 - prevalence
            new_p, new_n = p + 2 * prevalence, n + 2 * (1 - prevalence)
            hr = new_tp / new_p if new_p > 0 else np.nan
            far = new_fp / new_n if new_n > 0 else np.nan
            return hr, far
        raise ValueError(f"Invalid correction: {correction}")
