from typing import Union, List, Iterable

import pandas as pd

import peyes.Config.constants as cnst
from peyes.Analysis.Pipelines.BaseComparisonPipeline import BaseComparisonPipeline
from peyes.GazeDetectors.EngbertDetector import EngbertDetector
from peyes.Visualization import distributions_grid as dg
from peyes.Analysis.helpers import group_and_aggregate
from peyes.Analysis.figures import save_figure


class EngbertLambdasPipeline(BaseComparisonPipeline):
    _LAMBDA_STR = "λ"

    def _run_impl(self, lambdas: Iterable[float] = None, verbose=False, **kwargs):
        if lambdas is None:
            detectors = self._get_default_detectors()
        else:
            detectors = list(EngbertDetector(lambdaa=lmda) for lmda in sorted(set(lambdas)))
        results = super()._run_impl(
            detectors=detectors,
            allow_cross_matching=False,
            verbose=verbose,
        )
        self._velocity_threshold_figure(detector_results=results[2])
        return results

    def _velocity_threshold_figure(self, detector_results: pd.DataFrame):
        thresholds = pd.concat(
            [detector_results[f"{EngbertLambdasPipeline._LAMBDA_STR}=1"].map(lambda cell: cell['thresh_Vx']),
             detector_results[f"{EngbertLambdasPipeline._LAMBDA_STR}=1"].map(lambda cell: cell['thresh_Vy'])],
            axis=1, keys=["Vx", "Vy"]
        )
        agg_thresholds = group_and_aggregate(thresholds, cnst.STIMULUS)
        threshold_fig = dg.distributions_grid(
            agg_thresholds,
            title=f"{self.dataset_name.upper()}:\t\tVelocity-Threshold Distribution"
        )
        save_figure(threshold_fig, self._output_dir, "Velocity-Threshold Distribution")

    @classmethod
    def _get_default_detectors(cls) -> Union[EngbertDetector, List[EngbertDetector]]:
        return list(EngbertDetector(lambdaa=lmda) for lmda in range(1, 7))

    @staticmethod
    def _column_mapper(colname: str) -> str:
        if EngbertLambdasPipeline._LAMBDA_STR not in colname:
            return colname
        lambda_index = colname.index(EngbertLambdasPipeline._LAMBDA_STR)
        comma_index = colname.index(",")
        sub_col = colname[lambda_index:comma_index]
        new_name = sub_col.replace("'", "").replace("\"", "").replace(":", "=")
        return new_name
