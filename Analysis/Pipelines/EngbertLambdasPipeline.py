import os
from typing import Union, List

import pandas as pd

import Config.constants as cnst
from Analysis.Pipelines.BaseComparisonPipeline import BaseComparisonPipeline
from GazeDetectors.EngbertDetector import EngbertDetector
import Analysis.helpers as hlp
from Visualization import distributions_grid as dg


class EngbertLambdasPipeline(BaseComparisonPipeline):
    _LAMBDA_STR = "λ"

    def run(self, verbose=False, **kwargs):
        lambdas = kwargs.get("lambdas", None)
        if lambdas is None:
            detectors = self._get_default_detectors()
        else:
            detectors = list({EngbertDetector(lambdaa=lmda) for lmda in lambdas})
        results = super().run(
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
        agg_thresholds = hlp.group_and_aggregate(thresholds, cnst.STIMULUS)
        threshold_fig = dg.distributions_grid(
            agg_thresholds,
            title=f"{self.dataset_name.upper()}:\t\tVelocity-Threshold Distribution"
        )
        threshold_fig.write_html(os.path.join(self._dataset_dir, "Velocity-Threshold Distribution.html"))

    @classmethod
    def _get_default_detectors(cls) -> Union[EngbertDetector, List[EngbertDetector]]:
        return list({EngbertDetector(lambdaa=lmda) for lmda in range(1, 7)})

    @staticmethod
    def _column_mapper(colname: str) -> str:
        if EngbertLambdasPipeline._LAMBDA_STR not in colname:
            return colname
        lambda_index = colname.index(EngbertLambdasPipeline._LAMBDA_STR)
        comma_index = colname.index(",")
        sub_col = colname[lambda_index:comma_index]
        new_name = sub_col.replace("'", "").replace("\"", "").replace(":", "=")
        return new_name
