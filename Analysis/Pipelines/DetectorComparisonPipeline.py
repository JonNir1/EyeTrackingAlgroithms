from typing import List, Callable

import pandas as pd

import Config.experiment_config as cnfg
from Analysis.Pipelines.BaseComparisonPipeline import BaseComparisonPipeline
from GazeDetectors.BaseDetector import BaseDetector


class DetectorComparisonPipeline(BaseComparisonPipeline):

    def _preprocess(self, verbose=False) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        return self.load_and_detect(
            column_mapper=lambda col: col[:col.index("ector")] if "ector" in col else col,
            verbose=verbose,
        )
