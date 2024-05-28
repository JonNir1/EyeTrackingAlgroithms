from typing import List, Union

from peyes.Analysis.Pipelines.BaseComparisonPipeline import BaseComparisonPipeline
from peyes.GazeDetectors.BaseDetector import BaseDetector


class DetectorComparisonPipeline(BaseComparisonPipeline):
    _DETECTOR_STR = "Detector"

    @staticmethod
    def _column_mapper(colname: str) -> str:
        if DetectorComparisonPipeline._DETECTOR_STR not in colname:
            return colname
        ector_index = colname.index(DetectorComparisonPipeline._DETECTOR_STR[3:])
        return colname[:ector_index]

    @classmethod
    def _get_default_detectors(cls) -> Union[BaseDetector, List[BaseDetector]]:
        from peyes.GazeDetectors.IVTDetector import IVTDetector
        from peyes.GazeDetectors.IDTDetector import IDTDetector
        from peyes.GazeDetectors.EngbertDetector import EngbertDetector
        from peyes.GazeDetectors.NHDetector import NHDetector
        from peyes.GazeDetectors.REMoDNaVDetector import REMoDNaVDetector
        return [IVTDetector(), IDTDetector(), EngbertDetector(), NHDetector(), REMoDNaVDetector()]
