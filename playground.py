from DataSetLoaders.DataSetFactory import DataSetFactory
from GazeDetectors.EngbertDetector import EngbertDetector


irf = DataSetFactory.load("IRF")
eng = EngbertDetector()

samples, events, detector_results = DataSetFactory.detect(irf, raters=["RZ"], detectors=[eng])

