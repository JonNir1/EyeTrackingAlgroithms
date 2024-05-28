# %%
# DetectorComparisonPipeline
from peyes.Analysis.Pipelines.DetectorComparisonPipeline import DetectorComparisonPipeline

lund_results = DetectorComparisonPipeline(dataset_name="Lund2013", reference_rater="RA").run(verbose=True)
irf_results = DetectorComparisonPipeline(dataset_name="IRF", reference_rater="RZ").run(verbose=True)

# # %%
# # EngbertLambdasPipeline
from peyes.Analysis.Pipelines.EngbertLambdasPipeline import EngbertLambdasPipeline

lund_results = EngbertLambdasPipeline(dataset_name="Lund2013", reference_rater="RA").run(lambdas=range(1, 7), verbose=True)
irf_results = EngbertLambdasPipeline(dataset_name="IRF", reference_rater="RZ").run(lambdas=range(1, 7), verbose=True)

# # %%
# # MultiIterationsPipeline - Engbert

from peyes.GazeDetectors.EngbertDetector import EngbertDetector
from peyes.Analysis.Pipelines.MultiIterationsPipeline import MultiIterationsPipeline

engbert_lund_results = MultiIterationsPipeline(dataset_name="Lund2013", detector=EngbertDetector()).run(verbose=True)
engbert_irf_results = MultiIterationsPipeline(dataset_name="IRF", detector=EngbertDetector()).run(verbose=True)

# # %%
# # MultiIterationsPipeline - NH

from peyes.GazeDetectors.NHDetector import NHDetector
from peyes.Analysis.Pipelines.MultiIterationsPipeline import MultiIterationsPipeline

nh_lund_results = MultiIterationsPipeline(dataset_name="Lund2013", detector=NHDetector()).run(verbose=True)
nh_irf_results = MultiIterationsPipeline(dataset_name="IRF", detector=NHDetector()).run(verbose=True)
