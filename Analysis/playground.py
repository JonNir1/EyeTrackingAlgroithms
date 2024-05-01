from Analysis.Pipelines.DetectorComparisonPipeline import DetectorComparisonPipeline
from Analysis.Pipelines.EngbertLambdasPipeline import EngbertLambdasPipeline
# from Analysis.Pipelines.MultiIterationsPipeline import MultiIterationsPipeline

DATASET_NAME = "Lund2013"

# %%
# DetectorComparisonPipeline

p = DetectorComparisonPipeline(dataset_name=DATASET_NAME, reference_rater="RA")
results = p.run(verbose=True, create_figures=True)

# %%
# EngbertLambdasPipeline

p = EngbertLambdasPipeline(dataset_name=DATASET_NAME, reference_rater="RA", lambdas=range(1, 7))
results = p.run(verbose=True, create_figures=True)

