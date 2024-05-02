# from Analysis.Pipelines.MultiIterationsPipeline import MultiIterationsPipeline

# %%
# DetectorComparisonPipeline
from Analysis.Pipelines.DetectorComparisonPipeline import DetectorComparisonPipeline

p = DetectorComparisonPipeline(dataset_name="Lund2013", reference_rater="RA")
results = p.run(verbose=True, create_figures=True)

# %%
# EngbertLambdasPipeline
from Analysis.Pipelines.EngbertLambdasPipeline import EngbertLambdasPipeline

p = EngbertLambdasPipeline(dataset_name="Lund2013", reference_rater="RA", lambdas=range(1, 7))
results = p.run(verbose=True, create_figures=True)

