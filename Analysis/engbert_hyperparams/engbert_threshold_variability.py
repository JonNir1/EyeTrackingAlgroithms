import pandas as pd
import plotly.io as pio

import Config.constants as cnst
from GazeDetectors.EngbertDetector import EngbertDetector
from Visualization.distributions_grid import *

from Analysis.BaseAnalyzer import BaseAnalyzer

pio.renderers.default = "browser"

DATASET = "Lund2013"
LAMBDA_STR = "λ"
COL_MAPPER = lambda col: col[col.index(LAMBDA_STR):col.index(",")].replace("'", "") if LAMBDA_STR in col else col
DETECTORS = [EngbertDetector(lambdaa=1)]

# %%
_, _, detector_results_df, _, _ = BaseAnalyzer.preprocess_dataset(DATASET,
                                                                  detectors=DETECTORS,
                                                                  column_mapper=COL_MAPPER,
                                                                  verbose=True)

# %%
# Velocity-Threshold Distribution
thresholds = pd.concat([detector_results_df[f"{LAMBDA_STR}:0"].map(lambda cell: cell['thresh_Vx']),
                        detector_results_df[f"{LAMBDA_STR}:0"].map(lambda cell: cell['thresh_Vy'])],
                       axis=1, keys=["Vx", "Vy"])
agg_thresholds = BaseAnalyzer.group_and_aggregate(thresholds)
threshold_distribution_fig = distributions_grid(agg_thresholds,
                                                title=f"{DATASET.upper()}:\t\tVelocity-Threshold Distribution")
threshold_distribution_fig.show()
