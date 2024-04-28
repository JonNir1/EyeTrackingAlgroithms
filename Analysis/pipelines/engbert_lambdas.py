import os

import numpy as np
import pandas as pd

import Config.constants as cnst
import Config.experiment_config as cnfg
from GazeDetectors.EngbertDetector import EngbertDetector
from Analysis.run_pipeline import run_pipeline
import Analysis.helpers as hlp
from Visualization import distributions_grid as dg

LAMBDA_STR = "λ"
DATASET_NAME = "Lund2013"
PIPELINE_NAME = "Engbert_Lambdas"
REFERENCE_RATER = "RA"

results = run_pipeline(
    DATASET_NAME,
    PIPELINE_NAME,
    REFERENCE_RATER,
    verbose=True,
    column_mapper=lambda col: col[col.index(LAMBDA_STR):col.index(",")].replace("'", "").replace("\"", "") if LAMBDA_STR in col else col,
    detectors=[EngbertDetector(lambdaa=lmda) for lmda in np.arange(1, 7)],
)

# %%
# Velocity-Threshold Distribution
_, _, detector_results, _, _, _, _, _ = results
thresholds = pd.concat([detector_results[f"{LAMBDA_STR}:1"].map(lambda cell: cell['thresh_Vx']),
                        detector_results[f"{LAMBDA_STR}:1"].map(lambda cell: cell['thresh_Vy'])],
                       axis=1, keys=["Vx", "Vy"])
agg_thresholds = hlp.group_and_aggregate(thresholds, cnst.STIMULUS)
threshold_fig = dg.distributions_grid(
    agg_thresholds,
    title=f"{DATASET_NAME.upper()}:\t\tVelocity-Threshold Distribution"
)
threshold_fig.write_html(
    os.path.join(cnfg.OUTPUT_DIR, DATASET_NAME, PIPELINE_NAME, "Velocity-Threshold Distribution.html")
)
