import numpy as np
import pandas as pd

import Config.constants as cnst
from GazeDetectors.EngbertDetector import EngbertDetector

import Analysis.helpers as hlp
import Analysis.figures as figs

DATASET = "Lund2013"
LAMBDA_STR = "λ"
COL_MAPPER = lambda col: col[col.index(LAMBDA_STR):col.index(",")].replace("'", "") if LAMBDA_STR in col else col
DETECTORS = [EngbertDetector(lambdaa=lmda) for lmda in np.arange(7)]

# %%
samples_df, events_df, detector_results_df, matches, comparison_columns = hlp.preprocess_dataset(DATASET,
                                                                                                 column_mapper=COL_MAPPER,
                                                                                                 verbose=True)

# %%
# Velocity-Threshold Distribution
thresholds = pd.concat([detector_results_df[f"{LAMBDA_STR}:0"].map(lambda cell: cell['thresh_Vx']),
                        detector_results_df[f"{LAMBDA_STR}:0"].map(lambda cell: cell['thresh_Vy'])],
                       axis=1, keys=["Vx", "Vy"])
agg_thresholds = hlp.group_and_aggregate(thresholds, group_by=cnst.STIMULUS)
threshold_distribution_fig = figs.distributions_grid(agg_thresholds,
                                                     title="Velocity-Threshold Distribution")
threshold_distribution_fig.show()


# TODO: copy from notebook to here and use helpers.py instead of comparisons.py

