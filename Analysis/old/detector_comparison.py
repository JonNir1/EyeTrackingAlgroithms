from Analysis.old.run_pipeline import run_pipeline

DATASET_NAME = "Lund2013"
PIPELINE_NAME = "Detector_Comparison"
REFERENCE_RATER = "RA"

results = run_pipeline(
    DATASET_NAME,
    PIPELINE_NAME,
    REFERENCE_RATER,
    verbose=True,
    column_mapper=lambda col: col[:col.index("ector")] if "ector" in col else col
)
