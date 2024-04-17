import plotly.io as pio

from Analysis.detector_comparison.DetectorComparisonAnalyzer import DetectorComparisonAnalyzer
from Visualization.distributions_grid import distributions_grid

pio.renderers.default = "browser"

DATASET = "Lund2013"
COL_MAPPER = lambda col: col[:col.index("ector")] if "ector" in col else col

samples, events, _, _, comparison_columns = DetectorComparisonAnalyzer.preprocess_dataset(DATASET,
                                                                                          column_mapper=COL_MAPPER,
                                                                                          verbose=True)

# %%
#############################################
# Sample Metrics
all_event_metrics = DetectorComparisonAnalyzer.analyze(events, None, samples, verbose=True)
sample_metrics = all_event_metrics[DetectorComparisonAnalyzer.SAMPLE_METRICS_STR]
print(f"Available sample metrics: {list(sample_metrics.keys())}")

# show feature distributions
sample_metric_figures = {}
for metric in sample_metrics.keys():
    data = sample_metrics[metric]
    fig = distributions_grid(
        data=data[comparison_columns],
        title=f"{DATASET.upper()}:\t\tSample-Level {metric.title()}",
        pdf_min_val=0 if "Transition Matrix" not in metric else None,
        pdf_max_val=1 if "Transition Matrix" not in metric else None,
        column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
    )
    sample_metric_figures[metric] = fig
    fig.show()

# show p-value heatmaps
# TODO

del data
