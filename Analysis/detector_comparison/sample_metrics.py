import plotly.io as pio

from Analysis.Analyzers.SamplesAnalyzer import SamplesAnalyzer
from Visualization.distributions_grid import distributions_grid
from Visualization.p_value_heatmap import heatmap_grid

pio.renderers.default = "browser"

DATASET = "Lund2013"
COL_MAPPER = lambda col: col[:col.index("ector")] if "ector" in col else col

STAT_TEST_NAME = "Wilcoxon"
CRITICAL_VALUE = 0.05
CORRECTION = "Bonferroni"

samples, comparison_columns = SamplesAnalyzer.preprocess_dataset(DATASET, column_mapper=COL_MAPPER, verbose=True)

# %%
#############################################
# Sample Metrics
sample_metrics, sample_metric_stats = SamplesAnalyzer.analyze(samples, test_name=STAT_TEST_NAME, verbose=True)
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
sample_metric_pvalue_figures = {}
for metric in sample_metric_stats.keys():
    p_values = sample_metric_stats[metric].xs("p-value", axis=1, level=2)
    fig = heatmap_grid(
        p_values,
        title=f"{DATASET.upper()}:\t\tStatistical Comparison of Sample-Level {metric.title()}",
        critical_value=CRITICAL_VALUE,
        correction=CORRECTION,
        add_annotations=True,
        ignore_above_critical=True
    )
    sample_metric_pvalue_figures[metric] = fig
    fig.show()

del data, p_values, metric
