import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_DEFAULT_COLORSCALE = "Viridis"


def heatmap_grid(data: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Creates a grid of heatmaps showing each row of the DataFrame as a square matrix of heatmaps colored by the p-values
    of the comparisons between the columns. The number of rows and columns in the grid will depend on the number of
    unique values in each level of the index of the DataFrame.

    :param data: DataFrame with the p-values of the comparisons. The index should have 1 or 2 levels, with the first
        level corresponding to the columns of the grid and the second level corresponding to the rows of the grid.
    :param kwargs: Additional arguments to pass to the plot:
        - title: Title of the figure.
        - critical_value: Significance level for the critical value.
        - correction: Correction method for the critical value.
        - colorscale: Color scale to use in the heatmaps.
        - ignore_above_critical: If True, gives the same color to p-values above the critical value.
        - add_annotations: If True, adds stat-sig annotations to the heatmaps.
    """
    col_titles = data.index.get_level_values(0).unique().to_list()
    row_titles = [""] if data.index.nlevels == 1 else data.index.get_level_values(1).unique().to_list()
    nrows, ncols = len(row_titles), len(col_titles)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        row_titles=row_titles,
        column_titles=col_titles,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    for r, row_title in enumerate(row_titles):
        for c, col_title in enumerate(col_titles):
            indexer = col_title if data.index.nlevels == 1 else (col_title, row_title)
            p_values = data.loc[indexer, :].unstack().T
            critical_value = _calc_critical_value(p_values,
                                                  critical_value=kwargs.get("critical_value", 0.05),
                                                  correction=kwargs.get("correction", None))
            hm = _create_heatmap_trace(p_values,
                                       colorscale=kwargs.get("colorscale", _DEFAULT_COLORSCALE),
                                       max_val=critical_value if kwargs.get("ignore_above_critical", False) else 1)
            fig.add_trace(hm, row=r+1, col=c+1)
            if kwargs.get("add_annotations", True):
                scat = _create_stat_sig_annotations(p_values, critical_value)
                fig.add_trace(scat, row=r+1, col=c+1)
    fig.update_layout(
        title_text=kwargs.get("title", "P-Values Heatmap")
    )
    return fig


def _create_heatmap_trace(p_values: pd.DataFrame, colorscale: str, max_val: float = 1) -> go.Heatmap:
    assert 0 < max_val <= 1, "Max value must be between 0 and 1"
    # reorder data so that we have a lower-left triangular matrix (NaNs at the top-right)
    p_values = p_values.loc[
        p_values.isnull().sum(axis=1).sort_values(ascending=True).index,
        p_values.isnull().sum(axis=0).sort_values(ascending=True).index
    ]
    # create the heatmap trace
    hm = go.Heatmap(
        z=p_values.values,
        x=p_values.columns,
        y=p_values.index,
        zmin=0,
        zmax=max_val,
        colorscale=colorscale,
    )
    return hm


def _create_stat_sig_annotations(p_values: pd.DataFrame, critical_value: float) -> go.Scatter:
    assert 0 < critical_value <= 1, "Critical value must be between 0 and 1"
    annotations = np.full_like(p_values.values, "", dtype=object)
    annotations[(critical_value / 5 <= p_values) & (p_values < critical_value)] = "*"
    annotations[(critical_value / 50 <= p_values) & (p_values < critical_value / 5)] = "**"
    annotations[p_values < critical_value / 50] = "***"
    annotations[np.isnan(p_values)] = ""
    annotations = pd.DataFrame(annotations, index=p_values.index, columns=p_values.columns).stack()
    scat = go.Scatter(
        x=annotations.index.get_level_values(1).values,
        y=annotations.index.get_level_values(0).values,
        text=annotations.values,
        mode='text',
        textposition='middle center',
        textfont=dict(size=20, color='red'),
        showlegend=False)
    return scat


def _calc_critical_value(p_values: pd.DataFrame, critical_value: float = 0.05, correction: str = None) -> float:
    assert 0 < critical_value <= 1, "Critical value must be between 0 and 1"
    if correction is None:
        return critical_value
    correction = correction.lower().strip()
    if correction in {"bonf", "bonferroni", "holm", "holm-bonferroni"}:
        num_comparisons = p_values.size - p_values.isnull().sum().sum()
        return critical_value / num_comparisons
    raise ValueError(f"Invalid correction method: {correction}")
