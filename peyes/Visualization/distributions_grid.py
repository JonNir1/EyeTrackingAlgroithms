from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def distributions_grid(data: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Creates a grid of violin plots showing the distribution of values in each cell of the DataFrame, or a grid of bar
    plots showing the count of each value in each cell of the DataFrame.
    :param data: DataFrame with the data to plot. Each cell should contain a list of values.
    :param kwargs: Additional arguments to pass to the plot:
        - title: Title of the plot.
        - show_counts: If True, show bar plots instead of violin plots.
        - column_title_mapper: Function to map column names to titles.
        - row_title_mapper: Function to map row names to titles.
        - points: Points to show in violin plots (see go.Violin).
        - side: Side of the violin plot to show (see go.Violin).
        - pdf_min_val & pdf_max_val: Min and max values to show in the PDF of the violin plot. Must both be provided to
            apply the limits.
    """
    fig = make_subplots(
        rows=data.index.size,
        cols=data.columns.size,
        shared_xaxes=True,
        shared_yaxes=True,
        row_titles=[kwargs.get("row_title_mapper", lambda x: x)(r) for r in data.index],
        column_titles=[kwargs.get("column_title_mapper", lambda x: x)(c) for c in data.columns]
    )
    kwargs["points"] = "all"
    kwargs["orientation"] = "v"
    fig = _add_traces(fig, data, **kwargs)
    fig.update_layout(
        title_text=kwargs.get("title", "Distributions"),
        showlegend=False
    )
    return fig


def multi_distributions_grid(multi_data: Dict[str, pd.DataFrame], **kwargs) -> go.Figure:
    dfs = list(multi_data.values())
    assert all(df.shape == dfs[0].shape for df in dfs), "All DataFrames must have the same shape."
    assert all(df.index.equals(dfs[0].index) for df in dfs), "All DataFrames must have the same index."
    assert all(df.columns.equals(dfs[0].columns) for df in dfs), "All DataFrames must have the same columns."

    index, columns = dfs[0].index, dfs[0].columns
    fig = make_subplots(
        rows=index.size,
        cols=columns.size,
        shared_xaxes=True,
        shared_yaxes=True,
        row_titles=[kwargs.get("row_title_mapper", lambda x: x)(r) for r in index],
        column_titles=[kwargs.get("column_title_mapper", lambda x: x)(c) for c in columns]
    )
    colormap = px.colors.qualitative.Plotly
    kwargs["points"] = False
    kwargs["orientation"] = "h"
    for i, (label, df) in enumerate(multi_data.items()):
        fig = _add_traces(fig, df, label=label, color=colormap[i], **kwargs)
    fig.update_traces(width=len(multi_data))
    fig.update_layout(
        title_text=kwargs.get("title", "Distributions"),
        showlegend=True,
    )
    return fig


def _add_traces(fig: go.Figure, data: pd.DataFrame, **kwargs):
    for row_num, row_name in enumerate(data.index):
        for col_num, col_name in enumerate(data.columns):
            cell = data.loc[row_name, col_name]
            label = kwargs.get("label", "")
            if kwargs.get("show_counts", False):
                trace = go.Bar(x=cell.index, y=cell.values, name=label)
            else:
                trace_params = dict(showlegend=False,
                                    name=label,
                                    line_color=kwargs.get("color", "blue"),
                                    points=kwargs.get("points", "all"),
                                    side=kwargs.get("side", "positive"))
                trace = go.Violin(y=cell, **trace_params) if kwargs.get("orientation", "v") == "v" else go.Violin(x=cell, **trace_params)
                if kwargs.get("pdf_min_val", None) and kwargs.get("pdf_max_val", None):
                    trace.update(span=[kwargs["pdf_min_val"], kwargs["pdf_max_val"]], spanmode="manual")
            fig.add_trace(
                row=row_num + 1,
                col=col_num + 1,
                trace=trace,
            )
    return fig
