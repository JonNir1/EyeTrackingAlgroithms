import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import Config.constants as cnst

_HEATMAP_COLORMAP = "HOT"


def similarity_heatmap(data: pd.DataFrame, title: str, similarity_measure: str) -> go.Figure:
    fig = px.imshow(data.T,
                    title=title,
                    color_continuous_scale=_HEATMAP_COLORMAP,
                    labels=dict(x=cnst.TRIAL.capitalize(), y="Detectors", color=similarity_measure),
                    x=data.index.get_level_values(cnst.TRIAL),
                    aspect="auto")
    fig.update_layout(
        xaxis=dict(side="top"),
        yaxis=dict(tickmode="array",
                   tickvals=np.arange(len(data.columns)),
                   ticktext=data.columns.tolist()
                   ),
        coloraxis=dict(colorbar_x=1.,
                       colorbar_title_side='right',
                       ),
    )
    return fig


def distributions_grid(data: pd.DataFrame, **kwargs) -> go.Figure:
    """
    Creates a grid of violin plots showing the distribution of values in each cell of the DataFrame, or a grid of bar
    plots showing the count of each value in each cell of the DataFrame.
    :param data: DataFrame with the data to plot. Each cell should contain a list of values.
    :param kwargs: Additional arguments to pass to the plot:
        - title: Title of the plot.
        - column_title_mapper: Function to map column names to titles.
        - row_title_mapper: Function to map row names to titles.
        - max_bins: Maximum number of bins to use in histograms (see go.Histogram).
        - points: Points to show in violin plots (see go.Violin).
        - side: Side of the violin plot to show (see go.Violin).
    """
    fig = make_subplots(
        rows=data.index.size,
        cols=data.columns.size,
        shared_xaxes=True,
        shared_yaxes=True,
        row_titles=[kwargs.get("row_title_mapper", lambda x: x)(r) for r in data.index],
        column_titles=[kwargs.get("column_title_mapper", lambda x: x)(c) for c in data.columns]
    )
    for row_num, row_name in enumerate(data.index):
        for col_num, col_name in enumerate(data.columns):
            cell = data.loc[row_name, col_name]
            if kwargs.get("show_counts", False):
                trace = go.Bar(x=cell.index, y=cell.values, name=col_name)
            else:
                trace = go.Violin(y=cell,
                                  showlegend=False,
                                  name="",
                                  points=kwargs.get("points", "all"),
                                  side=kwargs.get("side", "positive"))
                if kwargs.get("pdf_min_val", None) and kwargs.get("pdf_max_val", None):
                    trace.update(span=[kwargs["pdf_min_val"], kwargs["pdf_max_val"]], spanmode="manual")
            fig.add_trace(
                row=row_num + 1,
                col=col_num + 1,
                trace=trace,
            )
    fig.update_layout(
        title_text=kwargs.get("title", "Distributions"),
        showlegend=False
    )
    return fig
