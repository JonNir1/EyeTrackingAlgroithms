import numpy as np
import plotly.graph_objects as go

import Config.constants as cnst
import Config.experiment_config as cnfg


def add_scarfplot(fig: go.Figure,
                  t: np.ndarray,
                  events: np.ndarray,
                  ymin: float,
                  ymax: float,
                  **colorbar_kwargs) -> go.Figure:
    """ Adds a scarfplot to the figure. """
    assert len(t) == len(events)
    _borders, centers, colormap = _discrete_colormap({e for e in cnst.EVENT_LABELS},
                                                     {k: v[cnst.COLOR] for k, v in cnfg.EVENT_MAPPING.items()})
    scarfplot = go.Heatmap(
        x=t,
        y=[ymin, ymax],
        z=[events],
        zmin=np.nanmin([e.value for e in cnst.EVENT_LABELS]),
        zmax=np.nanmax([e.value for e in cnst.EVENT_LABELS]),
        colorscale=colormap,
        colorbar=dict(
            len=colorbar_kwargs.get("len", 0.75),
            thickness=colorbar_kwargs.get("thickness", 25),
            tickvals=centers,
            ticktext=[e.name for e in cnst.EVENT_LABELS],
        ),
        showscale=colorbar_kwargs.get("showscale", True),
    )
    fig.add_trace(scarfplot)
    return fig


def _discrete_colormap(s: set, colors: dict):
    borders, centers = _calculate_bins(s)
    colormap = []
    for i, e in enumerate(s):
        colormap.extend([(borders[i], colors[e]), (borders[i+1], colors[e])])
    return borders, centers, colormap


def _calculate_bins(s: set):
    borders = np.arange(len(s) + 1)
    normalized_borders = borders / np.max(borders)
    centers = ((borders[1:] + borders[:-1]) / 2).tolist()
    return normalized_borders, centers
