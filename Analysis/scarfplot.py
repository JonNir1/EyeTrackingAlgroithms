import numpy as np
import plotly.graph_objects as go

import Config.constants as cnst
import Config.experiment_config as cnfg


def compare_scarfplots(t: np.ndarray, *events, **kwargs) -> go.Figure:
    """
    Creates a figure with multiple scarfplots stacked on top of each other.
    :param t: The time axis.
    :param events: The events to be plotted.

    :keyword names: The names of the scarfplots, defaults to [0, 1, 2, ...].
    :keyword scarf_size: The width (in y-axis units) of each scarfplot, defaults to 1.
    :keyword colorbar_length: The length of the colorbar (range [0, 1] where 1 is the full height of the plot), defaults to 0.75.
    :keyword colorbar_thickness: The thickness of the colorbar, defaults to 25.
    :keyword colorbar_show: Whether to show the colorbar, defaults to True.

    :return: The figure with the scarfplots.
    """
    num_scarfs = len(events)
    names = kwargs.get("names", [str(i) for i in range(num_scarfs)])
    scarf_size = kwargs.get("scarf_size", 1)
    assert len(names) == num_scarfs
    fig = go.Figure()
    for i, e in enumerate(events):
        ymin, ymax = 2 * i * scarf_size, (2 * i + 1) * scarf_size
        fig = add_scarfplot(fig, t, e, ymin, ymax, **kwargs)
    return fig


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
            len=colorbar_kwargs.get("colorbar_length", 0.75),
            thickness=colorbar_kwargs.get("colorbar_thickness", 25),
            tickvals=centers,
            ticktext=[e.name for e in cnst.EVENT_LABELS],
        ),
        showscale=colorbar_kwargs.get("colorbar_show", True),
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
