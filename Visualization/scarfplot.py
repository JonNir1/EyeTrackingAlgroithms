import numpy as np
import plotly.graph_objects as go

import Config.constants as cnst
import Config.experiment_config as cnfg


def _discrete_colormap(colors: dict):
    borders = np.arange(len(colors) + 1)
    centers = (borders[1:] + borders[:-1]) / 2
    borders = borders / len(colors)  # Normalize to [0, 1]
    colormap = []
    for i, key in enumerate(sorted(colors.keys())):
        colormap.extend([(borders[i], colors[key]), (borders[i + 1], colors[key])])
    return colormap, centers


_COLORMAP, _TICKVALS = _discrete_colormap({label: value[cnst.COLOR] for label, value in cnfg.EVENT_MAPPING.items()})
_ZMIN, _ZMAX = np.nanmin([e.value for e in cnfg.EVENT_LABELS]), np.nanmax([e.value for e in cnfg.EVENT_LABELS])
_TICKTEXT = [e.name for e in cnfg.EVENT_LABELS]



def scarfplots_comparison_figure(t: np.ndarray, *events, **kwargs) -> go.Figure:
    """
    Creates a figure with multiple scarfplots stacked on top of each other.
    :param t: The time axis.
    :param events: The events to be plotted.

    :keyword scarf_size: The width (in y-axis units) of each scarfplot, defaults to 1.
    keyword title: The title of the figure, defaults to "Scarfplot Comparison".
    :keyword names: The names of the scarfplots, defaults to [0, 1, 2, ...].
    :keyword colorbar_length: The length of the colorbar (range [0, 1] where 1 is the full height of the plot), defaults to 0.75.
    :keyword colorbar_thickness: The thickness of the colorbar, defaults to 25.
    :keyword colorbar_show: Whether to show the colorbar, defaults to True.

    :return: The figure with the scarfplots.
    """
    num_scarfs = len(events)
    scarf_size = kwargs.get("scarf_size", 1)
    fig = go.Figure()
    for i, e in enumerate(events):
        ymin, ymax = 2 * i * scarf_size, (2 * i + 1) * scarf_size
        fig = add_scarfplot(fig, t, e.values, ymin, ymax, **kwargs)
    # Update layout
    names = kwargs.get("names", [str(i) for i in range(num_scarfs)])
    assert len(names) == num_scarfs
    fig.update_layout(
        title=kwargs.get("title", "Scarfplot Comparison"),
        yaxis=dict(
            range=[0, (2 * num_scarfs - 1) * scarf_size],
            tickmode='array',
            tickvals=[(2 * i + 0.5) * scarf_size for i in range(num_scarfs)],
            ticktext=names,
        ),
    )
    return fig


def add_scarfplot(fig: go.Figure,
                  t: np.ndarray,
                  events: np.ndarray,
                  ymin: float,
                  ymax: float,
                  **colorbar_kwargs) -> go.Figure:
    """ Adds a scarfplot to the figure. """
    assert len(t) == len(events)
    colorbar_length = colorbar_kwargs.get("colorbar_length", 1)
    scarfplot = go.Heatmap(
        x=t,
        y=[ymin, ymax],
        z=[events],
        zmin=_ZMIN,
        zmax=_ZMAX,
        colorscale=_COLORMAP,
        colorbar=dict(
            len=colorbar_length,
            thickness=colorbar_kwargs.get("colorbar_thickness", 25),
            tickvals=_TICKVALS * colorbar_length,
            ticktext=_TICKTEXT,
        ),
        showscale=colorbar_kwargs.get("colorbar_show", True),
    )
    fig.add_trace(scarfplot)
    return fig
