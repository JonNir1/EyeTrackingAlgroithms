import numpy as np
import pandas as pd
import plotly.express as px

import peyes.Config.constants as cnst


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

