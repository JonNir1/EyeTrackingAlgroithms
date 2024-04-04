import time
from typing import List, Union, Optional

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

import Config.constants as cnst
from GazeEvents.BaseEvent import BaseEvent
from GazeDetectors.EngbertDetector import EngbertDetector
from DataSetLoaders.DataSetFactory import DataSetFactory

pio.renderers.default = "browser"

###################################

DATASET_NAME = "Lund2013"
RATERS = ["MN", "RA"]
DETECTORS = [EngbertDetector(lambdaa=lmda) for lmda in np.arange(0.5, 6.1, 0.5)]

start = time.time()

lund_dataset = DataSetFactory.load(DATASET_NAME)
lund_samples, lund_events = DataSetFactory.process(lund_dataset, RATERS, DETECTORS)

end = time.time()
print(f"Elapsed time: {end - start:.2f} seconds")
del start, end


###################################


def count_event_labels(events: List[Union[cnst.EVENT_LABELS, BaseEvent]]) -> pd.Series:
    labels = pd.Series([e.event_label if isinstance(e, BaseEvent) else e for e in events])
    counts = labels.value_counts()
    if counts.empty:
        return pd.Series({l: 0 for l in cnst.EVENT_LABELS})
    if len(counts) == len(cnst.EVENT_LABELS):
        return counts
    missing_labels = pd.Series({l: 0 for l in cnst.EVENT_LABELS if l not in counts.index})
    return pd.concat([counts, missing_labels]).sort_index()


def group_and_aggregate(data: pd.DataFrame, group_by: Optional[str] = None) -> pd.DataFrame:
    """ Group the data by the given criteria and aggregate the values in each group. """
    # create a Series of all measure values in each column
    group_all = pd.Series(event_counts.sum(axis=0), index=event_counts.columns, name="all")
    if group_by is None:
        return pd.DataFrame(group_all).T
    # create a list of values per group & column, and add a row for "all" group
    grouped_vals = data.groupby(level=group_by).agg(list).map(sum)
    grouped_vals = pd.concat([grouped_vals.T, group_all], axis=1).T  # add "all" row
    return grouped_vals


sample_counts = lund_samples.map(count_event_labels)
event_counts = lund_events.map(count_event_labels)

###################################

event_counts_grouped = group_and_aggregate(event_counts, group_by=cnst.STIMULUS)

fig = make_subplots(rows=event_counts_grouped.index.size,
                    cols=event_counts_grouped.columns.size // 2,
                    row_titles=event_counts_grouped.index.tolist(),
                    column_titles=[col[col.index("λ"): col.index(",")] if ":" in col else col
                                   for col in event_counts_grouped.columns[::2]],
                    shared_xaxes=True, shared_yaxes=False)
for row_num, row_name in enumerate(event_counts_grouped.index):
    for col_num, col_name in enumerate(event_counts_grouped.columns):
        if col_num % 2 == 0:
            continue
        group = event_counts_grouped.loc[row_name, col_name]
        new_trace = go.Bar(x=group.index, y=group.values, name=col_name)
        fig.add_trace(
            row=row_num + 1,
            col=col_num // 2 + 1,
            trace=new_trace,
        )
fig.update_layout(title_text="Event Label Counts",
                  showlegend=True)
# for ax in fig['layout']:
#     if ax.startswith("xaxis"):
#         fig.layout[ax].update(
#             dict(
#                 tickmode='array',
#                 tickvals=[l.value for l in cnst.EVENT_LABELS],
#                 ticktext=[l.name for l in cnst.EVENT_LABELS],
#             )
#         )

fig.show()
