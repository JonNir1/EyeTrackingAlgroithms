import os
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import Config.constants as cnst
import Analysis.helpers as hlp
from Visualization import scarfplot
from Visualization import distributions_grid as dg

FEATURES_WITHIN_EVENT = {
    "Start Time", "End Time", "Duration", "Amplitude", "Azimuth", "Peak Velocity",
}
FEATURES_BETWEEN_EVENTS = {
    "L2 Timing Difference", "IoU", "Overlap Time", "CoM Distance", "Dispersion Ratio",
}


def save_figure(fig: go.Figure, output_dir: str, filename: str):
    fig.write_html(os.path.join(output_dir, f"{filename}.html"))
    fig.write_json(os.path.join(output_dir, f"{filename}.json"))


def create_comparison_scarfplots(
        samples_df: pd.DataFrame,
        output_dir: str
) -> Dict[str, go.Figure]:
    figures = {}
    for i, idx in enumerate(samples_df.index):
        num_samples = samples_df.loc[idx].map(len).max()  # Number of samples in the longest detected sequence
        t = np.arange(num_samples)
        detected_labels = samples_df.loc[idx]
        is_all_nan = detected_labels.apply(lambda arr: pd.isna(arr).all())
        detected_labels = detected_labels.loc[~is_all_nan]
        fig = scarfplot.scarfplots_comparison_figure(t, *detected_labels.to_list(), names=detected_labels.index)
        save_figure(fig, output_dir, f"{idx}")
        figures[str(idx)] = fig
    return figures


def create_sample_metric_distributions(
        metrics: dict,
        dataset_name: str,
        output_dir: str,
        columns: list = None
) -> Dict[str, go.Figure]:
    figures = {}
    for metric in metrics.keys():
        if metric in {"Count", "Counts"}:
            fig = _create_counts_grid(metrics[metric], dataset_name, cnst.SAMPLES)
        elif metric == "Confusion Matrix":
            # skip confusion matrix as it is not a distribution
            # TODO: add confusion matrix visualization
            continue
        else:
            data = metrics[metric][columns] if columns is not None else metrics[metric]
            grouped = hlp.group_and_aggregate(data, group_by=cnst.STIMULUS)
            fig = dg.distributions_grid(
                data=grouped,
                title=f"{dataset_name.upper()}:\t\tSample-Level {metric.title()}",
                pdf_min_val=0 if "Transition Matrix" not in metric else None,
                pdf_max_val=1 if "Transition Matrix" not in metric else None,
                column_title_mapper=lambda col: f"{col[0]}→{col[1]}"
            )
        save_figure(fig, output_dir, f"{metric}")
        figures[metric] = fig
    return figures


def create_event_feature_distributions(
        features: dict,
        dataset_name: str,
        output_dir: str,
        columns: list = None
) -> Dict[str, go.Figure]:
    figures = {}
    for feature in features.keys():
        data = features[feature][columns] if columns is not None else features[feature]
        if feature in {"Count", "Counts"}:
            fig = _create_counts_grid(data, dataset_name, cnst.EVENTS)
        else:
            grouped = hlp.group_and_aggregate(data, group_by=cnst.STIMULUS)
            fig = dg.distributions_grid(
                data=grouped,
                title=f"{dataset_name.upper()}:\t\t{feature.title()} Distribution",
                show_counts=False,
                pdf_min_val=0,
                pdf_max_val=1,
            )
        save_figure(fig, output_dir, f"{feature}")
        figures[feature] = fig
    return figures


def create_matched_event_feature_distributions(
        matched_features: dict,
        dataset_name: str,
        output_dir: str,
        columns: list = None
) -> Dict[str, go.Figure]:
    figures = {}
    for feature in matched_features.keys():
        multi_data = {}
        for scheme, df in matched_features[feature].items():
            scheme_df = df[columns] if columns else df
            if feature in FEATURES_WITHIN_EVENT:   # calculate difference of within-event features
                scheme_df = scheme_df.map(
                    lambda cell: [v[0] - v[1] for v in cell if not np.any(pd.isna(v))] if np.all(pd.notnull(cell)) else np.nan
                )
                title = f"{dataset_name.upper()}:\t\tDifference of Matched {feature.title()} Distribution"
            elif feature in FEATURES_BETWEEN_EVENTS:
                title = f"{dataset_name.upper()}:\t\t{feature.title()} Distribution"
            else:
                raise ValueError(f"Feature {feature} is not supported for matched event distributions.")
            multi_data[scheme] = hlp.group_and_aggregate(scheme_df, group_by=cnst.STIMULUS)
        fig = dg.multi_distributions_grid(
            multi_data=multi_data,
            title=title,
            column_title_mapper=lambda col: f"{col[0]}→{col[1]}",
            pdf_min_val=0 if feature in {"IoU", "Overlap Time"} else None,
            pdf_max_val=1 if feature in {"IoU", "Overlap Time"} else None,
        )
        save_figure(fig, output_dir, f"{feature}")
        figures[feature] = fig
    return figures


def create_matching_ratio_distributions(
        ratios: dict,
        dataset_name: str,
        output_dir: str,
        columns: list = None
) -> go.Figure:
    for scheme in ratios.keys():
        data = ratios[scheme][columns] if columns is not None else ratios[scheme]
        ratios[scheme] = hlp.group_and_aggregate(data, group_by=cnst.STIMULUS)
    fig = dg.multi_distributions_grid(
        multi_data=ratios,
        title=f"{dataset_name.upper()}:\t\tMatch Ratio Distribution",
        column_title_mapper=lambda col: f"{col[0]}→{col[1]}",
        pdf_min_val=0,
        pdf_max_val=1,
    )
    save_figure(fig, output_dir, f"Match Ratios")
    return fig


def _create_counts_grid(
        counts: pd.DataFrame,
        dataset_name: str,
        count_of: str,
) -> go.Figure:
    grouped = counts.groupby(level=cnst.STIMULUS).agg(list).map(sum)
    if len(grouped.index) > 1:
        # there is more than one group, so add a row for "all" groups
        group_all = pd.Series(counts.sum(axis=0), index=counts.columns, name="all")
        grouped = pd.concat([grouped.T, group_all], axis=1).T  # add "all" row
    title = f"{dataset_name.upper()}:\t\t{count_of.title()} Label Counts"
    fig = dg.distributions_grid(
        data=grouped,
        title=title,
        show_counts=True,
    )
    return fig
