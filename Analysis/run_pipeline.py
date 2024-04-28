import os

import Config.experiment_config as cnfg
from Analysis.PreProcessor import PreProcessor
import Analysis.helpers as hlp
import Analysis.figures as figs

pipelines = {
    "Detector_Comparison": {
        "column_mapper": lambda col: col[:col.index("ector")] if "ector" in col else col
    },
}


def run_pipeline(
        dataset_name: str,
        pipeline_name: str,
        reference_rater: str,
        verbose=False,
        **kwargs
):
    results = PreProcessor.load_or_run(
        dataset_name,
        pipeline_name,
        verbose=verbose,
        **kwargs
    )
    _event_figures(dataset_name, pipeline_name, reference_rater, results)
    _fixation_figures(dataset_name, pipeline_name, reference_rater, results)
    _saccade_figures(dataset_name, pipeline_name, reference_rater, results)
    return results


def _event_figures(
        dataset_name: str, pipeline_name: str, reference_rater: str, pre_processing_results: tuple
):
    figures_dir = os.path.join(cnfg.OUTPUT_DIR, dataset_name, pipeline_name, "All_Events")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    samples, events, _, matches, sample_metrics, event_features, match_ratios, matched_features = pre_processing_results

    # Per-Trial Scarfplots
    scarfplot_dir = os.path.join(figures_dir, "scarfplots")
    if not os.path.exists(scarfplot_dir):
        os.makedirs(scarfplot_dir)
    _ = figs.create_comparison_scarfplots(samples, scarfplot_dir)

    # Sample Metric Distributions Per-Stimulus Type
    sample_metrics_dir = os.path.join(figures_dir, "sample_metrics")
    if not os.path.exists(sample_metrics_dir):
        os.makedirs(sample_metrics_dir)
    rater_detector_pairs = [pair for pair in hlp.extract_rater_detector_pairs(samples) if pair[0] == reference_rater]
    _ = figs.create_sample_metric_distributions(
        sample_metrics, dataset_name, sample_metrics_dir, rater_detector_pairs
    )

    # Event Feature Distributions Per-Stimulus Type
    event_features_dir = os.path.join(figures_dir, "event_features")
    if not os.path.exists(event_features_dir):
        os.makedirs(event_features_dir)
    _ = figs.create_event_feature_distributions(
        event_features, dataset_name, event_features_dir, columns=None
    )

    # Event Matching Feature Distributions Per-Stimulus Type
    matched_events_features_dir = os.path.join(figures_dir, "matched_event_features")
    if not os.path.exists(matched_events_features_dir):
        os.makedirs(matched_events_features_dir)
    rater_detector_pairs = [pair for pair in hlp.extract_rater_detector_pairs(events) if pair[0] == reference_rater]
    _matched_event_feature_figures = figs.create_matched_event_feature_distributions(
        matched_features, dataset_name, matched_events_features_dir, rater_detector_pairs
    )
    _matching_ratio_fig = figs.create_matching_ratio_distributions(
        match_ratios, dataset_name, matched_events_features_dir, rater_detector_pairs
    )


def _fixation_figures(
        dataset_name: str, pipeline_name: str, reference_rater: str, pre_processing_results: tuple
):
    figures_dir = os.path.join(cnfg.OUTPUT_DIR, dataset_name, pipeline_name, "Fixations")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    _, events, _, matches, _, _, _, _ = pre_processing_results

    # Filter Out Non-Fixation Events
    fixations = events.map(lambda cell: [event for event in cell if event.event_label == cnfg.EVENT_LABELS.FIXATION])
    fixation_matches = {scheme: df.map(
        lambda cell: {k: v for k, v in cell.items() if
                      k.event_label == cnfg.EVENT_LABELS.FIXATION} if cell is not None else None
    ) for scheme, df in matches.items()}

    del events, matches

    # Calculate Fixation-Specific Features
    fixation_features = PreProcessor.calculate_event_features(
        fixations,
        feature_names={"Duration", "Peak Velocity"},
        verbose=True,
    )
    fixation_match_ratios = PreProcessor.calculate_match_ratios(
        fixations,
        fixation_matches,
        verbose=True,
    )
    fixation_matched_features = PreProcessor.calculate_matched_event_features(
        fixation_matches,
        verbose=True,
    )
    fixation_matched_features["CoM Distance"] = {scheme: fixation_matches[scheme].map(
        lambda cell: [k.center_distance(v) for k, v in cell.items()] if cell is not None else None
    ) for scheme in fixation_matches.keys()}
    fixation_matched_features["Dispersion Ratio"] = {scheme: fixation_matches[scheme].map(
        lambda cell: [k.dispersion / v.dispersion for k, v in cell.items()] if cell is not None else None
    ) for scheme in fixation_matches.keys()}

    # Fixation Feature Distributions Per-Stimulus Type
    fixations_features_dir = os.path.join(figures_dir, "fixation_features")
    if not os.path.exists(fixations_features_dir):
        os.makedirs(fixations_features_dir)
    _fixation_feature_figures = figs.create_event_feature_distributions(
        fixation_features, dataset_name, fixations_features_dir, columns=None
    )

    # Fixation Matching Feature Distributions Per-Stimulus Type
    matched_fixations_features_dir = os.path.join(figures_dir, "matched_fixation_features")
    if not os.path.exists(matched_fixations_features_dir):
        os.makedirs(matched_fixations_features_dir)
    rater_detector_pairs = [pair for pair in hlp.extract_rater_detector_pairs(fixations) if pair[0] == reference_rater]
    _matched_fixation_feature_figures = figs.create_matched_event_feature_distributions(
        fixation_matched_features, dataset_name, matched_fixations_features_dir, rater_detector_pairs
    )
    _matching_ratio_fig = figs.create_matching_ratio_distributions(
        fixation_match_ratios, dataset_name, matched_fixations_features_dir, rater_detector_pairs
    )


def _saccade_figures(
        dataset_name: str, pipeline_name: str, reference_rater: str, pre_processing_results: tuple
):
    figures_dir = os.path.join(cnfg.OUTPUT_DIR, dataset_name, pipeline_name, "Saccades")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    _, events, _, matches, _, _, _, _ = pre_processing_results

    # Filter Out Non-Saccade Events
    saccades = events.map(lambda cell: [event for event in cell if event.event_label == cnfg.EVENT_LABELS.SACCADE])
    saccade_matches = {scheme: df.map(
        lambda cell: {k: v for k, v in cell.items() if
                      k.event_label == cnfg.EVENT_LABELS.SACCADE} if cell is not None else None
    ) for scheme, df in matches.items()}

    del events, matches

    # Calculate Saccade-Specific Features
    saccade_features = PreProcessor.calculate_event_features(
        saccades,
        feature_names=PreProcessor.EVENT_FEATURES - {"Count", "Micro-Saccade Ratio"},
        verbose=True,
    )
    saccade_match_ratios = PreProcessor.calculate_match_ratios(
        saccades,
        saccade_matches,
        verbose=True,
    )
    saccade_matched_features = PreProcessor.calculate_matched_event_features(
        saccade_matches,
        verbose=True,
    )

    # Saccade Feature Distributions Per-Stimulus Type
    saccade_features_dir = os.path.join(figures_dir, "saccade_features")
    if not os.path.exists(saccade_features_dir):
        os.makedirs(saccade_features_dir)
    _event_feature_figures = figs.create_event_feature_distributions(
        saccade_features, dataset_name, saccade_features_dir, columns=None
    )

    # Saccade Matching Feature Distributions Per-Stimulus Type
    matched_saccades_features_dir = os.path.join(figures_dir, "matched_saccade_features")
    if not os.path.exists(matched_saccades_features_dir):
        os.makedirs(matched_saccades_features_dir)
    rater_detector_pairs = [pair for pair in hlp.extract_rater_detector_pairs(saccades) if pair[0] == reference_rater]
    _matched_saccade_feature_figures = figs.create_matched_event_feature_distributions(
        saccade_matched_features, dataset_name, matched_saccades_features_dir, rater_detector_pairs
    )
    _matching_ratio_fig = figs.create_matching_ratio_distributions(
        saccade_match_ratios, dataset_name, matched_saccades_features_dir, rater_detector_pairs
    )
