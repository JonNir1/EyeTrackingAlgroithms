import Config.constants as cnst
import Analysis.Detector_Comparisons.helpers as hlp

SHOW_DISTRIBUTIONS = True

# %%
lund_preprocessed = hlp.preprocess_dataset("Lund2013", verbose=True)
lund_samples, lund_events, lund_detector_results, lund_event_matches, lund_comparison_columns = lund_preprocessed
del lund_preprocessed

# %%
# Sample-by-Sample Metrics
lund_sample_metrics = hlp.calculate_sample_metrics(lund_samples,
                                                   lund_comparison_columns,
                                                   show_distributions=SHOW_DISTRIBUTIONS,
                                                   verbose=True)

# %%
# Event Features
lund_event_features = hlp.extract_features(lund_events,
                                           show_distributions=SHOW_DISTRIBUTIONS,
                                           verbose=True)
lund_fixation_features = hlp.extract_features(lund_events,
                                              ignore_events=[v for v in cnst.EVENT_LABELS if
                                                             v != cnst.EVENT_LABELS.FIXATION],
                                              show_distributions=SHOW_DISTRIBUTIONS,
                                              verbose=True)
lund_saccade_features = hlp.extract_features(lund_events,
                                             ignore_events=[v for v in cnst.EVENT_LABELS if
                                                            v != cnst.EVENT_LABELS.SACCADE],
                                             show_distributions=SHOW_DISTRIBUTIONS,
                                             verbose=True)

# %%
# Matched-Event Metrics

