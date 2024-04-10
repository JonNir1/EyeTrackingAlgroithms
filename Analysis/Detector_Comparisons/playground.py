import Config.constants as cnst
import Analysis.helpers as hlp

DATASET = "Lund2013"
COL_MAPPER = lambda col: col[:col.index("ector")] if "ector" in col else col

samples, events, _, event_matches, comparison_columns = hlp.preprocess_dataset(DATASET,
                                                                               column_mapper=COL_MAPPER,
                                                                               verbose=True)

# %%
# All-Event Metrics
sample_metrics = hlp.calc_sample_metrics(samples, verbose=True)
event_features = hlp.extract_features(events, verbose=True)
event_matching_ratios = hlp.calc_event_matching_ratios(events, event_matches, verbose=True)
event_matching_feature_diffs = hlp.calc_matched_events_feature(event_matches, verbose=True)

# %%
# Fixation Metrics
fixation_features = hlp.extract_features(events,
                                         ignore_events={v for v in cnst.EVENT_LABELS if
                                                        v != cnst.EVENT_LABELS.FIXATION},
                                         verbose=VERBOSE)
fixation_matching_ratios = hlp.calc_event_matching_ratios(events,
                                                          event_matches,
                                                          ignore_events={v for v in cnst.EVENT_LABELS if
                                                                         v != cnst.EVENT_LABELS.FIXATION},
                                                          verbose=True)
fixation_matching_feature_diffs = hlp.calc_matched_events_feature(event_matches,
                                                                  ignore_events={v for v in cnst.EVENT_LABELS if
                                                                                 v != cnst.EVENT_LABELS.FIXATION},
                                                                  verbose=True)

# %%
# Saccade Metrics
saccade_features = hlp.extract_features(events,
                                        ignore_events={v for v in cnst.EVENT_LABELS if
                                                       v != cnst.EVENT_LABELS.SACCADE},
                                        verbose=VERBOSE)
saccade_matching_ratios = hlp.calc_event_matching_ratios(events,
                                                         event_matches,
                                                         ignore_events={v for v in cnst.EVENT_LABELS if
                                                                        v != cnst.EVENT_LABELS.SACCADE},
                                                         verbose=True)
saccade_matching_feature_diffs = hlp.calc_matched_events_feature(event_matches,
                                                                 ignore_events={v for v in cnst.EVENT_LABELS if
                                                                                v != cnst.EVENT_LABELS.SACCADE},
                                                                 verbose=True)
