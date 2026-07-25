# Per-Candidate Predictive AI Data Foundation milestone -- the data
# layer for a FUTURE localized bottleneck-prediction model
# (P(candidate becomes congested within horizon T | information
# available at time t), one row per SCENARIO x TIMESTEP x CANDIDATE
# instead of the existing SCENARIO x ONE WHOLE-BUILDING ROW datasets
# in dataset_builder/). See docs/architecture/localized_predictive_ai_
# dataset.md for the full design.
#
# NO MODEL IS TRAINED BY THIS PACKAGE. This package only builds and
# audits the dataset a future training milestone would consume.
