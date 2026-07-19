# =====================================================
# Human Decision Engine -- moves civilian/firefighter behavior from
# scenario-authored actions (Human Population & Assisted Evacuation
# Modeling's own assisting_occupant_id/rescue_target_occupant_id
# fields) to dynamic, in-simulation decision making.
#
# This package contains no DecisionStrategy/RouteChoiceStrategy of its
# own -- it is consumed BY new strategies in behavior_library
# (dynamic_human_strategies.py, dynamic_firefighter_strategies.py),
# which in turn delegate the actual movement mechanics (speed
# multipliers, route synchronization) to the *existing*,
# unmodified behavior_library.assistance_strategies/
# firefighter_strategies. The Behavior Library itself is not replaced;
# this package only supplies a different, dynamic way of deciding
# which traits those existing strategies see.
#
# Every decision here is deterministic, rule-based (category/hazard/
# proximity thresholds), and reproducible from a DecisionContext alone
# -- never hardcoded per-occupant, never unseeded randomness.
# =====================================================
