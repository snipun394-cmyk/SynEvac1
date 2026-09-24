"""The approved label-free default (decision D2) for the AI-learning experiment.

SYMMETRIC_SPLIT_DEADBAND, margin 0: the existing, unmodified
`deadband_heuristic.deadband_heuristic_action` evaluated on an observation
whose hub split is REPLACED by the label-free hub total split evenly
(Q_A = Q_B = waiting_H_total / 2). Closed form: PARITY, unless the hazard
has actually started (residual H->CE1 capacity 2/6) and at least one person
is waiting at the hub, then FAVOR_B.

The ONLY input is the 26-feature admissible vector (see ADMISSIBLE_FEATURES).
The hidden Q_A_at_H / Q_B_at_H values are never an input: the 27-D vector
the rule needs is rebuilt here from admissible features alone. Used for:
continuation after every candidate, selector / abstention / OOD fallback,
the closed-loop default, and pre-decision state generation. The hidden-label
GLOBAL_DEADBAND heuristic is a reference arm only (P2b) and is not imported
here.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)
import audit_pre_training_ai_dataset as _paths  # noqa: E402,F401  (sets up DATA_ROOT import paths)

from deadband_heuristic import deadband_heuristic_action  # noqa: E402
from observation import OBSERVATION_INDEX  # noqa: E402

POLICY_ID = "SYMMETRIC_SPLIT_DEADBAND_margin0"
MARGIN = 0.0

ADMISSIBLE_FEATURES = [
    "waiting_R1", "waiting_R2", "waiting_C1", "waiting_R3", "waiting_R4", "waiting_C2", "waiting_CE1", "waiting_CE2",
    "in_transit_R1_C1", "in_transit_R2_C1", "in_transit_C1_H_1tick", "in_transit_C1_H_2tick",
    "in_transit_R3_C2", "in_transit_R4_C2", "in_transit_C2_H_1tick", "in_transit_C2_H_2tick",
    "in_transit_H_CE1_1tick", "in_transit_H_CE1_2tick", "in_transit_H_CE2_1tick", "in_transit_H_CE2_2tick",
    "in_transit_CE1_EXITA", "in_transit_CE2_EXITB",
    "residual_cap_H_CE1_normalized", "normalized_time", "normalized_remaining", "waiting_H_total",
]
assert len(ADMISSIBLE_FEATURES) == 26
_HIDDEN = ("Q_A_at_H", "Q_B_at_H")


def rebuild_rule_input(x26) -> np.ndarray:
    """27-D input for the unmodified rule, built ONLY from admissible features."""
    x = np.asarray(x26, dtype=np.float64).reshape(-1)
    if x.shape[0] != 26:
        raise ValueError(f"label-free default takes exactly the 26 admissible features, got {x.shape}")
    vals = dict(zip(ADMISSIBLE_FEATURES, x))
    obs = np.zeros(len(OBSERVATION_INDEX), dtype=np.float64)
    for i, name in enumerate(OBSERVATION_INDEX):
        if name in _HIDDEN:
            obs[i] = vals["waiting_H_total"] / 2.0
        else:
            obs[i] = vals[name]
    return obs


def label_free_default_action(x26) -> int:
    return int(deadband_heuristic_action(rebuild_rule_input(x26), MARGIN))


def policy_hash() -> str:
    """Hash of this module plus the unmodified rule it calls."""
    h = hashlib.sha256()
    h.update(POLICY_ID.encode())
    for p in (os.path.abspath(__file__),
              os.path.join(_paths.DATA_ROOT, "deadband_heuristic.py"),
              os.path.join(_paths.DATA_ROOT, "capacity_aware_heuristic.py")):
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()
