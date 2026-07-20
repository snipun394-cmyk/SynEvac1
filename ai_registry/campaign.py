from typing import Tuple

from models.building import Building

from ai_training.dataset import ScenarioRecord, TrainingDataset

import ai_features as af


# =====================================================
# Simulation Ground Truth -> live-compatible TrainingDataset. Reuses
# ai_features.simulation_extractor.extract_canonical_training_row() --
# the SAME BuildingState-routed extractor a genuine live deployment's
# BuildingState-side path also calls (ai_features.building_state_
# extractor.extract_canonical_features()) -- so this is not a second,
# independently-defined "training feature" implementation (Phase 2's
# central requirement, restated for this milestone's own dataset
# generation).
#
# TARGETS remain simulation Ground Truth (Phase 6's own explicit rule)
# -- only INPUT features are constrained to the canonical schema; every
# value in `outcome`/`ground_truth` on the returned records is
# untouched, read straight from the legacy TrainingDataset exactly as
# ai_training's own existing models already read it.
# =====================================================


def build_live_compatible_dataset(legacy_dataset: TrainingDataset, building: Building) -> TrainingDataset:

    live_records = []

    for record in legacy_dataset.records:

        ignition_zone_id = record.features.get("ignition_zone")
        total_occupants = record.features.get("total_occupants") or 0

        live_row = af.extract_canonical_training_row(
            building, total_occupants=total_occupants, ignition_zone_id=ignition_zone_id,
        )

        live_records.append(ScenarioRecord(
            scenario_id=record.scenario_id,
            features=live_row,
            outcome=record.outcome,
            zone_results=record.zone_results,
            timeline=record.timeline,
            ground_truth=record.ground_truth,
            decision_policy=record.decision_policy,
        ))

    return TrainingDataset(
        campaign_dir=legacy_dataset.campaign_dir, records=live_records, errors=legacy_dataset.errors,
    )


# =====================================================


def scenario_ids_of(dataset: TrainingDataset) -> Tuple[str, ...]:

    return tuple(dataset.scenario_ids)
