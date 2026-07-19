from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import training_dataset


# =====================================================
# Phase 1 -- reading. ai_training is a pure consumer of whatever the
# Training Dataset Toolkit (training_dataset/) already exported to a
# completed Campaign Studio output directory: this module never
# generates scenarios, reruns simulations, or writes anything back
# under campaign_dir. Every value below is read verbatim from
# training_dataset.load_campaign()'s own SimulationSample/
# CampaignDataset -- no CSV/JSON parsing is reimplemented here.
# =====================================================


# The three scenario_features.csv columns that identify a row rather
# than describe it -- never a predictive feature. Mirrors
# training_dataset.manifest._NON_FEATURE_COLUMNS (kept as an
# independent constant rather than imported, since ai_training must
# never depend on training_dataset's export-manifest internals, only
# on its public load_campaign() loading contract).
NON_FEATURE_SCENARIO_COLUMNS = ("scenario_id", "definition_id", "seed")


@dataclass(frozen=True)
class ScenarioRecord:

    # One accepted scenario's full set of already-computed artifacts,
    # unpacked from training_dataset.SimulationSample -- ai_training's
    # own copy of that shape (rather than importing SimulationSample
    # directly) so every ai_training module depends on one local type,
    # not on training_dataset's internal dataclass staying stable.

    scenario_id: str
    features: Dict[str, Any]
    outcome: Dict[str, Any]
    zone_results: Tuple[Dict[str, Any], ...]
    timeline: Tuple[Dict[str, Any], ...]
    ground_truth: Optional[Dict[str, Any]]
    decision_policy: Optional[Dict[str, Any]]


class TrainingDataset:

    # A read-only view over one already-exported campaign directory.
    # Every accessor below returns a *new* list of *copied* dicts --
    # callers (preprocessing/models) are free to mutate what they get
    # back without risking corruption of the underlying export.

    def __init__(
        self, campaign_dir: str, records: List[ScenarioRecord], errors: Tuple[Any, ...] = (),
    ):

        self.campaign_dir = campaign_dir
        self.records = records
        self.errors = tuple(errors)

    # =====================================================

    @property
    def scenario_ids(self) -> List[str]:

        return [record.scenario_id for record in self.records]

    def __len__(self) -> int:

        return len(self.records)

    def __iter__(self):

        return iter(self.records)

    def __getitem__(self, index):

        return self.records[index]

    # =====================================================
    # Per-source row accessors -- one dict per scenario for the
    # scenario-level sources (features/outcome/ground_truth), flattened
    # across every scenario for the multi-row sources (zone_results/
    # timeline). Every list is aligned with self.records by index/
    # order, so e.g. scenario_feature_rows()[i] and outcome_rows()[i]
    # always describe the same scenario.
    # =====================================================

    def scenario_feature_rows(self) -> List[Dict[str, Any]]:

        return [dict(record.features) for record in self.records]

    def outcome_rows(self) -> List[Dict[str, Any]]:

        return [dict(record.outcome) for record in self.records]

    def zone_result_rows(self) -> List[Dict[str, Any]]:

        return [dict(row) for record in self.records for row in record.zone_results]

    def timeline_rows(self) -> List[Dict[str, Any]]:

        return [dict(row) for record in self.records for row in record.timeline]

    def ground_truth_rows(self) -> List[Optional[Dict[str, Any]]]:

        # Parallel to scenario_feature_rows()/outcome_rows() (one entry
        # per scenario, in the same order) rather than filtered down to
        # only the scenarios that have one -- callers that zip() this
        # against scenario_feature_rows() need the None placeholders to
        # keep both lists the same length/order.

        return [
            dict(record.ground_truth) if record.ground_truth is not None else None
            for record in self.records
        ]

    def decision_policy_rows(self) -> List[Optional[Dict[str, Any]]]:

        return [
            dict(record.decision_policy) if record.decision_policy is not None else None
            for record in self.records
        ]


# =====================================================


def load_campaign_dataset(
    campaign_dir,
    *,
    strict: bool = True,
    require_timeline: bool = True,
    require_ground_truth: bool = True,
    require_decision_policy: bool = True,
) -> TrainingDataset:

    # The one place ai_training touches the simulation platform's own
    # output -- a thin, read-only wrapper over
    # training_dataset.load_campaign(). strict/require_* mirror that
    # function's own parameters exactly (same meaning, same defaults);
    # ai_training adds no additional loading behaviour of its own.

    campaign_dataset = training_dataset.load_campaign(
        campaign_dir,
        strict=strict,
        require_timeline=require_timeline,
        require_ground_truth=require_ground_truth,
        require_decision_policy=require_decision_policy,
    )

    records = [
        ScenarioRecord(
            scenario_id=sample.scenario_id,
            features=sample.scenario_features,
            outcome=sample.simulation_outcome,
            zone_results=tuple(sample.zone_results),
            timeline=tuple(sample.timeline),
            ground_truth=sample.ground_truth,
            decision_policy=sample.decision_policy,
        )
        for sample in campaign_dataset.samples
    ]

    return TrainingDataset(
        campaign_dir=campaign_dataset.campaign_dir, records=records, errors=campaign_dataset.errors,
    )
