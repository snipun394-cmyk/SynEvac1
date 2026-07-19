import os

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from models.building import Building
from scenario.scenario import Scenario
from simulation_runtime.result import TickResult
from simulator.multi_agent_result import MultiAgentSimulationResult

from dataset_builder.exporters import (
    export_scenario_features_csv,
    export_simulation_outcomes_csv,
    export_zone_results_csv,
)
from dataset_builder.feature_extractor import extract_scenario_features
from dataset_builder.labels import extract_simulation_outcome, extract_zone_results


@dataclass(frozen=True)
class SimulationRun:

    # Everything the Dataset Builder needs from one completed
    # simulation run -- nothing here is generated, validated, or
    # simulated by this package; it is only ever handed in by the
    # caller once a run has already finished.

    scenario: Scenario
    building: Building
    movement_result: MultiAgentSimulationResult

    # Whether the Simulation Runtime reached its own end condition
    # (SimulationRuntime.is_finished) -- passed in explicitly rather
    # than inferred, since the Dataset Builder never touches the
    # Runtime itself.
    simulation_finished: bool = True

    # Human Population & Assisted Evacuation Modeling Phase 6 --
    # additive, the same optional tick_results field dataset_builder.
    # timeline.TimelineRun already carries (this package's own sibling
    # dataclass), needed here only for Fallen_Count/Possible_Injury_Count
    # (ground_truth.human_behavior's own hazard-and-congestion-derived
    # signal -- see that module's docstring for why). Every existing
    # SimulationRun construction that predates this field is unaffected;
    # Fallen_Count/Possible_Injury_Count are simply 0 when it is omitted,
    # never fabricated.
    tick_results: Tuple[TickResult, ...] = field(default_factory=tuple)

    # Human Decision Engine Phase 6 -- additive, the same optional-and-
    # empty-by-default shape tick_results already establishes. A plain
    # list of human_decision_engine.events.DecisionEvent.to_dict()'s
    # own shape, supplied only when the caller used
    # behaviour_profile_resolver.dynamic_registrar.
    # register_population_dynamic() this run; every Help_Decision_Count/
    # Firefighter_Task_Count/... column is simply 0 when omitted, never
    # fabricated.
    decision_events: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)


class DatasetBuilder:

    # Extraction only -- see feature_extractor.py/labels.py for the
    # per-row logic. Adding a future dataset (timeline, RL episode,
    # ...) means adding one more build_*()/key here, never changing
    # the shape of the ones that already exist.

    def __init__(self, runs: Sequence[SimulationRun]):

        self.runs = list(runs)

    # =====================================================

    def build_scenario_features(self) -> List[Dict[str, Any]]:

        return [extract_scenario_features(run) for run in self.runs]

    # =====================================================

    def build_simulation_outcomes(self) -> List[Dict[str, Any]]:

        return [extract_simulation_outcome(run) for run in self.runs]

    # =====================================================

    def build_zone_results(self) -> List[Dict[str, Any]]:

        rows = []

        for run in self.runs:
            rows.extend(extract_zone_results(run))

        return rows

    # =====================================================

    def build_all(self) -> Dict[str, List[Dict[str, Any]]]:

        return {
            "scenario_features": self.build_scenario_features(),
            "simulation_outcomes": self.build_simulation_outcomes(),
            "zone_results": self.build_zone_results(),
        }

    # =====================================================

    def export_all(self, output_dir: str) -> Dict[str, str]:

        os.makedirs(output_dir, exist_ok=True)

        datasets = self.build_all()

        paths = {
            "scenario_features": os.path.join(output_dir, "scenario_features.csv"),
            "simulation_outcomes": os.path.join(output_dir, "simulation_outcomes.csv"),
            "zone_results": os.path.join(output_dir, "zone_results.csv"),
        }

        export_scenario_features_csv(
            datasets["scenario_features"], paths["scenario_features"],
        )
        export_simulation_outcomes_csv(
            datasets["simulation_outcomes"], paths["simulation_outcomes"],
        )
        export_zone_results_csv(
            datasets["zone_results"], paths["zone_results"],
        )

        return paths
