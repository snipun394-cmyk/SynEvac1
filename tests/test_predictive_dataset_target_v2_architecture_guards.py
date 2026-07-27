import ast
import unittest
from pathlib import Path

from predictive_dataset import target_generator
from predictive_dataset.target_generator import CONGESTION_THRESHOLD, generate_candidate_label


# =====================================================
# Predictive Congestion Target V2 milestone, Phase 25/26 -- (1)
# architecture guards proving target_generator_v2.py stays on the
# leakage boundary's target-only side, the same way target_generator.py
# itself already is (tests/test_predictive_dataset_architecture_guards.py),
# and (2) a direct reproducibility check that Target V1 (target_
# generator.py) was NOT modified by this milestone -- this milestone's
# own Hard Rule #3 / Phase 26 "Target V1 MUST remain reproducible. Do
# not silently rewrite historical datasets."
# =====================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTIVE_DATASET_DIR = REPO_ROOT / "predictive_dataset"


def _imported_module_names(file_path: Path):

    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    names = set()

    for node in ast.walk(tree):

        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)

        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)

    return names


class TargetGeneratorV2ArchitectureGuardTests(unittest.TestCase):

    def test_target_generator_v2_never_imported_by_simulation_extractor(self):

        imports = _imported_module_names(PREDICTIVE_DATASET_DIR / "simulation_extractor.py")
        self.assertFalse(any("target_generator_v2" in name for name in imports))

    def test_target_generator_v2_never_imported_by_simulation_extractor_v2_1(self):

        imports = _imported_module_names(PREDICTIVE_DATASET_DIR / "simulation_extractor_v2_1.py")
        self.assertFalse(any("target_generator_v2" in name for name in imports))

    def test_target_generator_v2_never_imported_by_live_extractor(self):

        imports = _imported_module_names(PREDICTIVE_DATASET_DIR / "live_extractor.py")
        self.assertFalse(any("target_generator_v2" in name for name in imports))

    def test_target_generator_v2_never_imported_by_live_extractor_v2_1(self):

        imports = _imported_module_names(PREDICTIVE_DATASET_DIR / "live_extractor_v2_1.py")
        self.assertFalse(any("target_generator_v2" in name for name in imports))

    def test_target_semantics_analysis_never_imported_by_any_extractor(self):
        """The shared episode-sweep primitives target_generator_v2.py
        builds on are equally future-inspecting -- must stay out of
        every extractor, sim or live, v1 or v2_1."""

        for extractor_file in (
            "simulation_extractor.py", "simulation_extractor_v2_1.py",
            "live_extractor.py", "live_extractor_v2_1.py",
        ):
            with self.subTest(extractor_file=extractor_file):
                imports = _imported_module_names(PREDICTIVE_DATASET_DIR / extractor_file)
                self.assertFalse(any("target_semantics_analysis" in name for name in imports))

    def test_target_generator_v2_does_not_import_simulator_execution_engine(self):
        """target_generator_v2.py, like target_generator.py itself, must
        only ever READ an already-completed movement_result -- never
        depend on the simulator's own execution/scheduling internals."""

        imports = _imported_module_names(PREDICTIVE_DATASET_DIR / "target_generator_v2.py")
        self.assertFalse(any(name == "simulator.coordinator" for name in imports))


class TargetV1ReproducibilityTests(unittest.TestCase):
    """Target V1 (predictive_dataset/target_generator.py) must be
    completely untouched by this milestone -- verified by checking its
    own public API/constants are exactly what V1's own milestones
    already established and documented (docs/architecture/
    localized_predictive_ai_dataset.md, predictive_dataset_campaign_v1.md)."""

    def test_congestion_threshold_constant_unchanged(self):
        self.assertEqual(CONGESTION_THRESHOLD, 2)

    def test_generate_candidate_label_signature_and_behavior_unchanged(self):
        """A minimal, direct behavioral check -- 2 concurrent occupants
        (even a zero-duration touch) still registers as V1-congested,
        exactly like every V1/V2/V2.1/V2.2 milestone already relied on."""

        from navigation.edge import Edge
        from navigation.node import Node
        from pathfinding.route import Route
        from simulator.multi_agent_result import MultiAgentSimulationResult, OccupantTimeline, OccupantTimelineStep
        from simulator.occupant import OccupantState

        edge = Edge(id="edge-1", edge_type=Edge.DOOR, from_node="zone-a", to_node="zone-b", walking_distance=5.0)
        zone_a = Node(id="zone-a", name="zone-a", floor_id="floor-1", node_type=Node.ZONE)
        zone_b = Node(id="zone-b", name="zone-b", floor_id="floor-1", node_type=Node.ZONE)

        step1 = OccupantTimelineStep(index=0, from_node=zone_a, to_node=zone_b, edge=edge, queue_wait_time=0.0, start_time=0.0, end_time=10.0)
        step2 = OccupantTimelineStep(index=0, from_node=zone_a, to_node=zone_b, edge=edge, queue_wait_time=0.0, start_time=10.0, end_time=20.0)

        route = Route(nodes=[], edges=[edge], total_cost=0.0, total_distance=0.0)
        occ1 = OccupantTimeline(occupant_id="occ-1", route=route, steps=[step1], state=OccupantState.ARRIVED, depart_time=0.0, arrival_time=10.0)
        occ2 = OccupantTimeline(occupant_id="occ-2", route=route, steps=[step2], state=OccupantState.ARRIVED, depart_time=10.0, arrival_time=20.0)

        movement_result = MultiAgentSimulationResult(occupants={"occ-1": occ1, "occ-2": occ2}, total_evacuation_time=20.0)

        # V1's own (disclosed, target-V2-superseded) zero-duration
        # handoff behavior: observing just before the handoff, V1
        # registers a positive within-window crossing at t=10.
        label = generate_candidate_label("edge-1", movement_result, 5.0, 20.0)

        self.assertFalse(label.currently_congested)
        self.assertTrue(label.target)  # V1's own, now-disclosed-as-artifact behavior -- unchanged, not "fixed" in place

    def test_target_generator_module_exposes_no_v2_symbols(self):
        """target_generator.py must not have been edited to add V2
        concepts in place -- Target V2 lives entirely in its own,
        separate module."""

        self.assertFalse(hasattr(target_generator, "TARGET_VERSION_V2"))
        self.assertFalse(hasattr(target_generator, "MIN_PERSISTENCE_SECONDS"))
        self.assertFalse(hasattr(target_generator, "generate_candidate_label_v2"))


if __name__ == "__main__":
    unittest.main()
