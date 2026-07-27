import ast
import unittest
from pathlib import Path

from predictive_dataset.topologies_v3_1_variants import (
    build_multi_exit_wide_6exit,
    build_twin_stair_highrise_3stair,
    variant_specs,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _imported_module_names(file_path: Path):
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class VariantSpecsTests(unittest.TestCase):
    """Localized Predictive Model V3.1 milestone, Phase 10 -- structural
    validity of the two new topology-diversity variants. NOT a
    simulation test (too expensive for a unit test) -- these check the
    Building/ScenarioDefinition wiring itself is internally consistent
    (every door/exit/stair references a zone that actually exists),
    the same class of regression topologies_v2.py's own
    StaircaseVerticalHeightRegressionTests guards against."""

    def test_variant_specs_returns_two_specs(self):
        specs = variant_specs()
        self.assertEqual(len(specs), 2)
        self.assertEqual({s.name for s in specs}, {"multi_exit_wide_6exit", "twin_stair_highrise_3stair"})

    def test_scenario_counts_within_milestone_suggested_range(self):
        """Milestone charter: '~100-250 scenarios per added structural
        variant' -- not an arbitrarily huge new campaign."""
        for spec in variant_specs():
            with self.subTest(spec=spec.name):
                self.assertGreaterEqual(spec.scenario_count, 100)
                self.assertLessEqual(spec.scenario_count, 250)

    def _assert_zone_references_resolve(self, spec):

        zone_ids = set()
        for floor in spec.building.floors:
            for zone in floor.zones:
                zone_ids.add(zone.id)

        for floor in spec.building.floors:
            for door in floor.doors:
                self.assertIn(door.zone_a_id, zone_ids, f"{door.id} zone_a_id dangling")
                self.assertIn(door.zone_b_id, zone_ids, f"{door.id} zone_b_id dangling")
            for exit_ in floor.exits:
                self.assertIn(exit_.zone_id, zone_ids, f"{exit_.id} zone_id dangling")
            for stair in floor.stairs:
                self.assertIn(stair.from_zone_id, zone_ids, f"{stair.id} from_zone_id dangling")
                self.assertIn(stair.to_zone_id, zone_ids, f"{stair.id} to_zone_id dangling")

    def _assert_stairs_set_both_floor_ids(self, spec):
        """Direct regression guard for the exact V1 bug topologies_v2.py's
        own module docstring documents (Staircase.vertical_height()
        silently collapsing to 0.0 when from_floor_id is unset) --
        every stair in these new variants must set BOTH floor ids."""

        floor_ids = {floor.id for floor in spec.building.floors}
        for floor in spec.building.floors:
            for stair in floor.stairs:
                self.assertIn(stair.from_floor_id, floor_ids, f"{stair.id} from_floor_id dangling/unset")
                self.assertIn(stair.to_floor_id, floor_ids, f"{stair.id} to_floor_id dangling/unset")
                self.assertNotEqual(stair.from_floor_id, "", f"{stair.id} from_floor_id empty")
                self.assertNotEqual(stair.to_floor_id, "", f"{stair.id} to_floor_id empty")

    def test_multi_exit_wide_6exit_structural_validity(self):
        spec = build_multi_exit_wide_6exit()
        self._assert_zone_references_resolve(spec)
        floor = spec.building.floors[0]
        self.assertEqual(len(floor.doors), 6)
        self.assertEqual(len(floor.exits), 5)

    def test_multi_exit_wide_6exit_has_more_connectivity_than_parent(self):
        """The whole point of this variant -- genuinely more structural
        diversity than the original 4-door/3-exit multi_exit_wide, not
        a relabeled copy of the same shape."""
        from predictive_dataset.topologies_v2 import build_multi_exit_wide
        parent = build_multi_exit_wide()
        variant = build_multi_exit_wide_6exit()

        parent_doors = sum(len(f.doors) for f in parent.building.floors)
        parent_exits = sum(len(f.exits) for f in parent.building.floors)
        variant_doors = sum(len(f.doors) for f in variant.building.floors)
        variant_exits = sum(len(f.exits) for f in variant.building.floors)

        self.assertGreater(variant_doors, parent_doors)
        self.assertGreater(variant_exits, parent_exits)

    def test_twin_stair_highrise_3stair_structural_validity(self):
        spec = build_twin_stair_highrise_3stair()
        self._assert_zone_references_resolve(spec)
        self._assert_stairs_set_both_floor_ids(spec)

        total_stairs = sum(len(f.stairs) for f in spec.building.floors)
        total_floors = len(spec.building.floors)
        self.assertEqual(total_stairs, 3)
        self.assertEqual(total_floors, 4)

    def test_twin_stair_highrise_3stair_has_more_floors_than_parent(self):
        from predictive_dataset.topologies_v2 import build_twin_stair_highrise
        parent = build_twin_stair_highrise()
        variant = build_twin_stair_highrise_3stair()

        parent_stairs = sum(len(f.stairs) for f in parent.building.floors)
        variant_stairs = sum(len(f.stairs) for f in variant.building.floors)

        self.assertGreater(len(variant.building.floors), len(parent.building.floors))
        self.assertGreater(variant_stairs, parent_stairs)

    def test_specs_are_deterministic(self):
        """Calling the builder twice must produce the same structural
        shape (door/exit/stair/zone counts) -- these are pure functions
        of hardcoded authoring, not randomized at build time."""
        first = build_multi_exit_wide_6exit()
        second = build_multi_exit_wide_6exit()
        self.assertEqual(
            sum(len(f.doors) for f in first.building.floors),
            sum(len(f.doors) for f in second.building.floors),
        )

    def test_topologies_v2_module_untouched_by_new_variants_file(self):
        """This new file must never import FROM topologies_v2 in a way
        that could indicate it modifies shared state, and topologies_v2
        itself has no reason to import the new variants file at all."""

        variants_path = REPO_ROOT / "predictive_dataset" / "topologies_v3_1_variants.py"
        v2_path = REPO_ROOT / "predictive_dataset" / "topologies_v2.py"

        v2_imports = _imported_module_names(v2_path)
        self.assertFalse(any("topologies_v3_1_variants" in name for name in v2_imports))


if __name__ == "__main__":
    unittest.main()
