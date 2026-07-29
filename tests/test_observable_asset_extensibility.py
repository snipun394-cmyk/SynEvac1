import unittest
from dataclasses import dataclass
from typing import Dict, Tuple

from models.building import Building
from models.floor import Floor
from models.staircase import Staircase, StairObservableRegion

from camera_calibration.asset_lookup import ObservableAssetKind, build_assets_by_floor, covered_asset_ids, locate_asset
from camera_calibration.stair_lookup import STAIR_ASSET_KIND

from observable_assets.facts import compute_asset_occupancy_snapshot
from observable_assets.models import ObservationStatus


# =====================================================
# Observable Asset Perception Framework milestone, Phase 8 -- proves,
# WITHOUT implementing real Door perception (no models.door.Door change,
# no live_camera_pipeline/crowd_intelligence/building_state change),
# that a second observable asset KIND can be registered using nothing
# but the existing framework's own public interface:
#
#   1. a tiny object satisfying the SAME contract Staircase already does
#      (`.id`, `.observable_region_for_floor(floor_id)`,
#      `.contains_world_point(floor_id, world_position)`)
#   2. one ObservableAssetKind registration record
#   3. passing it alongside STAIR_ASSET_KIND to the exact same
#      build_assets_by_floor()/locate_asset()/covered_asset_ids()/
#      compute_asset_occupancy_snapshot() calls Stair already uses
#
# Zero changes to camera_calibration.asset_lookup, observable_assets, or
# any other production module were needed to make this pass -- this
# test is the demonstration, not a step toward a real Door feature (see
# docs/architecture/observable_asset_perception.md Sec "Future
# extensibility").
# =====================================================


@dataclass(frozen=True)
class _FakeObservableRegion:

    # A minimal stand-in for models.staircase.StairObservableRegion --
    # this test does not need the real class, only something with a
    # `.contains(x, y)` method, proving the framework never assumes
    # StairObservableRegion specifically, only the CONTRACT.

    center_x: float
    center_y: float
    half_size: float = 1.0

    def contains(self, x: float, y: float) -> bool:
        return abs(x - self.center_x) <= self.half_size and abs(y - self.center_y) <= self.half_size


@dataclass(frozen=True)
class _FakeDoorLikeAsset:

    # Deliberately NOT models.door.Door -- a test-only object satisfying
    # exactly the same observable-asset contract Staircase already does,
    # to prove the framework depends on the CONTRACT, not on Staircase
    # itself. Single-floor (unlike Staircase's two-floor shape) --
    # exactly the kind of shape difference a real Door/Exit/Assembly
    # Point/etc. would also have, and the framework does not care.

    id: str
    floor_id: str
    region: _FakeObservableRegion

    def observable_region_for_floor(self, floor_id: str):
        return self.region if floor_id == self.floor_id else None

    def contains_world_point(self, floor_id: str, world_position) -> bool:
        region = self.observable_region_for_floor(floor_id)
        if region is None:
            return False
        x, y = world_position
        return region.contains(x, y)


def _build_fake_doors_by_floor(fake_doors) -> "Dict[str, Tuple[_FakeDoorLikeAsset, ...]]":

    # The Door-equivalent of camera_calibration.stair_lookup.
    # build_stairs_by_floor() -- a per-kind adapter a real Door
    # integration would write, reading wherever Door geometry actually
    # lives (floor.doors, in a real integration). Here it just returns a
    # fixed, hand-built mapping -- this test is about the FRAMEWORK's
    # extensibility, not about building a real Door-to-floor index.

    by_floor: Dict[str, list] = {}

    for door in fake_doors:
        by_floor.setdefault(door.floor_id, []).append(door)

    return {floor_id: tuple(doors) for floor_id, doors in by_floor.items()}


class RegisterASecondAssetKindTests(unittest.TestCase):

    def _make_building_with_stair_and_fake_door(self):

        building = Building(name="Extensibility Test Building")
        floor = Floor(name="Floor 1", display_order=0)
        building.add_floor(floor)

        stair = Staircase(name="Stair S1", from_floor_id=floor.id, to_floor_id=floor.id)
        stair.from_observable_region = StairObservableRegion(center_x=0.0, center_y=0.0, width=2.0, depth=2.0)
        floor.add_stair(stair)

        fake_door = _FakeDoorLikeAsset(
            id="FAKE-DOOR-1", floor_id=floor.id, region=_FakeObservableRegion(center_x=10.0, center_y=10.0),
        )

        return building, floor, stair, fake_door

    def test_1_a_second_kind_registers_with_one_new_dataclass_instance(self):

        building, floor, stair, fake_door = self._make_building_with_stair_and_fake_door()

        fake_door_kind = ObservableAssetKind(
            asset_type="Door", build_by_floor=lambda _building: _build_fake_doors_by_floor([fake_door]),
        )

        # Registering the second kind is exactly this: one more entry in
        # the tuple passed to build_assets_by_floor() -- no change to
        # ObservableAssetKind, build_assets_by_floor, locate_asset, or
        # covered_asset_ids themselves.
        registered_kinds = (STAIR_ASSET_KIND, fake_door_kind)

        assets_by_floor = build_assets_by_floor(building, registered_kinds)

        types_present = {asset_type for asset_type, _asset in assets_by_floor[floor.id]}
        self.assertEqual(types_present, {"Stair", "Door"})

    def test_2_locate_asset_distinguishes_the_two_types_by_position(self):

        building, floor, stair, fake_door = self._make_building_with_stair_and_fake_door()

        fake_door_kind = ObservableAssetKind(
            asset_type="Door", build_by_floor=lambda _building: _build_fake_doors_by_floor([fake_door]),
        )
        assets_by_floor = build_assets_by_floor(building, (STAIR_ASSET_KIND, fake_door_kind))

        stair_match = locate_asset(assets_by_floor[floor.id], floor.id, (0.0, 0.0))
        door_match = locate_asset(assets_by_floor[floor.id], floor.id, (10.0, 10.0))
        neither_match = locate_asset(assets_by_floor[floor.id], floor.id, (500.0, 500.0))

        self.assertEqual((stair_match.asset_id, stair_match.asset_type), (stair.id, "Stair"))
        self.assertEqual((door_match.asset_id, door_match.asset_type), ("FAKE-DOOR-1", "Door"))
        self.assertIsNone(neither_match.asset_id)

    def test_3_ambiguity_across_two_different_types_is_never_arbitrarily_resolved(self):

        # A Door and a Stair whose observable regions genuinely overlap
        # -- matching is type-agnostic, so this is exactly as unresolved
        # as two overlapping Stairs would be (see tests/test_stair_
        # observable_geometry.py's own equivalent single-type case).
        building, floor, stair, _unused = self._make_building_with_stair_and_fake_door()

        overlapping_door = _FakeDoorLikeAsset(
            id="FAKE-DOOR-OVERLAP", floor_id=floor.id, region=_FakeObservableRegion(center_x=0.0, center_y=0.0),
        )
        fake_door_kind = ObservableAssetKind(
            asset_type="Door", build_by_floor=lambda _building: _build_fake_doors_by_floor([overlapping_door]),
        )
        assets_by_floor = build_assets_by_floor(building, (STAIR_ASSET_KIND, fake_door_kind))

        result = locate_asset(assets_by_floor[floor.id], floor.id, (0.0, 0.0))

        self.assertIsNone(result.asset_id)
        self.assertIsNone(result.asset_type)
        self.assertTrue(result.ambiguous)

    def test_4_covered_asset_ids_treats_the_new_type_identically_no_special_casing(self):

        building, floor, stair, fake_door = self._make_building_with_stair_and_fake_door()

        fake_door_kind = ObservableAssetKind(
            asset_type="Door", build_by_floor=lambda _building: _build_fake_doors_by_floor([fake_door]),
        )
        assets_by_floor = build_assets_by_floor(building, (STAIR_ASSET_KIND, fake_door_kind))

        covered_with_camera = covered_asset_ids(assets_by_floor, frozenset({floor.id}))
        covered_without_camera = covered_asset_ids(assets_by_floor, frozenset())

        self.assertEqual(covered_with_camera, {stair.id, "FAKE-DOOR-1"})
        self.assertEqual(covered_without_camera, frozenset())

    def test_5_one_snapshot_holds_both_asset_types_via_observations_of_type(self):

        building, floor, stair, fake_door = self._make_building_with_stair_and_fake_door()

        fake_door_kind = ObservableAssetKind(
            asset_type="Door", build_by_floor=lambda _building: _build_fake_doors_by_floor([fake_door]),
        )
        assets_by_floor = build_assets_by_floor(building, (STAIR_ASSET_KIND, fake_door_kind))
        covered = covered_asset_ids(assets_by_floor, frozenset({floor.id}))

        snapshot = compute_asset_occupancy_snapshot(
            asset_ids_by_type={"Stair": [stair.id], "Door": ["FAKE-DOOR-1"]},
            occupant_ids_by_asset={stair.id: ("OCC-1",), "FAKE-DOOR-1": ("OCC-2", "OCC-3")},
            covered_asset_ids=covered, timestamp=0.0,
        )

        stair_observations = snapshot.observations_of_type("Stair")
        door_observations = snapshot.observations_of_type("Door")

        self.assertEqual(len(stair_observations), 1)
        self.assertEqual(stair_observations[0].occupant_count, 1)
        self.assertEqual(stair_observations[0].status, ObservationStatus.OBSERVED)

        self.assertEqual(len(door_observations), 1)
        self.assertEqual(door_observations[0].occupant_count, 2)
        self.assertEqual(door_observations[0].status, ObservationStatus.OBSERVED)

    def test_6_stair_only_registry_is_unaffected_by_the_extensibility_proof(self):

        # Sanity/regression guard: DEFAULT_OBSERVABLE_ASSET_KINDS (what
        # every real caller actually uses today) still contains only
        # Stair -- this test file proves EXTENSIBILITY, it does not
        # itself register Door into production behavior anywhere.
        from camera_calibration.stair_lookup import DEFAULT_OBSERVABLE_ASSET_KINDS

        self.assertEqual(len(DEFAULT_OBSERVABLE_ASSET_KINDS), 1)
        self.assertEqual(DEFAULT_OBSERVABLE_ASSET_KINDS[0].asset_type, "Stair")


if __name__ == "__main__":
    unittest.main()
