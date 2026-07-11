import unittest

from dataclasses import FrozenInstanceError

from occupancy.observation import OccupancyObservation
from occupancy.provider import ManualOccupancyProvider, OccupancyProvider
from occupancy.snapshot import OccupancySnapshot


class OccupancyObservationTests(unittest.TestCase):

    def test_defaults_mean_no_reading(self):

        observation = OccupancyObservation()

        self.assertIsNone(observation.occupant_count)
        self.assertIsNone(observation.confidence)

    def test_is_frozen(self):

        observation = OccupancyObservation()

        with self.assertRaises(FrozenInstanceError):
            observation.occupant_count = 5


class OccupancySnapshotTests(unittest.TestCase):

    def test_default_snapshot_has_no_observations(self):

        snapshot = OccupancySnapshot()

        self.assertEqual(len(snapshot.observations), 0)
        self.assertEqual(snapshot.timestamp, 0.0)

    def test_missing_node_returns_a_default_observation_not_none(self):

        snapshot = OccupancySnapshot(
            observations={"z1": OccupancyObservation(occupant_count=3.0)},
        )

        self.assertEqual(snapshot.observation_at("z1"), OccupancyObservation(occupant_count=3.0))
        self.assertEqual(snapshot.observation_at("z2"), OccupancyObservation())

    def test_is_frozen(self):

        snapshot = OccupancySnapshot()

        with self.assertRaises(FrozenInstanceError):
            snapshot.timestamp = 5.0

    def test_observations_mapping_is_read_only(self):

        snapshot = OccupancySnapshot(observations={"z1": OccupancyObservation()})

        with self.assertRaises(TypeError):
            snapshot.observations["z2"] = OccupancyObservation()

    def test_snapshot_ids_are_unique_by_default(self):

        first = OccupancySnapshot()
        second = OccupancySnapshot()

        self.assertNotEqual(first.snapshot_id, second.snapshot_id)


class OccupancyProviderTests(unittest.TestCase):

    def test_base_interface_is_not_implemented(self):

        provider = OccupancyProvider()

        with self.assertRaises(NotImplementedError):
            provider.snapshot_at(0.0)


class ManualOccupancyProviderTests(unittest.TestCase):

    def test_defaults_to_an_empty_snapshot(self):

        provider = ManualOccupancyProvider()
        snapshot = provider.snapshot_at(100.0)

        self.assertEqual(len(snapshot.observations), 0)

    def test_returns_the_same_snapshot_regardless_of_requested_time(self):

        fixed = OccupancySnapshot(observations={"z1": OccupancyObservation(occupant_count=4.0)})
        provider = ManualOccupancyProvider(fixed)

        self.assertIs(provider.snapshot_at(0.0), fixed)
        self.assertIs(provider.snapshot_at(999.0), fixed)


class OccupancyPackageIndependenceTests(unittest.TestCase):

    def test_occupancy_package_never_imports_hazard_navigation_or_downstream_layers(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "occupancy"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(navigation|pathfinding|simulator|behavior|models|designer|"
            r"hazard|hazard_evolution|sensors)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"{path.name} imports a Hazard/Navigation/downstream layer directly -- "
                f"Occupancy must stay completely independent of HazardSnapshot and "
                f"everything it depends on",
            )


if __name__ == "__main__":
    unittest.main()
