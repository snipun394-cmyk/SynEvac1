import unittest

from live_system.live_occupants_gateway import EngineLiveOccupantsGateway
from live_occupants.models import LiveOccupantsSnapshot


class _FakeManager:

    def __init__(self, occupants=(), canonical_count=None, raise_on_compute=False):
        self._occupants = tuple(occupants)
        self._canonical_count = canonical_count if canonical_count is not None else len(self._occupants)
        self._raise = raise_on_compute

    def all_occupants(self):
        if self._raise:
            raise RuntimeError("boom")
        return self._occupants

    def canonical_occupancy(self, time):

        class _Facts:
            def __init__(self, count):
                self.total_observed_count = count

        return _Facts(self._canonical_count)


class EngineLiveOccupantsGatewayTests(unittest.TestCase):

    def test_empty_manager_produces_an_empty_snapshot(self):

        gateway = EngineLiveOccupantsGateway(_FakeManager())
        snapshot = gateway.compute(1.0)

        self.assertIsInstance(snapshot, LiveOccupantsSnapshot)
        self.assertEqual(snapshot.occupants, ())
        self.assertEqual(snapshot.current_occupant_count, 0)
        self.assertEqual(snapshot.timestamp, 1.0)

    def test_occupants_tuple_is_passed_through_verbatim(self):

        gateway = EngineLiveOccupantsGateway(_FakeManager(occupants=("occ-A", "occ-B"), canonical_count=1))
        snapshot = gateway.compute(2.0)

        self.assertEqual(snapshot.occupants, ("occ-A", "occ-B"))

    def test_current_occupant_count_comes_from_canonical_occupancy_not_len_occupants(self):

        # The critical guard this milestone's own predecessor audit
        # established -- historical/TEMPORARILY_LOST entries may appear
        # in `occupants` without inflating current_occupant_count.
        gateway = EngineLiveOccupantsGateway(_FakeManager(occupants=("t1", "t2", "t3"), canonical_count=1))
        snapshot = gateway.compute(3.0)

        self.assertEqual(len(snapshot.occupants), 3)
        self.assertEqual(snapshot.current_occupant_count, 1)

    def test_a_failing_manager_never_crashes_the_cycle(self):

        gateway = EngineLiveOccupantsGateway(_FakeManager(raise_on_compute=True))

        self.assertIsNone(gateway.compute(4.0))


if __name__ == "__main__":
    unittest.main()
