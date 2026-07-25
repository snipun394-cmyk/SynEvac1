import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from navigation.edge import Edge
from navigation.node import Node

from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates


def make_building():

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
            Zone(id="zone-2", name="Office", x=20.0, y=0.0, width=6.0, height=6.0),
        ],
        doors=[Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[Exit(id="exit-1", zone_id="zone-1"), Exit(id="exit-2", zone_id="zone-2")],
        stairs=[Staircase(id="stair-1", from_zone_id="zone-1", to_zone_id="zone-3", to_floor_id="floor-2")],
    )
    floor2 = Floor(name="Upper", id="floor-2", zones=[Zone(id="zone-3", name="Attic")])

    return Building(name="Test Building", id="building-1", floors=[floor1, floor2])


class CandidateIdentityTests(unittest.TestCase):

    def test_candidate_ids_are_exactly_the_underlying_object_ids(self):

        building = make_building()
        candidates = enumerate_candidates(building)

        candidate_ids = {candidate.candidate_id for candidate in candidates}

        self.assertEqual(candidate_ids, {"door-1", "exit-1", "exit-2", "stair-1"})

    def test_candidate_types_match_edge_vocabulary(self):

        building = make_building()
        candidates = {c.candidate_id: c for c in enumerate_candidates(building)}

        self.assertEqual(candidates["door-1"].candidate_type, Edge.DOOR)
        self.assertEqual(candidates["exit-1"].candidate_type, Edge.EXIT)
        self.assertEqual(candidates["stair-1"].candidate_type, Edge.STAIR)

    def test_exit_zone_ids_excludes_the_shared_outside_node(self):

        building = make_building()
        candidates = {c.candidate_id: c for c in enumerate_candidates(building)}

        self.assertEqual(candidates["exit-1"].zone_ids, ("zone-1",))
        self.assertNotIn(Node.OUTSIDE_NODE_ID, candidates["exit-1"].zone_ids)

    def test_door_zone_ids_includes_both_adjacent_zones(self):

        building = make_building()
        candidates = {c.candidate_id: c for c in enumerate_candidates(building)}

        self.assertEqual(set(candidates["door-1"].zone_ids), {"zone-1", "zone-2"})

    def test_identity_is_stable_across_repeated_calls_not_row_order(self):

        building = make_building()

        first = enumerate_candidates(building)
        second = enumerate_candidates(building)

        self.assertEqual(first, second)

    def test_edges_by_candidate_id_keys_match_candidate_ids(self):

        building = make_building()
        candidates = enumerate_candidates(building)
        edges = edges_by_candidate_id(building)

        self.assertEqual(set(edges.keys()), {c.candidate_id for c in candidates})

        for candidate in candidates:
            self.assertEqual(edges[candidate.candidate_id].id, candidate.candidate_id)

    def test_building_with_no_candidates_returns_empty_tuple(self):

        floor = Floor(name="Empty", id="floor-1", zones=[Zone(id="zone-1", name="Room")])
        building = Building(name="Empty Building", id="building-2", floors=[floor])

        self.assertEqual(enumerate_candidates(building), ())


if __name__ == "__main__":
    unittest.main()
