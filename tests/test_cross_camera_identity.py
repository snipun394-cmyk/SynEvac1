import unittest

from tracking.track_state import TrackState
from tracking.tracked_human import TrackedHuman

from cross_camera_identity.identity_registry import IdentityRegistry
from cross_camera_identity.matching import RuleBasedCrossCameraMatcher
from cross_camera_identity.resolver import RuleBasedCrossCameraIdentityResolver
from cross_camera_identity.topology import CameraTopology, build_topology_from_navigation_graph
from cross_camera_identity.transition_model import TransitionModel


# =====================================================
# Cross-Camera Identity Resolution (ReID Framework) milestone, Phase 9
# -- deterministic, offline unit tests. No randomness anywhere in this
# file: every global_id assignment is fully reproducible.
# =====================================================


def th(track_id="T1", camera_id="CAM-A", confidence=0.9, state=TrackState.TRACKED, age=5, frames_seen=5, frames_missing=0, last_timestamp=0.0):

    return TrackedHuman(
        track_id=track_id, camera_id=camera_id, bounding_box=(0.0, 0.0, 10.0, 20.0),
        confidence=confidence, state=state, age=age, frames_seen=frames_seen,
        frames_missing=frames_missing, last_timestamp=last_timestamp,
    )


def make_resolver(topology=None, timeout_seconds=30.0, **matcher_kwargs):

    topology = topology or CameraTopology()
    registry = IdentityRegistry()
    transition_model = TransitionModel(topology, timeout_seconds=timeout_seconds)
    matcher = RuleBasedCrossCameraMatcher(**matcher_kwargs)

    return RuleBasedCrossCameraIdentityResolver(
        topology=topology, registry=registry, transition_model=transition_model, matcher=matcher,
    )


class SingleCameraTests(unittest.TestCase):

    def test_1_single_camera_same_track_keeps_same_global_id(self):

        resolver = make_resolver()

        first = resolver.resolve("CAM-A", 0.0, [th(last_timestamp=0.0)], {})
        second = resolver.resolve("CAM-A", 1.0, [th(last_timestamp=1.0)], {})

        self.assertEqual(first[0].global_id, second[0].global_id)
        self.assertTrue(first[0].global_id.startswith("OCC-"))


class TwoAndThreeCameraTests(unittest.TestCase):

    def _adjacent_topology(self):

        topology = CameraTopology()
        topology.add_transition("CAM-A", "CAM-B", min_transition_time=1.0, max_transition_time=10.0)
        topology.add_transition("CAM-B", "CAM-C", min_transition_time=1.0, max_transition_time=10.0)
        return topology

    def test_2_two_cameras_transition_preserves_global_id(self):

        topology = self._adjacent_topology()
        resolver = make_resolver(topology)

        resolver.resolve("CAM-A", 0.0, [th(track_id="T1", camera_id="CAM-A", last_timestamp=0.0)], {})
        global_id_a = resolver.resolve("CAM-A", 0.0, [th(track_id="T1", camera_id="CAM-A", last_timestamp=0.0)], {})[0].global_id

        resolver.resolve("CAM-A", 1.0, [th(track_id="T1", camera_id="CAM-A", state=TrackState.EXPIRED)], {})

        arrival = resolver.resolve("CAM-B", 5.0, [th(track_id="T9", camera_id="CAM-B", last_timestamp=5.0)], {})

        self.assertEqual(arrival[0].global_id, global_id_a)

    def test_3_three_cameras_chain_preserves_global_id_across_two_transitions(self):

        topology = self._adjacent_topology()
        resolver = make_resolver(topology)

        global_id = resolver.resolve("CAM-A", 0.0, [th(track_id="T1", camera_id="CAM-A", last_timestamp=0.0)], {})[0].global_id

        resolver.resolve("CAM-A", 1.0, [th(track_id="T1", camera_id="CAM-A", state=TrackState.EXPIRED)], {})
        arrival_b = resolver.resolve("CAM-B", 5.0, [th(track_id="T2", camera_id="CAM-B", last_timestamp=5.0)], {})
        self.assertEqual(arrival_b[0].global_id, global_id)

        resolver.resolve("CAM-B", 6.0, [th(track_id="T2", camera_id="CAM-B", state=TrackState.EXPIRED)], {})
        arrival_c = resolver.resolve("CAM-C", 10.0, [th(track_id="T3", camera_id="CAM-C", last_timestamp=10.0)], {})
        self.assertEqual(arrival_c[0].global_id, global_id)


class TransitionPlausibilityTests(unittest.TestCase):

    def test_4_plausible_camera_transition_within_window_matches(self):

        topology = CameraTopology()
        topology.add_transition("CAM-A", "CAM-B", min_transition_time=2.0, max_transition_time=8.0)
        resolver = make_resolver(topology)

        global_id = resolver.resolve("CAM-A", 0.0, [th(track_id="T1", camera_id="CAM-A", last_timestamp=0.0)], {})[0].global_id
        resolver.resolve("CAM-A", 0.5, [th(track_id="T1", camera_id="CAM-A", state=TrackState.EXPIRED)], {})

        arrival = resolver.resolve("CAM-B", 5.0, [th(track_id="T2", camera_id="CAM-B", last_timestamp=5.0)], {})

        self.assertEqual(arrival[0].global_id, global_id)

    def test_5_impossible_transition_mints_a_new_identity(self):

        # Topology explicitly knows both cameras but registers NO edge
        # between them -- closed-world rejection (topology.py's own
        # documented "at least one camera known -> explicit absence"
        # rule).
        topology = CameraTopology()
        topology.add_camera("CAM-A")
        topology.add_camera("CAM-B")
        resolver = make_resolver(topology)

        original_id = resolver.resolve("CAM-A", 0.0, [th(track_id="T1", camera_id="CAM-A", last_timestamp=0.0)], {})[0].global_id
        resolver.resolve("CAM-A", 0.5, [th(track_id="T1", camera_id="CAM-A", state=TrackState.EXPIRED)], {})

        arrival = resolver.resolve("CAM-B", 1.0, [th(track_id="T2", camera_id="CAM-B", last_timestamp=1.0)], {})

        self.assertNotEqual(arrival[0].global_id, original_id)

    def test_5_transition_outside_time_window_mints_a_new_identity(self):

        topology = CameraTopology()
        topology.add_transition("CAM-A", "CAM-B", min_transition_time=1.0, max_transition_time=3.0)
        resolver = make_resolver(topology)

        original_id = resolver.resolve("CAM-A", 0.0, [th(track_id="T1", camera_id="CAM-A", last_timestamp=0.0)], {})[0].global_id
        resolver.resolve("CAM-A", 0.5, [th(track_id="T1", camera_id="CAM-A", state=TrackState.EXPIRED)], {})

        # Arrives WAY too late for this edge's own window (100s, vs. a
        # max of 3s) -- implausible even though the edge itself exists.
        arrival = resolver.resolve("CAM-B", 100.0, [th(track_id="T2", camera_id="CAM-B", last_timestamp=100.0)], {})

        self.assertNotEqual(arrival[0].global_id, original_id)


class DisappearanceTests(unittest.TestCase):

    def test_6_long_disappearance_beyond_timeout_expires_the_identity(self):

        topology = CameraTopology()
        topology.add_transition("CAM-A", "CAM-B", min_transition_time=0.0, max_transition_time=1000.0)
        resolver = make_resolver(topology, timeout_seconds=10.0)

        original_id = resolver.resolve("CAM-A", 0.0, [th(track_id="T1", camera_id="CAM-A", last_timestamp=0.0)], {})[0].global_id
        resolver.resolve("CAM-A", 0.5, [th(track_id="T1", camera_id="CAM-A", state=TrackState.EXPIRED)], {})

        # Nothing else happens for a long time -- the unbound identity
        # itself should be purged by the time anyone else asks.
        resolver.resolve("CAM-C", 50.0, [], {})  # unrelated cycle, far past the 10s timeout

        arrival = resolver.resolve("CAM-B", 51.0, [th(track_id="T2", camera_id="CAM-B", last_timestamp=51.0)], {})

        self.assertNotEqual(arrival[0].global_id, original_id)

    def test_7_short_disappearance_within_timeout_still_matches(self):

        topology = CameraTopology()
        topology.add_transition("CAM-A", "CAM-B", min_transition_time=0.0, max_transition_time=1000.0)
        resolver = make_resolver(topology, timeout_seconds=30.0)

        original_id = resolver.resolve("CAM-A", 0.0, [th(track_id="T1", camera_id="CAM-A", last_timestamp=0.0)], {})[0].global_id
        resolver.resolve("CAM-A", 0.5, [th(track_id="T1", camera_id="CAM-A", state=TrackState.EXPIRED)], {})

        arrival = resolver.resolve("CAM-B", 5.0, [th(track_id="T2", camera_id="CAM-B", last_timestamp=5.0)], {})

        self.assertEqual(arrival[0].global_id, original_id)


class MultipleAndCrossingOccupantTests(unittest.TestCase):

    def test_8_multiple_independent_occupants_resolved_independently(self):

        topology = CameraTopology()
        topology.add_transition("CAM-A", "CAM-B", min_transition_time=0.0, max_transition_time=100.0)
        resolver = make_resolver(topology)

        id_1 = resolver.resolve("CAM-A", 0.0, [th(track_id="T1", camera_id="CAM-A", last_timestamp=0.0)], {})[0].global_id
        id_2 = resolver.resolve("CAM-A", 0.0, [th(track_id="T2", camera_id="CAM-A", last_timestamp=0.0)], {})[0].global_id

        self.assertNotEqual(id_1, id_2)

        resolver.resolve("CAM-A", 1.0, [
            th(track_id="T1", camera_id="CAM-A", state=TrackState.EXPIRED),
            th(track_id="T2", camera_id="CAM-A", state=TrackState.EXPIRED),
        ], {})

        arrivals = resolver.resolve("CAM-B", 5.0, [
            th(track_id="T3", camera_id="CAM-B", last_timestamp=5.0),
            th(track_id="T4", camera_id="CAM-B", last_timestamp=5.0),
        ], {})

        arrived_ids = {a.global_id for a in arrivals}
        self.assertEqual(arrived_ids, {id_1, id_2})

    def test_9_crossing_occupants_between_two_cameras_never_swap_identity(self):

        topology = CameraTopology()
        topology.add_transition("CAM-A", "CAM-B", min_transition_time=0.0, max_transition_time=100.0)
        resolver = make_resolver(topology)

        # Person 1 starts on CAM-A, Person 2 starts on CAM-B.
        id_1 = resolver.resolve("CAM-A", 0.0, [th(track_id="T1", camera_id="CAM-A", last_timestamp=0.0)], {})[0].global_id
        id_2 = resolver.resolve("CAM-B", 0.0, [th(track_id="T2", camera_id="CAM-B", last_timestamp=0.0)], {})[0].global_id

        # Both depart their original cameras...
        resolver.resolve("CAM-A", 1.0, [th(track_id="T1", camera_id="CAM-A", state=TrackState.EXPIRED)], {})
        resolver.resolve("CAM-B", 1.0, [th(track_id="T2", camera_id="CAM-B", state=TrackState.EXPIRED)], {})

        # ...and cross paths: Person 1 arrives at CAM-B, Person 2 arrives at CAM-A.
        arrival_at_b = resolver.resolve("CAM-B", 5.0, [th(track_id="T3", camera_id="CAM-B", last_timestamp=5.0)], {})
        arrival_at_a = resolver.resolve("CAM-A", 5.0, [th(track_id="T4", camera_id="CAM-A", last_timestamp=5.0)], {})

        self.assertEqual(arrival_at_b[0].global_id, id_1)
        self.assertEqual(arrival_at_a[0].global_id, id_2)


class SimultaneousEntryAndExitTests(unittest.TestCase):

    def test_10_simultaneous_entries_competing_for_one_candidate_only_one_matches(self):

        topology = CameraTopology()
        topology.add_transition("CAM-A", "CAM-B", min_transition_time=0.0, max_transition_time=100.0)
        resolver = make_resolver(topology)

        original_id = resolver.resolve("CAM-A", 0.0, [th(track_id="T1", camera_id="CAM-A", last_timestamp=0.0)], {})[0].global_id
        resolver.resolve("CAM-A", 1.0, [th(track_id="T1", camera_id="CAM-A", state=TrackState.EXPIRED)], {})

        # Two NEW arrivals on CAM-B in the SAME cycle, only one real
        # departed candidate exists -- both cannot claim it.
        arrivals = resolver.resolve("CAM-B", 5.0, [
            th(track_id="T2", camera_id="CAM-B", last_timestamp=5.0),
            th(track_id="T3", camera_id="CAM-B", last_timestamp=5.0),
        ], {})

        matched_to_original = [a for a in arrivals if a.global_id == original_id]
        self.assertEqual(len(matched_to_original), 1)
        self.assertEqual(len({a.global_id for a in arrivals}), 2)  # the other got its own new id

    def test_11_simultaneous_exits_on_one_camera_release_independently(self):

        resolver = make_resolver()

        resolver.resolve("CAM-A", 0.0, [
            th(track_id="T1", camera_id="CAM-A", last_timestamp=0.0),
            th(track_id="T2", camera_id="CAM-A", last_timestamp=0.0),
        ], {})

        expired = resolver.resolve("CAM-A", 1.0, [
            th(track_id="T1", camera_id="CAM-A", state=TrackState.EXPIRED),
            th(track_id="T2", camera_id="CAM-A", state=TrackState.EXPIRED),
        ], {})

        self.assertEqual(expired, ())  # no observation for a departure itself
        self.assertEqual(len(resolver.registry.unbound_records()), 2)  # both kept alive, unbound


class IdentityPersistenceAndExpirationTests(unittest.TestCase):

    def test_12_identity_persists_across_many_consecutive_tracked_cycles(self):

        resolver = make_resolver()

        global_id = None
        for i in range(10):
            result = resolver.resolve("CAM-A", float(i), [th(camera_id="CAM-A", last_timestamp=float(i))], {})
            if global_id is None:
                global_id = result[0].global_id
            self.assertEqual(result[0].global_id, global_id)

    def test_13_identity_expiration_is_a_strict_boundary(self):

        topology = CameraTopology()
        transition_model = TransitionModel(topology, timeout_seconds=10.0)

        registry = IdentityRegistry()
        global_id = registry.create("CAM-A", "T1", timestamp=0.0)
        registry.release("CAM-A", "T1")

        record = registry.get(global_id)

        self.assertFalse(transition_model.is_expired(record, now=10.0))   # exactly at the boundary -- not yet expired
        self.assertTrue(transition_model.is_expired(record, now=10.0001))  # just past it -- expired


class RegistryCleanupTests(unittest.TestCase):

    def test_14_registry_cleanup_leaves_no_leaked_entries(self):

        # 5 KNOWN but mutually-disconnected cameras (closed-world --
        # topology.py's own "at least one camera known -> explicit
        # absence" rule) -- each one's departure must produce its own
        # independent identity, never plausibly matched to an unrelated
        # camera's departure just because they happened close in time.
        topology = CameraTopology()
        for i in range(5):
            topology.add_camera(f"CAM-{i}")

        resolver = make_resolver(topology, timeout_seconds=5.0)

        for i in range(5):
            resolver.resolve(f"CAM-{i}", 0.0, [th(track_id="T1", camera_id=f"CAM-{i}", last_timestamp=0.0)], {})
            resolver.resolve(f"CAM-{i}", 0.5, [th(track_id="T1", camera_id=f"CAM-{i}", state=TrackState.EXPIRED)], {})

        self.assertEqual(len(resolver.registry), 5)

        # Long past every one of those identities' own timeout.
        resolver.resolve("CAM-999", 100.0, [], {})

        self.assertEqual(len(resolver.registry), 0)


class CameraTopologyEdgeCaseTests(unittest.TestCase):

    def test_15_unknown_cameras_fall_back_to_default_window(self):

        topology = CameraTopology()  # completely empty -- open world

        plausible, expected = topology.is_plausible_transition(
            "CAM-X", "CAM-Y", elapsed_seconds=5.0,
            default_min_transition_time=0.0, default_max_transition_time=10.0,
        )

        self.assertTrue(plausible)
        self.assertEqual(expected, 5.0)

    def test_15_known_camera_without_a_registered_edge_is_closed_world(self):

        topology = CameraTopology()
        topology.add_camera("CAM-X")

        plausible, expected = topology.is_plausible_transition(
            "CAM-X", "CAM-Y", elapsed_seconds=1.0,
            default_min_transition_time=0.0, default_max_transition_time=10.0,
        )

        self.assertFalse(plausible)
        self.assertIsNone(expected)

    def test_15_registered_transition_boundaries_are_inclusive(self):

        topology = CameraTopology()
        topology.add_transition("CAM-A", "CAM-B", min_transition_time=2.0, max_transition_time=8.0)

        plausible_min, _ = topology.is_plausible_transition("CAM-A", "CAM-B", 2.0, 0.0, 100.0)
        plausible_max, _ = topology.is_plausible_transition("CAM-A", "CAM-B", 8.0, 0.0, 100.0)
        implausible_below, _ = topology.is_plausible_transition("CAM-A", "CAM-B", 1.999, 0.0, 100.0)
        implausible_above, _ = topology.is_plausible_transition("CAM-A", "CAM-B", 8.001, 0.0, 100.0)

        self.assertTrue(plausible_min)
        self.assertTrue(plausible_max)
        self.assertFalse(implausible_below)
        self.assertFalse(implausible_above)

    def test_15_possible_destinations_reflects_registered_transitions_only(self):

        topology = CameraTopology()
        topology.add_transition("CAM-A", "CAM-B", 0.0, 10.0)
        topology.add_transition("CAM-A", "CAM-C", 0.0, 10.0, bidirectional=False)

        self.assertEqual(topology.possible_destinations("CAM-A"), ("CAM-B", "CAM-C"))
        self.assertEqual(topology.possible_destinations("CAM-C"), ())  # one-directional edge only

    def test_15_bidirectional_transition_registers_both_directions(self):

        topology = CameraTopology()
        topology.add_transition("CAM-A", "CAM-B", min_transition_time=1.0, max_transition_time=5.0)

        forward = topology.transition("CAM-A", "CAM-B")
        backward = topology.transition("CAM-B", "CAM-A")

        self.assertIsNotNone(forward)
        self.assertIsNotNone(backward)
        self.assertEqual(forward.expected_transition_time, 3.0)


class MinTrackAgeAndDepartureMotionTests(unittest.TestCase):

    def test_min_track_age_rejects_a_brand_new_flickery_track(self):

        topology = CameraTopology()
        topology.add_transition("CAM-A", "CAM-B", 0.0, 100.0)
        resolver = make_resolver(topology, min_track_age_for_matching=3)

        original_id = resolver.resolve("CAM-A", 0.0, [th(track_id="T1", camera_id="CAM-A", last_timestamp=0.0)], {})[0].global_id
        resolver.resolve("CAM-A", 1.0, [th(track_id="T1", camera_id="CAM-A", state=TrackState.EXPIRED)], {})

        # age=1 -- below min_track_age_for_matching=3 -- must not match.
        arrival = resolver.resolve("CAM-B", 5.0, [th(track_id="T2", camera_id="CAM-B", age=1, last_timestamp=5.0)], {})

        self.assertNotEqual(arrival[0].global_id, original_id)


class NavigationGraphTopologyDerivationTests(unittest.TestCase):

    # Phase 4 -- honestly deriving a CameraTopology from a real
    # navigation.graph.NavigationGraph (Digital Twin zone adjacency)
    # instead of hand-building one. Uses the real NavigationGraph/Node/
    # Edge classes directly (not a fake), proving this integration
    # against the actual Digital Twin data shape.

    def test_derives_adjacency_from_real_navigation_graph_zone_edges(self):

        from navigation.graph import NavigationGraph
        from navigation.node import Node
        from navigation.edge import Edge

        graph = NavigationGraph()
        graph.add_node(Node(id="zone-1", name="Zone 1", floor_id="floor-1", node_type=Node.ZONE))
        graph.add_node(Node(id="zone-2", name="Zone 2", floor_id="floor-1", node_type=Node.ZONE))
        graph.add_edge(Edge(id="door-1", edge_type=Edge.DOOR, from_node="zone-1", to_node="zone-2", walking_distance=6.0))

        camera_zone_ids = {"CAM-A": ("zone-1",), "CAM-B": ("zone-2",)}

        topology = build_topology_from_navigation_graph(graph, camera_zone_ids, walking_speed_m_per_s=1.5)

        transition = topology.transition("CAM-A", "CAM-B")

        self.assertIsNotNone(transition)
        self.assertAlmostEqual(transition.expected_transition_time, 6.0 / 1.5)

    def test_cameras_sharing_a_zone_are_treated_as_immediately_adjacent(self):

        from navigation.graph import NavigationGraph

        graph = NavigationGraph()

        camera_zone_ids = {"CAM-A": ("zone-1",), "CAM-B": ("zone-1",)}

        topology = build_topology_from_navigation_graph(graph, camera_zone_ids)

        transition = topology.transition("CAM-A", "CAM-B")

        self.assertIsNotNone(transition)
        self.assertEqual(transition.min_transition_time, 0.0)

    def test_unconnected_cameras_produce_no_transition(self):

        from navigation.graph import NavigationGraph
        from navigation.node import Node

        graph = NavigationGraph()
        graph.add_node(Node(id="zone-1", name="Zone 1", floor_id="floor-1", node_type=Node.ZONE))
        graph.add_node(Node(id="zone-2", name="Zone 2", floor_id="floor-1", node_type=Node.ZONE))
        # No edge between them at all.

        camera_zone_ids = {"CAM-A": ("zone-1",), "CAM-B": ("zone-2",)}

        topology = build_topology_from_navigation_graph(graph, camera_zone_ids)

        self.assertIsNone(topology.transition("CAM-A", "CAM-B"))


if __name__ == "__main__":
    unittest.main()
