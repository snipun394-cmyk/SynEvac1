import unittest

from models.building import Building
from models.floor import Floor
from models.staircase import Staircase

from occupancy.observation import OccupancyObservation
from occupancy.snapshot import OccupancySnapshot

from pathfinding.route import Route

from simulator.multi_agent_result import MultiAgentSimulationResult, OccupantTimeline, OccupantTimelineStep
from simulator.occupant import OccupantState

from live_occupants.manager import LiveOccupantManager

from observable_assets.models import AssetObservation, ObservableAssetSnapshot, ObservationStatus

from predictive_dataset.candidate import edges_by_candidate_id, enumerate_candidates
from predictive_dataset.live_extractor_v2_1 import (
    build_stair_flow_snapshot_for_prediction, extract_live_experimental_candidate_features,
)
from predictive_dataset.simulation_extractor_v2_1 import build_alternative_route_counts, extract_experimental_candidate_features


# =====================================================
# Stair Predictive-Feature Live Parity milestone, Phase 7/8 -- controlled
# equivalence tests. Both sides are driven through their REAL, actually-
# wired public entry points (extract_experimental_candidate_features()/
# extract_live_experimental_candidate_features()), never a private
# helper in isolation -- proving the WIRED path, not just the underlying
# formula.
# =====================================================


def make_building_with_stair():

    floor_1 = Floor(name="Floor 1", id="floor-1", display_order=0, height=3.0)
    floor_2 = Floor(name="Floor 2", id="floor-2", display_order=1, height=3.0)

    stair = Staircase(
        id="stair-1", name="S1", from_floor_id="floor-1", to_floor_id="floor-2",
        from_zone_id="zone-1", to_zone_id="zone-2", width=1.5,
    )
    floor_1.stairs.append(stair)

    building = Building(name="Building", id="building-1", floors=[floor_1, floor_2])

    # Zones referenced by the Staircase's own from_zone_id/to_zone_id --
    # required for NavigationGraphGenerator to resolve a valid Stair edge.
    from models.zone import Zone
    floor_1.zones.append(Zone(id="zone-1", name="Zone 1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="floor-1"))
    floor_2.zones.append(Zone(id="zone-2", name="Zone 2", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="floor-2"))

    return building, floor_1, floor_2, stair


class StairFlowRateSimLiveParityTests(unittest.TestCase):

    def setUp(self):

        self.building, self.floor_1, self.floor_2, self.stair = make_building_with_stair()
        self.edges = edges_by_candidate_id(self.building)
        self.candidates = {c.candidate_id: c for c in enumerate_candidates(self.building)}
        self.alt_counts = build_alternative_route_counts(tuple(self.candidates.values()))
        self.stair_edge = self.edges["stair-1"]
        self.stair_candidate = self.candidates["stair-1"]

    # ---------------------------------------------------
    # SIM side: build a completed-crossing step per requested end_time,
    # through the edge's own Stair type.
    # ---------------------------------------------------

    def _sim_flow_rate(self, completion_times, observation_time, window=None):

        steps = [
            OccupantTimelineStep(
                index=0, from_node=None, to_node=None, edge=self.stair_edge,
                queue_wait_time=0.0, start_time=end_time - 3.0, end_time=end_time,
            )
            for end_time in completion_times
        ]
        occupants = {
            f"occ-{i}": OccupantTimeline(
                occupant_id=f"occ-{i}", route=Route(nodes=[], edges=[self.stair_edge], total_cost=0.0, total_distance=0.0),
                steps=[step], state=OccupantState.ARRIVED, depart_time=step.start_time, arrival_time=step.end_time,
            )
            for i, step in enumerate(steps)
        }
        movement_result = MultiAgentSimulationResult(occupants=occupants, total_evacuation_time=observation_time)
        occupancy_snapshot = OccupancySnapshot(observations={"zone-1": OccupancyObservation(occupant_count=0)})

        kwargs = {}
        if window is not None:
            # _recent_flow_rate's own `window` parameter is only reachable
            # through the base extractor's default; when a non-default
            # window is requested for an illustrative scenario, call the
            # private function directly (still the SAME formula the
            # public wrapper always uses with FLOW_WINDOW_SECONDS).
            from predictive_dataset.simulation_extractor_v2_1 import _recent_flow_rate
            return _recent_flow_rate(movement_result, "stair-1", observation_time, window=window)

        features = extract_experimental_candidate_features(
            self.stair_candidate, self.stair_edge, observation_time,
            building=self.building, movement_result=movement_result, occupancy_snapshot=occupancy_snapshot,
            alternative_route_counts=self.alt_counts,
        )
        return features["candidate_recent_flow_rate"]

    # ---------------------------------------------------
    # LIVE side: drive the REAL LiveOccupantManager so occupants genuinely
    # enter AND exit the stair (entry then exit -- a completed traversal),
    # completing at the requested timestamp.
    # ---------------------------------------------------

    def _observed_snapshot(self):

        # A genuinely CONFIRMED-observed stair (a calibrated camera
        # covers it right now) -- what lets a real zero surface as `0`,
        # not `None`. See build_stair_flow_snapshot_for_prediction()'s
        # own docstring.
        return ObservableAssetSnapshot(
            observations={
                "stair-1": AssetObservation(asset_id="stair-1", asset_type="Stair", status=ObservationStatus.OBSERVED),
            },
        )

    def _live_flow_rate(self, completion_times, observation_time, observed=True):

        manager = LiveOccupantManager()

        for i, end_time in enumerate(completion_times):

            occupant_id = f"occ-{i}"
            entry_time = end_time - 3.0

            manager.update(
                occupant_id, "CAM-1", f"T-{i}", "zone-1", self.floor_1.id, (1.0, 1.0), 0.0, None, 0.9, entry_time - 1.0,
            )
            manager.update(
                occupant_id, "CAM-1", f"T-{i}", None, self.floor_1.id, (2.0, 2.0), 0.5, None, 0.9, entry_time,
                stair_id="stair-1",
            )
            manager.update(
                occupant_id, "CAM-1", f"T-{i}", "zone-2", self.floor_2.id, (3.0, 3.0), 0.5, None, 0.9, end_time,
            )

        stair_flow_snapshot = build_stair_flow_snapshot_for_prediction(
            [self.stair], manager.all_occupants(), self.building, observation_time,
            observable_assets=self._observed_snapshot() if observed else None,
        )

        features = extract_live_experimental_candidate_features(
            self.stair_candidate, self.stair_edge, observation_time,
            building=self.building, crowd_snapshot=None, occupancy_facts=None,
            alternative_route_counts=self.alt_counts, evacuation_snapshot=None, occupants=None,
            stair_flow_snapshot=stair_flow_snapshot,
        )
        return features["candidate_recent_flow_rate"]

    # --- controlled scenarios (Phase 7) ---

    def test_zero_flow(self):

        sim_rate = self._sim_flow_rate([], 100.0)
        live_rate = self._live_flow_rate([], 100.0)

        self.assertEqual(sim_rate, 0)
        self.assertEqual(live_rate, 0)

    def test_one_completed_entry(self):

        sim_rate = self._sim_flow_rate([50.0], 100.0)
        live_rate = self._live_flow_rate([50.0], 100.0)

        self.assertEqual(sim_rate, live_rate)
        self.assertEqual(sim_rate, 1)

    def test_multiple_completed_crossings(self):

        # The milestone's own illustrative example: occupants entering
        # at t=5, t=10, t=15 within a 20s predictive window, observed at
        # t=20 -- all three complete (a short 3s traversal) within the
        # window.
        completion_times = [8.0, 13.0, 18.0]  # entry + 3s travel time each
        sim_rate = self._sim_flow_rate(completion_times, 20.0, window=20.0)

        # Live side, using the SAME 20s window explicitly (illustrating
        # the milestone's own example -- the actually-wired feature
        # always uses FLOW_WINDOW_SECONDS=60.0, proved separately below).
        from stair_flow.compute import compute_stair_flow_snapshot

        manager = LiveOccupantManager()
        for i, end_time in enumerate(completion_times):
            occupant_id = f"occ-{i}"
            entry_time = end_time - 3.0
            manager.update(occupant_id, "CAM-1", f"T-{i}", "zone-1", self.floor_1.id, (1.0, 1.0), 0.0, None, 0.9, entry_time - 1.0)
            manager.update(occupant_id, "CAM-1", f"T-{i}", None, self.floor_1.id, (2.0, 2.0), 0.5, None, 0.9, entry_time, stair_id="stair-1")
            manager.update(occupant_id, "CAM-1", f"T-{i}", "zone-2", self.floor_2.id, (3.0, 3.0), 0.5, None, 0.9, end_time)

        snapshot = compute_stair_flow_snapshot(
            [self.stair], manager.all_occupants(), self.building, timestamp=20.0, window_seconds=20.0,
        )
        live_rate = snapshot.for_stair("stair-1").exits

        self.assertEqual(sim_rate, live_rate)
        self.assertEqual(sim_rate, 3)

    def test_entry_and_exit_same_occupant_counts_as_one_completion(self):

        sim_rate = self._sim_flow_rate([60.0], 100.0)
        live_rate = self._live_flow_rate([60.0], 100.0)

        self.assertEqual(sim_rate, live_rate)
        self.assertEqual(sim_rate, 1)

    def test_simultaneous_completions(self):

        sim_rate = self._sim_flow_rate([70.0, 70.0, 70.0], 100.0)
        live_rate = self._live_flow_rate([70.0, 70.0, 70.0], 100.0)

        self.assertEqual(sim_rate, live_rate)
        self.assertEqual(sim_rate, 3)

    def test_boundary_timestamp_excludes_window_start(self):

        # window = (time - 60, time]; a completion at EXACTLY time-60
        # must be excluded on both sides.
        sim_rate = self._sim_flow_rate([40.0], 100.0)  # window_start == 40.0
        live_rate = self._live_flow_rate([40.0], 100.0)

        self.assertEqual(sim_rate, 0)
        self.assertEqual(live_rate, 0)

    def test_boundary_timestamp_includes_observation_time(self):

        sim_rate = self._sim_flow_rate([100.0], 100.0)  # completion == time itself
        live_rate = self._live_flow_rate([100.0], 100.0)

        self.assertEqual(sim_rate, 1)
        self.assertEqual(live_rate, 1)

    def test_incomplete_traversal_is_not_counted_on_either_side(self):

        # Occupant enters the stair but has NOT yet exited by observation
        # time -- an in-progress, incomplete traversal. Neither side may
        # count it (the genuinely-differing epistemics case: live
        # `entries` WOULD show 1 here, but the parity-proven quantity is
        # `exits`, which correctly agrees with sim's own "completed
        # crossings only" definition).
        manager = LiveOccupantManager()
        manager.update("occ-0", "CAM-1", "T-0", "zone-1", self.floor_1.id, (1.0, 1.0), 0.0, None, 0.9, 90.0)
        manager.update(
            "occ-0", "CAM-1", "T-0", None, self.floor_1.id, (2.0, 2.0), 0.5, None, 0.9, 95.0, stair_id="stair-1",
        )
        # No exit update -- still on the stair as of observation_time=100.0.

        stair_flow_snapshot = build_stair_flow_snapshot_for_prediction(
            [self.stair], manager.all_occupants(), self.building, 100.0,
        )
        self.assertEqual(stair_flow_snapshot.for_stair("stair-1").entries, 1)  # entry evidence exists...
        self.assertEqual(stair_flow_snapshot.for_stair("stair-1").exits, 0)    # ...but no completion yet

        features = extract_live_experimental_candidate_features(
            self.stair_candidate, self.stair_edge, 100.0,
            building=self.building, crowd_snapshot=None, occupancy_facts=None,
            alternative_route_counts=self.alt_counts, evacuation_snapshot=None, occupants=None,
            stair_flow_snapshot=stair_flow_snapshot,
        )

        sim_rate = self._sim_flow_rate([], 100.0)  # sim: no completed step exists at all
        self.assertEqual(features["candidate_recent_flow_rate"], 0)
        self.assertEqual(features["candidate_recent_flow_rate"], sim_rate)

    def test_unknown_when_no_observation_available_sim_still_computes_ground_truth(self):

        # Phase 6's own disclosed epistemic difference: simulation has
        # omniscient ground truth (always 0, never None, for a scenario
        # with no crossings) -- live has none without SOME confirmed
        # observation basis (no observable_assets supplied here, and no
        # window evidence either). Not a parity bug -- documented in
        # docs/architecture/stair_predictive_feature_live_parity.md.
        sim_rate = self._sim_flow_rate([], 100.0)
        live_rate = self._live_flow_rate([], 100.0, observed=False)

        self.assertEqual(sim_rate, 0)
        self.assertIsNone(live_rate)

    def test_default_behavior_unchanged_when_no_stair_flow_snapshot_supplied(self):

        # Backward compatibility -- Stair falls back to the ORIGINAL
        # zone-transition proxy exactly like Door, when the new evidence
        # is not supplied at all.
        features = extract_live_experimental_candidate_features(
            self.stair_candidate, self.stair_edge, 100.0,
            building=self.building, crowd_snapshot=None, occupancy_facts=None,
            alternative_route_counts=self.alt_counts, evacuation_snapshot=None, occupants=None,
            stair_flow_snapshot=None,
        )
        self.assertIsNone(features["candidate_recent_flow_rate"])  # occupants=None -> proxy returns None


class MultiCameraPredictiveFeatureParityTests(unittest.TestCase):
    """Phase 8 -- three cameras all observing the same canonical
    occupants traversing the same Stair must produce the IDENTICAL
    candidate_recent_flow_rate as one camera would."""

    def setUp(self):

        self.building, self.floor_1, self.floor_2, self.stair = make_building_with_stair()
        self.edges = edges_by_candidate_id(self.building)
        self.candidates = {c.candidate_id: c for c in enumerate_candidates(self.building)}
        self.alt_counts = build_alternative_route_counts(tuple(self.candidates.values()))

    def _flow_rate_with_camera_count(self, camera_count):

        manager = LiveOccupantManager()
        camera_ids = [f"CAM-{chr(65 + i)}" for i in range(camera_count)]  # CAM-A, CAM-B, CAM-C, ...

        for occ_index, entry_time in enumerate((5.0, 10.0, 15.0)):

            occupant_id = f"occ-{occ_index}"
            exit_time = entry_time + 3.0

            manager.update(
                occupant_id, camera_ids[0], f"T-{occ_index}-0", "zone-1", self.floor_1.id,
                (1.0, 1.0), 0.0, None, 0.9, entry_time - 1.0,
            )

            # Every camera observes the SAME entry this same cycle.
            for cam_idx, camera_id in enumerate(camera_ids):
                manager.update(
                    occupant_id, camera_id, f"T-{occ_index}-{cam_idx}", None, self.floor_1.id,
                    (2.0, 2.0), 0.5, None, 0.9, entry_time, stair_id="stair-1",
                )

            # Every camera observes the SAME exit this same cycle.
            for cam_idx, camera_id in enumerate(camera_ids):
                manager.update(
                    occupant_id, camera_id, f"T-{occ_index}-{cam_idx}", "zone-2", self.floor_2.id,
                    (3.0, 3.0), 0.5, None, 0.9, exit_time,
                )

        from predictive_dataset.live_extractor_v2_1 import build_stair_flow_snapshot_for_prediction

        stair_flow_snapshot = build_stair_flow_snapshot_for_prediction(
            [self.stair], manager.all_occupants(), self.building, 20.0,
        )

        features = extract_live_experimental_candidate_features(
            self.candidates["stair-1"], self.edges["stair-1"], 20.0,
            building=self.building, crowd_snapshot=None, occupancy_facts=None,
            alternative_route_counts=self.alt_counts, evacuation_snapshot=None, occupants=None,
            stair_flow_snapshot=stair_flow_snapshot,
        )
        return features["candidate_recent_flow_rate"]

    def test_one_two_and_three_cameras_produce_identical_flow_rate(self):

        one_camera = self._flow_rate_with_camera_count(1)
        two_cameras = self._flow_rate_with_camera_count(2)
        three_cameras = self._flow_rate_with_camera_count(3)

        self.assertEqual(one_camera, 3)
        self.assertEqual(one_camera, two_cameras)
        self.assertEqual(two_cameras, three_cameras)


if __name__ == "__main__":
    unittest.main()
