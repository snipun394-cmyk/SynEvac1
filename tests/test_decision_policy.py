import unittest

from models.building import Building
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from scenario.metadata import ScenarioMetadata
from scenario.occupant import ScenarioOccupant
from scenario.scenario import Scenario

from ground_truth.labels import GroundTruth

from decision_policy import DecisionInputs, DecisionPolicy, generate_policy
from decision_policy import announcement_policy, exit_policy, rescue_policy, stair_policy, zone_policy
from decision_policy.exit_policy import CLOSE, HIGH_CONGESTION, KEEP_OPEN, OPEN
from decision_policy.stair_policy import AVOID, CONGESTED, USE
from decision_policy.zone_policy import EVACUATE_IMMEDIATELY, SHELTER_IN_PLACE, WAIT


# =====================================================
# Fixtures
# =====================================================


def make_building():

    floor_1 = Floor(
        name="Ground", id="floor-1", display_order=0,
        zones=[
            Zone(id="zone-a", name="Zone A", x=0.0, y=0.0, width=4.0, height=5.0, floor_id="floor-1"),
            Zone(id="zone-b", name="Zone B", x=5.0, y=0.0, width=4.0, height=5.0, floor_id="floor-1"),
            Zone(id="zone-c", name="Zone C", x=10.0, y=0.0, width=4.0, height=5.0, floor_id="floor-1"),
            Zone(id="zone-d", name="Zone D", x=15.0, y=0.0, width=4.0, height=5.0, floor_id="floor-1"),
            Zone(id="zone-e", name="Zone E", x=20.0, y=0.0, width=4.0, height=5.0, floor_id="floor-1"),
        ],
        exits=[
            Exit(id="exit-normal", floor_id="floor-1", zone_id="zone-d", capacity=50, is_blocked=False),
            Exit(id="exit-congested", floor_id="floor-1", zone_id="zone-c", capacity=1, is_blocked=False),
            Exit(id="exit-hazard", floor_id="floor-1", zone_id="zone-a", capacity=50, is_blocked=False),
            Exit(id="exit-closed", floor_id="floor-1", zone_id="zone-b", capacity=50, is_blocked=True),
        ],
        stairs=[
            Staircase(
                id="stair-safe", from_floor_id="floor-1", to_floor_id="floor-2",
                from_zone_id="zone-d", to_zone_id="zone-f",
            ),
            Staircase(
                id="stair-hazard", from_floor_id="floor-1", to_floor_id="floor-2",
                from_zone_id="zone-a", to_zone_id="zone-f",
            ),
        ],
    )

    floor_2 = Floor(
        name="First Floor", id="floor-2", display_order=1,
        zones=[Zone(id="zone-f", name="Zone F", floor_id="floor-2", width=4.0, height=4.0)],
    )

    return Building(name="Test Building", id="building-1", floors=[floor_1, floor_2])


def make_scenario(building, extra_occupants=()):

    metadata = ScenarioMetadata(
        scenario_id="scn-1", definition_id="def-1", definition_content_hash="hash-1",
        generation_version="v1", seed=1, created_at="2026-01-01T00:00:00",
    )

    occupants = (
        ScenarioOccupant(
            occupant_id="occ-a", zone_id="zone-a", floor_id="floor-1",
            position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
        ),
        ScenarioOccupant(
            occupant_id="occ-b", zone_id="zone-b", floor_id="floor-1",
            position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
        ),
        ScenarioOccupant(
            occupant_id="occ-c", zone_id="zone-c", floor_id="floor-1",
            position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
        ),
        ScenarioOccupant(
            occupant_id="occ-d", zone_id="zone-d", floor_id="floor-1",
            position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
        ),
        ScenarioOccupant(
            occupant_id="occ-e", zone_id="zone-e", floor_id="floor-1",
            position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
        ),
    ) + tuple(extra_occupants)

    return Scenario(metadata=metadata, occupants=occupants)


def make_ground_truth(**overrides):

    defaults = dict(
        scenario_id="scn-1",
        definition_id="def-1",
        building_cleared=False,
        people_trapped=1,
        zone_route_stats=[
            {"zone_id": "zone-a", "preferred_exit": "exit-hazard", "preferred_stair": None,
             "average_travel_distance": 5.0, "average_travel_time": 5.0},
            {"zone_id": "zone-b", "preferred_exit": "exit-closed", "preferred_stair": None,
             "average_travel_distance": 5.0, "average_travel_time": 5.0},
            {"zone_id": "zone-c", "preferred_exit": "exit-congested", "preferred_stair": None,
             "average_travel_distance": 5.0, "average_travel_time": 5.0},
            {"zone_id": "zone-d", "preferred_exit": "exit-normal", "preferred_stair": "stair-safe",
             "average_travel_distance": 5.0, "average_travel_time": 5.0},
            # zone-e deliberately has no entry -- simulates "no viable route".
        ],
        zone_risk_scores=[
            {"zone_id": "zone-a", "risk_score": 0.9},
            {"zone_id": "zone-b", "risk_score": 0.5},
            {"zone_id": "zone-c", "risk_score": 0.1},
            {"zone_id": "zone-d", "risk_score": 0.05},
            {"zone_id": "zone-e", "risk_score": None},
        ],
        exit_risk_scores=[],
        stair_risk_scores=[
            {"stair_id": "stair-safe", "risk_score": 0.1},
            {"stair_id": "stair-hazard", "risk_score": 0.1},
        ],
        hazard_spread_order=("zone-a",),
        maximum_hazard_zone="zone-a",
        first_hazardous_zone="zone-a",
        exits_exceeding_capacity=("exit-congested",),
        worst_exit=None,
        stairs_exceeding_capacity=(),
        worst_stair=None,
    )
    defaults.update(overrides)

    return GroundTruth(**defaults)


# =====================================================


class ZonePolicyTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.scenario = make_scenario(self.building)
        self.ground_truth = make_ground_truth()
        self.zones = [
            zone for floor in self.building.ordered_floors() for zone in floor.zones
        ]

    def _decisions(self, **overrides):

        ground_truth = make_ground_truth(**overrides) if overrides else self.ground_truth

        return {
            entry["zone_id"]: entry
            for entry in zone_policy.compute_zone_decisions(
                scenario=self.scenario, ground_truth=ground_truth, zones=self.zones,
            )
        }

    def test_critical_risk_shelters_in_place(self):

        decisions = self._decisions()
        self.assertEqual(decisions["zone-a"]["action"], SHELTER_IN_PLACE)

    def test_hazard_spread_zone_evacuates_immediately(self):

        decisions = self._decisions()
        # zone-b is not in hazard_spread_order/critical risk, but has
        # elevated risk (0.5 >= 0.35).
        self.assertEqual(decisions["zone-b"]["action"], EVACUATE_IMMEDIATELY)

    def test_congested_preferred_exit_produces_wait(self):

        decisions = self._decisions()
        self.assertEqual(decisions["zone-c"]["action"], WAIT)
        self.assertEqual(decisions["zone-c"]["recommended_exit"], "exit-congested")

    def test_low_risk_uncongested_evacuates_immediately_by_default(self):

        decisions = self._decisions()
        self.assertEqual(decisions["zone-d"]["action"], EVACUATE_IMMEDIATELY)
        self.assertEqual(decisions["zone-d"]["recommended_stair"], "stair-safe")

    def test_no_viable_route_with_occupants_shelters_in_place(self):

        decisions = self._decisions()
        self.assertEqual(decisions["zone-e"]["action"], SHELTER_IN_PLACE)
        self.assertIsNone(decisions["zone-e"]["recommended_exit"])

    def test_current_fire_zone_id_forces_evacuate_immediately(self):

        zones_only = {
            entry["zone_id"]: entry
            for entry in zone_policy.compute_zone_decisions(
                scenario=self.scenario, ground_truth=self.ground_truth, zones=self.zones,
                current_fire_zone_ids=frozenset({"zone-d"}),
            )
        }

        self.assertEqual(zones_only["zone-d"]["action"], EVACUATE_IMMEDIATELY)


class ExitPolicyTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.scenario = make_scenario(self.building)
        self.ground_truth = make_ground_truth()
        self.exits = [
            exit_obj for floor in self.building.ordered_floors() for exit_obj in floor.exits
        ]
        self.zone_decisions = zone_policy.compute_zone_decisions(
            scenario=self.scenario, ground_truth=self.ground_truth,
            zones=[zone for floor in self.building.ordered_floors() for zone in floor.zones],
        )

    def _decisions(self):

        return {
            entry["exit_id"]: entry
            for entry in exit_policy.compute_exit_decisions(
                scenario=self.scenario, ground_truth=self.ground_truth,
                exits=self.exits, zone_decisions=self.zone_decisions,
            )
        }

    def test_exit_in_hazardous_zone_is_closed(self):

        self.assertEqual(self._decisions()["exit-hazard"]["status"], CLOSE)

    def test_exit_exceeding_capacity_is_high_congestion(self):

        self.assertEqual(self._decisions()["exit-congested"]["status"], HIGH_CONGESTION)

    def test_closed_exit_recommended_by_a_zone_should_be_opened(self):

        # exit-closed is is_blocked=True and zone-b's zone_route_stats
        # recommends it.
        self.assertEqual(self._decisions()["exit-closed"]["status"], OPEN)

    def test_closed_exit_not_needed_stays_closed(self):

        ground_truth = make_ground_truth(
            people_trapped=0,
            zone_route_stats=[
                entry for entry in self.ground_truth.zone_route_stats
                if entry["zone_id"] != "zone-b"
            ],
        )
        decisions = {
            entry["exit_id"]: entry
            for entry in exit_policy.compute_exit_decisions(
                scenario=self.scenario, ground_truth=ground_truth,
                exits=self.exits, zone_decisions=[],
            )
        }

        self.assertEqual(decisions["exit-closed"]["status"], CLOSE)

    def test_normal_open_exit_is_keep_open(self):

        self.assertEqual(self._decisions()["exit-normal"]["status"], KEEP_OPEN)

    def test_scenario_override_marks_an_otherwise_open_exit_closed(self):

        from scenario.engineering_state import ScenarioExitState

        scenario = Scenario(
            metadata=self.scenario.metadata, occupants=self.scenario.occupants,
            exit_states=(ScenarioExitState(exit_id="exit-normal", is_open=False),),
        )
        ground_truth = make_ground_truth(people_trapped=0, zone_route_stats=[])

        decisions = {
            entry["exit_id"]: entry
            for entry in exit_policy.compute_exit_decisions(
                scenario=scenario, ground_truth=ground_truth, exits=self.exits, zone_decisions=[],
            )
        }

        self.assertEqual(decisions["exit-normal"]["status"], CLOSE)


class StairPolicyTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.stairs = [
            stair for floor in self.building.ordered_floors() for stair in floor.stairs
        ]

    def test_stair_connected_to_hazardous_zone_is_avoided(self):

        ground_truth = make_ground_truth()

        decisions = {
            entry["stair_id"]: entry
            for entry in stair_policy.compute_stair_decisions(
                ground_truth=ground_truth, stairs=self.stairs,
            )
        }

        self.assertEqual(decisions["stair-hazard"]["status"], AVOID)
        self.assertEqual(decisions["stair-safe"]["status"], USE)

    def test_high_risk_score_is_avoided_even_without_hazard_connectivity(self):

        ground_truth = make_ground_truth(
            hazard_spread_order=(), maximum_hazard_zone=None, first_hazardous_zone=None,
            stair_risk_scores=[
                {"stair_id": "stair-safe", "risk_score": 0.9},
                {"stair_id": "stair-hazard", "risk_score": 0.0},
            ],
        )

        decisions = {
            entry["stair_id"]: entry
            for entry in stair_policy.compute_stair_decisions(
                ground_truth=ground_truth, stairs=self.stairs,
            )
        }

        self.assertEqual(decisions["stair-safe"]["status"], AVOID)
        self.assertEqual(decisions["stair-hazard"]["status"], USE)

    def test_moderate_risk_score_is_congested(self):

        ground_truth = make_ground_truth(
            hazard_spread_order=(), maximum_hazard_zone=None, first_hazardous_zone=None,
            stair_risk_scores=[
                {"stair_id": "stair-safe", "risk_score": 0.4},
                {"stair_id": "stair-hazard", "risk_score": 0.0},
            ],
        )

        decisions = {
            entry["stair_id"]: entry
            for entry in stair_policy.compute_stair_decisions(
                ground_truth=ground_truth, stairs=self.stairs,
            )
        }

        self.assertEqual(decisions["stair-safe"]["status"], CONGESTED)

    def test_current_fire_zone_id_forces_avoid(self):

        ground_truth = make_ground_truth(
            hazard_spread_order=(), maximum_hazard_zone=None, first_hazardous_zone=None,
            stair_risk_scores=[
                {"stair_id": "stair-safe", "risk_score": 0.0},
                {"stair_id": "stair-hazard", "risk_score": 0.0},
            ],
        )

        decisions = {
            entry["stair_id"]: entry
            for entry in stair_policy.compute_stair_decisions(
                ground_truth=ground_truth, stairs=self.stairs,
                current_fire_zone_ids=frozenset({"zone-d"}),
            )
        }

        self.assertEqual(decisions["stair-safe"]["status"], AVOID)


class AnnouncementPolicyTests(unittest.TestCase):

    def test_generates_one_announcement_per_present_action_in_priority_order(self):

        zone_decisions = [
            {"zone_id": "zone-a", "action": WAIT},
            {"zone_id": "zone-b", "action": SHELTER_IN_PLACE},
            {"zone_id": "zone-c", "action": EVACUATE_IMMEDIATELY},
            {"zone_id": "zone-d", "action": EVACUATE_IMMEDIATELY},
        ]

        announcements = announcement_policy.generate_announcements(zone_decisions)

        self.assertEqual(len(announcements), 3)
        self.assertEqual(announcements[0]["priority"], "CRITICAL")
        self.assertEqual(announcements[0]["target_zones"], ["zone-b"])
        self.assertEqual(announcements[1]["priority"], "HIGH")
        self.assertEqual(announcements[1]["target_zones"], ["zone-c", "zone-d"])
        self.assertEqual(announcements[2]["priority"], "NORMAL")
        self.assertEqual(announcements[2]["target_zones"], ["zone-a"])

    def test_no_zones_produces_no_announcements(self):

        self.assertEqual(announcement_policy.generate_announcements([]), [])

    def test_missing_action_category_is_omitted(self):

        zone_decisions = [{"zone_id": "zone-a", "action": EVACUATE_IMMEDIATELY}]
        announcements = announcement_policy.generate_announcements(zone_decisions)

        self.assertEqual(len(announcements), 1)
        self.assertEqual(announcements[0]["priority"], "HIGH")


class RescuePolicyTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.zones = [
            zone for floor in self.building.ordered_floors() for zone in floor.zones
        ]

    def test_building_cleared_produces_no_rescue_targets(self):

        scenario = make_scenario(self.building)
        ground_truth = make_ground_truth(building_cleared=True)

        priorities, order = rescue_policy.compute_rescue_priorities(
            scenario=scenario, ground_truth=ground_truth, zones=self.zones,
        )

        self.assertTrue(all(entry["rescue_priority"] == "NONE" for entry in priorities))
        self.assertEqual(order, ())

    def test_zone_with_no_occupants_is_none(self):

        scenario = make_scenario(self.building)
        ground_truth = make_ground_truth()

        priorities, order = rescue_policy.compute_rescue_priorities(
            scenario=scenario, ground_truth=ground_truth, zones=self.zones,
        )

        by_zone = {entry["zone_id"]: entry for entry in priorities}
        # zone-f has no occupants in this scenario's fixture.
        self.assertEqual(by_zone["zone-f"]["rescue_priority"], "NONE")

    def test_rescue_order_ranks_by_impact_not_raw_risk_score(self):

        # occupant-heavy extra zone with moderate risk should rank above
        # a lightly-populated zone with a higher risk score, since
        # impact_score = risk_score * occupant_count.
        extra_occupants = [
            ScenarioOccupant(
                occupant_id=f"extra-{i}", zone_id="zone-b", floor_id="floor-1",
                position=(0.0, 0.0), behaviour_profile_id="Adult_Default",
            )
            for i in range(10)
        ]
        scenario = make_scenario(self.building, extra_occupants=extra_occupants)
        ground_truth = make_ground_truth()

        priorities, order = rescue_policy.compute_rescue_priorities(
            scenario=scenario, ground_truth=ground_truth, zones=self.zones,
        )

        # zone-a: risk 0.9 * 1 occupant = 0.9
        # zone-b: risk 0.5 * 11 occupants = 5.5 -- ranks first despite
        # a lower risk score.
        self.assertEqual(order[0], "zone-b")
        self.assertIn("zone-a", order)
        self.assertLess(order.index("zone-b"), order.index("zone-a"))

    def test_firefighter_deployment_priority_combines_zones_stairs_and_exits(self):

        scenario = make_scenario(self.building)
        ground_truth = make_ground_truth()

        priorities, order = rescue_policy.compute_rescue_priorities(
            scenario=scenario, ground_truth=ground_truth, zones=self.zones,
        )

        stairs = [stair for floor in self.building.ordered_floors() for stair in floor.stairs]
        exits = [exit_obj for floor in self.building.ordered_floors() for exit_obj in floor.exits]

        stair_decisions = stair_policy.compute_stair_decisions(
            ground_truth=ground_truth, stairs=stairs,
        )
        zone_decisions = zone_policy.compute_zone_decisions(
            scenario=scenario, ground_truth=ground_truth, zones=self.zones,
        )
        exit_decisions = exit_policy.compute_exit_decisions(
            scenario=scenario, ground_truth=ground_truth, exits=exits, zone_decisions=zone_decisions,
        )

        deployment = rescue_policy.compute_firefighter_deployment_priority(
            rescue_priorities=priorities, rescue_order=order,
            stair_decisions=stair_decisions, exit_decisions=exit_decisions,
        )

        target_types = [entry["target_type"] for entry in deployment]

        # Zones (rescue order) come first, then stairs, then exits --
        # and ranks are sequential starting at 1.
        self.assertEqual(target_types.count("zone"), len(order))
        self.assertEqual([entry["rank"] for entry in deployment], list(range(1, len(deployment) + 1)))

        by_target_id = {entry["target_id"]: entry for entry in deployment}
        self.assertEqual(by_target_id["stair-hazard"]["target_type"], "stair")
        self.assertEqual(
            by_target_id["stair-hazard"]["reason"],
            "Stair marked AVOID -- unsafe or severely bottlenecked",
        )
        self.assertIn("exit-hazard", by_target_id)
        self.assertEqual(by_target_id["exit-hazard"]["target_type"], "exit")

        # Category order preserved: every zone rank precedes every
        # stair rank, which precedes every exit rank.
        last_zone_rank = max(e["rank"] for e in deployment if e["target_type"] == "zone")
        first_stair_rank = min(e["rank"] for e in deployment if e["target_type"] == "stair")
        first_exit_rank = min(e["rank"] for e in deployment if e["target_type"] == "exit")
        self.assertLess(last_zone_rank, first_stair_rank)
        self.assertLess(first_stair_rank, first_exit_rank)


class GeneratePolicyIntegrationTests(unittest.TestCase):

    def setUp(self):

        self.building = make_building()
        self.scenario = make_scenario(self.building)
        self.ground_truth = make_ground_truth()

    def test_generate_policy_matches_individual_submodule_calls(self):

        inputs = DecisionInputs(
            building=self.building, scenario=self.scenario, ground_truth=self.ground_truth,
        )

        result = generate_policy(inputs)

        self.assertEqual(result.scenario_id, "scn-1")
        self.assertGreater(len(result.zone_decisions), 0)
        self.assertGreater(len(result.exit_decisions), 0)
        self.assertGreater(len(result.stair_decisions), 0)
        self.assertGreater(len(result.announcements), 0)

    def test_timeline_rows_current_fire_zones_influences_decisions(self):

        inputs_without_timeline = DecisionInputs(
            building=self.building, scenario=self.scenario, ground_truth=self.ground_truth,
        )
        without = generate_policy(inputs_without_timeline)
        zone_d_action_without = next(
            d["action"] for d in without.zone_decisions if d["zone_id"] == "zone-d"
        )
        self.assertEqual(zone_d_action_without, EVACUATE_IMMEDIATELY)

        inputs_with_timeline = DecisionInputs(
            building=self.building, scenario=self.scenario, ground_truth=self.ground_truth,
            timeline_rows=(
                {"scenario_id": "scn-1", "simulation_time": 5.0, "current_fire_zones": "zone-d"},
            ),
        )
        with_fire = generate_policy(inputs_with_timeline)
        stair_safe_status = next(
            d["status"] for d in with_fire.stair_decisions if d["stair_id"] == "stair-safe"
        )

        # zone-d/stair-safe are connected -- with zone-d now on fire per
        # the latest Timeline Dataset row, stair-safe should be avoided.
        self.assertEqual(stair_safe_status, AVOID)

    def test_empty_timeline_rows_behaves_identically_to_omitting_them(self):

        explicit_empty = DecisionInputs(
            building=self.building, scenario=self.scenario, ground_truth=self.ground_truth,
            timeline_rows=(),
        )
        omitted = DecisionInputs(
            building=self.building, scenario=self.scenario, ground_truth=self.ground_truth,
        )

        self.assertEqual(
            generate_policy(explicit_empty).to_dict(), generate_policy(omitted).to_dict(),
        )

    def test_determinism_across_repeated_calls(self):

        inputs = DecisionInputs(
            building=self.building, scenario=self.scenario, ground_truth=self.ground_truth,
        )

        first = generate_policy(inputs)
        second = generate_policy(inputs)

        self.assertEqual(first.to_dict(), second.to_dict())


class SerializationTests(unittest.TestCase):

    def test_round_trip_preserves_all_fields(self):

        building = make_building()
        scenario = make_scenario(building)
        ground_truth = make_ground_truth()

        policy = generate_policy(
            DecisionInputs(building=building, scenario=scenario, ground_truth=ground_truth),
        )

        restored = DecisionPolicy.from_dict(policy.to_dict())

        self.assertEqual(policy.to_dict(), restored.to_dict())
        self.assertEqual(policy, restored)

    def test_from_dict_defaults_missing_optional_fields(self):

        minimal = DecisionPolicy.from_dict({"scenario_id": "scn-x"})

        self.assertEqual(minimal.zone_decisions, ())
        self.assertEqual(minimal.rescue_order, ())
        self.assertEqual(minimal.firefighter_deployment_priority, ())


if __name__ == "__main__":
    unittest.main()
