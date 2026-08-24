import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from scenario_definition import (
    EngineeringConstraints,
    FireDefinition,
    FixedValue,
    OccupantDefinition,
    ScenarioDefinition,
    UniformRange,
    WeightedOptions,
)

from campaign_feasibility import compute_exact_candidate_validity


# =====================================================
# Fixtures
# =====================================================


def _fire(**overrides):

    defaults = dict(growth_parameter_distribution=FixedValue(200.0))
    defaults.update(overrides)
    return FireDefinition(**defaults)


def _two_zone_building():

    door = Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")
    floor = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="R1", x=0.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-2", name="R2", x=10.0, y=0.0, width=8.0, height=8.0),
        ],
        doors=[door],
        exits=[Exit(id="exit-1", zone_id="zone-2")],
    )
    return Building(name="Two Zone", id="b-1", floors=[floor])


# =====================================================
# 1/2 -- deterministic reachable / unreachable occupancy
# =====================================================


class DeterministicOccupancyTests(unittest.TestCase):

    def test_deterministic_reachable_occupancy_is_fully_valid(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": FixedValue(1)},
                behaviour_profile_distribution={"zone-1": FixedValue("Staff_Default")},
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertAlmostEqual(result.p_valid, 1.0, places=9)
        self.assertTrue(result.exact)

    def test_deterministic_unreachable_occupancy_is_fully_invalid(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": FixedValue(1)},
                behaviour_profile_distribution={"zone-1": FixedValue("Staff_Default")},
            ),
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": FixedValue("LOCKED")},
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertAlmostEqual(result.p_valid, 0.0, places=9)
        self.assertTrue(result.exact)


# =====================================================
# 3 -- probabilistic occupancy including zero
# =====================================================


class ProbabilisticOccupancyIncludingZeroTests(unittest.TestCase):

    def test_unreachable_zone_with_positive_probability_of_zero_occupants(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": WeightedOptions({0: 0.3, 2: 0.7})},
                behaviour_profile_distribution={"zone-1": FixedValue("Staff_Default")},
            ),
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": FixedValue("LOCKED")},
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        # zone-1 is unreachable (door locked) -- the candidate is
        # valid iff zone-1 happens to receive zero occupants this
        # attempt, exactly P(0) = 0.3.
        self.assertAlmostEqual(result.p_valid, 0.3, places=9)
        self.assertAlmostEqual(result.p_invalid, 0.7, places=9)
        self.assertTrue(result.exact)

    def test_discrete_integer_range_occupancy_p_zero_matches_hand_derivation(self):

        # UniformRange(0, 3, discrete=True) -- rng.randint(0,3) is
        # uniform over {0,1,2,3} -- P(zero) = 1/4 exactly.
        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": UniformRange(0, 3, discrete=True)},
                behaviour_profile_distribution={"zone-1": FixedValue("Staff_Default")},
            ),
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": FixedValue("LOCKED")},
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertAlmostEqual(result.p_valid, 0.25, places=9)
        self.assertTrue(result.exact)

    def test_continuous_range_occupancy_p_zero_matches_hand_derivation(self):

        # UniformRange(0.0, 2.0, discrete=False) continuous -- P(zero)
        # = P(round(x) <= 0) = P(x < 0.5) = 0.5 / 2.0 = 0.25 exactly
        # (an interval-length ratio, computed analytically, not
        # sampled).
        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": UniformRange(0.0, 2.0, discrete=False)},
                behaviour_profile_distribution={"zone-1": FixedValue("Staff_Default")},
            ),
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": FixedValue("LOCKED")},
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertAlmostEqual(result.p_valid, 0.25, places=9)
        self.assertTrue(result.exact)


# =====================================================
# 4 -- mixed reachable and unreachable zones
# =====================================================


class MixedReachabilityTests(unittest.TestCase):

    def test_reachable_zone_occupancy_never_reduces_validity(self):

        zones = [
            Zone(id="zone-1", name="R1", x=0, y=0, width=8, height=8),
            Zone(id="zone-2", name="R2", x=10, y=0, width=8, height=8),
            Zone(id="zone-3", name="R3", x=20, y=0, width=8, height=8),
        ]
        doors = [Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")]
        exits = [Exit(id="exit-1", zone_id="zone-2"), Exit(id="exit-2", zone_id="zone-3")]
        floor = Floor(name="Ground", id="floor-1", zones=zones, doors=doors, exits=exits)
        building = Building(name="Mixed", id="b-mixed", floors=[floor])

        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=OccupantDefinition(
                occupancy_distribution={
                    "zone-1": WeightedOptions({0: 0.4, 1: 0.6}),  # unreachable (door locked)
                    "zone-3": FixedValue(1),  # always reachable directly via exit-2
                },
                behaviour_profile_distribution={
                    "zone-1": FixedValue("Staff_Default"), "zone-3": FixedValue("Staff_Default"),
                },
            ),
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": FixedValue("LOCKED")},
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        # zone-3 is always occupied and always reachable -- contributes
        # NO invalidity, regardless of its guaranteed presence. Only
        # zone-1 (unreachable) needs to be empty: P(valid) = P(zone-1=0)
        # = 0.4 exactly, unreduced by zone-3's own occupancy.
        self.assertAlmostEqual(result.p_valid, 0.4, places=9)


# =====================================================
# 5 -- multiple independently sampled uncertain zones
# =====================================================


class IndependentZonesTests(unittest.TestCase):

    def test_two_independent_unreachable_zones_combine_as_a_product(self):

        zones = [
            Zone(id="zone-X", name="X", x=0, y=0, width=8, height=8),
            Zone(id="zone-Y", name="Y", x=0, y=20, width=8, height=8),
            Zone(id="zone-exitX", name="EX", x=10, y=0, width=8, height=8),
            Zone(id="zone-exitY", name="EY", x=10, y=20, width=8, height=8),
        ]
        doors = [
            Door(id="door-X", normally_open=True, zone_a_id="zone-X", zone_b_id="zone-exitX"),
            Door(id="door-Y", normally_open=True, zone_a_id="zone-Y", zone_b_id="zone-exitY"),
        ]
        exits = [Exit(id="exit-X", zone_id="zone-exitX"), Exit(id="exit-Y", zone_id="zone-exitY")]
        floor = Floor(name="Ground", id="floor-1", zones=zones, doors=doors, exits=exits)
        building = Building(name="Multi", id="b-multi", floors=[floor])

        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-X"})),
            occupant=OccupantDefinition(
                occupancy_distribution={
                    "zone-X": WeightedOptions({0: 0.2, 1: 0.8}),
                    "zone-Y": WeightedOptions({0: 0.5, 1: 0.5}),
                },
                behaviour_profile_distribution={
                    "zone-X": FixedValue("Staff_Default"), "zone-Y": FixedValue("Staff_Default"),
                },
            ),
            engineering=EngineeringConstraints(
                door_state_distribution={
                    "door-X": FixedValue("LOCKED"), "door-Y": FixedValue("LOCKED"),
                },
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        # Both zones unreachable (their own doors locked), independent
        # -- P(valid) = P(X=0) x P(Y=0) = 0.2 x 0.5 = 0.1 exactly.
        self.assertAlmostEqual(result.p_valid, 0.2 * 0.5, places=9)


# =====================================================
# 6 -- engineering-state + occupancy interaction
# =====================================================


class EngineeringStateOccupancyInteractionTests(unittest.TestCase):

    def test_weighted_combination_of_door_state_and_occupancy(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": WeightedOptions({0: 0.4, 1: 0.6})},
                behaviour_profile_distribution={"zone-1": FixedValue("Staff_Default")},
            ),
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": WeightedOptions({"OPEN": 0.7, "LOCKED": 0.3})},
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        # If door open (0.7): zone-1 reachable -> valid regardless of
        # occupancy -> contributes 0.7.
        # If door locked (0.3): zone-1 unreachable -> valid only if
        # zone-1 gets zero occupants (0.4) -> contributes 0.3*0.4=0.12.
        # Total: 0.7 + 0.12 = 0.82 exactly.
        self.assertAlmostEqual(result.p_valid, 0.82, places=9)
        self.assertEqual(result.total_states_considered, 2)


# =====================================================
# 7 -- shared infrastructure regression (Phase 2A's own central
# correctness test, re-verified unaffected by Phase 2B)
# =====================================================


class SharedInfrastructureRegressionTests(unittest.TestCase):

    def test_shared_bottleneck_door_still_evaluated_jointly_not_multiplied(self):

        zones = [
            Zone(id="zone-A", name="A", x=0.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-B", name="B", x=0.0, y=20.0, width=8.0, height=8.0),
            Zone(id="zone-hub", name="Hub", x=10.0, y=10.0, width=8.0, height=8.0),
            Zone(id="zone-exitroom", name="ExitRoom", x=20.0, y=10.0, width=8.0, height=8.0),
        ]
        doors = [
            Door(id="door-A-hub", normally_open=True, zone_a_id="zone-A", zone_b_id="zone-hub"),
            Door(id="door-B-hub", normally_open=True, zone_a_id="zone-B", zone_b_id="zone-hub"),
            Door(
                id="door-shared", normally_open=True,
                zone_a_id="zone-hub", zone_b_id="zone-exitroom",
            ),
        ]
        floor = Floor(
            name="Ground", id="floor-1", zones=zones, doors=doors,
            exits=[Exit(id="exit-1", zone_id="zone-exitroom")],
        )
        building = Building(name="Shared Bottleneck", id="b-shared", floors=[floor])

        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-A"})),
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-A": FixedValue(1), "zone-B": FixedValue(1)},
                behaviour_profile_distribution={
                    "zone-A": FixedValue("Staff_Default"), "zone-B": FixedValue("Staff_Default"),
                },
            ),
            engineering=EngineeringConstraints(
                door_state_distribution={
                    "door-shared": WeightedOptions({"OPEN": 0.5, "LOCKED": 0.5}),
                },
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertAlmostEqual(result.p_valid, 0.5, places=9)
        self.assertNotAlmostEqual(result.p_valid, 0.25, places=9)


# =====================================================
# 8 -- zero occupants globally
# =====================================================


class ZeroOccupantsGloballyTests(unittest.TestCase):

    def test_zero_occupants_globally_ignores_navigation_entirely(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(),
            occupant=OccupantDefinition(),
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": FixedValue("LOCKED")},
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        # No occupancy_distribution entries at all -- NAVIGATION never
        # runs (validate_navigation()'s own early return); a locked
        # door is therefore irrelevant to validity.
        self.assertAlmostEqual(result.p_valid, 1.0, places=9)
        self.assertEqual(result.analyzed_zone_ids, frozenset())

    def test_zero_occupants_globally_with_min_open_exits_still_applies_structurally(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(),
            occupant=OccupantDefinition(),
            engineering=EngineeringConstraints(
                exit_state_distribution={"exit-1": WeightedOptions({True: 0.6, False: 0.4})},
                min_open_exits=1,
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertAlmostEqual(result.p_valid, 0.6, places=9)


# =====================================================
# 9 -- exactness status
# =====================================================


class ExactnessStatusTests(unittest.TestCase):

    def test_fully_exact_when_all_relevant_uncertainty_is_covered(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": WeightedOptions({0: 0.3, 1: 0.7})},
                behaviour_profile_distribution={"zone-1": FixedValue("Staff_Default")},
            ),
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": WeightedOptions({"OPEN": 0.5, "LOCKED": 0.5})},
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertTrue(result.exact)
        self.assertTrue(result.exact_for_analyzed_dimensions)
        self.assertEqual(result.unresolved_occupancy_zone_ids, frozenset())

    def test_genuinely_unresolvable_occupancy_distribution_is_not_mislabeled_exact(self):

        # A degenerate WeightedOptions whose weights sum to <= 0 cannot
        # be resolved to any exact P(zero) (sample() itself would be in
        # undefined territory) -- this module must disclose it, never
        # silently guess a value or falsely claim exactness.
        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": WeightedOptions({0: 0.0, 1: 0.0})},
                behaviour_profile_distribution={"zone-1": FixedValue("Staff_Default")},
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertFalse(result.exact)
        self.assertIn("zone-1", result.unresolved_occupancy_zone_ids)
        self.assertTrue(any("occupancy" in w.lower() for w in result.warnings))


# =====================================================
# 10 -- state-space protection: occupancy uncertainty must never
# multiply into the enumerated engineering-state space
# =====================================================


class OccupancyStateSpaceProtectionTests(unittest.TestCase):

    def test_many_independent_uncertain_occupancy_zones_do_not_explode_state_count(self):

        # 20 independent, unreachable, uncertain-occupancy zones, with
        # NO engineering-state uncertainty at all (every connecting
        # door is FixedValue LOCKED, deterministic). If occupancy were
        # (incorrectly) enumerated as a Cartesian product, this would
        # be 2^20 combinations; the correct closed-form product
        # approach keeps total_states_considered at exactly 1.
        zones = [Zone(id="zone-hub", name="Hub", x=0.0, y=0.0, width=8.0, height=8.0)]
        doors = []
        exits = [Exit(id="exit-1", zone_id="zone-hub")]
        occupancy_distribution = {}
        behaviour_distribution = {}
        door_state_distribution = {}

        for i in range(20):

            zone_id = f"zone-{i}"
            door_id = f"door-{i}"

            zones.append(Zone(id=zone_id, name=f"Z{i}", x=float(10 + i * 10), y=0.0, width=8.0, height=8.0))
            doors.append(Door(id=door_id, normally_open=True, zone_a_id="zone-hub", zone_b_id=zone_id))

            occupancy_distribution[zone_id] = WeightedOptions({0: 0.5, 1: 0.5})
            behaviour_distribution[zone_id] = FixedValue("Staff_Default")
            door_state_distribution[door_id] = FixedValue("LOCKED")

        floor = Floor(name="Ground", id="floor-1", zones=zones, doors=doors, exits=exits)
        building = Building(name="Star", id="b-star", floors=[floor])

        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-hub"})),
            occupant=OccupantDefinition(
                occupancy_distribution=occupancy_distribution,
                behaviour_profile_distribution=behaviour_distribution,
            ),
            engineering=EngineeringConstraints(door_state_distribution=door_state_distribution),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertEqual(result.total_states_considered, 1)
        self.assertTrue(result.exact)
        # All 20 zones unreachable (their own doors locked) and
        # independent -- P(valid) = 0.5^20 exactly.
        self.assertAlmostEqual(result.p_valid, 0.5 ** 20, places=12)


# =====================================================
# 11 -- existing Phase 2A regression (cross-check; the full Phase 2A
# suite, tests/test_campaign_feasibility_exact_validity.py, is run
# separately and must also pass unchanged)
# =====================================================


class Phase2ARegressionCrossCheckTests(unittest.TestCase):

    def test_fully_deterministic_case_from_phase_2a_is_unaffected(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": FixedValue(1)},
                behaviour_profile_distribution={"zone-1": FixedValue("Staff_Default")},
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertEqual(result.p_valid, 1.0)
        self.assertEqual(result.p_invalid, 0.0)
        self.assertTrue(result.exact)


if __name__ == "__main__":
    unittest.main()
