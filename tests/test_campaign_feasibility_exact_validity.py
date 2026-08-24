import unittest

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
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

from campaign_feasibility import CandidateValidityResult, compute_exact_candidate_validity
from campaign_feasibility.exact_validity import DEFAULT_MAX_ENUMERATED_STATES


# =====================================================
# Fixtures
# =====================================================


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


def _occupant_only(zone_id, count=1):

    return OccupantDefinition(
        occupancy_distribution={zone_id: FixedValue(count)},
        behaviour_profile_distribution={zone_id: FixedValue("Staff_Default")},
    )


def _fire(**overrides):

    defaults = dict(growth_parameter_distribution=FixedValue(200.0))
    defaults.update(overrides)
    return FireDefinition(**defaults)


def _make_chain_building(zone_count=5):

    zones = [
        Zone(id=f"zone-{i}", name=f"R{i}", x=float(i * 10), y=0.0, width=8.0, height=8.0)
        for i in range(1, zone_count + 1)
    ]
    doors = [
        Door(id=f"door-{i}", normally_open=True, zone_a_id=f"zone-{i}", zone_b_id=f"zone-{i + 1}")
        for i in range(1, zone_count)
    ]
    exits = [Exit(id="exit-1", zone_id=f"zone-{zone_count}")]
    floor = Floor(name="Ground", id="floor-1", zones=zones, doors=doors, exits=exits)
    return Building(name="Chain", id="chain-1", floors=[floor])


# =====================================================
# 1/2 -- fully deterministic valid / invalid
# =====================================================


class DeterministicCaseTests(unittest.TestCase):

    def test_fully_deterministic_valid_case(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=_occupant_only("zone-1"),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertEqual(result.p_valid, 1.0)
        self.assertEqual(result.p_invalid, 0.0)
        self.assertTrue(result.exact)
        self.assertTrue(result.exact_for_analyzed_dimensions)
        self.assertEqual(result.total_states_considered, 1)
        self.assertFalse(result.state_space_too_large)

    def test_fully_deterministic_invalid_case(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=_occupant_only("zone-1"),
            engineering=EngineeringConstraints(
                door_state_distribution={"door-1": FixedValue("LOCKED")},
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertEqual(result.p_valid, 0.0)
        self.assertEqual(result.p_invalid, 1.0)
        self.assertTrue(result.exact)


# =====================================================
# 3 -- simple probabilistic door, analytically known
# =====================================================


class ProbabilisticDoorTests(unittest.TestCase):

    def test_simple_weighted_door_matches_analytical_probability(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=_occupant_only("zone-1"),
            engineering=EngineeringConstraints(
                door_state_distribution={
                    "door-1": WeightedOptions({"OPEN": 0.6, "LOCKED": 0.4}),
                },
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertAlmostEqual(result.p_valid, 0.6, places=9)
        self.assertAlmostEqual(result.p_invalid, 0.4, places=9)
        self.assertEqual(result.total_states_considered, 2)
        self.assertTrue(result.exact)

    def test_three_way_weighted_door_collapses_correctly(self):

        # OPEN and CLOSED are both traversable; only LOCKED is not --
        # the exact P(valid) must equal the COMBINED weight of OPEN+CLOSED,
        # not just OPEN alone.
        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=_occupant_only("zone-1"),
            engineering=EngineeringConstraints(
                door_state_distribution={
                    "door-1": WeightedOptions({"OPEN": 0.5, "CLOSED": 0.3, "LOCKED": 0.2}),
                },
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertAlmostEqual(result.p_valid, 0.8, places=9)


# =====================================================
# 4 -- fire-origin probability, mixed weights
# =====================================================


class FireOriginProbabilityTests(unittest.TestCase):

    def test_weighted_fire_preference_with_mixed_safe_and_lethal_origins(self):

        building = _make_chain_building(zone_count=3)
        definition = ScenarioDefinition(
            fire=_fire(
                ignition_zone_preference=WeightedOptions({"zone-2": 0.3, "zone-3": 0.7}),
            ),
            occupant=_occupant_only("zone-1"),
        )

        # zone-1's only path to Outside is zone-1 -> zone-2 -> zone-3 -> exit.
        # Excluding zone-2 or zone-3 both disconnect zone-1 -- both LETHAL.
        result = compute_exact_candidate_validity(building, definition)

        self.assertAlmostEqual(result.p_valid, 0.0, places=9)
        self.assertAlmostEqual(result.p_invalid, 1.0, places=9)

    def test_uniform_fire_with_mixed_safe_and_lethal_origins_matches_hand_derivation(self):

        # 5-zone chain, only zone-2 occupied. Hand-derived in the
        # implementation's own verification: fire=zone-1 or zone-2(self)
        # -> SAFE; fire=zone-3/4/5 -> LETHAL. P(valid) = 2/5 = 0.4 exactly.
        building = _make_chain_building(zone_count=5)
        definition = ScenarioDefinition(
            fire=_fire(),
            occupant=_occupant_only("zone-2"),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertAlmostEqual(result.p_valid, 0.4, places=9)
        self.assertAlmostEqual(result.p_invalid, 0.6, places=9)


# =====================================================
# 5 -- joint shared-route case (the central correctness requirement)
# =====================================================


class JointEvaluationTests(unittest.TestCase):

    def test_shared_bottleneck_door_is_evaluated_jointly_not_multiplied(self):

        # zone-A and zone-B both occupied, both must pass through the
        # SAME shared door to reach the exit. A naive per-zone
        # calculation would compute P(A ok) x P(B ok) = 0.5 x 0.5 =
        # 0.25. The correct joint answer is exactly 0.5, since both
        # zones' fate is determined by the SAME single random draw.
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
        self.assertEqual(result.analyzed_zone_ids, frozenset({"zone-A", "zone-B"}))


# =====================================================
# 6 -- door + exit / door + stair interaction
# =====================================================


class CombinedEngineeringStateTests(unittest.TestCase):

    def test_door_and_exit_interaction(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=_occupant_only("zone-1"),
            engineering=EngineeringConstraints(
                door_state_distribution={
                    "door-1": WeightedOptions({"OPEN": 0.8, "LOCKED": 0.2}),
                },
                exit_state_distribution={
                    "exit-1": WeightedOptions({True: 0.9, False: 0.1}),
                },
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        # Independent objects -- exact joint probability is the product.
        self.assertAlmostEqual(result.p_valid, 0.8 * 0.9, places=9)
        self.assertEqual(result.total_states_considered, 4)

    def test_door_and_stair_interaction(self):

        floor1 = Floor(
            name="Ground", id="floor-1",
            zones=[Zone(id="zone-1", name="R1", x=0.0, y=0.0, width=8.0, height=8.0)],
            stairs=[Staircase(id="stair-1", from_zone_id="zone-1", to_zone_id="zone-2", to_floor_id="floor-2")],
        )
        floor2 = Floor(
            name="Upper", id="floor-2",
            zones=[Zone(id="zone-2", name="R2", x=0.0, y=0.0, width=8.0, height=8.0)],
            exits=[Exit(id="exit-1", zone_id="zone-2")],
        )
        building = Building(name="Two Floor", id="b-stair", floors=[floor1, floor2])

        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=_occupant_only("zone-1"),
            engineering=EngineeringConstraints(
                stair_state_distribution={
                    "stair-1": WeightedOptions({"AVAILABLE": 0.7, "CLOSED": 0.3}),
                },
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertAlmostEqual(result.p_valid, 0.7, places=9)


# =====================================================
# 7 -- min_open_exits
# =====================================================


class MinOpenExitsTests(unittest.TestCase):

    def test_min_open_exits_changes_exact_validity_probability(self):

        zones = [
            Zone(id="zone-1", name="R1", x=0.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-2", name="R2", x=10.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-3", name="R3", x=20.0, y=0.0, width=8.0, height=8.0),
        ]
        doors = [
            Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2"),
            Door(id="door-2", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-3"),
        ]
        exits = [Exit(id="exit-2", zone_id="zone-2"), Exit(id="exit-3", zone_id="zone-3")]
        floor = Floor(name="Ground", id="floor-1", zones=zones, doors=doors, exits=exits)
        building = Building(name="Two Exit", id="b-two-exit", floors=[floor])

        exit_distribution = {
            "exit-2": WeightedOptions({True: 0.7, False: 0.3}),
            "exit-3": WeightedOptions({True: 0.7, False: 0.3}),
        }

        definition_min1 = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=_occupant_only("zone-1"),
            engineering=EngineeringConstraints(
                exit_state_distribution=exit_distribution, min_open_exits=1,
            ),
        )
        definition_min2 = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=_occupant_only("zone-1"),
            engineering=EngineeringConstraints(
                exit_state_distribution=exit_distribution, min_open_exits=2,
            ),
        )

        result_min1 = compute_exact_candidate_validity(building, definition_min1)
        result_min2 = compute_exact_candidate_validity(building, definition_min2)

        # min_open_exits=1: valid whenever AT LEAST ONE of the two
        # independent 70%-open exits is open and reachable ->
        # 1 - (0.3 * 0.3) = 0.91.
        self.assertAlmostEqual(result_min1.p_valid, 0.91, places=9)

        # min_open_exits=2: valid only when BOTH are open -> 0.7*0.7=0.49.
        self.assertAlmostEqual(result_min2.p_valid, 0.49, places=9)

        self.assertLess(result_min2.p_valid, result_min1.p_valid)

    def test_min_open_exits_structural_check_applies_even_with_zero_occupants(self):

        # No occupancy_distribution at all -- NAVIGATION never runs
        # (validate_navigation()'s own early return), but the
        # STRUCTURAL MIN_OPEN_EXITS_UNSATISFIED check has no such
        # guard and still applies.
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
        self.assertEqual(result.analyzed_zone_ids, frozenset())


# =====================================================
# 8 -- LETHAL fire-origin pruning
# =====================================================


class LethalPruningTests(unittest.TestCase):

    def test_lethal_fire_origin_pruning_is_mathematically_correct(self):

        # zone-2 occupied in a 5-chain, with an ADDITIONAL uncertain
        # door on zone-2's own reachable side, to prove pruning doesn't
        # corrupt the probability even when engineering-state
        # enumeration IS needed for the SAFE branches.
        building = _make_chain_building(zone_count=5)
        definition = ScenarioDefinition(
            fire=_fire(),
            occupant=_occupant_only("zone-2"),
            engineering=EngineeringConstraints(
                door_state_distribution={
                    "door-1": WeightedOptions({"OPEN": 0.5, "LOCKED": 0.5}),
                },
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertTrue(result.pruned)

        # SAFE fire zones for zone-2: zone-1 (door-1 is on THIS path,
        # since zone-1 is on the OTHER side of door-1 from zone-2 --
        # door-1 connects zone-1<->zone-2, not on zone-2's OWN route to
        # the exit, so it must NOT affect this branch) and zone-2
        # itself (self-exclusion, always safe). LETHAL: zone-3/4/5.
        # door-1 sits between zone-1 and zone-2, off zone-2's own path
        # to the exit (zone2->zone3->zone4->zone5->exit) -- it should
        # be pruned as topologically irrelevant to zone-2's own
        # reachability and therefore NOT enumerated at all, leaving
        # P(valid) identical to the pruning-free hand-derivation
        # (0.4, Section FireOriginProbabilityTests) regardless of
        # door-1's own probability.
        self.assertAlmostEqual(result.p_valid, 0.4, places=9)

    def test_pruning_avoids_enumerating_the_lethal_branch(self):

        # A LETHAL-heavy configuration with several independently
        # uncertain doors -- verifies that (a) LETHAL fire branches are
        # pruned (never enumerated at all, `pruned=True`) while (b) the
        # remaining SAFE branch's own engineering-state space is still
        # enumerated exactly and correctly. Only fire=zone-1 (self,
        # vacuously safe) is a SAFE branch here -- zone-2..5 are all
        # LETHAL (FireOriginProbabilityTests' own hand-derivation,
        # which holds regardless of door probabilities: excluding any
        # of zone-2/3/4/5 as a NODE removes zone-1's only route
        # topologically, independent of any door's own state).
        #
        # This module's own relevant-edge prune is a weaker (and much
        # cheaper), topological-connectivity-only test -- "is this
        # edge in the same connected component as an occupied zone" --
        # not a tight minimal-cut relevant-edge-set, so door-2/3/4
        # (all on zone-1's own single-chain path, hence in the same
        # connected component) remain part of the SAFE branch's own
        # enumeration; the exact probability this test checks proves
        # that inclusion doesn't corrupt correctness.
        building = _make_chain_building(zone_count=5)

        door_distribution = {f"door-{i}": WeightedOptions({"OPEN": 0.9, "LOCKED": 0.1}) for i in range(2, 5)}

        definition = ScenarioDefinition(
            fire=_fire(),
            occupant=_occupant_only("zone-1"),
            engineering=EngineeringConstraints(door_state_distribution=door_distribution),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertTrue(result.pruned)
        # SAFE branch weight (fire=zone-1 only) is 0.2 (1 of 5 uniform
        # fire zones); within that branch, zone-1 must traverse
        # door-1 (degenerate, always open) then door-2/3/4 (each 0.9
        # traversable, independent) to reach the exit:
        # P(valid) = 0.2 * 0.9 * 0.9 * 0.9 = 0.1458 exactly.
        self.assertAlmostEqual(result.p_valid, 0.2 * 0.9 ** 3, places=9)
        # 3 independently-uncertain doors in the one SAFE branch -> 8
        # combinations -- not 8 x 5 (every fire zone), proving the
        # LETHAL branches genuinely were never enumerated.
        self.assertEqual(result.total_states_considered, 8)


# =====================================================
# 9 -- state-space limit
# =====================================================


class StateSpaceLimitTests(unittest.TestCase):

    def test_oversized_enumeration_is_declined_not_executed(self):

        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=_occupant_only("zone-1"),
            engineering=EngineeringConstraints(
                door_state_distribution={
                    "door-1": WeightedOptions({"OPEN": 0.5, "LOCKED": 0.5}),
                },
            ),
        )

        result = compute_exact_candidate_validity(building, definition, max_states=0)

        self.assertTrue(result.state_space_too_large)
        self.assertFalse(result.exact)
        self.assertFalse(result.exact_for_analyzed_dimensions)
        self.assertIsNone(result.p_valid)
        self.assertEqual(result.total_states_considered, 0)
        self.assertTrue(result.warnings)

    def test_default_limit_is_a_named_documented_constant(self):

        self.assertIsInstance(DEFAULT_MAX_ENUMERATED_STATES, int)
        self.assertGreater(DEFAULT_MAX_ENUMERATED_STATES, 0)


# =====================================================
# 10 -- probability invariants
# =====================================================


class ProbabilityInvariantTests(unittest.TestCase):

    def test_valid_and_invalid_mass_sum_to_one(self):

        building = _make_chain_building(zone_count=4)
        definition = ScenarioDefinition(
            fire=_fire(),
            occupant=_occupant_only("zone-2"),
            engineering=EngineeringConstraints(
                door_state_distribution={
                    "door-3": WeightedOptions({"OPEN": 0.6, "LOCKED": 0.4}),
                },
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertAlmostEqual(result.p_valid + result.p_invalid, 1.0, places=9)
        self.assertEqual(result.valid_states + result.invalid_states, result.total_states_considered)

    def test_discrete_uniform_range_occupancy_is_now_resolved_exactly_by_phase_2b(self):

        # Phase 2A (before Phase 2B existed) could not resolve a
        # UniformRange occupancy distribution to an exact probability
        # and disclosed it as unresolved. Phase 2B closes this gap
        # (see _p_zero_occupancy()) -- this is a genuine capability
        # improvement, re-verified here rather than left asserting the
        # old, now-superseded limitation. See
        # ProbabilityInvariantTests (Phase 2B section, below) for the
        # dedicated coverage of a genuinely UNRESOLVABLE occupancy
        # distribution.
        building = _two_zone_building()
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-1"})),
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": UniformRange(0, 3, discrete=True)},
                behaviour_profile_distribution={"zone-1": FixedValue("Staff_Default")},
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        self.assertTrue(result.exact)
        self.assertEqual(result.unresolved_occupancy_zone_ids, frozenset())
        # zone-1 is reachable (doors normally open) -- occupancy
        # uncertainty is therefore irrelevant to validity here
        # (occupied or not, zone-1 can always escape).
        self.assertAlmostEqual(result.p_valid, 1.0, places=9)


# =====================================================
# 11 -- existing Phase 1 regression (imported directly to prove the
# Phase 1 public API/behavior is completely unaffected by this module's
# existence)
# =====================================================


class Phase1RegressionTests(unittest.TestCase):

    def test_phase1_analyze_campaign_feasibility_still_works_unchanged(self):

        from campaign_feasibility import analyze_campaign_feasibility

        building = _make_chain_building(zone_count=5)
        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-2"})),
            occupant=_occupant_only("zone-1"),
        )

        report = analyze_campaign_feasibility(building, definition)

        self.assertTrue(report.has_errors)
        self.assertEqual(report.zone_results[0].status, "ERROR")


if __name__ == "__main__":
    unittest.main()
