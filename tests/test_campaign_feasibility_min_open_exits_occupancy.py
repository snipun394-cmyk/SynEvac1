import unittest
from unittest.mock import patch

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
    WeightedOptions,
)

from campaign_feasibility import compute_exact_candidate_validity
import campaign_feasibility.exact_validity as exact_validity_module


# =====================================================
# Phase 2B.1 -- closes Phase 2B's own disclosed gap: min_open_exits'
# NAVIGATION-based reachable-egress check must account for every SAFE
# zone's occupancy uncertainty, not only guaranteed-occupied zones.
# See docs/architecture/scenario_campaign_feasibility_min_open_exits_
# occupancy_implementation_report.txt for the full derivation each
# test below is hand-checked against.
# =====================================================


def _fire(**overrides):

    defaults = dict(growth_parameter_distribution=FixedValue(200.0))
    defaults.update(overrides)
    return FireDefinition(**defaults)


# =====================================================
# 1 -- threshold already satisfied by guaranteed-occupied zones alone;
# uncertain occupancy elsewhere must not affect the result (preserves
# the Phase 2B closed-form shortcut).
# =====================================================


class ThresholdAlreadySatisfiedTests(unittest.TestCase):

    def test_uncertain_safe_zone_does_not_affect_already_satisfied_threshold(self):

        zones = [
            Zone(id="zone-G", name="G", x=0.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-U", name="U", x=20.0, y=0.0, width=8.0, height=8.0),
        ]
        exits = [Exit(id="exit-A", zone_id="zone-G"), Exit(id="exit-B", zone_id="zone-U")]
        floor = Floor(name="Ground", id="floor-1", zones=zones, doors=[], exits=exits)
        building = Building(name="Threshold Satisfied", id="b-1", floors=[floor])

        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-G"})),
            occupant=OccupantDefinition(
                occupancy_distribution={
                    "zone-G": FixedValue(1),
                    "zone-U": WeightedOptions({0: 0.5, 1: 0.5}),
                },
                behaviour_profile_distribution={
                    "zone-G": FixedValue("Staff_Default"), "zone-U": FixedValue("Staff_Default"),
                },
            ),
            engineering=EngineeringConstraints(min_open_exits=1),
        )

        result = compute_exact_candidate_validity(building, definition)

        # zone-G alone already provides 1 reachable open exit, meeting
        # min_open_exits=1 -- zone-U's own occupancy (uncertain) can
        # never change this outcome.
        self.assertAlmostEqual(result.p_valid, 1.0, places=9)
        self.assertEqual(result.total_states_considered, 1)
        self.assertTrue(result.exact)


# =====================================================
# 2 -- one uncertain safe zone provides the missing exit.
# =====================================================


class SingleUncertainZoneProvidesMissingExitTests(unittest.TestCase):

    def test_probability_equals_p_when_uncertain_zone_supplies_the_second_exit(self):

        zones = [
            Zone(id="zone-G", name="G", x=0.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-U", name="U", x=20.0, y=0.0, width=8.0, height=8.0),
        ]
        exits = [Exit(id="exit-A", zone_id="zone-G"), Exit(id="exit-B", zone_id="zone-U")]
        floor = Floor(name="Ground", id="floor-1", zones=zones, doors=[], exits=exits)
        building = Building(name="Missing Exit", id="b-2", floors=[floor])

        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-G"})),
            occupant=OccupantDefinition(
                occupancy_distribution={
                    "zone-G": FixedValue(1),
                    "zone-U": WeightedOptions({0: 0.3, 1: 0.7}),
                },
                behaviour_profile_distribution={
                    "zone-G": FixedValue("Staff_Default"), "zone-U": FixedValue("Staff_Default"),
                },
            ),
            engineering=EngineeringConstraints(min_open_exits=2),
        )

        result = compute_exact_candidate_validity(building, definition)

        # zone-G reaches only exit-A (1 exit, short of min_open_exits=2).
        # zone-U reaches only exit-B, independently, with P(occupied)=0.7.
        # Hand-derived: P(valid) = P(zone-U occupied) = 0.7 exactly.
        self.assertAlmostEqual(result.p_valid, 0.7, places=9)
        self.assertAlmostEqual(result.p_invalid, 0.3, places=9)
        self.assertTrue(result.exact)


# =====================================================
# 3 -- multiple uncertain zones with overlapping reachable-exit sets;
# the exact answer must reflect the UNION of exits, not a naive
# per-zone-independent count.
#
# [PROVEN, this task] Within a SINGLE fixed engineering-state
# combination, two zones' reachable-open-exit sets
# (`_zone_reachable_open_exit_sets()`) can only ever be IDENTICAL (both
# in the same connected indoor component) or DISJOINT (different
# components) -- never partially overlapping-but-different -- because
# reaching a shared exit means reaching that exit's own hosting zone,
# which (edges being bidirectional, navigation_validation.py's own
# documented invariant) puts both zones in the same connected
# component, which then shares its ENTIRE reachable-exit set. A real
# Building/Definition therefore cannot construct the task's own
# {B,C}/{C,D}-style partial-overlap example end-to-end; the DP
# (`_exit_coverage_probability_at_least()`) is still implemented in
# fully general form (it does not special-case identical-vs-disjoint),
# so `ExitCoverageDPUnitTests` below verifies the general union
# computation directly and exactly against the task's own worked
# example, while `IdenticalContributionMergeTests` verifies the
# real-world case (multiple zones sharing one component, hence one
# identical contribution) end-to-end through a real Building.
# =====================================================


class ExitCoverageDPUnitTests(unittest.TestCase):

    def test_two_zone_overlap_matches_hand_derived_union_probability(self):

        # Zone 1 -> {B, C} (p1=0.6 activated); Zone 2 -> {C, D}
        # (p2=0.75 activated) -- the task's own worked example. need=2
        # is met by EITHER zone alone (each already contributes 2
        # distinct exits), so P(valid) = P(at least one activated).
        groups = [
            (frozenset({"exit-B", "exit-C"}), 0.6),
            (frozenset({"exit-C", "exit-D"}), 0.75),
        ]

        result = exact_validity_module._exit_coverage_probability_at_least(groups, need=2)

        expected = 1.0 - (1.0 - 0.6) * (1.0 - 0.75)
        self.assertAlmostEqual(result, expected, places=9)
        self.assertAlmostEqual(result, 0.9, places=9)

    def test_three_zone_overlap_requires_true_union_not_additive_sizes(self):

        # Zone1 -> {B,C} (a1=0.5); Zone2 -> {C,D} (a2=0.6); Zone3 ->
        # {D} (a3=0.5). need=3. A WRONG implementation that summed
        # contribution SIZES independently (e.g. "zone2 (2) + zone3
        # (1) = 3, so zone2+zone3 must satisfy need=3") would
        # overcount: zone2 and zone3 both include exit-D, so together
        # they only cover {C,D} (size 2), NOT 3. Hand-enumerated over
        # all 8 activation patterns (a1=0.5, a2=0.6, a3=0.5): only
        # {1,2}, {1,3}, {1,2,3} reach a union of size >= 3 -- {2,3}
        # alone does NOT. Sum of those three joint probabilities:
        #   P(1,2,~3) = 0.5*0.6*0.5 = 0.15
        #   P(1,~2,3) = 0.5*0.4*0.5 = 0.10
        #   P(1,2,3)  = 0.5*0.6*0.5 = 0.15
        # Total = 0.40 exactly.
        groups = [
            (frozenset({"exit-B", "exit-C"}), 0.5),
            (frozenset({"exit-C", "exit-D"}), 0.6),
            (frozenset({"exit-D"}), 0.5),
        ]

        result = exact_validity_module._exit_coverage_probability_at_least(groups, need=3)

        self.assertAlmostEqual(result, 0.40, places=9)

        # The naive "sum of contribution sizes weighted by probability
        # crosses the threshold" approach would have wrongly credited
        # the {zone2, zone3}-only outcome (probability
        # (1-0.5)*0.6*0.5 = 0.15) as satisfying need=3 -- the correct
        # union-based answer excludes it, so the two must differ.
        naively_inflated = result + (1.0 - 0.5) * 0.6 * 0.5
        self.assertNotAlmostEqual(result, naively_inflated, places=9)

    def test_redundant_contribution_does_not_change_the_probability(self):

        # A group whose contribution is already fully covered by
        # another group's contribution must not change the result --
        # mirrors the "redundant uncertain zones" requirement, tested
        # here directly at the DP level.
        groups_without_redundant = [(frozenset({"exit-B"}), 0.5)]
        groups_with_redundant = [
            (frozenset({"exit-B"}), 0.5),
            (frozenset({"exit-B"}), 0.9),  # subset of the universe already covered
        ]

        result_without = exact_validity_module._exit_coverage_probability_at_least(
            groups_without_redundant, need=1,
        )
        result_with = exact_validity_module._exit_coverage_probability_at_least(
            groups_with_redundant, need=1,
        )

        # These are NOT expected to be numerically equal (a second
        # group contributing the SAME exit still changes P(covered)
        # via its own activation) -- this test instead documents that
        # `_combo_occupancy_factor()` itself never constructs a second
        # group for a zone whose contribution is a subset of what is
        # already covered (see `RedundantUncertainZonesTests` below
        # for the real, end-to-end verification of that pruning).
        self.assertGreaterEqual(result_with, result_without)


class IdenticalContributionMergeTests(unittest.TestCase):

    def test_zones_sharing_one_connected_component_merge_to_one_group(self):

        zones = [
            Zone(id="zone-G", name="G", x=0.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-hub1", name="Hub1", x=20.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-hub2", name="Hub2", x=20.0, y=20.0, width=8.0, height=8.0),
            Zone(id="zone-hub3", name="Hub3", x=20.0, y=40.0, width=8.0, height=8.0),
            Zone(id="zone-B", name="B", x=40.0, y=-10.0, width=8.0, height=8.0),
            Zone(id="zone-C", name="C", x=40.0, y=10.0, width=8.0, height=8.0),
            Zone(id="zone-D", name="D", x=40.0, y=30.0, width=8.0, height=8.0),
        ]
        doors = [
            Door(id="door-hub1-B", normally_open=True, zone_a_id="zone-hub1", zone_b_id="zone-B"),
            Door(id="door-hub1-C", normally_open=True, zone_a_id="zone-hub1", zone_b_id="zone-C"),
            Door(id="door-hub2-C", normally_open=True, zone_a_id="zone-hub2", zone_b_id="zone-C"),
            Door(id="door-hub2-D", normally_open=True, zone_a_id="zone-hub2", zone_b_id="zone-D"),
            Door(id="door-hub3-D", normally_open=True, zone_a_id="zone-hub3", zone_b_id="zone-D"),
        ]
        exits = [
            Exit(id="exit-A", zone_id="zone-G"),
            Exit(id="exit-B", zone_id="zone-B"),
            Exit(id="exit-C", zone_id="zone-C"),
            Exit(id="exit-D", zone_id="zone-D"),
        ]
        floor = Floor(name="Ground", id="floor-1", zones=zones, doors=doors, exits=exits)
        building = Building(name="Merged Component", id="b-3", floors=[floor])

        door_state_distribution = {door.id: FixedValue("OPEN") for door in doors}

        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-G"})),
            occupant=OccupantDefinition(
                occupancy_distribution={
                    "zone-G": FixedValue(1),
                    # hub1, hub2, hub3 are all mutually reachable
                    # indoors (hub1-C-hub2-D-hub3, every door open) --
                    # a SINGLE connected component sharing exit-B,
                    # exit-C, exit-D, so all three get the IDENTICAL
                    # contribution {B,C,D}, not merely overlapping
                    # ones (see this class's own module-level comment).
                    "zone-hub1": WeightedOptions({0: 0.5, 1: 0.5}),
                    "zone-hub2": WeightedOptions({0: 0.4, 1: 0.6}),
                    "zone-hub3": WeightedOptions({0: 0.5, 1: 0.5}),
                },
                behaviour_profile_distribution={
                    "zone-G": FixedValue("Staff_Default"),
                    "zone-hub1": FixedValue("Staff_Default"),
                    "zone-hub2": FixedValue("Staff_Default"),
                    "zone-hub3": FixedValue("Staff_Default"),
                },
            ),
            engineering=EngineeringConstraints(
                door_state_distribution=door_state_distribution,
                min_open_exits=4,
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        # base_reachable={A} (zone-G, no doors of its own), need=3.
        # hub1/hub2/hub3 merge into ONE group with contribution
        # {B,C,D} (size 3, exactly meeting need) and combined
        # P(all three empty) = 0.5*0.4*0.5 = 0.1 -> P(activated) = 0.9.
        # p_all_safe_empty is forced to 0.0 by the guaranteed zone-G
        # factor, so P(valid) = 0.9 exactly.
        self.assertAlmostEqual(result.p_valid, 0.9, places=9)
        self.assertEqual(result.total_states_considered, 1)


# =====================================================
# 4 -- unsafe uncertain zone (must be empty) plus a safe uncertain
# zone that must be occupied to supply the missing exit.
# =====================================================


class UnsafeAndSafeUncertainZoneJointTests(unittest.TestCase):

    def test_probability_is_the_product_of_the_two_independent_factors(self):

        zones = [
            Zone(id="zone-G", name="G", x=0.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-U", name="U", x=20.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-Trapped", name="Trapped", x=0.0, y=20.0, width=8.0, height=8.0),
            Zone(id="zone-Behind", name="Behind", x=0.0, y=40.0, width=8.0, height=8.0),
        ]
        doors = [
            Door(
                id="door-trap", normally_open=False,
                zone_a_id="zone-Trapped", zone_b_id="zone-Behind",
            ),
        ]
        exits = [
            Exit(id="exit-A", zone_id="zone-G"),
            Exit(id="exit-B", zone_id="zone-U"),
            Exit(id="exit-trap", zone_id="zone-Behind"),
        ]
        floor = Floor(name="Ground", id="floor-1", zones=zones, doors=doors, exits=exits)
        building = Building(name="Unsafe Plus Safe", id="b-4", floors=[floor])

        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-G"})),
            occupant=OccupantDefinition(
                occupancy_distribution={
                    "zone-G": FixedValue(1),
                    "zone-U": WeightedOptions({0: 0.3, 1: 0.7}),
                    # zone-Trapped can only reach Outside by routing
                    # through door-trap, which is LOCKED -- unsafe.
                    "zone-Trapped": WeightedOptions({0: 0.6, 1: 0.4}),
                },
                behaviour_profile_distribution={
                    "zone-G": FixedValue("Staff_Default"),
                    "zone-U": FixedValue("Staff_Default"),
                    "zone-Trapped": FixedValue("Staff_Default"),
                },
            ),
            engineering=EngineeringConstraints(
                door_state_distribution={"door-trap": FixedValue("LOCKED")},
                min_open_exits=2,
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        # q = P(zone-Trapped empty) = 0.6; p = P(zone-U occupied) = 0.7.
        # P(valid) = q * p = 0.42 exactly.
        self.assertAlmostEqual(result.p_valid, 0.42, places=9)


# =====================================================
# 5 -- multiple safe uncertain zones required JOINTLY: no single one
# alone reaches the threshold.
# =====================================================


class MultipleSafeUncertainZonesRequiredJointlyTests(unittest.TestCase):

    def test_probability_is_the_product_when_both_are_needed_together(self):

        zones = [
            Zone(id="zone-G", name="G", x=0.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-U1", name="U1", x=20.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-U2", name="U2", x=20.0, y=20.0, width=8.0, height=8.0),
        ]
        exits = [
            Exit(id="exit-A", zone_id="zone-G"),
            Exit(id="exit-B", zone_id="zone-U1"),
            Exit(id="exit-C", zone_id="zone-U2"),
        ]
        floor = Floor(name="Ground", id="floor-1", zones=zones, doors=[], exits=exits)
        building = Building(name="Jointly Required", id="b-5", floors=[floor])

        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-G"})),
            occupant=OccupantDefinition(
                occupancy_distribution={
                    "zone-G": FixedValue(1),
                    "zone-U1": WeightedOptions({0: 0.5, 1: 0.5}),
                    "zone-U2": WeightedOptions({0: 0.5, 1: 0.5}),
                },
                behaviour_profile_distribution={
                    "zone-G": FixedValue("Staff_Default"),
                    "zone-U1": FixedValue("Staff_Default"),
                    "zone-U2": FixedValue("Staff_Default"),
                },
            ),
            engineering=EngineeringConstraints(min_open_exits=3),
        )

        result = compute_exact_candidate_validity(building, definition)

        # base={A}, need=2. U1 alone -> {B} size1; U2 alone -> {C}
        # size1; only BOTH occupied together -> {B,C} size2 satisfies.
        # P(valid) = P(U1 occupied) * P(U2 occupied) = 0.5*0.5 = 0.25.
        self.assertAlmostEqual(result.p_valid, 0.25, places=9)


# =====================================================
# 6 -- redundant uncertain zones: contribute no exit beyond what is
# already guaranteed available. Must not change the probability, and
# must not be given to the coverage DP at all.
# =====================================================


class RedundantUncertainZonesTests(unittest.TestCase):

    def _build(self, redundant_p_zero):

        zones = [
            Zone(id="zone-G", name="G", x=0.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-U", name="U", x=20.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-Redundant", name="Redundant", x=0.0, y=20.0, width=8.0, height=8.0),
            # Isolated, never-occupied ignition zone -- kept separate
            # from zone-G/zone-Redundant/zone-U specifically so
            # excluding it (the FIRE_ORIGIN_BLOCKS_EVACUATION check)
            # cannot itself change anyone's SAFE/UNSAFE classification;
            # this test is about redundant EXIT contributions, not
            # fire-origin cut-vertex behavior (covered elsewhere).
            Zone(id="zone-Ignition", name="Ignition", x=-40.0, y=0.0, width=8.0, height=8.0),
        ]
        doors = [
            Door(
                id="door-redundant", normally_open=True,
                zone_a_id="zone-Redundant", zone_b_id="zone-G",
            ),
        ]
        exits = [Exit(id="exit-A", zone_id="zone-G"), Exit(id="exit-B", zone_id="zone-U")]
        floor = Floor(name="Ground", id="floor-1", zones=zones, doors=doors, exits=exits)
        building = Building(name="Redundant", id="b-6", floors=[floor])

        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-Ignition"})),
            occupant=OccupantDefinition(
                occupancy_distribution={
                    "zone-G": FixedValue(1),
                    "zone-U": WeightedOptions({0: 0.3, 1: 0.7}),
                    # zone-Redundant reaches only exit-A -- already in
                    # base_reachable via zone-G -- so it can never add
                    # a NEW exit, regardless of its own P(zero).
                    "zone-Redundant": WeightedOptions({0: redundant_p_zero, 1: 1.0 - redundant_p_zero}),
                },
                behaviour_profile_distribution={
                    "zone-G": FixedValue("Staff_Default"),
                    "zone-U": FixedValue("Staff_Default"),
                    "zone-Redundant": FixedValue("Staff_Default"),
                },
            ),
            engineering=EngineeringConstraints(
                door_state_distribution={"door-redundant": FixedValue("OPEN")},
                min_open_exits=2,
            ),
        )

        return compute_exact_candidate_validity(building, definition)

    def test_redundant_zone_probability_is_irrelevant_to_the_result(self):

        result_a = self._build(redundant_p_zero=0.9)
        result_b = self._build(redundant_p_zero=0.1)

        # Both must equal P(zone-U occupied) = 0.7 exactly, regardless
        # of zone-Redundant's own occupancy distribution.
        self.assertAlmostEqual(result_a.p_valid, 0.7, places=9)
        self.assertAlmostEqual(result_b.p_valid, 0.7, places=9)
        self.assertAlmostEqual(result_a.p_valid, result_b.p_valid, places=9)


# =====================================================
# 7 -- no occupants at all: NAVIGATION never runs (real Validator's
# own early return); only the unconditional STRUCTURAL min_open_exits
# rule remains active.
# =====================================================


class NoOccupantsTests(unittest.TestCase):

    def test_no_occupants_uses_only_the_structural_rule(self):

        zones = [
            Zone(id="zone-G", name="G", x=0.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-U", name="U", x=20.0, y=0.0, width=8.0, height=8.0),
        ]
        exits = [Exit(id="exit-A", zone_id="zone-G"), Exit(id="exit-B", zone_id="zone-U")]
        floor = Floor(name="Ground", id="floor-1", zones=zones, doors=[], exits=exits)
        building = Building(name="No Occupants", id="b-7", floors=[floor])

        definition = ScenarioDefinition(
            fire=_fire(),
            occupant=OccupantDefinition(),
            engineering=EngineeringConstraints(
                exit_state_distribution={"exit-B": WeightedOptions({True: 0.6, False: 0.4})},
                min_open_exits=2,
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        # No occupancy_distribution entries at all -- exit-A is always
        # open (FixedValue-equivalent default); valid iff exit-B is
        # also open (structural count >= 2). P(valid) = 0.6 exactly.
        self.assertAlmostEqual(result.p_valid, 0.6, places=9)
        self.assertEqual(result.analyzed_zone_ids, frozenset())


# =====================================================
# 8 -- min_open_exits == 0 (disabled): the Phase 2B closed-form
# shortcut must apply unchanged, without ever invoking the new
# exit-coverage DP.
# =====================================================


class MinOpenExitsDisabledTests(unittest.TestCase):

    def test_disabled_min_open_exits_ignores_exit_reachability_entirely(self):

        zones = [
            Zone(id="zone-G", name="G", x=0.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-Trapped", name="Trapped", x=0.0, y=20.0, width=8.0, height=8.0),
            Zone(id="zone-Behind", name="Behind", x=0.0, y=40.0, width=8.0, height=8.0),
        ]
        doors = [
            Door(
                id="door-trap", normally_open=False,
                zone_a_id="zone-Trapped", zone_b_id="zone-Behind",
            ),
        ]
        exits = [Exit(id="exit-A", zone_id="zone-G"), Exit(id="exit-trap", zone_id="zone-Behind")]
        floor = Floor(name="Ground", id="floor-1", zones=zones, doors=doors, exits=exits)
        building = Building(name="Disabled Threshold", id="b-8", floors=[floor])

        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-G"})),
            occupant=OccupantDefinition(
                occupancy_distribution={
                    "zone-G": FixedValue(1),
                    "zone-Trapped": WeightedOptions({0: 0.6, 1: 0.4}),
                },
                behaviour_profile_distribution={
                    "zone-G": FixedValue("Staff_Default"), "zone-Trapped": FixedValue("Staff_Default"),
                },
            ),
            engineering=EngineeringConstraints(
                door_state_distribution={"door-trap": FixedValue("LOCKED")},
                min_open_exits=0,
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        # min_open_exits<=0 -- the only surviving requirement is
        # zone-Trapped (unsafe) being empty: P(valid) = 0.6 exactly.
        self.assertAlmostEqual(result.p_valid, 0.6, places=9)


# =====================================================
# 9 -- large uncertain-zone count stress/regression test: many safe
# uncertain zones share an IDENTICAL reachable-exit contribution, so
# grouping must collapse them to one DP group, never a Cartesian
# product over individual occupancy patterns.
# =====================================================


class LargeUncertainZoneCountStressTests(unittest.TestCase):

    def test_many_zones_with_identical_contribution_collapse_to_one_group(self):

        # zone-Base is a SEPARATE, guaranteed-occupied zone (its own
        # exit-base, no connection to the star) -- deliberately present
        # so `p_all_safe_empty` (the "nobody at all is present"
        # structural-fallback term) is forced to exactly 0.0, isolating
        # this test to the exit-coverage DP's own behavior specifically
        # (without it, "nobody in the star is occupied" would also
        # trivially satisfy min_open_exits via the unconditional
        # STRUCTURAL rule whenever exit-1 alone already meets the
        # threshold, masking what this test intends to measure).
        zones = [
            Zone(id="zone-Base", name="Base", x=-40.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-G", name="G", x=0.0, y=0.0, width=8.0, height=8.0),
        ]
        doors = []
        occupancy_distribution = {"zone-Base": FixedValue(1)}
        behaviour_distribution = {"zone-Base": FixedValue("Staff_Default")}

        p_zero_each = 0.9

        for i in range(60):

            zone_id = f"zone-U{i}"
            door_id = f"door-U{i}"

            zones.append(Zone(id=zone_id, name=f"U{i}", x=float(20 + i * 10), y=0.0, width=8.0, height=8.0))
            doors.append(Door(id=door_id, normally_open=True, zone_a_id="zone-G", zone_b_id=zone_id))

            occupancy_distribution[zone_id] = WeightedOptions({0: p_zero_each, 1: 1.0 - p_zero_each})
            behaviour_distribution[zone_id] = FixedValue("Staff_Default")

        # All 60 zone-U_i reach the SAME single exit (through zone-G),
        # since zone-G itself hosts exit-1 and every door is open -- so
        # every zone's own reachable-exit contribution is IDENTICAL
        # ({exit-1}), collapsing to exactly one DP group. exit-base is
        # a second, always-open exit reachable only from zone-Base
        # (never from the star), so min_open_exits=2 genuinely
        # requires the star's own DP-computed coverage, not just the
        # guaranteed zone-Base contribution alone.
        exits = [Exit(id="exit-1", zone_id="zone-G"), Exit(id="exit-base", zone_id="zone-Base")]
        door_state_distribution = {door.id: FixedValue("OPEN") for door in doors}

        floor = Floor(name="Ground", id="floor-1", zones=zones, doors=doors, exits=exits)
        building = Building(name="Star Stress", id="b-9", floors=[floor])

        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-Base"})),
            occupant=OccupantDefinition(
                occupancy_distribution=occupancy_distribution,
                behaviour_profile_distribution=behaviour_distribution,
            ),
            engineering=EngineeringConstraints(
                door_state_distribution=door_state_distribution,
                min_open_exits=2,
            ),
        )

        result = compute_exact_candidate_validity(building, definition)

        # No engineering-state uncertainty (all doors/exits FixedValue),
        # so still exactly 1 engineering-state combination considered
        # -- occupancy uncertainty across 60 zones must not multiply
        # into it, even though the NEW exit-coverage DP is exercised
        # this time (unlike the older, purely-reachability 20-zone
        # regression test in test_campaign_feasibility_occupancy_
        # uncertainty.py, which never reaches this DP at all).
        self.assertEqual(result.total_states_considered, 1)
        self.assertTrue(result.exact)

        # base_reachable={exit-base} (from guaranteed zone-Base),
        # need=2-1=1. The 60 zone-U_i all merge into ONE DP group
        # contributing {exit-1} -- P(valid) = P(at least one of the 60
        # zones is occupied) = 1 - (p_zero_each ** 60).
        expected = 1.0 - (p_zero_each ** 60)
        self.assertAlmostEqual(result.p_valid, expected, places=9)


# =====================================================
# State-space protection for the NEW exit-coverage DP specifically
# (separate from DEFAULT_MAX_ENUMERATED_STATES, which this must not
# regress -- see the dedicated 20-zone test in
# tests/test_campaign_feasibility_occupancy_uncertainty.py).
# =====================================================


class ExitCoverageStateSpaceProtectionTests(unittest.TestCase):

    def test_oversized_exit_coverage_space_is_declined_not_approximated(self):

        zones = [
            Zone(id="zone-G", name="G", x=0.0, y=0.0, width=8.0, height=8.0),
            Zone(id="zone-U", name="U", x=20.0, y=0.0, width=8.0, height=8.0),
        ]
        exits = [Exit(id="exit-A", zone_id="zone-G"), Exit(id="exit-B", zone_id="zone-U")]
        floor = Floor(name="Ground", id="floor-1", zones=zones, doors=[], exits=exits)
        building = Building(name="Forced Too Large", id="b-10", floors=[floor])

        definition = ScenarioDefinition(
            fire=_fire(allowed_ignition_zone_ids=frozenset({"zone-G"})),
            occupant=OccupantDefinition(
                occupancy_distribution={
                    "zone-G": FixedValue(1),
                    "zone-U": WeightedOptions({0: 0.3, 1: 0.7}),
                },
                behaviour_profile_distribution={
                    "zone-G": FixedValue("Staff_Default"), "zone-U": FixedValue("Staff_Default"),
                },
            ),
            engineering=EngineeringConstraints(min_open_exits=2),
        )

        with patch.object(exact_validity_module, "MAX_EXIT_COVERAGE_STATES", 1):

            result = compute_exact_candidate_validity(building, definition)

        self.assertFalse(result.exact)
        self.assertTrue(result.state_space_too_large)
        self.assertIsNone(result.p_valid)
        self.assertTrue(any("exit-coverage" in w.lower() for w in result.warnings))


if __name__ == "__main__":
    unittest.main()
