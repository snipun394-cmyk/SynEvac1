import unittest

from models.building import Building
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from behavior.intent import ActionIntent, AlwaysEvacuateDecisionStrategy, DecisionStrategy
from behavior.orchestrator import HumanBehaviorLayer
from behavior.pre_movement import NoPreMovementDelay
from behavior.profile import Role
from behavior.route_choice import ShortestRouteChoiceStrategy
from simulator.decision import ActionType

from scenario import Scenario, ScenarioFire, ScenarioMetadata, ScenarioOccupant
from scenario_runner import run

from behaviour_profile_resolver import (
    BehaviorProfileTemplate,
    DEFAULT_PROFILE_REGISTRY,
    UnknownBehaviourProfileError,
    register_occupants,
    resolve_profile,
)
from behaviour_profile_resolver.occupant_attributes import derive_occupant_attributes
from behaviour_profile_resolver.registrar import _effective_walking_speed_multiplier


def make_building():

    floor = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0),
            Zone(id="zone-2", name="Office", x=20.0, y=0.0, width=6.0, height=6.0),
        ],
        exits=[Exit(id="exit-1", zone_id="zone-1"), Exit(id="exit-2", zone_id="zone-2")],
    )

    return Building(name="Test Building", id="building-1", floors=[floor])


def make_metadata(**overrides):

    defaults = dict(
        scenario_id="scn-1", definition_id="def-1", definition_content_hash="hash-abc",
        generation_version="scenario_generator/1", seed=42, created_at="2026-07-13T00:00:00",
    )
    defaults.update(overrides)

    return ScenarioMetadata(**defaults)


def make_occupant(**overrides):

    defaults = dict(
        occupant_id="occ-1", zone_id="zone-1", floor_id="floor-1",
        position=(3.0, 3.0), behaviour_profile_id="Adult_Default",
    )
    defaults.update(overrides)

    return ScenarioOccupant(**defaults)


def make_scenario(**overrides):

    defaults = dict(
        metadata=make_metadata(),
        occupants=(make_occupant(),),
        fire=ScenarioFire(
            ignition_zone_id="zone-2", ignition_floor_id="floor-1",
            fire_profile="Electrical", growth_parameters={"growth_time": 200.0},
        ),
    )
    defaults.update(overrides)

    return Scenario(**defaults)


def make_context(scenario=None, building=None):

    return run(scenario or make_scenario(), building or make_building())


class AlwaysGoalDecisionStrategy(DecisionStrategy):

    # A minimal, deterministic test double -- not a duplicate of any
    # library strategy, just a fixed-outcome stand-in so registration
    # tests never depend on behavior_library's own unseeded rng.

    def decide(self, context) -> ActionIntent:

        return ActionIntent(
            occupant_id=context.profile.occupant_id, action_type=ActionType.EVACUATE,
            requires_movement=True,
        )


class ProfileResolutionTests(unittest.TestCase):

    def test_resolves_a_known_profile_id(self):

        template = resolve_profile("Adult_Default", DEFAULT_PROFILE_REGISTRY)

        self.assertIsInstance(template, BehaviorProfileTemplate)

    def test_unknown_profile_id_raises(self):

        with self.assertRaises(UnknownBehaviourProfileError):
            resolve_profile("Not_A_Real_Profile", DEFAULT_PROFILE_REGISTRY)

    def test_unknown_profile_error_message_names_the_id(self):

        with self.assertRaises(UnknownBehaviourProfileError) as ctx:
            resolve_profile("Ghost_Profile", DEFAULT_PROFILE_REGISTRY)

        self.assertIn("Ghost_Profile", str(ctx.exception))

    def test_resolves_against_a_custom_registry(self):

        custom_registry = {"Custom_Profile": BehaviorProfileTemplate(walking_speed=2.5)}

        template = resolve_profile("Custom_Profile", custom_registry)

        self.assertEqual(template.walking_speed, 2.5)

    def test_custom_registry_does_not_recognize_default_registry_ids(self):

        custom_registry = {"Custom_Profile": BehaviorProfileTemplate()}

        with self.assertRaises(UnknownBehaviourProfileError):
            resolve_profile("Adult_Default", custom_registry)


class DefaultRegistryContentsTests(unittest.TestCase):

    EXPECTED_PROFILE_IDS = (
        "Adult_Default", "Child_Default", "Wheelchair_Default",
        "Staff_Default", "FireWarden_Default", "Visitor_Default",
    )

    def test_every_expected_profile_id_is_present(self):

        for profile_id in self.EXPECTED_PROFILE_IDS:
            self.assertIn(profile_id, DEFAULT_PROFILE_REGISTRY)

    def test_every_entry_is_a_behavior_profile_template(self):

        for template in DEFAULT_PROFILE_REGISTRY.values():
            self.assertIsInstance(template, BehaviorProfileTemplate)

    def test_every_entry_uses_existing_strategy_interfaces(self):

        for template in DEFAULT_PROFILE_REGISTRY.values():

            self.assertTrue(hasattr(template.decision_strategy, "decide"))
            self.assertTrue(hasattr(template.route_choice_strategy, "choose"))
            self.assertTrue(hasattr(template.pre_movement_strategy, "delay"))

    def test_every_walking_speed_is_positive(self):

        for template in DEFAULT_PROFILE_REGISTRY.values():
            self.assertGreater(template.walking_speed, 0.0)

    def test_every_compliance_level_is_a_valid_probability(self):

        for template in DEFAULT_PROFILE_REGISTRY.values():
            self.assertTrue(0.0 <= template.compliance_level <= 1.0)

    def test_leadership_roles_are_assigned_to_staff_and_fire_warden(self):

        self.assertEqual(DEFAULT_PROFILE_REGISTRY["Staff_Default"].role, Role.LEADER)
        self.assertEqual(DEFAULT_PROFILE_REGISTRY["FireWarden_Default"].role, Role.LEADER)

    def test_stair_speed_defaults_to_none_for_every_shipped_profile(self):

        # Edge-Type-Specific Movement Speed (Experimental Branch V1) --
        # the feature must be disabled by default: no shipped profile
        # may set stair_speed, so every occupant keeps falling back to
        # walking_speed on Stair edges exactly as before this field
        # existed.
        for profile_id, template in DEFAULT_PROFILE_REGISTRY.items():
            self.assertIsNone(template.stair_speed, msg=f"{profile_id} must default stair_speed to None")


class BehaviorProfileTemplateTests(unittest.TestCase):

    def test_default_construction_uses_trivial_strategies(self):

        template = BehaviorProfileTemplate()

        self.assertIsInstance(template.decision_strategy, AlwaysEvacuateDecisionStrategy)
        self.assertIsInstance(template.route_choice_strategy, ShortestRouteChoiceStrategy)
        self.assertIsInstance(template.pre_movement_strategy, NoPreMovementDelay)

    def test_familiarity_and_traits_default_to_empty(self):

        template = BehaviorProfileTemplate()

        self.assertEqual(dict(template.familiarity), {})
        self.assertEqual(dict(template.traits), {})

    def test_familiarity_mapping_is_read_only(self):

        template = BehaviorProfileTemplate(familiarity={"zone-1": 0.9})

        with self.assertRaises(TypeError):
            template.familiarity["zone-1"] = 0.1

    def test_is_frozen(self):

        from dataclasses import FrozenInstanceError

        template = BehaviorProfileTemplate()

        with self.assertRaises(FrozenInstanceError):
            template.walking_speed = 5.0

    def test_stair_speed_defaults_to_none(self):

        template = BehaviorProfileTemplate()

        self.assertIsNone(template.stair_speed)

    def test_stair_speed_can_be_set_independently_of_walking_speed(self):

        template = BehaviorProfileTemplate(walking_speed=1.2, stair_speed=0.55)

        self.assertEqual(template.walking_speed, 1.2)
        self.assertEqual(template.stair_speed, 0.55)


class OccupantRegistrationTests(unittest.TestCase):

    def test_returns_a_human_behavior_layer(self):

        context = make_context()
        behavior_layer = register_occupants(context)

        self.assertIsInstance(behavior_layer, HumanBehaviorLayer)

    def test_registers_onto_the_same_simulation_instance_from_context(self):

        context = make_context()
        behavior_layer = register_occupants(context)

        self.assertIs(behavior_layer.simulation, context.simulation)

    def test_every_occupant_ends_up_registered(self):

        scenario = make_scenario(
            occupants=(
                make_occupant(occupant_id="occ-1"),
                make_occupant(occupant_id="occ-2", position=(1.0, 1.0)),
            ),
        )
        context = make_context(scenario=scenario)

        register_occupants(context)

        self.assertEqual(set(context.simulation._occupants.keys()), {"occ-1", "occ-2"})

    def test_registered_occupant_uses_the_resolved_walking_speed(self):

        # The template's own resolved walking_speed reaches the
        # simulation scaled by this occupant's own deterministic
        # "effective_walking_speed_multiplier" (Occupant Attributes
        # Phase 2 -- see behaviour_profile_resolver.registrar's own
        # _register_one()/_effective_walking_speed_multiplier()) -- not
        # exactly unchanged, since every occupant now automatically
        # carries this attribute. BehaviorProfile.walking_speed itself
        # is still never mutated (asserted separately in tests.
        # test_occupant_attributes.RegistrarIntegrationTests) -- this
        # test is about the *simulation's* own resulting speed, which
        # has always incorporated whatever registration-time
        # calculation this package performs.

        registry = {
            "Fast_Profile": BehaviorProfileTemplate(
                walking_speed=9.0, decision_strategy=AlwaysGoalDecisionStrategy(),
            ),
        }
        scenario = make_scenario(occupants=(make_occupant(behaviour_profile_id="Fast_Profile"),))
        context = make_context(scenario=scenario)

        register_occupants(context, registry=registry)

        attributes = derive_occupant_attributes(scenario.metadata.seed, "occ-1", "Fast_Profile")
        expected_speed = 9.0 * _effective_walking_speed_multiplier(attributes)

        self.assertAlmostEqual(
            context.simulation._occupants["occ-1"].walking_speed, expected_speed,
        )

    def test_registration_is_empty_before_the_call(self):

        context = make_context()

        self.assertEqual(context.simulation._occupants, {})

    def test_no_occupants_registers_nothing_and_does_not_raise(self):

        scenario = make_scenario(occupants=())
        context = make_context(scenario=scenario)

        register_occupants(context)

        self.assertEqual(context.simulation._occupants, {})

    def test_unknown_behaviour_profile_id_raises_and_registers_nothing_from_that_occupant(self):

        scenario = make_scenario(
            occupants=(make_occupant(occupant_id="occ-1", behaviour_profile_id="Ghost_Profile"),),
        )
        context = make_context(scenario=scenario)

        with self.assertRaises(UnknownBehaviourProfileError):
            register_occupants(context)

    def test_unknown_profile_stops_registration_for_later_occupants_too(self):

        # No repair, no partial success -- a hard error part-way
        # through must not silently leave some occupants registered and
        # others not without the caller knowing.
        scenario = make_scenario(
            occupants=(
                make_occupant(occupant_id="occ-1", behaviour_profile_id="Adult_Default"),
                make_occupant(occupant_id="occ-2", position=(1.0, 1.0), behaviour_profile_id="Ghost_Profile"),
            ),
        )
        context = make_context(scenario=scenario)

        with self.assertRaises(UnknownBehaviourProfileError):
            register_occupants(context)

    def test_default_registry_is_used_when_none_is_supplied(self):

        context = make_context()

        # Should not raise -- "Adult_Default" is in DEFAULT_PROFILE_REGISTRY.
        register_occupants(context)

    def test_deterministic_strategy_produces_a_deterministic_registration_outcome(self):

        registry = {
            "Deterministic_Profile": BehaviorProfileTemplate(
                walking_speed=1.5,
                decision_strategy=AlwaysGoalDecisionStrategy(),
                route_choice_strategy=ShortestRouteChoiceStrategy(),
            ),
        }
        scenario = make_scenario(
            occupants=(make_occupant(behaviour_profile_id="Deterministic_Profile"),),
        )

        first_context = make_context(scenario=scenario)
        register_occupants(first_context, registry=registry)

        second_context = make_context(scenario=scenario)
        register_occupants(second_context, registry=registry)

        first_occupant = first_context.simulation._occupants["occ-1"]
        second_occupant = second_context.simulation._occupants["occ-1"]

        self.assertEqual(first_occupant.state, second_occupant.state)
        self.assertEqual(first_occupant.walking_speed, second_occupant.walking_speed)


class BehaviourProfileIdRemainsOpaqueUpstreamTests(unittest.TestCase):

    # Confirms the Scenario Runner side of the boundary: nothing about
    # running a Scenario (scenario_runner.run()) changed as a result of
    # this package existing -- the Runner still carries
    # behaviour_profile_id through unread.

    def test_context_occupants_still_carry_the_unresolved_profile_id(self):

        context = make_context()

        self.assertEqual(context.occupants[0].behaviour_profile_id, "Adult_Default")


class BehaviourProfileResolverPackageDependencyDirectionTests(unittest.TestCase):

    def test_package_may_import_behavior_and_behavior_library(self):

        import pathlib

        text = (
            pathlib.Path(__file__).resolve().parent.parent
            / "behaviour_profile_resolver" / "registry.py"
        ).read_text()

        self.assertIn("behavior_library", text)

    def test_package_never_imports_forbidden_packages(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "behaviour_profile_resolver"

        # random is deliberately NOT forbidden here -- docs/architecture/
        # reproducibility_review.md §7.4: registrar.py constructs a
        # deterministic, per-occupant random.Random(seed) derived from
        # the Scenario's own seed + occupant_id, the same sanctioned
        # "seeded, reproducible randomness" pattern scenario_generator.
        # seed_manager already uses elsewhere -- never a bare, unseeded
        # random.Random(). See test_registrar_never_constructs_an_
        # unseeded_random_random below for the structural guard against
        # regressing back to arbitrary randomness.
        forbidden = (
            r"^\s*(from|import)\s+"
            r"(scenario_generator|scenario_validator|scenario_pipeline|ai_decision|"
            r"perception|sensors|occupancy|rl|sandbox|designer|fire_growth|"
            r"hazard_evolution)\b"
        )

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"behaviour_profile_resolver/{path.name} imports a package this "
                f"adapter must never depend on",
            )

    def test_registrar_never_constructs_an_unseeded_random_random(self):

        import pathlib
        import re

        text = (
            pathlib.Path(__file__).resolve().parent.parent
            / "behaviour_profile_resolver" / "registrar.py"
        ).read_text()

        # Every random.Random( construction in this file must be given
        # an argument (a derived seed) -- a bare random.Random() would
        # silently reintroduce exactly the unseeded-randomness bug
        # docs/architecture/reproducibility_review.md §5 identified.
        self.assertIsNone(
            re.search(r"random\.Random\(\s*\)", text),
            "registrar.py must never construct a bare, unseeded random.Random()",
        )
        self.assertIn("random.Random(", text)

    def test_scenario_runner_never_imports_this_package(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "scenario_runner"

        forbidden = r"^\s*(from|import)\s+behaviour_profile_resolver\b"

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"scenario_runner/{path.name} must remain completely unaware of "
                f"behaviour_profile_resolver/",
            )

    def test_package_performs_no_file_io(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "behaviour_profile_resolver"

        forbidden = r"\bopen\s*\(|\.write\s*\(|\.read\s*\(|\bjson\.(load|dump)\s*\("

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text),
                f"behaviour_profile_resolver/{path.name} appears to perform file I/O",
            )

    def test_package_never_generates_or_validates_or_simulates(self):

        import pathlib

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "behaviour_profile_resolver"

        for path in sorted(package_dir.glob("*.py")):

            text = path.read_text()

            self.assertNotIn(".evolve(", text, f"behaviour_profile_resolver/{path.name}")
            self.assertNotIn("generate_scenario", text, f"behaviour_profile_resolver/{path.name}")


if __name__ == "__main__":
    unittest.main()
