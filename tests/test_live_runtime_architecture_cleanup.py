import pathlib
import re
import unittest

from live_runtime.factory import build_offline_demo_runtime

from tests.live_runtime_fixtures import make_demo_building


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# =====================================================
# Live Runtime Architecture Cleanup milestone -- Phase 8's own
# dependency guards (proving new production code cannot accidentally
# start using the retired live_system.integration adapter generation
# again) plus Phase 9/10's own current-runtime and operator-workflow
# regression proof, run as part of THIS milestone's own dedicated test
# module rather than trusting the full suite alone.
# =====================================================


_DELETED_SYMBOLS = (
    "SensorFusionPerceptionGateway",
    "PredictorAIInferenceGateway",
    "GeneratePolicyDecisionPolicyGateway",
    "DashboardCommandCenterGateway",
    "FeatureRowBuilder",
    "DecisionInputsBuilder",
)

# Files that are EXPECTED to still mention a deleted symbol's name, in
# prose only (never a Python import) -- integration.py's own module
# docstring explains what was removed and why (Phase 5's own "make its
# legacy/deprecated status explicit" requirement), and the two
# historical architecture docs are explicitly preserved as a record of
# the investigation that led here (Phase 11's own "label historical
# rather than falsifying history" instruction), not rewritten.
_ALLOWED_MENTIONS = frozenset({
    "live_system/integration.py",
    "docs/architecture/live_system_integration_audit.md",
    "docs/architecture/live_command_center_integration.md",
    "docs/architecture/live_runtime_architecture_cleanup.md",
    "docs/architecture/synevac_end_to_end_architecture_review.md",
    # Pre-existing, comment-only analogical cross-references (never an
    # import) to the deleted classes' own former documented conventions
    # -- e.g. "the same pattern DashboardCommandCenterGateway already
    # established." Confirmed by Phase 1's own usage audit to be prose
    # only; left as-is rather than rewritten, matching this milestone's
    # own "do not perform broad refactors" instruction. A FUTURE new
    # mention anywhere else still fails this guard, forcing a conscious
    # decision rather than silent copy-paste drift.
    "command_center/data_source.py",
    "command_center/incident_data.py",
    "human_decision_engine/view.py",
    "live_system/live_advisory_gateway.py",
    "live_system/state_manager.py",
    "perception/providers/human_observation_provider.py",
})


class DeletedSymbolsHaveNoRemainingImportsTests(unittest.TestCase):

    def test_no_python_source_imports_a_deleted_symbol(self):

        import_pattern = re.compile(
            r"^\s*(from\s+\S+\s+import\s+.*\b(%s)\b|import\s+.*\b(%s)\b)"
            % ("|".join(_DELETED_SYMBOLS), "|".join(_DELETED_SYMBOLS)),
            re.MULTILINE,
        )

        offenders = []

        for path in REPO_ROOT.rglob("*.py"):

            if "__pycache__" in path.parts:
                continue

            relative = path.relative_to(REPO_ROOT).as_posix()

            text = path.read_text(encoding="utf-8")

            if import_pattern.search(text):
                offenders.append(relative)

        self.assertEqual(
            offenders, [],
            f"The following files still IMPORT a deleted live_system.integration symbol "
            f"({', '.join(_DELETED_SYMBOLS)}): {offenders}. These classes were deleted by the "
            f"Live Runtime Architecture Cleanup milestone -- zero production callers, superseded "
            f"by build_live_runtime()'s own newer composition.",
        )

    def test_deleted_symbols_no_longer_exist_in_live_system_integration(self):

        import live_system.integration as integration

        for symbol in _DELETED_SYMBOLS:
            self.assertFalse(
                hasattr(integration, symbol),
                f"live_system.integration.{symbol} still exists -- Phase 5 expected it deleted.",
            )

    def test_deleted_symbols_no_longer_exported_from_live_system_package(self):

        import live_system

        for symbol in _DELETED_SYMBOLS:
            self.assertNotIn(symbol, live_system.__all__)
            self.assertFalse(hasattr(live_system, symbol))

    def test_surviving_protocols_and_recommendation_builder_still_exported(self):

        # Phase 6 -- these are KEPT (LiveOrchestrator's own unmodified
        # constructor signature still type-annotates against them), not
        # accidentally swept away along with the dead adapters.

        import live_system

        for symbol in (
            "PerceptionGateway", "AIInferenceGateway", "DecisionPolicyGateway",
            "CommandCenterGateway", "RecommendationBuilder",
        ):
            self.assertIn(symbol, live_system.__all__)
            self.assertTrue(hasattr(live_system, symbol))


class NoNonHistoricalMentionsOutsideAllowedFilesTests(unittest.TestCase):

    def test_every_remaining_mention_is_inside_an_allowed_or_test_file(self):

        # A looser, name-only scan (not import-specific) -- catches
        # comment/docstring cross-references too, which are fine
        # anywhere, but this positively confirms the ONLY places left
        # are the ones this milestone's own Phase 5/11 decisions
        # explicitly kept (integration.py's own docstring, the two
        # historical docs, this milestone's own new doc, and this test
        # module itself).

        name_pattern = re.compile("|".join(re.escape(s) for s in _DELETED_SYMBOLS))

        unexpected = []

        for path in REPO_ROOT.rglob("*"):

            if path.suffix not in (".py", ".md"):
                continue

            if "__pycache__" in path.parts:
                continue

            relative = path.relative_to(REPO_ROOT).as_posix()

            if relative in _ALLOWED_MENTIONS or relative == "tests/test_live_runtime_architecture_cleanup.py":
                continue

            if relative == "tests/test_live_system.py":
                # Explicitly allowed -- its own removal-notice comment
                # names the deleted classes by name (Phase 7's own
                # "document what was removed and why" convention).
                continue

            text = path.read_text(encoding="utf-8")

            if name_pattern.search(text):
                unexpected.append(relative)

        self.assertEqual(unexpected, [])


class LiveRuntimeNeverDependsOnIntegrationConcretesTests(unittest.TestCase):

    def test_live_runtime_package_never_imports_live_system_integration(self):

        # live_runtime/ (factory.py, runtime.py) is the CURRENT
        # production composition root -- it must never import
        # live_system.integration at all, concrete classes or
        # Protocols, confirming the retired generation genuinely has no
        # path back into production composition.

        package_dir = REPO_ROOT / "live_runtime"

        forbidden = re.compile(r"^\s*(from|import)\s+live_system\.integration\b", re.MULTILINE)

        for path in package_dir.glob("*.py"):

            text = path.read_text(encoding="utf-8")

            self.assertIsNone(
                forbidden.search(text),
                f"live_runtime/{path.name} imports live_system.integration -- the current "
                f"production composition root must never depend on the retired generation.",
            )

    def test_live_runtime_launcher_package_never_imports_live_system_integration(self):

        # Same guard, one layer up -- the Application Live Runtime
        # Launcher milestone's own live_runtime_launcher/ package.

        package_dir = REPO_ROOT / "live_runtime_launcher"

        forbidden = re.compile(r"^\s*(from|import)\s+live_system\.integration\b", re.MULTILINE)

        for path in package_dir.glob("*.py"):

            text = path.read_text(encoding="utf-8")

            self.assertIsNone(forbidden.search(text))


# =====================================================
# Phase 9 -- current runtime E2E regression. Proves cleanup did not
# damage the actual, current production composition path: Designer's
# LiveRuntimeSession -> build_live_runtime()/build_offline_demo_runtime()
# -> LiveOrchestrator -> BuildingState -> Crowd -> Evacuation Progress
# -> Trajectory -> Emergency Response -> AI (optional/unconfigured) ->
# Recommendation -> Guidance -> Dynamic Signage -> Advisory (optional)
# -> StateManager -> Command Center.
# =====================================================


class CurrentRuntimeE2ERegressionTests(unittest.TestCase):

    def setUp(self):

        self.building = make_demo_building()
        self.runtime = build_offline_demo_runtime(self.building)
        self.runtime.start()

    def tearDown(self):
        self.runtime.stop()

    def test_full_stage_chain_still_reaches_command_center(self):

        self.runtime.run_cycle(1.0)

        snapshot = self.runtime.command_center_data_source.current_snapshot()

        self.assertIsNotNone(snapshot.building_state)
        self.assertIsNotNone(snapshot.evacuation_progress)
        self.assertIsNotNone(snapshot.trajectory_intelligence)
        self.assertIsNotNone(snapshot.emergency_response)
        self.assertIsNotNone(snapshot.evacuation_recommendation)
        self.assertIsNotNone(snapshot.evacuation_guidance)
        self.assertIsNotNone(snapshot.dynamic_signage)

        # AI/Advisory remain honestly unconfigured -- this cleanup
        # milestone adds no AI wiring, and none of the deleted
        # integration.py adapters were ever how AI reached BuildingState
        # in production anyway.
        self.assertIsNone(self.runtime.orchestrator.live_ai_gateway)
        self.assertIsNone(snapshot.ai_prediction_snapshot)
        self.assertIsNone(snapshot.advisory_report)

    def test_state_manager_and_command_center_share_the_same_runtime(self):

        self.runtime.run_cycle(1.0)

        self.assertIs(
            self.runtime.command_center_data_source._state_manager, self.runtime.orchestrator.state_manager,
        )


# =====================================================
# Phase 10 -- offline operator workflow regression. Re-proves Voice/
# Dynamic Sign/Building Control all still require an explicit operator
# approval, and that provider ownership (SIMULATION under
# build_offline_demo_runtime()) is unchanged by this cleanup.
# =====================================================


class OperatorAuthorityRegressionTests(unittest.TestCase):

    def setUp(self):

        self.building = make_demo_building()
        self.runtime = build_offline_demo_runtime(self.building)
        self.runtime.start()

    def tearDown(self):
        self.runtime.stop()

    def test_nothing_dispatches_automatically_after_several_cycles(self):

        for cycle in range(3):
            self.runtime.run_cycle(float(cycle))

        gateway = self.runtime.operator_action_gateway

        self.assertEqual(len(self.runtime.voice_evacuation_controller.broadcast_log.all_instructions()), 0)
        self.assertEqual(self.runtime.building_control_controller.all_requests(), ())
        self.assertEqual(gateway.all_signage_instructions(), ())

    def test_voice_signage_and_building_control_providers_are_simulation_not_none(self):

        # Provider ownership unchanged by this cleanup -- still exactly
        # the Simulation* providers build_offline_demo_runtime() has
        # always defaulted in, never live_system.integration's
        # DashboardCommandCenterGateway or any other retired adapter.

        gateway = self.runtime.operator_action_gateway

        self.assertIsNotNone(gateway.voice_controller)
        self.assertIsNotNone(gateway.control_controller)
        self.assertIsNotNone(gateway.signage_controller)

        self.assertEqual(type(self.runtime.voice_evacuation_controller.provider).__name__, "SimulationVoiceOutputProvider")
        self.assertEqual(type(self.runtime.building_control_controller.provider).__name__, "SimulationControlProvider")


if __name__ == "__main__":
    unittest.main()
