import unittest
from pathlib import Path

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from scenario.firefighter import ScenarioFirefighter
from scenario.metadata import ScenarioMetadata
from scenario.occupant import ScenarioOccupant
from scenario.scenario import Scenario

from ground_truth.labels import GroundTruth
from decision_policy.policy import DecisionPolicy
from decision_policy.exit_policy import CLOSE
from decision_policy.stair_policy import AVOID

from advisory_system.orchestrator import AdvisoryOrchestrator
from advisory_system.recommendation_models import AdvisoryInputs, BuildingRecommendation

from simulation_interactive.action_executor import Action, ActionResult, InteractiveActionType

from building_control.advisory_adapter import translate_recommendation, translate_report
from building_control.controller import BuildingControlController
from building_control.providers import BuildingControlProvider, SimulationControlProvider
from building_control.requests import ControlInstruction, ControlRequest, ControlResult
from building_control.types import (
    ApprovalMode,
    ControlAction,
    ControlSystemType,
    RequestSource,
    RequestStatus,
)


# =====================================================
# Fixtures -- same shape as tests/test_advisory_system.py's own
# make_building()/make_scenario()/make_ground_truth()/make_decision_policy(),
# reused here (not imported, kept local -- this file must stand alone
# for the architecture-guard tests below, which read every building_control
# source file's own imports).
# =====================================================


def make_building():

    floor = Floor(
        name="Ground", id="floor-1",
        zones=[
            Zone(id="zone-a", name="Cafeteria", x=0.0, y=0.0, width=10.0, height=8.0, floor_id="floor-1"),
            Zone(id="zone-b", name="Laboratory", x=20.0, y=0.0, width=6.0, height=6.0, floor_id="floor-1"),
        ],
        doors=[Door(id="door-1", name="D1", zone_a_id="zone-a", zone_b_id="zone-b", floor_id="floor-1")],
        exits=[
            Exit(id="exit-1", name="Exit 1", zone_id="zone-a", floor_id="floor-1"),
        ],
        stairs=[Staircase(id="stair-1", name="Stair 1", from_zone_id="zone-b", to_zone_id="zone-a", to_floor_id="floor-1")],
    )

    return Building(name="Test Building", id="building-1", floors=[floor])


def make_metadata():

    return ScenarioMetadata(
        scenario_id="scn-1", definition_id="def-1", definition_content_hash="h",
        generation_version="v1", seed=1, created_at="2026-07-16T00:00:00",
    )


def make_scenario():

    occupants = (
        ScenarioOccupant(
            occupant_id="occ-1", zone_id="zone-a", floor_id="floor-1",
            position=(1.0, 1.0), behaviour_profile_id="Adult_Default",
        ),
    )
    firefighters = (
        ScenarioFirefighter(
            firefighter_id="ff-1", team_id="team-0", entry_zone_id="zone-a",
            floor_id="floor-1", position=(0.0, 0.0), arrival_time=30.0,
            behaviour_profile_id="Firefighter_Default",
        ),
    )

    return Scenario(metadata=make_metadata(), occupants=occupants, firefighters=firefighters)


def make_ground_truth(
    *, maximum_hazard_zone=None, hazard_spread_order=(), zone_risk_scores=None,
    stair_risk_scores=None, recommendations=(),
):

    return GroundTruth(
        scenario_id="scn-1", definition_id="def-1",
        total_evacuation_time=150.0, building_cleared=False,
        reachable_occupants=1, unreachable_occupants=0,
        people_trapped=1, people_evacuated=0,
        worst_exit=None, zone_route_stats=[],
        maximum_hazard_zone=maximum_hazard_zone, hazard_spread_order=hazard_spread_order,
        first_hazardous_zone=maximum_hazard_zone,
        doors_that_became_bottlenecks=(), exits_underutilized=(), exits_exceeding_capacity=(),
        stairs_exceeding_capacity=(),
        zone_risk_scores=zone_risk_scores or [], stair_risk_scores=stair_risk_scores or [],
        recommendations=recommendations,
        helping_group_count=0, fallen_count=0, possible_injury_count=0,
    )


def make_decision_policy(*, zone_decisions=(), exit_decisions=(), stair_decisions=(), announcements=()):

    return DecisionPolicy(
        scenario_id="scn-1",
        zone_decisions=zone_decisions, exit_decisions=exit_decisions,
        stair_decisions=stair_decisions, announcements=announcements,
        rescue_priorities=[], rescue_order=(),
    )


class RecordingActionExecutor:

    # A minimal stand-in for simulation_interactive.action_executor.
    # ActionExecutor -- records every Action it is handed and always
    # reports applied=True, so SimulationControlProvider's Door/Exit
    # composition can be exercised without standing up a full
    # SimulationContext/RouteManager. See action_executor.py's own
    # real ActionExecutor.apply() for the interface this mirrors.

    def __init__(self, applied=True, reason=None):
        self.applied = applied
        self.reason = reason
        self.calls = []

    def apply(self, action: Action, time: float) -> ActionResult:
        self.calls.append(action)
        return ActionResult(action=action, applied=self.applied, reason=self.reason)


def make_recommendation(action, target_type, target_id, reason="test", confidence=0.7):

    return BuildingRecommendation(
        action=action, target_type=target_type, target_id=target_id,
        reason=reason, confidence=confidence, expected_engineering_benefit="test benefit",
    )


# =====================================================
# 1-7: per-system control requests + target validation.
# =====================================================


class ControlRequestExecutionTests(unittest.TestCase):

    def setUp(self):
        self.building = make_building()

    def _controller(self, executor=None, should_fail=None):
        provider = SimulationControlProvider(self.building, executor, should_fail=should_fail)
        return BuildingControlController(self.building, provider), provider

    def _submit_and_approve(self, controller, **kwargs):
        request = ControlRequest(source=RequestSource.OPERATOR, reason="test", **kwargs)
        controller.submit(request)
        controller.approve(request.request_id)
        return request

    # ---- 1: Door OPEN ----

    def test_door_open_request_confirmed_via_action_executor(self):

        executor = RecordingActionExecutor(applied=True)
        controller, _ = self._controller(executor)

        request = self._submit_and_approve(
            controller, system_type=ControlSystemType.DOOR, target_id="door-1",
            requested_action=ControlAction.OPEN,
        )

        self.assertEqual(controller.status_of(request.request_id), RequestStatus.CONFIRMED)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(executor.calls[0].action_type, InteractiveActionType.OPEN_DOOR)
        self.assertEqual(executor.calls[0].target_id, "door-1")

    # ---- 2: Door CLOSE ----

    def test_door_close_request_confirmed_via_action_executor(self):

        executor = RecordingActionExecutor(applied=True)
        controller, _ = self._controller(executor)

        request = self._submit_and_approve(
            controller, system_type=ControlSystemType.DOOR, target_id="door-1",
            requested_action=ControlAction.CLOSE,
        )

        self.assertEqual(controller.status_of(request.request_id), RequestStatus.CONFIRMED)
        self.assertEqual(executor.calls[0].action_type, InteractiveActionType.CLOSE_DOOR)

    # ---- 3: Exit OPEN ----

    def test_exit_open_request_confirmed_via_action_executor(self):

        executor = RecordingActionExecutor(applied=True)
        controller, _ = self._controller(executor)

        request = self._submit_and_approve(
            controller, system_type=ControlSystemType.EXIT, target_id="exit-1",
            requested_action=ControlAction.OPEN,
        )

        self.assertEqual(controller.status_of(request.request_id), RequestStatus.CONFIRMED)
        self.assertEqual(executor.calls[0].action_type, InteractiveActionType.OPEN_EXIT)

    # ---- 4: Exit CLOSE ----

    def test_exit_close_request_confirmed_via_action_executor(self):

        executor = RecordingActionExecutor(applied=True)
        controller, _ = self._controller(executor)

        request = self._submit_and_approve(
            controller, system_type=ControlSystemType.EXIT, target_id="exit-1",
            requested_action=ControlAction.CLOSE,
        )

        self.assertEqual(controller.status_of(request.request_id), RequestStatus.CONFIRMED)
        self.assertEqual(executor.calls[0].action_type, InteractiveActionType.CLOSE_EXIT)

    # ---- No executor available (e.g. Command Center's static replay) ----

    def test_door_request_honestly_fails_without_an_action_executor(self):

        controller, _ = self._controller(executor=None)

        request = self._submit_and_approve(
            controller, system_type=ControlSystemType.DOOR, target_id="door-1",
            requested_action=ControlAction.OPEN,
        )

        self.assertEqual(controller.status_of(request.request_id), RequestStatus.FAILED)
        self.assertEqual(len(controller.snapshot().entries), 0)

    # ---- 5: Stair Pressurization ACTIVATE ----

    def test_stair_pressurization_activate_confirmed_state_only(self):

        controller, provider = self._controller()

        request = self._submit_and_approve(
            controller, system_type=ControlSystemType.STAIR_PRESSURIZATION, target_id="stair-1",
            requested_action=ControlAction.ACTIVATE,
        )

        self.assertEqual(controller.status_of(request.request_id), RequestStatus.CONFIRMED)
        state = provider.state_only_states()
        self.assertEqual(state[(ControlSystemType.STAIR_PRESSURIZATION, "stair-1")], ControlAction.ACTIVATE)

        entries = controller.snapshot().entries
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].system_type, ControlSystemType.STAIR_PRESSURIZATION)

    # ---- 6: Smoke Exhaust ACTIVATE ----

    def test_smoke_exhaust_activate_confirmed_state_only(self):

        controller, provider = self._controller()

        request = self._submit_and_approve(
            controller, system_type=ControlSystemType.SMOKE_EXHAUST, target_id="zone-b",
            requested_action=ControlAction.ACTIVATE,
        )

        self.assertEqual(controller.status_of(request.request_id), RequestStatus.CONFIRMED)
        self.assertIn((ControlSystemType.SMOKE_EXHAUST, "zone-b"), provider.state_only_states())

    # ---- 7: Deluge ACTIVATE ----

    def test_deluge_activate_confirmed_state_only(self):

        controller, provider = self._controller()

        request = self._submit_and_approve(
            controller, system_type=ControlSystemType.DELUGE, target_id="zone-b",
            requested_action=ControlAction.ACTIVATE,
        )

        self.assertEqual(controller.status_of(request.request_id), RequestStatus.CONFIRMED)
        self.assertIn((ControlSystemType.DELUGE, "zone-b"), provider.state_only_states())

    # ---- 8: Invalid target rejection ----

    def test_invalid_target_is_rejected(self):

        controller, _ = self._controller()

        request = ControlRequest(
            system_type=ControlSystemType.DOOR, target_id="no-such-door",
            requested_action=ControlAction.OPEN, reason="test", source=RequestSource.OPERATOR,
        )
        controller.submit(request)

        self.assertEqual(controller.status_of(request.request_id), RequestStatus.REJECTED)

    # ---- 9: Wrong target type rejection ----

    def test_wrong_target_type_is_rejected(self):

        controller, _ = self._controller()

        # zone-b exists, but as a Zone -- not valid for a DOOR request.
        request = ControlRequest(
            system_type=ControlSystemType.DOOR, target_id="zone-b",
            requested_action=ControlAction.OPEN, reason="test", source=RequestSource.OPERATOR,
        )
        controller.submit(request)

        self.assertEqual(controller.status_of(request.request_id), RequestStatus.REJECTED)


# =====================================================
# 10-18: request lifecycle / approval workflow / provider feedback.
# =====================================================


class RequestLifecycleTests(unittest.TestCase):

    def setUp(self):
        self.building = make_building()
        self.provider = SimulationControlProvider(self.building)
        self.controller = BuildingControlController(self.building, self.provider)

    def _pressurize_request(self, target_id="stair-1"):
        return ControlRequest(
            system_type=ControlSystemType.STAIR_PRESSURIZATION, target_id=target_id,
            requested_action=ControlAction.ACTIVATE, reason="test", source=RequestSource.OPERATOR,
        )

    # ---- 10: Request remains pending without approval ----

    def test_request_remains_pending_without_approval(self):

        request = self._pressurize_request()
        self.controller.submit(request)

        self.assertEqual(self.controller.status_of(request.request_id), RequestStatus.PENDING_APPROVAL)
        self.assertEqual(len(self.controller.snapshot().entries), 0)
        self.assertEqual(self.controller.snapshot().pending_count, 1)

    # ---- 11: Approved request dispatches ----

    def test_approved_request_dispatches_and_confirms(self):

        request = self._pressurize_request()
        self.controller.submit(request)
        self.controller.approve(request.request_id)

        statuses = [event.to_status for event in self.controller.history() if event.request_id == request.request_id]
        self.assertIn(RequestStatus.DISPATCHED, statuses)
        self.assertIn(RequestStatus.CONFIRMED, statuses)
        self.assertLess(statuses.index(RequestStatus.DISPATCHED), statuses.index(RequestStatus.CONFIRMED))

    # ---- 12: Rejected request never dispatches ----

    def test_rejected_request_never_dispatches(self):

        request = self._pressurize_request()
        self.controller.submit(request)
        self.controller.reject(request.request_id)

        statuses = [event.to_status for event in self.controller.history() if event.request_id == request.request_id]
        self.assertNotIn(RequestStatus.DISPATCHED, statuses)
        self.assertEqual(self.controller.status_of(request.request_id), RequestStatus.REJECTED)

        with self.assertRaises(ValueError):
            self.controller.approve(request.request_id)

    # ---- 13: Cancelled request never dispatches ----

    def test_cancelled_request_never_dispatches(self):

        request = self._pressurize_request()
        self.controller.submit(request)
        self.controller.cancel(request.request_id)

        statuses = [event.to_status for event in self.controller.history() if event.request_id == request.request_id]
        self.assertNotIn(RequestStatus.DISPATCHED, statuses)
        self.assertEqual(self.controller.status_of(request.request_id), RequestStatus.CANCELLED)

        with self.assertRaises(ValueError):
            self.controller.approve(request.request_id)

    # ---- 14: Provider failure does not become CONFIRMED ----

    def test_provider_failure_does_not_become_confirmed(self):

        provider = SimulationControlProvider(self.building, should_fail=lambda instruction: True)
        controller = BuildingControlController(self.building, provider)

        request = self._pressurize_request()
        controller.submit(request)
        controller.approve(request.request_id)

        self.assertEqual(controller.status_of(request.request_id), RequestStatus.FAILED)
        self.assertEqual(len(controller.snapshot().entries), 0)

    # ---- 15: Provider confirmation becomes CONFIRMED ----

    def test_provider_confirmation_becomes_confirmed(self):

        request = self._pressurize_request()
        self.controller.submit(request)
        self.controller.approve(request.request_id)

        self.assertEqual(self.controller.status_of(request.request_id), RequestStatus.CONFIRMED)
        self.assertEqual(len(self.controller.snapshot().entries), 1)

    # ---- 16: Duplicate request handling ----

    def test_duplicate_pending_request_is_deduplicated(self):

        first = self._pressurize_request()
        second = self._pressurize_request()

        self.controller.submit(first)
        result = self.controller.submit(second)

        self.assertEqual(result.request_id, first.request_id)
        self.assertEqual(len(self.controller.all_requests()), 1)

    # ---- 17: Contradictory sequential requests preserved in history ----

    def test_contradictory_requests_preserve_history(self):

        activate = self._pressurize_request()
        self.controller.submit(activate)
        self.controller.approve(activate.request_id)

        deactivate = ControlRequest(
            system_type=ControlSystemType.STAIR_PRESSURIZATION, target_id="stair-1",
            requested_action=ControlAction.DEACTIVATE, reason="reconsidered", source=RequestSource.OPERATOR,
        )
        self.controller.submit(deactivate)
        self.controller.approve(deactivate.request_id)

        self.assertEqual(len(self.controller.all_requests()), 2)

        entries = self.controller.snapshot().entries
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].action, ControlAction.DEACTIVATE)

        request_ids_in_history = {event.request_id for event in self.controller.history()}
        self.assertIn(activate.request_id, request_ids_in_history)
        self.assertIn(deactivate.request_id, request_ids_in_history)

    # ---- 18: Deterministic ordering ----

    def test_pending_requests_ordering_is_deterministic(self):

        first = self._pressurize_request(target_id="stair-1")
        second = ControlRequest(
            system_type=ControlSystemType.SMOKE_EXHAUST, target_id="zone-b",
            requested_action=ControlAction.ACTIVATE, reason="test", source=RequestSource.OPERATOR,
        )

        self.controller.submit(first)
        self.controller.submit(second)

        order_a = [r.request_id for r in self.controller.pending_requests()]
        order_b = [r.request_id for r in self.controller.pending_requests()]

        self.assertEqual(order_a, order_b)
        self.assertEqual(order_a, [first.request_id, second.request_id])


# =====================================================
# 19-20: simulation-only auto-approval.
# =====================================================


class AutoApprovalTests(unittest.TestCase):

    def setUp(self):
        self.building = make_building()

    # ---- 19: Simulation-only auto-approval works only when explicitly enabled ----

    def test_auto_approve_simulation_confirms_without_manual_approval(self):

        provider = SimulationControlProvider(self.building)
        controller = BuildingControlController(
            self.building, provider, approval_mode=ApprovalMode.AUTO_APPROVE_SIMULATION,
        )

        request = ControlRequest(
            system_type=ControlSystemType.STAIR_PRESSURIZATION, target_id="stair-1",
            requested_action=ControlAction.ACTIVATE, reason="test", source=RequestSource.OPERATOR,
        )
        controller.submit(request)

        self.assertEqual(controller.status_of(request.request_id), RequestStatus.CONFIRMED)

    def test_default_approval_mode_requires_manual_approval(self):

        provider = SimulationControlProvider(self.building)
        controller = BuildingControlController(self.building, provider)

        request = ControlRequest(
            system_type=ControlSystemType.STAIR_PRESSURIZATION, target_id="stair-1",
            requested_action=ControlAction.ACTIVATE, reason="test", source=RequestSource.OPERATOR,
        )
        controller.submit(request)

        self.assertEqual(controller.status_of(request.request_id), RequestStatus.PENDING_APPROVAL)

    # ---- 20: Live mode cannot accidentally use simulation auto-approval ----

    def test_non_simulation_provider_refuses_auto_approve_simulation(self):

        class _FakeLiveProvider(BuildingControlProvider):
            is_simulation_only = False

            def execute(self, instruction):
                return ControlResult(instruction_id=instruction.instruction_id, confirmed=True)

        with self.assertRaises(ValueError):
            BuildingControlController(
                self.building, _FakeLiveProvider(), approval_mode=ApprovalMode.AUTO_APPROVE_SIMULATION,
            )


# =====================================================
# 21-22: Advisory adapter + full scenario-backed end-to-end pipeline.
# =====================================================


class AdvisoryAdapterTests(unittest.TestCase):

    def test_translates_stair_pressurization_smoke_exhaust_and_deluge(self):

        pressurize = translate_recommendation(
            make_recommendation("Activate Stair Pressurization for Stair stair-1", "stair", "stair-1"),
        )
        self.assertEqual(pressurize.system_type, ControlSystemType.STAIR_PRESSURIZATION)
        self.assertEqual(pressurize.requested_action, ControlAction.ACTIVATE)
        self.assertEqual(pressurize.target_id, "stair-1")
        self.assertEqual(pressurize.source, RequestSource.ADVISORY_ADAPTER)
        self.assertIsNotNone(pressurize.source_recommendation_id)

        exhaust = translate_recommendation(
            make_recommendation("Activate Smoke Exhaust in Zone zone-b", "zone", "zone-b"),
        )
        self.assertEqual(exhaust.system_type, ControlSystemType.SMOKE_EXHAUST)

        deluge = translate_recommendation(
            make_recommendation("Activate Deluge in Zone zone-b", "zone", "zone-b"),
        )
        self.assertEqual(deluge.system_type, ControlSystemType.DELUGE)

    def test_translates_unlock_exit_and_close_door(self):

        unlock = translate_recommendation(make_recommendation("Unlock Exit exit-1", "exit", "exit-1"))
        self.assertEqual(unlock.system_type, ControlSystemType.EXIT)
        self.assertEqual(unlock.requested_action, ControlAction.OPEN)

        close_door = translate_recommendation(make_recommendation("Close Door door-1", "door", "door-1"))
        self.assertEqual(close_door.system_type, ControlSystemType.DOOR)
        self.assertEqual(close_door.requested_action, ControlAction.CLOSE)

    def test_skips_broadcast_voice_message(self):

        rec = make_recommendation("Broadcast Voice Message", "building", None)
        self.assertIsNone(translate_recommendation(rec))

    def test_skips_dynamic_exit_signs(self):

        rec = make_recommendation("Update Dynamic Exit Signs", "building", None)
        self.assertIsNone(translate_recommendation(rec))

    def test_skips_recommendations_with_no_control_system_target(self):

        for action, target_type, target_id in (
            ("Deploy staff to Stair stair-1", "stair", "stair-1"),
            ("Increase exit width at Exit exit-1", "exit", "exit-1"),
            ("Additional detector needed in Zone zone-b", "zone", "zone-b"),
            ("Additional camera needed in Zone zone-b", "zone", "zone-b"),
            ("Redirect occupants from Exit exit-1 to Exit exit-2", "exit", "exit-1"),
        ):
            self.assertIsNone(translate_recommendation(make_recommendation(action, target_type, target_id)))


class EndToEndValidationTests(unittest.TestCase):

    # Phase 13's own required end-to-end chain: Fire/Smoke Scenario ->
    # Ground Truth / Decision Policy -> AdvisoryReport -> BuildingRecommendation
    # -> ControlRequest -> Operator Approval -> BuildingControlController ->
    # SimulationControlProvider -> Confirmed Simulated State. Nothing here
    # is fabricated: the AdvisoryReport is produced by the real, unmodified
    # AdvisoryOrchestrator from a deterministic GroundTruth/DecisionPolicy
    # fixture, exactly like tests/test_advisory_system.py's own tests.

    def test_real_advisory_report_drives_control_requests_end_to_end(self):

        building = make_building()
        scenario = make_scenario()

        ground_truth = make_ground_truth(
            maximum_hazard_zone="zone-b", hazard_spread_order=("zone-b",),
            zone_risk_scores=[{"zone_id": "zone-b", "risk_score": 0.9}],
            stair_risk_scores=[{"stair_id": "stair-1", "risk_score": 0.8}],
        )
        decision_policy = make_decision_policy(
            zone_decisions=[], stair_decisions=[{"stair_id": "stair-1", "status": AVOID}],
        )

        inputs = AdvisoryInputs(building=building, scenario=scenario, ground_truth=ground_truth, decision_policy=decision_policy)
        report = AdvisoryOrchestrator().generate_report(inputs)

        actions = [entry.action for entry in report.building_recommendations]
        self.assertTrue(any(a.startswith("Activate Stair Pressurization") for a in actions))
        self.assertTrue(any(a.startswith("Activate Smoke Exhaust") for a in actions))
        self.assertTrue(any(a.startswith("Activate Deluge") for a in actions))

        requests = translate_report(report)
        self.assertGreaterEqual(len(requests), 3)

        executor = RecordingActionExecutor(applied=True)
        provider = SimulationControlProvider(building, executor)
        controller = BuildingControlController(building, provider)

        for request in requests:
            controller.submit(request)

        for request in controller.pending_requests():
            controller.approve(request.request_id)

        confirmed_systems = {entry.system_type for entry in controller.snapshot().entries}
        self.assertIn(ControlSystemType.STAIR_PRESSURIZATION, confirmed_systems)
        self.assertIn(ControlSystemType.SMOKE_EXHAUST, confirmed_systems)
        self.assertIn(ControlSystemType.DELUGE, confirmed_systems)

        # Every translated request traces back to the recommendation that
        # produced it (Phase 3: source_recommendation_id if available).
        for request in requests:
            self.assertEqual(request.source, RequestSource.ADVISORY_ADAPTER)
            self.assertIsNotNone(request.source_recommendation_id)


# =====================================================
# Phase 14 -- Architecture guards.
# =====================================================


import re

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUILDING_CONTROL_DIR = _REPO_ROOT / "building_control"

# Only actual `import x` / `from x import ...` statements -- matching
# comments/docstrings too (a plain substring search) would flag this
# very module's own explanatory prose about what it deliberately does
# NOT import, so import statements are what's checked, never prose.
_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)", re.MULTILINE)


def _imported_roots(directory: Path):

    roots = set()

    for path in sorted(directory.glob("*.py")):
        for match in _IMPORT_RE.finditer(path.read_text(encoding="utf-8")):
            roots.add(match.group(1).split(".")[0])

    return roots


def _source_text():

    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(_BUILDING_CONTROL_DIR.glob("*.py"))
    )


class ArchitectureGuardTests(unittest.TestCase):

    def test_building_control_does_not_import_ai_or_rl_training(self):

        roots = _imported_roots(_BUILDING_CONTROL_DIR)
        self.assertNotIn("ai_decision", roots)
        self.assertNotIn("rl_training", roots)

    def test_building_control_does_not_import_vendor_protocol_libraries(self):

        roots = {root.lower() for root in _imported_roots(_BUILDING_CONTROL_DIR)}

        for forbidden in ("modbus", "bacnet", "mqtt", "paho", "pymodbus", "bacpypes", "opcua"):
            self.assertNotIn(forbidden, roots)

    def test_building_control_does_not_import_advisory_system(self):

        # advisory_adapter.py works purely by duck-typing the
        # recommendation objects it is handed (.action/.target_type/
        # .target_id/.reason/.confidence) -- it never imports
        # advisory_system itself, so it structurally cannot construct
        # a new BuildingRecommendation or call into that package.
        self.assertNotIn("advisory_system", _imported_roots(_BUILDING_CONTROL_DIR))

    def test_building_control_does_not_talk_to_hardware_or_real_building_systems(self):

        roots = _imported_roots(_BUILDING_CONTROL_DIR)

        for forbidden in ("socket", "serial", "requests", "urllib", "http"):
            self.assertNotIn(forbidden, roots)

    def test_advisory_system_does_not_execute_controls(self):

        self.assertNotIn("building_control", _imported_roots(_REPO_ROOT / "advisory_system"))

    def test_facp_does_not_execute_advisory_recommendations(self):

        roots = _imported_roots(_REPO_ROOT / "facp")
        self.assertNotIn("advisory_system", roots)
        self.assertNotIn("building_control", roots)

    def test_voice_evacuation_remains_separate_from_building_control(self):

        self.assertNotIn("building_control", _imported_roots(_REPO_ROOT / "voice_evacuation"))
        self.assertNotIn("voice_evacuation", _imported_roots(_BUILDING_CONTROL_DIR))

    def test_confirmed_state_carries_no_confidence_field(self):

        # Recommendation confidence must never be treated as hardware-
        # state confidence (Phase 14's explicit guard) -- structurally
        # enforced by these types never carrying a confidence field at
        # all; only ControlRequest (the ask) has one.
        from building_control.requests import ControlInstruction, ControlResult
        from building_control.snapshot import ControlStateEntry

        self.assertNotIn("confidence", ControlInstruction.__dataclass_fields__)
        self.assertNotIn("confidence", ControlResult.__dataclass_fields__)
        self.assertNotIn("confidence", ControlStateEntry.__dataclass_fields__)

    def test_building_control_never_makes_evacuation_or_ai_decisions(self):

        roots = _imported_roots(_BUILDING_CONTROL_DIR)
        for forbidden in ("decision_policy", "behaviour_profile_resolver", "ai_decision"):
            self.assertNotIn(forbidden, roots)


if __name__ == "__main__":
    unittest.main()
