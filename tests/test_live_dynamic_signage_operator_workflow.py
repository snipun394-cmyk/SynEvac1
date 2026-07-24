import sys
import unittest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.obstacle import Obstacle
from models.staircase import Staircase
from models.zone import Zone
from models.dynamic_sign import DynamicEvacuationSign

from navigation.graph_builder import NavigationGraphGenerator

from live_occupants.manager import LiveOccupantManager

from building_state.models import BuildingState

from evacuation_recommendation.engine import EvacuationRecommendationEngine

from evacuation_guidance.engine import EvacuationGuidanceEngine

from dynamic_signage.controller import DynamicSignageController, SignageRequestStatus
from dynamic_signage.models import DynamicSignageSnapshot, SignIndication as SI, SignageInstruction, SignageStatus
from dynamic_signage.planner import DynamicSignagePlanner
from dynamic_signage.provider import DynamicSignageProvider, SignageApplyResult, SimulationDynamicSignageProvider

from command_center.data_source import CommandCenterMode
from command_center.live_operator_action_gateway import (
    LiveOperatorActionGateway, OperatorActionUnavailable, PROVIDER_CAPABILITY_LIVE_HARDWARE,
    PROVIDER_CAPABILITY_NO_PROVIDER, PROVIDER_CAPABILITY_SIMULATION, SignageApprovalBlocked,
)
from command_center.live_dynamic_signage_panel import LiveDynamicSignagePanel

from live_system.event_bus import EventBus
from live_system.state_manager import StateManager
from live_system.live_command_center_gateway import LiveCommandCenterDataSource

from tests.dynamic_signage_fixtures import make_guidance_snapshot, make_sign, make_signage_building
from tests.trajectory_intelligence_fixtures import make_building_state


# =====================================================
# Live Dynamic Sign Operator Approval & Dispatch Completion milestone --
# closes the one remaining connectivity gap the Designer/Asset
# Connectivity audit found: DynamicSignagePlanner/DynamicSignageController/
# LiveOperatorActionGateway already existed with a complete approval
# workflow, but (1) CommandCenterSnapshot carried no dynamic_signage
# field at all, so LiveCommandCenterDataSource never surfaced a planned
# signage instruction to Command Center, and (2) no Command Center panel
# ever called ingest_signage_instructions()/approve_signage_instruction()
# in production -- both closed here (command_center/data_source.py,
# live_system/live_command_center_gateway.py, the new command_center/
# live_dynamic_signage_panel.py wired into command_center/dashboard.py).
#
# This file proves the completed chain end to end: Evacuation
# Recommendation -> Evacuation Guidance -> Dynamic Signage Planner ->
# Command Center -> Human Operator Approve/Reject -> LiveOperatorAction
# Gateway -> DynamicSignageController -> SimulationDynamicSignageProvider
# -> status/history, plus every safety-boundary/failure-mode requirement
# the milestone specifies.
# =====================================================


def _build_reroute_building():

    # Mirrors tests/test_obstacle_navigation_integration.py's own
    # `_build_worked_building()` topology exactly (already proven, in
    # that milestone's own test suite, to migrate a recommendation/
    # guidance plan from E1 to E2 the moment an obstacle blocks D1) --
    # reconstructed locally rather than imported so this file stays
    # self-contained. A Dynamic Sign posted in ZONE-A (the occupied
    # zone, itself neither exit's own zone) will show a directional
    # arrow toward whichever door is the current meaningful next step.

    building = Building(name="Signage Reroute Building")
    floor = building.create_floor(name="Ground")

    zone_a = Zone(id="ZONE-A", name="A", floor_id=floor.id, x=0.0, y=0.0, width=10.0, height=10.0)
    zone_b = Zone(id="ZONE-B", name="B", floor_id=floor.id, x=20.0, y=0.0, width=10.0, height=10.0)
    zone_c = Zone(id="ZONE-C", name="C", floor_id=floor.id, x=0.0, y=50.0, width=10.0, height=10.0)
    floor.add_zone(zone_a)
    floor.add_zone(zone_b)
    floor.add_zone(zone_c)

    door_1 = Door(
        id="D1", name="D1", floor_id=floor.id, start_point=(10.0, 5.0), end_point=(20.0, 5.0),
        zone_a_id="ZONE-A", zone_b_id="ZONE-B",
    )
    door_2 = Door(
        id="D2", name="D2", floor_id=floor.id, start_point=(5.0, 10.0), end_point=(5.0, 50.0),
        zone_a_id="ZONE-A", zone_b_id="ZONE-C",
    )
    floor.add_door(door_1)
    floor.add_door(door_2)

    exit_1 = Exit(id="E1", name="E1", floor_id=floor.id, start_point=(20.0, 0.0), end_point=(20.0, 10.0), zone_id="ZONE-B")
    exit_2 = Exit(id="E2", name="E2", floor_id=floor.id, start_point=(0.0, 50.0), end_point=(0.0, 60.0), zone_id="ZONE-C")
    floor.add_exit(exit_1)
    floor.add_exit(exit_2)

    return building, floor


def _occupant_manager(zone_id, floor_id):

    manager = LiveOccupantManager()
    manager.update("OCC-1", None, None, zone_id, floor_id, None, None, None, 0.9, 0.0)
    return manager


def _compute_chain(building, graph, occupant_manager, time):

    rec_engine = EvacuationRecommendationEngine(building, graph, occupant_manager)
    rec_snapshot = rec_engine.compute(time, building_state=BuildingState())

    guidance_engine = EvacuationGuidanceEngine(building, graph)
    guidance_snapshot = guidance_engine.compute(time, rec_snapshot, BuildingState())

    return guidance_snapshot


# =====================================================
# Phase 6 -- Obstacle-triggered route invalidation, full chain.
# =====================================================


class RouteInvalidationE2ETests(unittest.TestCase):

    def setUp(self):

        self.building, self.floor = _build_reroute_building()
        self.occupant_manager = _occupant_manager("ZONE-A", self.floor.id)

        self.sign = make_sign("DS-1", floor_id=self.floor.id, zone_ids=("ZONE-A",), position=(9.0, 5.0), orientation=0.0)

        self.planner = DynamicSignagePlanner(self.building)
        self.provider = SimulationDynamicSignageProvider()
        self.controller = DynamicSignageController(self.provider)
        self.gateway = LiveOperatorActionGateway(signage_controller=self.controller)

    def _cycle(self, time):

        graph = NavigationGraphGenerator().build(self.building)
        guidance_snapshot = _compute_chain(self.building, graph, self.occupant_manager, time)
        signage_snapshot = self.planner.compute(time, guidance_snapshot, [self.sign])

        return guidance_snapshot, signage_snapshot

    def test_initial_plan_points_toward_e1_and_requires_approval_before_dispatch(self):

        guidance_snapshot, signage_snapshot = self._cycle(0.0)
        plan = guidance_snapshot.zone("ZONE-A")

        self.assertEqual(plan.recommended_exit_id, "E1")

        instruction = signage_snapshot.instruction("DS-1")
        self.assertEqual(instruction.status, SignageStatus.ACTIVE)
        self.assertEqual(instruction.recommended_exit_id, "E1")
        self.assertEqual(instruction.target_asset_id, "D1")

        self.gateway.ingest_signage_instructions(signage_snapshot)
        self.assertIsNone(self.provider.current_indication("DS-1"))

        self.gateway.approve_signage_instruction(instruction, 0.5, guidance_snapshot)
        self.assertEqual(self.provider.current_indication("DS-1").target_asset_id, "D1")

    def test_obstacle_migrates_plan_supersedes_old_instruction_requires_fresh_approval(self):

        guidance_snapshot_1, signage_snapshot_1 = self._cycle(0.0)
        instruction_v1 = signage_snapshot_1.instruction("DS-1")

        self.gateway.ingest_signage_instructions(signage_snapshot_1)
        self.gateway.approve_signage_instruction(instruction_v1, 0.5, guidance_snapshot_1)
        self.assertEqual(self.provider.current_indication("DS-1").target_asset_id, "D1")

        # ---- Activate the obstacle -- E1's own door D1 becomes blocked ----
        obstacle = Obstacle(id="O1", floor_id=self.floor.id, x=14.0, y=4.0, length=2.0, width=2.0, active=True, traversability="Blocked")
        self.floor.obstacles.append(obstacle)

        guidance_snapshot_2, signage_snapshot_2 = self._cycle(1.0)
        plan_2 = guidance_snapshot_2.zone("ZONE-A")
        self.assertEqual(plan_2.recommended_exit_id, "E2")

        instruction_v2 = signage_snapshot_2.instruction("DS-1")
        self.assertEqual(instruction_v2.recommended_exit_id, "E2")
        self.assertEqual(instruction_v2.target_asset_id, "D2")
        self.assertNotEqual(instruction_v2.signage_revision, instruction_v1.signage_revision)

        # Old (revision 1) instruction is untouched history -- CONFIRMED,
        # never retroactively marked superseded, and never re-approvable.
        self.assertEqual(
            self.controller.status_of("DS-1", instruction_v1.signage_revision), SignageRequestStatus.CONFIRMED,
        )
        with self.assertRaises(ValueError):
            self.gateway.approve_signage_instruction(instruction_v1, 2.0, guidance_snapshot_2)

        # Provider still shows the OLD instruction until the NEW
        # revision is itself explicitly approved.
        submitted = self.gateway.ingest_signage_instructions(signage_snapshot_2)
        self.assertEqual({i.sign_id for i in submitted}, {"DS-1"})
        self.assertEqual(
            self.controller.status_of("DS-1", instruction_v2.signage_revision), SignageRequestStatus.PENDING_APPROVAL,
        )
        self.assertEqual(self.provider.current_indication("DS-1").target_asset_id, "D1")

        self.gateway.approve_signage_instruction(instruction_v2, 2.5, guidance_snapshot_2)
        self.assertEqual(self.provider.current_indication("DS-1").target_asset_id, "D2")
        self.assertEqual(self.provider.current_indication("DS-1").recommended_exit_id, "E2")


# =====================================================
# Phase 5 -- Revision safety, the milestone's own exact worked example
# shape (hand-built instructions, controller-level, no graph needed).
# =====================================================


class RevisionSafetyTests(unittest.TestCase):

    def setUp(self):

        self.provider = SimulationDynamicSignageProvider()
        self.controller = DynamicSignageController(self.provider)
        self.gateway = LiveOperatorActionGateway(signage_controller=self.controller)

    def test_stale_revision_cannot_be_approved_after_a_newer_one_supersedes_it(self):

        rev_4 = SignageInstruction(
            sign_id="DS-1", zone_id="Z1", recommended_exit_id="E1", indication=SI.LEFT,
            status=SignageStatus.ACTIVE, signage_revision=4, timestamp=0.0,
        )
        self.controller.submit(rev_4, 0.0)

        rev_5 = SignageInstruction(
            sign_id="DS-1", zone_id="Z1", recommended_exit_id="E2", indication=SI.RIGHT,
            status=SignageStatus.ACTIVE, signage_revision=5, timestamp=1.0,
        )
        self.controller.submit(rev_5, 1.0)

        self.assertEqual(self.controller.status_of("DS-1", 4), SignageRequestStatus.SUPERSEDED)
        self.assertEqual(self.controller.status_of("DS-1", 5), SignageRequestStatus.PENDING_APPROVAL)

        with self.assertRaises(ValueError):
            self.gateway.approve_signage_instruction(rev_4, 2.0)

        self.assertIsNone(self.provider.current_indication("DS-1"))

        self.gateway.approve_signage_instruction(rev_5, 2.0)
        self.assertEqual(self.provider.current_indication("DS-1").indication, SI.RIGHT)


# =====================================================
# Phase 7 -- Voice/Guidance consistency gates approval.
# =====================================================


class VoiceGuidanceConsistencyTests(unittest.TestCase):

    def setUp(self):

        self.building = make_signage_building()
        _, self.guidance_snapshot = make_guidance_snapshot(self.building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")

        self.provider = SimulationDynamicSignageProvider()
        self.controller = DynamicSignageController(self.provider)
        self.gateway = LiveOperatorActionGateway(signage_controller=self.controller)

    def test_approval_blocked_when_sign_disagrees_with_current_guidance(self):

        plan = self.guidance_snapshot.zone("z2")

        mismatched = SignageInstruction(
            sign_id="DS-4", zone_id="z2", recommended_exit_id="EXIT-WRONG",
            guidance_revision=plan.revision, indication=SI.STRAIGHT, status=SignageStatus.ACTIVE,
            signage_revision=1, timestamp=0.0,
        )
        self.controller.submit(mismatched, 0.0)

        with self.assertRaises(SignageApprovalBlocked):
            self.gateway.approve_signage_instruction(mismatched, 0.5, self.guidance_snapshot)

        # Never dispatched, still honestly PENDING_APPROVAL -- an
        # honest failure, not a silent no-op that pretends to succeed.
        self.assertEqual(self.controller.status_of("DS-4", 1), SignageRequestStatus.PENDING_APPROVAL)
        self.assertIsNone(self.provider.current_indication("DS-4"))

    def test_approval_succeeds_when_sign_agrees_with_current_guidance(self):

        plan = self.guidance_snapshot.zone("z2")

        matching = SignageInstruction(
            sign_id="DS-5", zone_id="z2", recommended_exit_id=plan.recommended_exit_id,
            guidance_revision=plan.revision, indication=SI.STRAIGHT, status=SignageStatus.ACTIVE,
            signage_revision=1, timestamp=0.0,
        )
        self.controller.submit(matching, 0.0)

        self.gateway.approve_signage_instruction(matching, 0.5, self.guidance_snapshot)
        self.assertIsNotNone(self.provider.current_indication("DS-5"))

    def test_no_guidance_snapshot_supplied_skips_the_gate_entirely(self):

        # Backward-compatible default (every pre-existing test/caller
        # that never passes guidance_snapshot at all) -- never a
        # behavior change for a caller that doesn't opt in.

        plan = self.guidance_snapshot.zone("z2")

        mismatched = SignageInstruction(
            sign_id="DS-6", zone_id="z2", recommended_exit_id="EXIT-WRONG",
            guidance_revision=plan.revision, indication=SI.STRAIGHT, status=SignageStatus.ACTIVE,
            signage_revision=1, timestamp=0.0,
        )
        self.controller.submit(mismatched, 0.0)

        self.gateway.approve_signage_instruction(mismatched, 0.5)
        self.assertIsNotNone(self.provider.current_indication("DS-6"))


# =====================================================
# Phase 8 -- Provider capability vocabulary.
# =====================================================


class _FakeLiveSignageProvider(DynamicSignageProvider):

    is_simulation_only = False

    def apply(self, instruction):
        return SignageApplyResult(confirmed=True, message="applied")


class _FailingSignageProvider(DynamicSignageProvider):

    is_simulation_only = True

    def apply(self, instruction):
        return SignageApplyResult(confirmed=False, message="simulated provider failure")


class ProviderCapabilityTests(unittest.TestCase):

    def test_no_provider_reports_no_provider(self):

        gateway = LiveOperatorActionGateway()
        self.assertEqual(gateway.signage_capability, PROVIDER_CAPABILITY_NO_PROVIDER)

        with self.assertRaises(OperatorActionUnavailable):
            gateway.approve_signage_instruction(
                SignageInstruction(sign_id="DS-1", status=SignageStatus.ACTIVE, signage_revision=1), 0.0,
            )

    def test_simulation_provider_reports_simulation(self):

        controller = DynamicSignageController(SimulationDynamicSignageProvider())
        gateway = LiveOperatorActionGateway(signage_controller=controller)

        self.assertEqual(gateway.signage_capability, PROVIDER_CAPABILITY_SIMULATION)

    def test_hypothetical_non_simulation_provider_reports_live_hardware(self):

        # Never fabricated -- only reached when a REAL provider with
        # is_simulation_only=False actually exists (Phase 8's own
        # explicit "do NOT fabricate LIVE capability").

        controller = DynamicSignageController(_FakeLiveSignageProvider())
        gateway = LiveOperatorActionGateway(signage_controller=controller)

        self.assertEqual(gateway.signage_capability, PROVIDER_CAPABILITY_LIVE_HARDWARE)

    def test_provider_reported_failure_becomes_failed_status_never_confirmed(self):

        controller = DynamicSignageController(_FailingSignageProvider())
        gateway = LiveOperatorActionGateway(signage_controller=controller)

        instruction = SignageInstruction(
            sign_id="DS-1", zone_id="z1", indication=SI.STRAIGHT, status=SignageStatus.ACTIVE, signage_revision=1,
        )
        controller.submit(instruction, 0.0)

        gateway.approve_signage_instruction(instruction, 1.0)

        self.assertEqual(controller.status_of("DS-1", 1), SignageRequestStatus.FAILED)


# =====================================================
# Phase 9 -- Rejection.
# =====================================================


class RejectionTests(unittest.TestCase):

    def setUp(self):

        self.provider = SimulationDynamicSignageProvider()
        self.controller = DynamicSignageController(self.provider)
        self.event_bus = EventBus()
        self.gateway = LiveOperatorActionGateway(signage_controller=self.controller, event_bus=self.event_bus)

    def test_reject_never_dispatches_and_remains_visible_in_history(self):

        instruction = SignageInstruction(
            sign_id="DS-1", zone_id="z1", indication=SI.STRAIGHT, status=SignageStatus.ACTIVE, signage_revision=1,
        )
        self.controller.submit(instruction, 0.0)

        self.gateway.reject_signage_instruction(instruction, 1.0)

        self.assertIsNone(self.provider.current_indication("DS-1"))
        self.assertEqual(self.controller.status_of("DS-1", 1), SignageRequestStatus.REJECTED)

        events = [event.to_status for event in self.controller.history() if event.request_key == "DS-1::1"]
        self.assertIn(SignageRequestStatus.REJECTED, events)

    def test_rejection_does_not_blacklist_the_sign_a_genuinely_new_revision_still_appears(self):

        instruction_v1 = SignageInstruction(
            sign_id="DS-1", zone_id="z1", indication=SI.STRAIGHT, status=SignageStatus.ACTIVE, signage_revision=1,
        )
        self.controller.submit(instruction_v1, 0.0)
        self.gateway.reject_signage_instruction(instruction_v1, 0.5)

        instruction_v2 = SignageInstruction(
            sign_id="DS-1", zone_id="z1", indication=SI.LEFT, status=SignageStatus.ACTIVE, signage_revision=2,
        )
        self.controller.submit(instruction_v2, 1.0)

        self.assertEqual(self.controller.status_of("DS-1", 2), SignageRequestStatus.PENDING_APPROVAL)

        self.gateway.approve_signage_instruction(instruction_v2, 1.5)
        self.assertEqual(self.provider.current_indication("DS-1").indication, SI.LEFT)


# =====================================================
# Phase 10 -- Multiple signs, independent decisions.
# =====================================================


class MultipleSignsIndependenceTests(unittest.TestCase):

    def test_approving_one_sign_never_affects_another(self):

        provider = SimulationDynamicSignageProvider()
        controller = DynamicSignageController(provider)
        gateway = LiveOperatorActionGateway(signage_controller=controller)

        ds1 = SignageInstruction(sign_id="DS-1", zone_id="ZONE-A", indication=SI.LEFT, status=SignageStatus.ACTIVE, signage_revision=1)
        ds2 = SignageInstruction(sign_id="DS-2", zone_id="ZONE-A", indication=SI.RIGHT, status=SignageStatus.ACTIVE, signage_revision=1)
        ds3 = SignageInstruction(sign_id="DS-3", zone_id="ZONE-B", indication=SI.STRAIGHT, status=SignageStatus.ACTIVE, signage_revision=1)

        for instruction in (ds1, ds2, ds3):
            controller.submit(instruction, 0.0)

        gateway.approve_signage_instruction(ds1, 1.0)
        gateway.reject_signage_instruction(ds2, 1.0)

        self.assertEqual(controller.status_of("DS-1", 1), SignageRequestStatus.CONFIRMED)
        self.assertEqual(controller.status_of("DS-2", 1), SignageRequestStatus.REJECTED)
        self.assertEqual(controller.status_of("DS-3", 1), SignageRequestStatus.PENDING_APPROVAL)

        self.assertIsNotNone(provider.current_indication("DS-1"))
        self.assertIsNone(provider.current_indication("DS-2"))
        self.assertIsNone(provider.current_indication("DS-3"))


# =====================================================
# Phase 11 -- Conflict handling: an Approve button existing is never
# itself dispatch permission.
# =====================================================


class ConflictHandlingTests(unittest.TestCase):

    def test_conflicted_sign_is_never_submitted_and_never_approvable(self):

        provider = SimulationDynamicSignageProvider()
        controller = DynamicSignageController(provider)
        gateway = LiveOperatorActionGateway(signage_controller=controller)

        conflict_instruction = SignageInstruction(
            sign_id="DS-1", zone_id=None, indication=SI.UNAVAILABLE,
            status=SignageStatus.CONFLICT, signage_revision=1, reason="conflicting zones",
        )
        snapshot = DynamicSignageSnapshot(timestamp=0.0, instructions={"DS-1": conflict_instruction}, conflicts={})

        submitted = gateway.ingest_signage_instructions(snapshot)
        self.assertEqual(submitted, ())
        self.assertEqual(gateway.pending_signage_instructions(), ())

        with self.assertRaises(KeyError):
            gateway.approve_signage_instruction(conflict_instruction, 1.0)


# =====================================================
# Panel rendering -- Phase 3's own required columns, Phase 7/11's own
# "an Approve button existing is never itself dispatch permission."
# =====================================================


class PanelRenderingTests(unittest.TestCase):

    def setUp(self):

        self.provider = SimulationDynamicSignageProvider()
        self.controller = DynamicSignageController(self.provider)
        self.gateway = LiveOperatorActionGateway(signage_controller=self.controller)
        self.panel = LiveDynamicSignagePanel()

        self.building = make_signage_building()
        _, self.guidance_snapshot = make_guidance_snapshot(self.building, zone_id="z2", floor_id="f1", exit_id="EXIT-1")
        self.plan = self.guidance_snapshot.zone("z2")

    def test_active_row_shows_approve_and_reject_buttons(self):

        instruction = SignageInstruction(
            sign_id="DS-1", zone_id="z2", recommended_exit_id=self.plan.recommended_exit_id,
            guidance_revision=self.plan.revision, indication=SI.STRAIGHT, status=SignageStatus.ACTIVE,
            signage_revision=1, timestamp=0.0,
        )
        snapshot = DynamicSignageSnapshot(timestamp=0.0, instructions={"DS-1": instruction}, conflicts={})

        self.panel.show_live(snapshot, self.guidance_snapshot, self.gateway, 0.0)

        self.assertEqual(self.panel.sign_table.rowCount(), 1)
        self.assertEqual(self.panel.sign_table.item(0, 7).text(), "OK")
        self.assertIsNotNone(self.panel.sign_table.cellWidget(0, 9))

    def test_inconsistent_row_shows_no_approve_button(self):

        instruction = SignageInstruction(
            sign_id="DS-1", zone_id="z2", recommended_exit_id="EXIT-WRONG",
            guidance_revision=self.plan.revision, indication=SI.STRAIGHT, status=SignageStatus.ACTIVE,
            signage_revision=1, timestamp=0.0,
        )
        snapshot = DynamicSignageSnapshot(timestamp=0.0, instructions={"DS-1": instruction}, conflicts={})

        self.panel.show_live(snapshot, self.guidance_snapshot, self.gateway, 0.0)

        self.assertIsNone(self.panel.sign_table.cellWidget(0, 9))
        self.assertIn("Approval unavailable", self.panel.sign_table.item(0, 9).text())

        # Never actually reached the controller either.
        self.assertEqual(self.gateway.pending_signage_instructions(), (instruction,))

    def test_conflict_row_shows_no_actionable_instruction(self):

        from dynamic_signage.models import SignageConflict

        instruction = SignageInstruction(
            sign_id="DS-1", zone_id=None, indication=SI.UNAVAILABLE, status=SignageStatus.CONFLICT,
            signage_revision=1, reason="conflicting zones", timestamp=0.0,
        )
        conflict = SignageConflict(sign_id="DS-1", conflicting_zone_ids=("z1", "z2"), reason="conflicting zones", timestamp=0.0)
        snapshot = DynamicSignageSnapshot(timestamp=0.0, instructions={"DS-1": instruction}, conflicts={"DS-1": conflict})

        self.panel.show_live(snapshot, self.guidance_snapshot, self.gateway, 0.0)

        self.assertIsNone(self.panel.sign_table.cellWidget(0, 9))
        self.assertEqual(self.panel.sign_table.item(0, 9).text(), "No actionable instruction -- nothing to approve")
        self.assertIn("CONFLICT", self.panel.sign_table.item(0, 7).text())
        self.assertEqual(self.gateway.pending_signage_instructions(), ())

    def test_approve_click_routes_through_gateway_and_dispatches(self):

        instruction = SignageInstruction(
            sign_id="DS-1", zone_id="z2", recommended_exit_id=self.plan.recommended_exit_id,
            guidance_revision=self.plan.revision, indication=SI.STRAIGHT, status=SignageStatus.ACTIVE,
            signage_revision=1, timestamp=0.0,
        )
        snapshot = DynamicSignageSnapshot(timestamp=0.0, instructions={"DS-1": instruction}, conflicts={})

        self.panel.show_live(snapshot, self.guidance_snapshot, self.gateway, 0.0)
        self.panel._on_approve(instruction)

        self.assertIsNotNone(self.provider.current_indication("DS-1"))
        self.assertEqual(self.panel.sign_table.item(0, 8).text(), SignageRequestStatus.CONFIRMED)

    def test_reject_click_routes_through_gateway_and_never_dispatches(self):

        instruction = SignageInstruction(
            sign_id="DS-1", zone_id="z2", recommended_exit_id=self.plan.recommended_exit_id,
            guidance_revision=self.plan.revision, indication=SI.STRAIGHT, status=SignageStatus.ACTIVE,
            signage_revision=1, timestamp=0.0,
        )
        snapshot = DynamicSignageSnapshot(timestamp=0.0, instructions={"DS-1": instruction}, conflicts={})

        self.panel.show_live(snapshot, self.guidance_snapshot, self.gateway, 0.0)
        self.panel._on_reject(instruction)

        self.assertIsNone(self.provider.current_indication("DS-1"))
        self.assertEqual(self.panel.sign_table.item(0, 8).text(), SignageRequestStatus.REJECTED)

    def test_no_gateway_still_renders_without_crashing(self):

        instruction = SignageInstruction(
            sign_id="DS-1", zone_id="z2", recommended_exit_id=self.plan.recommended_exit_id,
            guidance_revision=self.plan.revision, indication=SI.STRAIGHT, status=SignageStatus.ACTIVE,
            signage_revision=1, timestamp=0.0,
        )
        snapshot = DynamicSignageSnapshot(timestamp=0.0, instructions={"DS-1": instruction}, conflicts={})

        panel = LiveDynamicSignagePanel()
        panel.show_live(snapshot, self.guidance_snapshot, None, 0.0)

        self.assertEqual(panel.sign_table.rowCount(), 1)
        self.assertIsNotNone(panel.sign_table.cellWidget(0, 9))


# =====================================================
# Phase 12 -- Full offline E2E through the real production wiring:
# CommandCenterSnapshot.dynamic_signage / LiveDynamicSignagePanel /
# LiveOperatorActionGateway, driven by a real StateManager cycle.
# =====================================================


class ProductionWiringOfflineE2ETests(unittest.TestCase):

    def setUp(self):

        self.building, self.floor = _build_reroute_building()
        self.sign = make_sign("DS-1", floor_id=self.floor.id, zone_ids=("ZONE-A",), position=(9.0, 5.0), orientation=0.0)

        self.provider = SimulationDynamicSignageProvider()
        self.controller = DynamicSignageController(self.provider)
        self.gateway = LiveOperatorActionGateway(signage_controller=self.controller)

        self.state_manager = StateManager()
        self.data_source = LiveCommandCenterDataSource(self.state_manager, building=self.building)
        self.data_source.start()

        self.panel = LiveDynamicSignagePanel()

    def _drive_one_cycle(self, time):

        graph = NavigationGraphGenerator().build(self.building)
        occupant_manager = _occupant_manager("ZONE-A", self.floor.id)
        guidance_snapshot = _compute_chain(self.building, graph, occupant_manager, time)

        planner = DynamicSignagePlanner(self.building)
        signage_snapshot = planner.compute(time, guidance_snapshot, [self.sign])

        self.state_manager.update_building_state(BuildingState(timestamp=time), time)
        self.state_manager.update_dynamic_signage(signage_snapshot, time)

        return guidance_snapshot, signage_snapshot

    def test_snapshot_carries_dynamic_signage_and_operator_can_approve_from_the_panel(self):

        guidance_snapshot, signage_snapshot = self._drive_one_cycle(0.0)

        snapshot = self.data_source.current_snapshot()
        self.assertIs(snapshot.dynamic_signage, signage_snapshot)
        self.assertEqual(snapshot.dynamic_signage_timestamp, 0.0)

        self.panel.show_live(snapshot.dynamic_signage, snapshot.evacuation_guidance, self.gateway, snapshot.timestamp)

        # Nothing dispatched merely by rendering the panel.
        self.assertIsNone(self.provider.current_indication("DS-1"))

        instruction = signage_snapshot.instruction("DS-1")
        self.panel._on_approve(instruction)

        self.assertIsNotNone(self.provider.current_indication("DS-1"))
        self.assertEqual(self.provider.current_indication("DS-1").recommended_exit_id, "E1")

    def test_replay_mode_snapshot_never_fabricates_dynamic_signage(self):

        from command_center.data_source import ReplayCommandCenterDataSource

        replay = ReplayCommandCenterDataSource()
        snapshot = replay.current_snapshot()

        self.assertIsNone(snapshot.dynamic_signage)
        self.assertEqual(snapshot.mode, CommandCenterMode.REPLAY)


# =====================================================
# Phase 13 -- Failure modes: no crash, no fabricated success.
# =====================================================


class FailureModeTests(unittest.TestCase):

    def setUp(self):

        self.provider = SimulationDynamicSignageProvider()
        self.controller = DynamicSignageController(self.provider)
        self.gateway = LiveOperatorActionGateway(signage_controller=self.controller)

    def test_unavailable_instruction_is_never_submitted(self):

        unavailable = SignageInstruction(sign_id="DS-1", indication=SI.UNAVAILABLE, status=SignageStatus.UNAVAILABLE)
        snapshot = DynamicSignageSnapshot(timestamp=0.0, instructions={"DS-1": unavailable}, conflicts={})

        submitted = self.gateway.ingest_signage_instructions(snapshot)
        self.assertEqual(submitted, ())

    def test_duplicate_approval_click_raises_rather_than_double_dispatching(self):

        instruction = SignageInstruction(sign_id="DS-1", zone_id="z1", indication=SI.STRAIGHT, status=SignageStatus.ACTIVE, signage_revision=1)
        self.controller.submit(instruction, 0.0)

        self.gateway.approve_signage_instruction(instruction, 1.0)
        applied_count_after_first = len(self.provider.applied_instructions())

        with self.assertRaises(ValueError):
            self.gateway.approve_signage_instruction(instruction, 1.1)

        self.assertEqual(len(self.provider.applied_instructions()), applied_count_after_first)

    def test_duplicate_rejection_click_raises_rather_than_crashing_silently(self):

        instruction = SignageInstruction(sign_id="DS-1", zone_id="z1", indication=SI.STRAIGHT, status=SignageStatus.ACTIVE, signage_revision=1)
        self.controller.submit(instruction, 0.0)

        self.gateway.reject_signage_instruction(instruction, 1.0)

        with self.assertRaises(ValueError):
            self.gateway.reject_signage_instruction(instruction, 1.1)

    def test_approval_after_rejection_is_refused(self):

        instruction = SignageInstruction(sign_id="DS-1", zone_id="z1", indication=SI.STRAIGHT, status=SignageStatus.ACTIVE, signage_revision=1)
        self.controller.submit(instruction, 0.0)

        self.gateway.reject_signage_instruction(instruction, 1.0)

        with self.assertRaises(ValueError):
            self.gateway.approve_signage_instruction(instruction, 1.1)

        self.assertIsNone(self.provider.current_indication("DS-1"))

    def test_rejection_after_approval_is_refused(self):

        instruction = SignageInstruction(sign_id="DS-1", zone_id="z1", indication=SI.STRAIGHT, status=SignageStatus.ACTIVE, signage_revision=1)
        self.controller.submit(instruction, 0.0)

        self.gateway.approve_signage_instruction(instruction, 1.0)

        with self.assertRaises(ValueError):
            self.gateway.reject_signage_instruction(instruction, 1.1)

        self.assertIsNotNone(self.provider.current_indication("DS-1"))

    def test_panel_click_on_a_stale_row_never_crashes_and_never_dispatches(self):

        # "Command Center refresh between planning and click" (Phase 13)
        # -- a newer revision supersedes the row an operator is about to
        # click before the click is processed. The panel checks status
        # first (mirrors BuildingControlsPanel's own Replay-path
        # precedent) and must neither raise nor dispatch the stale row.

        panel = LiveDynamicSignagePanel()
        panel._gateway = self.gateway
        panel._signage_snapshot = DynamicSignageSnapshot(timestamp=0.0, instructions={}, conflicts={})
        panel._guidance_snapshot = None
        panel._time = 1.0

        instruction_v1 = SignageInstruction(sign_id="DS-1", zone_id="z1", indication=SI.STRAIGHT, status=SignageStatus.ACTIVE, signage_revision=1)
        self.controller.submit(instruction_v1, 0.0)

        instruction_v2 = SignageInstruction(sign_id="DS-1", zone_id="z1", indication=SI.LEFT, status=SignageStatus.ACTIVE, signage_revision=2)
        self.controller.submit(instruction_v2, 0.5)

        # instruction_v1 is now SUPERSEDED -- clicking Approve on it
        # (the stale row the operator was looking at) must not crash.
        panel._on_approve(instruction_v1)

        self.assertIsNone(self.provider.current_indication("DS-1"))

    def test_no_provider_configured_disables_approval_honestly(self):

        gateway = LiveOperatorActionGateway()

        instruction = SignageInstruction(sign_id="DS-1", zone_id="z1", indication=SI.STRAIGHT, status=SignageStatus.ACTIVE, signage_revision=1)

        with self.assertRaises(OperatorActionUnavailable):
            gateway.approve_signage_instruction(instruction, 0.0)

        with self.assertRaises(OperatorActionUnavailable):
            gateway.reject_signage_instruction(instruction, 0.0)


# =====================================================
# Phase 15 -- Dynamic Sign and Voice Guidance remain independently
# operator-controlled, even though both are consistency-checked
# against the same Guidance.
# =====================================================


class VoiceSignageIndependenceTests(unittest.TestCase):

    def test_approving_a_sign_never_broadcasts_voice(self):

        from voice_evacuation.controller import VoiceEvacuationController
        from voice_evacuation.provider import SimulationVoiceOutputProvider
        from speaker_manager.manager import SpeakerManager

        signage_controller = DynamicSignageController(SimulationDynamicSignageProvider())
        voice_controller = VoiceEvacuationController(SpeakerManager(), SimulationVoiceOutputProvider())

        gateway = LiveOperatorActionGateway(signage_controller=signage_controller, voice_controller=voice_controller)

        instruction = SignageInstruction(sign_id="DS-1", zone_id="z1", indication=SI.STRAIGHT, status=SignageStatus.ACTIVE, signage_revision=1)
        signage_controller.submit(instruction, 0.0)

        gateway.approve_signage_instruction(instruction, 1.0)

        self.assertEqual(len(voice_controller.broadcast_log.all_instructions()), 0)

    def test_approving_voice_never_approves_any_sign(self):

        from voice_evacuation.controller import VoiceEvacuationController
        from voice_evacuation.provider import SimulationVoiceOutputProvider
        from advisory_system.recommendation_models import CivilianAnnouncement
        from speaker_manager.manager import SpeakerManager
        from models.speaker import Speaker

        signage_controller = DynamicSignageController(SimulationDynamicSignageProvider())

        speaker_manager = SpeakerManager()
        speaker_manager.register_speaker(Speaker(name="SPK-1", floor_id="f1", zone_ids=("z1",)))
        voice_controller = VoiceEvacuationController(speaker_manager, SimulationVoiceOutputProvider())

        gateway = LiveOperatorActionGateway(signage_controller=signage_controller, voice_controller=voice_controller)

        instruction = SignageInstruction(sign_id="DS-1", zone_id="z1", indication=SI.STRAIGHT, status=SignageStatus.ACTIVE, signage_revision=1)
        signage_controller.submit(instruction, 0.0)

        announcement = CivilianAnnouncement(
            zone_id="z1", zone_name="Zone 1", announcement="Evacuate now.", confidence=0.9, reason="test",
            predicted_rset_improvement_seconds=None,
        )
        gateway.approve_voice_message(announcement, 1.0)

        self.assertEqual(gateway.signage_instruction_status(instruction), SignageRequestStatus.PENDING_APPROVAL)
        self.assertIsNone(signage_controller.provider.current_indication("DS-1"))


# =====================================================
# Phase 16 -- History retains every superseded/rejected/failed entry.
# =====================================================


class HistoryTests(unittest.TestCase):

    def test_history_never_erases_superseded_or_rejected_entries(self):

        provider = SimulationDynamicSignageProvider()
        controller = DynamicSignageController(provider)
        gateway = LiveOperatorActionGateway(signage_controller=controller)

        rejected = SignageInstruction(sign_id="DS-1", zone_id="z1", indication=SI.LEFT, status=SignageStatus.ACTIVE, signage_revision=1)
        controller.submit(rejected, 0.0)
        gateway.reject_signage_instruction(rejected, 0.5)

        superseded = SignageInstruction(sign_id="DS-2", zone_id="z1", indication=SI.RIGHT, status=SignageStatus.ACTIVE, signage_revision=1)
        controller.submit(superseded, 0.0)
        newer = SignageInstruction(sign_id="DS-2", zone_id="z1", indication=SI.STRAIGHT, status=SignageStatus.ACTIVE, signage_revision=2)
        controller.submit(newer, 1.0)

        history = gateway.signage_history()
        keys_and_statuses = {(event.request_key, event.to_status) for event in history}

        self.assertIn(("DS-1::1", SignageRequestStatus.REJECTED), keys_and_statuses)
        self.assertIn(("DS-2::1", SignageRequestStatus.SUPERSEDED), keys_and_statuses)
        self.assertIn(("DS-2::1", SignageRequestStatus.PENDING_APPROVAL), keys_and_statuses)


if __name__ == "__main__":
    unittest.main()
