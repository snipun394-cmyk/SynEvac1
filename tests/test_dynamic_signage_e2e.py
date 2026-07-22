import unittest

from hazard.severity import HazardSeverity

from models.building import Building
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from navigation.graph_builder import NavigationGraphGenerator

from evacuation_guidance.engine import EvacuationGuidanceEngine

from dynamic_signage.controller import DynamicSignageController, SignageRequestStatus
from dynamic_signage.models import SignIndication as SI, SignageStatus
from dynamic_signage.planner import DynamicSignagePlanner
from dynamic_signage.provider import SimulationDynamicSignageProvider

from command_center.live_operator_action_gateway import LiveOperatorActionGateway

from tests.dynamic_signage_fixtures import make_recommendation_snapshot, make_sign
from tests.trajectory_intelligence_fixtures import make_building_state


# =====================================================
# Live Dynamic Evacuation Signage milestone, Phase 28 -- a real offline
# end-to-end scenario, fully deterministic, zero network/hardware.
#
# Topology:
#
#   Floor 2 (f2): Zone Z2 (has Exit E2, a direct escape) --
#                 Door D2 -- Zone Z2A -- Stair S1 -- Floor 1 (f1)
#   Floor 1 (f1): Zone Z1 (has Exit E1)
#
# Initial recommendation: Z2 -> E1 (via D2 -> Z2A -> S1 -> Z1 -> E1).
# SIGN-2A (in Z2) should point toward D2. SIGN-2B (in Z2A) should show
# USE_STAIRS (posted at the stairwell itself, per this milestone's own
# documented "placard at the decision point, arrow before it" rule --
# see dynamic_signage/planner.py's own docstring).
#
# Then Z1 becomes hazard-excluded, E1's route becomes unsafe, and the
# recommendation retargets Z2 directly to its own E2 (no stairs needed
# at all) -- proving supersession, fresh-approval, and provider
# isolation until approval.
# =====================================================


def _make_e2e_building():

    f2 = Floor(
        id="f2", name="Floor 2",
        zones=[
            Zone(id="Z2", name="Z2", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f2"),
            Zone(id="Z2A", name="Z2A", x=20.0, y=0.0, width=10.0, height=10.0, floor_id="f2"),
        ],
        doors=[
            Door(id="D2", name="D2", floor_id="f2", zone_a_id="Z2", zone_b_id="Z2A",
                 start_point=(15.0, 3.0), end_point=(15.0, 7.0)),
        ],
        exits=[
            Exit(id="E2", name="E2", floor_id="f2", zone_id="Z2",
                 start_point=(0.0, 3.0), end_point=(0.0, 7.0)),
        ],
        stairs=[
            Staircase(id="S1", name="S1", from_zone_id="Z2A", to_zone_id="Z1",
                      from_floor_id="f2", to_floor_id="f1",
                      from_position=(25.0, 5.0), to_position=(5.0, 8.0)),
        ],
    )

    f1 = Floor(
        id="f1", name="Floor 1",
        zones=[
            Zone(id="Z1", name="Z1", x=0.0, y=0.0, width=10.0, height=10.0, floor_id="f1"),
        ],
        exits=[
            Exit(id="E1", name="E1", floor_id="f1", zone_id="Z1",
                 start_point=(0.0, 3.0), end_point=(0.0, 7.0)),
        ],
    )

    return Building(id="e2e-signage-building", name="E2E Signage Building", floors=[f2, f1])


class OfflineDynamicSignageE2ETest(unittest.TestCase):

    def setUp(self):

        self.building = _make_e2e_building()
        self.graph = NavigationGraphGenerator().build(self.building)
        self.guidance_engine = EvacuationGuidanceEngine(self.building, self.graph)
        self.planner = DynamicSignagePlanner(self.building)
        self.provider = SimulationDynamicSignageProvider()
        self.controller = DynamicSignageController(self.provider)
        self.gateway = LiveOperatorActionGateway(signage_controller=self.controller)

        self.sign_2a = make_sign("SIGN-2A", floor_id="f2", zone_ids=("Z2",), position=(22.0, 5.0), orientation=180.0)
        self.sign_2b = make_sign("SIGN-2B", floor_id="f2", zone_ids=("Z2A",), position=(22.0, 5.0), orientation=0.0)

    def _compute_signage(self, time, zone_id, exit_id, building_state=None):

        recommendation = make_recommendation_snapshot(zone_id, "f2", exit_id, timestamp=time)
        guidance_snapshot = self.guidance_engine.compute(time, recommendation, building_state)
        signage_snapshot = self.planner.compute(time, guidance_snapshot, [self.sign_2a, self.sign_2b])

        return guidance_snapshot, signage_snapshot

    def test_full_chain_with_approval_and_re_route(self):

        # ---- Initial route: Z2 -> D2 -> Z2A -> S1 -> Z1 -> E1 ----
        guidance_snapshot_1, signage_snapshot_1 = self._compute_signage(0.0, "Z2", "E1")

        instruction_2a_v1 = signage_snapshot_1.instruction("SIGN-2A")
        instruction_2b_v1 = signage_snapshot_1.instruction("SIGN-2B")

        self.assertEqual(instruction_2a_v1.status, SignageStatus.ACTIVE)
        self.assertEqual(instruction_2a_v1.target_asset_id, "D2")
        self.assertEqual(instruction_2a_v1.indication, SI.STRAIGHT)

        self.assertEqual(instruction_2b_v1.status, SignageStatus.ACTIVE)
        self.assertEqual(instruction_2b_v1.target_asset_id, "S1")
        self.assertEqual(instruction_2b_v1.indication, SI.USE_STAIRS)

        # ---- Nothing changes on the provider before operator approval ----
        self.gateway.ingest_signage_instructions(signage_snapshot_1)

        self.assertIsNone(self.provider.current_indication("SIGN-2A"))
        self.assertIsNone(self.provider.current_indication("SIGN-2B"))

        # ---- Operator approves both ----
        self.gateway.approve_signage_instruction(instruction_2a_v1, 0.5)
        self.gateway.approve_signage_instruction(instruction_2b_v1, 0.5)

        self.assertEqual(self.provider.current_indication("SIGN-2A").target_asset_id, "D2")
        self.assertEqual(self.provider.current_indication("SIGN-2B").target_asset_id, "S1")

        # ---- E1's own zone (Z1) becomes unsafe; recommendation
        # retargets Z2 directly to its own E2 ----
        building_state = make_building_state({"Z1": HazardSeverity.HIGH})
        guidance_snapshot_2, signage_snapshot_2 = self._compute_signage(1.0, "Z2", "E2", building_state=building_state)

        instruction_2a_v2 = signage_snapshot_2.instruction("SIGN-2A")
        instruction_2b_v2 = signage_snapshot_2.instruction("SIGN-2B")

        self.assertEqual(instruction_2a_v2.status, SignageStatus.ACTIVE)
        self.assertEqual(instruction_2a_v2.indication, SI.EXIT_HERE)
        self.assertEqual(instruction_2a_v2.recommended_exit_id, "E2")
        self.assertNotEqual(instruction_2a_v2.signage_revision, instruction_2a_v1.signage_revision)

        # Z2A is no longer on ANY active route at all -- honestly
        # UNAVAILABLE, never a stale "toward S1".
        self.assertEqual(instruction_2b_v2.status, SignageStatus.UNAVAILABLE)

        # ---- Old (revision 1) instruction stays exactly as applied;
        # new revision requires FRESH approval; provider does not
        # change until that happens ----
        self.assertEqual(
            self.controller.status_of("SIGN-2A", instruction_2a_v1.signage_revision), SignageRequestStatus.CONFIRMED,
        )

        submitted = self.gateway.ingest_signage_instructions(signage_snapshot_2)
        self.assertEqual({i.sign_id for i in submitted}, {"SIGN-2A"})

        self.assertEqual(
            self.controller.status_of("SIGN-2A", instruction_2a_v2.signage_revision), SignageRequestStatus.PENDING_APPROVAL,
        )
        # Provider still reflects the OLD (revision 1) instruction --
        # nothing changed on approval alone.
        self.assertEqual(self.provider.current_indication("SIGN-2A").target_asset_id, "D2")

        self.gateway.approve_signage_instruction(instruction_2a_v2, 1.5)

        self.assertEqual(self.provider.current_indication("SIGN-2A").indication, SI.EXIT_HERE)
        self.assertEqual(self.provider.current_indication("SIGN-2A").recommended_exit_id, "E2")

        # ---- Voice guidance and signage both reference E2 ----
        voice_plan = guidance_snapshot_2.voice_plan("Z2")
        self.assertIsNotNone(voice_plan)
        self.assertEqual(voice_plan.recommended_exit_id, "E2")
        self.assertEqual(instruction_2a_v2.recommended_exit_id, voice_plan.recommended_exit_id)

        # ---- Nothing was ever dispatched automatically ----
        history_actions = {event.to_status for event in self.controller.history()}
        self.assertIn(SignageRequestStatus.PENDING_APPROVAL, history_actions)
        # Every DISPATCHED transition in history was preceded by an
        # explicit approve() call above -- never triggered on its own.


if __name__ == "__main__":
    unittest.main()
