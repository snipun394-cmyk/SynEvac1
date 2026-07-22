import unittest

from models.building import Building
from models.floor import Floor
from models.smoke_detector import SmokeDetector
from models.zone import Zone

from facp.engine import SimulatedFACP
from facp.models import PanelState

from perception.models.smoke_detector_observation import SmokeDetectorReading

from live_runtime.factory import build_offline_demo_runtime


# =====================================================
# Digital Twin Asset -> Zone Assignment & Live FACP Runtime milestone,
# Phase 9/10 -- proves production LiveRuntime now actually TICKS a
# caller-supplied SimulatedFACP every cycle (previously only Designer's
# own debug runner ever called evaluate()), while preserving every
# existing state-machine guarantee (never auto-resets, re-alerts on a
# new source, requires an explicit reset() to clear).
# =====================================================


def _make_building_with_smoke_detector(alarm_active_by_id=None):

    floor = Floor(id="f1", name="Floor 1")
    floor.add_zone(Zone(id="z1", name="Z1", floor_id="f1", width=5.0, height=5.0))
    floor.add_smoke_detector(SmokeDetector(id="SD-1", name="SD-1", floor_id="f1", zone_ids=("z1",)))

    return Building(id="b1", name="B", floors=[floor])


class ProductionFACPTickingTests(unittest.TestCase):

    def test_facp_reaches_alarm_through_a_real_run_cycle(self):

        building = _make_building_with_smoke_detector()
        facp = SimulatedFACP(panel_id="FACP-PROD")

        def smoke_readings(time):
            return [SmokeDetectorReading(detector_id="SD-1", timestamp=time, alarm_active=True)]

        runtime = build_offline_demo_runtime(
            building, facp=facp, smoke_detector_reading_provider=smoke_readings,
        )
        runtime.start()

        snapshot = runtime.run_cycle(1.0)

        self.assertEqual(facp.panel_state, PanelState.ALARM)
        self.assertIsNotNone(snapshot.building_state.facp_status)
        self.assertEqual(snapshot.building_state.facp_status.panel_state, PanelState.ALARM)

        runtime.stop()

    def test_facp_unconfigured_stays_none_backward_compatible(self):

        # Regression: the pre-existing "no FACP configured" degraded
        # path (tests/test_live_runtime_failure_modes.py::
        # FACPUnavailableTests) must remain reachable -- facp=None
        # stays the honest default, never auto-constructed.
        building = _make_building_with_smoke_detector()

        runtime = build_offline_demo_runtime(building)
        runtime.start()

        snapshot = runtime.run_cycle(1.0)

        self.assertIsNone(snapshot.building_state.facp_status)

        runtime.stop()

    def test_acknowledge_still_requires_an_explicit_operator_call(self):

        building = _make_building_with_smoke_detector()
        facp = SimulatedFACP()

        def smoke_readings(time):
            return [SmokeDetectorReading(detector_id="SD-1", timestamp=time, alarm_active=True)]

        runtime = build_offline_demo_runtime(building, facp=facp, smoke_detector_reading_provider=smoke_readings)
        runtime.start()

        runtime.run_cycle(1.0)
        self.assertEqual(facp.panel_state, PanelState.ALARM)

        # Ticking again on its own never acknowledges anything.
        runtime.run_cycle(2.0)
        self.assertEqual(facp.panel_state, PanelState.ALARM)

        facp.acknowledge(3.0)
        self.assertEqual(facp.panel_state, PanelState.ALARM_ACKNOWLEDGED)

        runtime.stop()

    def test_restoration_does_not_auto_clear_the_panel(self):

        building = _make_building_with_smoke_detector()
        facp = SimulatedFACP()

        alarm_state = {"active": True}

        def smoke_readings(time):
            return [SmokeDetectorReading(detector_id="SD-1", timestamp=time, alarm_active=alarm_state["active"])]

        runtime = build_offline_demo_runtime(building, facp=facp, smoke_detector_reading_provider=smoke_readings)
        runtime.start()

        runtime.run_cycle(1.0)
        self.assertEqual(facp.panel_state, PanelState.ALARM)

        alarm_state["active"] = False
        runtime.run_cycle(2.0)

        # Restoration alone must never auto-clear -- panel stays
        # latched in the ALARM family until an explicit reset().
        self.assertEqual(facp.panel_state, PanelState.ALARM)
        self.assertEqual(facp.active_alarm_source_ids, ())

        facp.reset(3.0)
        self.assertEqual(facp.panel_state, PanelState.NORMAL)

        runtime.stop()

    def test_reset_does_not_clear_while_a_condition_still_active(self):

        building = _make_building_with_smoke_detector()
        facp = SimulatedFACP()

        def smoke_readings(time):
            return [SmokeDetectorReading(detector_id="SD-1", timestamp=time, alarm_active=True)]

        runtime = build_offline_demo_runtime(building, facp=facp, smoke_detector_reading_provider=smoke_readings)
        runtime.start()

        runtime.run_cycle(1.0)
        self.assertEqual(facp.panel_state, PanelState.ALARM)

        facp.reset(2.0)

        # Reset must not force NORMAL while the alarm condition is
        # still genuinely active.
        self.assertEqual(facp.panel_state, PanelState.ALARM)

        runtime.stop()

    def test_new_independent_alarm_source_re_alerts_after_silence(self):

        floor = Floor(id="f1", name="Floor 1")
        floor.add_zone(Zone(id="z1", name="Z1", floor_id="f1", width=5.0, height=5.0))
        floor.add_smoke_detector(SmokeDetector(id="SD-1", name="SD-1", floor_id="f1", zone_ids=("z1",)))
        floor.add_smoke_detector(SmokeDetector(id="SD-2", name="SD-2", floor_id="f1", zone_ids=("z1",)))
        building = Building(id="b1", name="B", floors=[floor])

        facp = SimulatedFACP()
        active_ids = {"SD-1"}

        def smoke_readings(time):
            return [
                SmokeDetectorReading(detector_id=sid, timestamp=time, alarm_active=sid in active_ids)
                for sid in ("SD-1", "SD-2")
            ]

        runtime = build_offline_demo_runtime(building, facp=facp, smoke_detector_reading_provider=smoke_readings)
        runtime.start()

        runtime.run_cycle(1.0)
        self.assertEqual(facp.panel_state, PanelState.ALARM)

        facp.silence(2.0)
        self.assertEqual(facp.panel_state, PanelState.ALARM_SILENCED)

        active_ids.add("SD-2")
        runtime.run_cycle(3.0)

        # A genuinely NEW alarm source re-alerts even though the panel
        # was already silenced.
        self.assertEqual(facp.panel_state, PanelState.ALARM)

        runtime.stop()

    def test_only_one_shared_facp_instance_is_ever_ticked(self):

        building = _make_building_with_smoke_detector()
        facp = SimulatedFACP(panel_id="THE-ONE")

        runtime = build_offline_demo_runtime(building, facp=facp)
        runtime.start()

        self.assertIs(runtime.facp, facp)
        runtime.run_cycle(1.0)
        self.assertIs(runtime.facp, facp)

        runtime.stop()


if __name__ == "__main__":
    unittest.main()
