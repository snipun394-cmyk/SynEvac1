import unittest

from models.building import Building
from models.floor import Floor
from models.fire_pump import FirePump
from models.fire_service_inlet import FireServiceInlet
from models.fire_water_tank import FireWaterTank
from models.jockey_pump import JockeyPump
from models.sensor_asset import HealthStatus

from fire_water_manager.manager import FireWaterInfrastructureManager
from fire_water_manager.snapshot import FireWaterSystemStatus


# =====================================================
# Fire Water Supply & Suppression Infrastructure milestone --
# FireWaterInfrastructureManager discovery/lookup/status/system-
# degradation tests, mirroring tests.test_fire_safety_asset_manager's
# own established structure.
# =====================================================


def _building_with(tanks=(), pumps=(), jockey_pumps=(), inlets=()):

    floor = Floor(id="f1", name="F1")

    for tank in tanks:
        floor.add_fire_water_tank(tank)

    for pump in pumps:
        floor.add_fire_pump(pump)

    for jockey_pump in jockey_pumps:
        floor.add_jockey_pump(jockey_pump)

    for inlet in inlets:
        floor.add_fire_service_inlet(inlet)

    return Building(id="b1", name="B", floors=[floor])


class DiscoveryTests(unittest.TestCase):

    def test_discovers_every_asset_across_floors(self):

        floor1 = Floor(id="f1", name="F1")
        floor1.add_fire_water_tank(FireWaterTank(id="T1", name="T1", floor_id="f1"))

        floor2 = Floor(id="f2", name="F2")
        floor2.add_fire_pump(FirePump(id="P1", name="P1", floor_id="f2"))

        building = Building(id="b1", name="B", floors=[floor1, floor2])

        manager = FireWaterInfrastructureManager()
        manager.discover_assets(building)

        self.assertEqual(len(manager.all_assets()), 2)

    def test_no_building_returns_empty(self):

        manager = FireWaterInfrastructureManager()
        result = manager.discover_assets(None)

        self.assertEqual(result, ())
        self.assertEqual(manager.all_assets(), ())

    def test_rediscovery_drops_removed_asset(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1")
        building = _building_with(tanks=[tank])

        manager = FireWaterInfrastructureManager()
        manager.discover_assets(building)
        self.assertEqual(len(manager.all_assets()), 1)

        building.floors[0].remove_fire_water_tank(tank)
        manager.discover_assets(building)

        self.assertEqual(manager.all_assets(), ())


class LookupTests(unittest.TestCase):

    def test_get_asset_unknown_returns_none(self):

        manager = FireWaterInfrastructureManager()
        self.assertIsNone(manager.get_asset("nope"))

    def test_assets_in_zone(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", zone_ids=("z1",))
        pump = FirePump(id="P1", name="P1", floor_id="f1", zone_ids=("z2",))

        building = _building_with(tanks=[tank], pumps=[pump])

        manager = FireWaterInfrastructureManager()
        manager.discover_assets(building)

        self.assertEqual({a.id for a in manager.assets_in_zone("z1")}, {"T1"})
        self.assertEqual({a.id for a in manager.assets_in_zone("z2")}, {"P1"})

    def test_typed_lookups(self):

        building = _building_with(
            tanks=[FireWaterTank(id="T1", name="T1", floor_id="f1")],
            pumps=[FirePump(id="P1", name="P1", floor_id="f1")],
            jockey_pumps=[JockeyPump(id="J1", name="J1", floor_id="f1")],
            inlets=[FireServiceInlet(id="I1", name="I1", floor_id="f1")],
        )

        manager = FireWaterInfrastructureManager()
        manager.discover_assets(building)

        self.assertEqual({t.id for t in manager.tanks()}, {"T1"})
        self.assertEqual({p.id for p in manager.fire_pumps()}, {"P1"})
        self.assertEqual({j.id for j in manager.jockey_pumps()}, {"J1"})
        self.assertEqual({i.id for i in manager.inlets()}, {"I1"})


class EnableDisableTests(unittest.TestCase):

    def test_disable_then_enable(self):

        pump = FirePump(id="P1", name="P1", floor_id="f1")
        building = _building_with(pumps=[pump])

        manager = FireWaterInfrastructureManager()
        manager.discover_assets(building)

        manager.disable_asset("P1")
        self.assertFalse(pump.active)

        manager.enable_asset("P1")
        self.assertTrue(pump.active)

    def test_disable_unknown_raises(self):

        manager = FireWaterInfrastructureManager()

        with self.assertRaises(KeyError):
            manager.disable_asset("nope")


class AssetStatusTests(unittest.TestCase):

    def test_tank_status(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", zone_ids=("z1",), capacity_liters=1000.0, current_level_liters=900.0)
        building = _building_with(tanks=[tank])

        manager = FireWaterInfrastructureManager()
        manager.discover_assets(building)

        status = manager.status_of("T1")

        self.assertEqual(status.asset_type, "FireWaterTank")
        self.assertEqual(status.zone_ids, ("z1",))
        self.assertEqual(status.state, "AVAILABLE")

    def test_pump_status(self):

        pump = FirePump(id="P1", name="P1", floor_id="f1", running=True)
        building = _building_with(pumps=[pump])

        manager = FireWaterInfrastructureManager()
        manager.discover_assets(building)

        self.assertEqual(manager.status_of("P1").state, "RUNNING")

    def test_status_of_unknown_raises(self):

        manager = FireWaterInfrastructureManager()

        with self.assertRaises(KeyError):
            manager.status_of("nope")


class SystemStatusTests(unittest.TestCase):

    def _manager_and_system(self, tank=None, pump=None):

        tanks = [tank] if tank else []
        pumps = [pump] if pump else []
        building = _building_with(tanks=tanks, pumps=pumps)

        system = building.create_fire_water_system("FW-1")
        if tank:
            system.tank_ids = (tank.id,)
        if pump:
            system.pump_ids = (pump.id,)

        manager = FireWaterInfrastructureManager()
        manager.discover_assets(building)

        return manager, system

    def test_healthy_tank_and_pump_is_available(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", capacity_liters=1000.0, current_level_liters=900.0)
        pump = FirePump(id="P1", name="P1", floor_id="f1")

        manager, system = self._manager_and_system(tank=tank, pump=pump)
        report = manager.system_status(system)

        self.assertEqual(report.status, FireWaterSystemStatus.SYSTEM_AVAILABLE)
        self.assertEqual(report.reasons, ())

    def test_stopped_pump_alone_is_not_degraded(self):

        # A fire pump normally sits STOPPED in automatic standby --
        # that alone must never read as degraded.
        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", capacity_liters=1000.0, current_level_liters=900.0)
        pump = FirePump(id="P1", name="P1", floor_id="f1", running=False)

        manager, system = self._manager_and_system(tank=tank, pump=pump)
        report = manager.system_status(system)

        self.assertEqual(report.status, FireWaterSystemStatus.SYSTEM_AVAILABLE)

    def test_pump_fault_degrades_system_with_named_reason(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", capacity_liters=1000.0, current_level_liters=900.0)
        pump = FirePump(id="P1", name="P1", floor_id="f1", health_status=HealthStatus.FAULT)

        manager, system = self._manager_and_system(tank=tank, pump=pump)
        report = manager.system_status(system)

        self.assertEqual(report.status, FireWaterSystemStatus.SYSTEM_DEGRADED)
        self.assertTrue(any("P1" in reason and "fault" in reason for reason in report.reasons))

    def test_tank_unavailable_degrades_system(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", active=False)
        pump = FirePump(id="P1", name="P1", floor_id="f1")

        manager, system = self._manager_and_system(tank=tank, pump=pump)
        report = manager.system_status(system)

        self.assertEqual(report.status, FireWaterSystemStatus.SYSTEM_DEGRADED)
        self.assertTrue(any("T1" in reason for reason in report.reasons))

    def test_all_supply_bad_is_unavailable(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", active=False)
        pump = FirePump(id="P1", name="P1", floor_id="f1", health_status=HealthStatus.FAULT)

        manager, system = self._manager_and_system(tank=tank, pump=pump)
        report = manager.system_status(system)

        self.assertEqual(report.status, FireWaterSystemStatus.SYSTEM_UNAVAILABLE)

    def test_no_supply_configured_is_unknown(self):

        building = _building_with()
        system = building.create_fire_water_system("FW-1")

        manager = FireWaterInfrastructureManager()
        manager.discover_assets(building)

        report = manager.system_status(system)

        self.assertEqual(report.status, FireWaterSystemStatus.UNKNOWN)
        self.assertIn("no supply assets configured", report.reasons)

    def test_dangling_reference_is_named_and_degrades(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", capacity_liters=1000.0, current_level_liters=900.0)
        building = _building_with(tanks=[tank])

        system = building.create_fire_water_system("FW-1")
        system.tank_ids = ("T1",)
        system.pump_ids = ("GHOST-PUMP",)

        manager = FireWaterInfrastructureManager()
        manager.discover_assets(building)

        report = manager.system_status(system)

        self.assertEqual(report.status, FireWaterSystemStatus.SYSTEM_DEGRADED)
        self.assertTrue(any("GHOST-PUMP" in reason and "not found" in reason for reason in report.reasons))

    def test_recovery_after_restoring_component(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", capacity_liters=1000.0, current_level_liters=900.0)
        pump = FirePump(id="P1", name="P1", floor_id="f1", health_status=HealthStatus.FAULT)

        manager, system = self._manager_and_system(tank=tank, pump=pump)

        self.assertEqual(manager.system_status(system).status, FireWaterSystemStatus.SYSTEM_DEGRADED)

        pump.health_status = HealthStatus.OK

        self.assertEqual(manager.system_status(system).status, FireWaterSystemStatus.SYSTEM_AVAILABLE)

    def test_dependent_assets_traced_without_health_assessment(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", capacity_liters=1000.0, current_level_liters=900.0)
        building = _building_with(tanks=[tank])

        system = building.create_fire_water_system("FW-1")
        system.tank_ids = ("T1",)
        system.sprinkler_ids = ("SPR-1",)
        system.hydrant_ids = ("HYD-1",)

        manager = FireWaterInfrastructureManager()
        manager.discover_assets(building)

        report = manager.system_status(system)

        self.assertEqual(report.sprinkler_ids, ("SPR-1",))
        self.assertEqual(report.hydrant_ids, ("HYD-1",))
        # Never assessed for health here (fire_safety_manager's own job).
        self.assertEqual(report.status, FireWaterSystemStatus.SYSTEM_AVAILABLE)


class SnapshotTests(unittest.TestCase):

    def test_snapshot_includes_entries_and_systems(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", capacity_liters=1000.0, current_level_liters=900.0)
        building = _building_with(tanks=[tank])
        system = building.create_fire_water_system("FW-1")
        system.tank_ids = ("T1",)

        manager = FireWaterInfrastructureManager()
        manager.discover_assets(building)

        snapshot = manager.snapshot()

        self.assertEqual(len(snapshot.entries), 1)
        self.assertEqual(len(snapshot.systems), 1)
        self.assertEqual(snapshot.systems[0].status, FireWaterSystemStatus.SYSTEM_AVAILABLE)

    def test_empty_manager_snapshot_is_empty(self):

        manager = FireWaterInfrastructureManager()
        snapshot = manager.snapshot()

        self.assertEqual(snapshot.entries, ())
        self.assertEqual(snapshot.systems, ())


if __name__ == "__main__":
    unittest.main()
