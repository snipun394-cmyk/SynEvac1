import unittest

from models.building import Building
from models.engineering_asset import EngineeringAsset
from models.fire_hydrant import FireHydrant
from models.fire_pump import FirePump
from models.fire_service_inlet import FireServiceInlet
from models.fire_water_system import FireWaterSystem, assign_asset_to_system, system_containing_asset
from models.fire_water_tank import FireWaterTank, TankOperationalState
from models.floor import Floor
from models.fire_safety_asset import PassiveFireSafetyAvailability
from models.jockey_pump import JockeyPump
from models.pump_asset import PumpAsset, PumpControlMode, PumpOperationalState
from models.sensor_asset import HealthStatus
from models.sprinkler import Sprinkler


# =====================================================
# Fire Water Supply & Suppression Infrastructure milestone -- model-
# level tests for FireWaterTank, FirePump, JockeyPump, FireServiceInlet,
# and FireWaterSystem's own membership helpers.
# =====================================================


class PumpAssetSharedBaseTests(unittest.TestCase):

    def test_fire_pump_and_jockey_pump_share_pump_asset_base(self):

        self.assertTrue(issubclass(FirePump, PumpAsset))
        self.assertTrue(issubclass(JockeyPump, PumpAsset))

    def test_fire_pump_object_type(self):

        self.assertEqual(FirePump(id="P1", name="P1", floor_id="f1").object_type, "FirePump")

    def test_jockey_pump_object_type(self):

        self.assertEqual(JockeyPump(id="J1", name="J1", floor_id="f1").object_type, "JockeyPump")

    def test_default_stopped(self):

        pump = FirePump(id="P1", name="P1", floor_id="f1")

        self.assertEqual(pump.compute_state(), PumpOperationalState.STOPPED)

    def test_running_true_reports_running(self):

        pump = FirePump(id="P1", name="P1", floor_id="f1", running=True)

        self.assertEqual(pump.compute_state(), PumpOperationalState.RUNNING)

    def test_inactive_is_unavailable_even_if_running(self):

        pump = FirePump(id="P1", name="P1", floor_id="f1", running=True, active=False)

        self.assertEqual(pump.compute_state(), PumpOperationalState.UNAVAILABLE)

    def test_fault_outranks_running(self):

        pump = FirePump(id="P1", name="P1", floor_id="f1", running=True, health_status=HealthStatus.FAULT)

        self.assertEqual(pump.compute_state(), PumpOperationalState.FAULT)

    def test_default_control_mode_is_automatic(self):

        pump = FirePump(id="P1", name="P1", floor_id="f1")

        self.assertEqual(pump.control_mode, PumpControlMode.AUTOMATIC)

    def test_round_trip_preserves_running_and_control_mode(self):

        pump = FirePump(
            id="P1", name="P1", floor_id="f1", zone_ids=("z1",), running=True, control_mode=PumpControlMode.MANUAL,
        )

        restored = FirePump.from_dict(pump.to_dict())

        self.assertEqual(restored.running, True)
        self.assertEqual(restored.control_mode, PumpControlMode.MANUAL)
        self.assertEqual(restored.zone_ids, ("z1",))

    def test_jockey_pump_round_trip(self):

        pump = JockeyPump(id="J1", name="J1", floor_id="f1", running=True)

        restored = JockeyPump.from_dict(pump.to_dict())

        self.assertEqual(restored.running, True)
        self.assertEqual(restored.object_type, "JockeyPump")

    def test_missing_running_key_defaults_false(self):

        data = FirePump(id="P1", name="P1", floor_id="f1", running=True).to_dict()
        del data["running"]

        restored = FirePump.from_dict(data)

        self.assertFalse(restored.running)


class FireWaterTankModelTests(unittest.TestCase):

    def test_object_type(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1")

        self.assertEqual(tank.object_type, "FireWaterTank")

    def test_is_engineering_asset_not_sensor(self):

        from models.sensor_asset import SensorAsset

        self.assertTrue(issubclass(FireWaterTank, EngineeringAsset))
        self.assertFalse(issubclass(FireWaterTank, SensorAsset))

    def test_no_measurement_is_available(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", capacity_liters=10000.0)

        self.assertIsNone(tank.current_level_liters)
        self.assertEqual(tank.compute_state(), TankOperationalState.AVAILABLE)

    def test_full_tank_is_available(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", capacity_liters=10000.0, current_level_liters=9000.0)

        self.assertEqual(tank.compute_state(), TankOperationalState.AVAILABLE)

    def test_low_level_below_fraction(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", capacity_liters=10000.0, current_level_liters=1000.0)

        self.assertEqual(tank.compute_state(), TankOperationalState.LOW_LEVEL)

    def test_zero_level_is_empty(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", capacity_liters=10000.0, current_level_liters=0.0)

        self.assertEqual(tank.compute_state(), TankOperationalState.EMPTY)

    def test_inactive_is_unavailable(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", active=False)

        self.assertEqual(tank.compute_state(), TankOperationalState.UNAVAILABLE)

    def test_fault_outranks_level(self):

        tank = FireWaterTank(
            id="T1", name="T1", floor_id="f1", capacity_liters=10000.0, current_level_liters=9000.0,
            health_status=HealthStatus.FAULT,
        )

        self.assertEqual(tank.compute_state(), TankOperationalState.FAULT)

    def test_round_trip_preserves_capacity_and_level(self):

        tank = FireWaterTank(
            id="T1", name="T1", floor_id="f1", zone_ids=("z1",), capacity_liters=5000.0, current_level_liters=2500.0,
        )

        restored = FireWaterTank.from_dict(tank.to_dict())

        self.assertEqual(restored.capacity_liters, 5000.0)
        self.assertEqual(restored.current_level_liters, 2500.0)
        self.assertEqual(restored.zone_ids, ("z1",))

    def test_round_trip_preserves_none_level(self):

        tank = FireWaterTank(id="T1", name="T1", floor_id="f1", capacity_liters=5000.0)

        restored = FireWaterTank.from_dict(tank.to_dict())

        self.assertIsNone(restored.current_level_liters)

    def test_missing_capacity_key_defaults_zero(self):

        data = FireWaterTank(id="T1", name="T1", floor_id="f1", capacity_liters=1000.0).to_dict()
        del data["capacity_liters"]

        restored = FireWaterTank.from_dict(data)

        self.assertEqual(restored.capacity_liters, 0.0)


class FireServiceInletModelTests(unittest.TestCase):

    def test_object_type(self):

        inlet = FireServiceInlet(id="I1", name="I1", floor_id="f1")

        self.assertEqual(inlet.object_type, "FireServiceInlet")

    def test_default_available(self):

        inlet = FireServiceInlet(id="I1", name="I1", floor_id="f1")

        self.assertEqual(inlet.compute_availability(), PassiveFireSafetyAvailability.AVAILABLE)

    def test_fault_health_is_fault(self):

        inlet = FireServiceInlet(id="I1", name="I1", floor_id="f1", health_status=HealthStatus.FAULT)

        self.assertEqual(inlet.compute_availability(), PassiveFireSafetyAvailability.FAULT)

    def test_round_trip_preserves_inlet_type(self):

        inlet = FireServiceInlet(id="I1", name="I1", floor_id="f1", inlet_type="Dry Riser Inlet", zone_ids=("z1",))

        restored = FireServiceInlet.from_dict(inlet.to_dict())

        self.assertEqual(restored.inlet_type, "Dry Riser Inlet")
        self.assertEqual(restored.zone_ids, ("z1",))

    def test_missing_type_key_defaults(self):

        data = FireServiceInlet(id="I1", name="I1", floor_id="f1", inlet_type="Sprinkler System Inlet").to_dict()
        del data["inlet_type"]

        restored = FireServiceInlet.from_dict(data)

        self.assertEqual(restored.inlet_type, "Wet Riser Inlet")


class FireWaterSystemModelTests(unittest.TestCase):

    def test_round_trip_preserves_all_membership_fields(self):

        system = FireWaterSystem(
            id="FW-1", name="FW-1",
            tank_ids=("T1",), pump_ids=("P1",), jockey_pump_ids=("J1",), inlet_ids=("I1",),
            sprinkler_ids=("S1", "S2"), hydrant_ids=("H1",), hose_reel_ids=("HR1",),
        )

        restored = FireWaterSystem.from_dict(system.to_dict())

        self.assertEqual(restored.tank_ids, ("T1",))
        self.assertEqual(restored.pump_ids, ("P1",))
        self.assertEqual(restored.jockey_pump_ids, ("J1",))
        self.assertEqual(restored.inlet_ids, ("I1",))
        self.assertEqual(restored.sprinkler_ids, ("S1", "S2"))
        self.assertEqual(restored.hydrant_ids, ("H1",))
        self.assertEqual(restored.hose_reel_ids, ("HR1",))

    def test_floor_round_trip_all_four_asset_types(self):

        floor = Floor(id="f1", name="F1")
        floor.add_fire_water_tank(FireWaterTank(id="T1", name="T1", floor_id="f1"))
        floor.add_fire_pump(FirePump(id="P1", name="P1", floor_id="f1"))
        floor.add_jockey_pump(JockeyPump(id="J1", name="J1", floor_id="f1"))
        floor.add_fire_service_inlet(FireServiceInlet(id="I1", name="I1", floor_id="f1"))

        restored = Floor.from_dict(floor.to_dict())

        self.assertEqual(restored.fire_water_tank_count, 1)
        self.assertEqual(restored.fire_pump_count, 1)
        self.assertEqual(restored.jockey_pump_count, 1)
        self.assertEqual(restored.fire_service_inlet_count, 1)

    def test_building_round_trip_preserves_systems(self):

        building = Building(id="b1", name="B", floors=[Floor(id="f1", name="F1")])
        system = building.create_fire_water_system("FW-1")
        system.tank_ids = ("T1",)

        restored = Building.from_dict(building.to_dict())

        self.assertEqual(len(restored.fire_water_systems), 1)
        self.assertEqual(restored.fire_water_systems[0].tank_ids, ("T1",))
        self.assertEqual(restored.fire_water_systems[0].name, "FW-1")

    def test_legacy_project_without_fire_water_systems_key_loads(self):

        building = Building(id="b1", name="B", floors=[Floor(id="f1", name="F1")])
        data = building.to_dict()
        del data["fire_water_systems"]

        restored = Building.from_dict(data)

        self.assertEqual(restored.fire_water_systems, [])

    def test_legacy_floor_without_new_asset_lists_loads(self):

        floor = Floor(id="f1", name="F1")
        data = floor.to_dict()

        for key in ("fire_water_tanks", "fire_pumps", "jockey_pumps", "fire_service_inlets"):
            del data[key]

        restored = Floor.from_dict(data)

        self.assertEqual(restored.fire_water_tanks, [])
        self.assertEqual(restored.fire_pumps, [])
        self.assertEqual(restored.jockey_pumps, [])
        self.assertEqual(restored.fire_service_inlets, [])


class MembershipHelperTests(unittest.TestCase):

    def test_assign_then_query_containing_system(self):

        building = Building(id="b1", name="B", floors=[Floor(id="f1", name="F1")])
        system = building.create_fire_water_system("FW-1")

        assign_asset_to_system(building, "P1", "pump_ids", system.id)

        self.assertEqual(system.pump_ids, ("P1",))
        found = system_containing_asset(building, "P1", "pump_ids")
        self.assertIs(found, system)

    def test_reassignment_moves_between_systems(self):

        building = Building(id="b1", name="B", floors=[Floor(id="f1", name="F1")])
        system_a = building.create_fire_water_system("FW-A")
        system_b = building.create_fire_water_system("FW-B")

        assign_asset_to_system(building, "P1", "pump_ids", system_a.id)
        self.assertEqual(system_a.pump_ids, ("P1",))

        assign_asset_to_system(building, "P1", "pump_ids", system_b.id)

        self.assertEqual(system_a.pump_ids, ())
        self.assertEqual(system_b.pump_ids, ("P1",))

    def test_assign_none_clears_membership(self):

        building = Building(id="b1", name="B", floors=[Floor(id="f1", name="F1")])
        system = building.create_fire_water_system("FW-1")

        assign_asset_to_system(building, "P1", "pump_ids", system.id)
        assign_asset_to_system(building, "P1", "pump_ids", None)

        self.assertEqual(system.pump_ids, ())

    def test_no_building_is_a_no_op(self):

        assign_asset_to_system(None, "P1", "pump_ids", "FW-1")
        self.assertIsNone(system_containing_asset(None, "P1", "pump_ids"))


if __name__ == "__main__":
    unittest.main()
