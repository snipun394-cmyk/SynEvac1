import unittest

from models.engineering_asset import EngineeringAsset
from models.fire_extinguisher import FireExtinguisher
from models.fire_hydrant import FireHydrant
from models.fire_safety_asset import PassiveFireSafetyAvailability
from models.floor import Floor
from models.hose_reel import HoseReel
from models.sensor_asset import HealthStatus, SensorAsset
from models.sprinkler import Sprinkler, SprinklerActivationState


# =====================================================
# Fire Suppression & Water-Based Safety Asset Digital Twin milestone --
# model-level tests for Sprinkler, FireExtinguisher, FireHydrant,
# HoseReel: activation/availability semantics and save/reload.
# =====================================================


class SprinklerModelTests(unittest.TestCase):

    def test_object_type_is_sprinkler(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1")

        self.assertEqual(sprinkler.object_type, "Sprinkler")

    def test_is_a_sensor_asset(self):

        self.assertTrue(issubclass(Sprinkler, SensorAsset))

    def test_default_activation_temperature(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1")

        self.assertEqual(sprinkler.activation_temperature, 68.0)

    def test_no_reading_is_normal(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1")

        self.assertEqual(sprinkler.compute_state(None), SprinklerActivationState.NORMAL)

    def test_below_threshold_is_normal(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1")

        self.assertEqual(sprinkler.compute_state(50.0), SprinklerActivationState.NORMAL)

    def test_at_threshold_is_activated(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1")

        self.assertEqual(sprinkler.compute_state(68.0), SprinklerActivationState.ACTIVATED)

    def test_above_threshold_is_activated(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1")

        self.assertEqual(sprinkler.compute_state(200.0), SprinklerActivationState.ACTIVATED)

    def test_activation_records_last_activation_time(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1")
        sprinkler.compute_state(200.0, time=5.0)

        self.assertEqual(sprinkler.last_activation_time, 5.0)

    def test_inactive_never_activates(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1", active=False)

        self.assertEqual(sprinkler.compute_state(200.0), SprinklerActivationState.NORMAL)

    def test_fault_health_outranks_activation(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1", health_status=HealthStatus.FAULT)

        self.assertEqual(sprinkler.compute_state(200.0), SprinklerActivationState.FAULT)

    def test_never_uses_detector_state_vocabulary(self):

        from models.sensor_asset import DetectorState

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1")
        state = sprinkler.compute_state(200.0)

        self.assertNotIsInstance(state, DetectorState)
        self.assertIsInstance(state, SprinklerActivationState)


class PassiveFireSafetyAssetModelTests(unittest.TestCase):

    def test_fire_extinguisher_object_type(self):

        extinguisher = FireExtinguisher(id="E1", name="E1", floor_id="f1")

        self.assertEqual(extinguisher.object_type, "FireExtinguisher")

    def test_fire_hydrant_object_type(self):

        hydrant = FireHydrant(id="H1", name="H1", floor_id="f1")

        self.assertEqual(hydrant.object_type, "FireHydrant")

    def test_hose_reel_object_type(self):

        hose_reel = HoseReel(id="HR1", name="HR1", floor_id="f1")

        self.assertEqual(hose_reel.object_type, "HoseReel")

    def test_none_are_sensor_assets(self):

        self.assertFalse(issubclass(FireExtinguisher, SensorAsset))
        self.assertFalse(issubclass(FireHydrant, SensorAsset))
        self.assertFalse(issubclass(HoseReel, SensorAsset))

    def test_all_are_engineering_assets(self):

        self.assertTrue(issubclass(FireExtinguisher, EngineeringAsset))
        self.assertTrue(issubclass(FireHydrant, EngineeringAsset))
        self.assertTrue(issubclass(HoseReel, EngineeringAsset))

    def test_default_is_available(self):

        for asset in (
            FireExtinguisher(id="E1", name="E1", floor_id="f1"),
            FireHydrant(id="H1", name="H1", floor_id="f1"),
            HoseReel(id="HR1", name="HR1", floor_id="f1"),
        ):
            self.assertEqual(asset.compute_availability(), PassiveFireSafetyAvailability.AVAILABLE)

    def test_inactive_is_unavailable(self):

        for asset in (
            FireExtinguisher(id="E1", name="E1", floor_id="f1", active=False),
            FireHydrant(id="H1", name="H1", floor_id="f1", active=False),
            HoseReel(id="HR1", name="HR1", floor_id="f1", active=False),
        ):
            self.assertEqual(asset.compute_availability(), PassiveFireSafetyAvailability.UNAVAILABLE)

    def test_offline_health_is_unavailable(self):

        for asset in (
            FireExtinguisher(id="E1", name="E1", floor_id="f1", health_status=HealthStatus.OFFLINE),
            FireHydrant(id="H1", name="H1", floor_id="f1", health_status=HealthStatus.OFFLINE),
            HoseReel(id="HR1", name="HR1", floor_id="f1", health_status=HealthStatus.OFFLINE),
        ):
            self.assertEqual(asset.compute_availability(), PassiveFireSafetyAvailability.UNAVAILABLE)

    def test_fault_health_is_fault(self):

        for asset in (
            FireExtinguisher(id="E1", name="E1", floor_id="f1", health_status=HealthStatus.FAULT),
            FireHydrant(id="H1", name="H1", floor_id="f1", health_status=HealthStatus.FAULT),
            HoseReel(id="HR1", name="HR1", floor_id="f1", health_status=HealthStatus.FAULT),
        ):
            self.assertEqual(asset.compute_availability(), PassiveFireSafetyAvailability.FAULT)

    def test_fault_outranks_inactive(self):

        extinguisher = FireExtinguisher(
            id="E1", name="E1", floor_id="f1", active=False, health_status=HealthStatus.FAULT,
        )

        self.assertEqual(extinguisher.compute_availability(), PassiveFireSafetyAvailability.FAULT)


class FireSafetyAssetSerializationTests(unittest.TestCase):

    def test_sprinkler_round_trip(self):

        sprinkler = Sprinkler(
            id="S1", name="S1", floor_id="f1", zone_ids=("z1",), position=(2.0, 3.0), activation_temperature=74.0,
        )

        restored = Sprinkler.from_dict(sprinkler.to_dict())

        self.assertEqual(restored.id, "S1")
        self.assertEqual(restored.zone_ids, ("z1",))
        self.assertEqual(restored.activation_temperature, 74.0)

    def test_sprinkler_missing_activation_temperature_defaults(self):

        data = Sprinkler(id="S1", name="S1", floor_id="f1", activation_temperature=90.0).to_dict()
        del data["activation_temperature"]

        restored = Sprinkler.from_dict(data)

        self.assertEqual(restored.activation_temperature, 68.0)

    def test_fire_extinguisher_round_trip(self):

        extinguisher = FireExtinguisher(
            id="E1", name="E1", floor_id="f1", zone_ids=("z1",), extinguisher_type="CO2",
        )

        restored = FireExtinguisher.from_dict(extinguisher.to_dict())

        self.assertEqual(restored.zone_ids, ("z1",))
        self.assertEqual(restored.extinguisher_type, "CO2")

    def test_fire_extinguisher_missing_type_defaults(self):

        data = FireExtinguisher(id="E1", name="E1", floor_id="f1", extinguisher_type="Foam").to_dict()
        del data["extinguisher_type"]

        restored = FireExtinguisher.from_dict(data)

        self.assertEqual(restored.extinguisher_type, "Water")

    def test_fire_hydrant_round_trip(self):

        hydrant = FireHydrant(
            id="H1", name="H1", floor_id="f1", zone_ids=("z1",), hydrant_type="Dry Riser Landing Valve",
        )

        restored = FireHydrant.from_dict(hydrant.to_dict())

        self.assertEqual(restored.zone_ids, ("z1",))
        self.assertEqual(restored.hydrant_type, "Dry Riser Landing Valve")

    def test_fire_hydrant_missing_type_defaults(self):

        data = FireHydrant(id="H1", name="H1", floor_id="f1", hydrant_type="External Hydrant").to_dict()
        del data["hydrant_type"]

        restored = FireHydrant.from_dict(data)

        self.assertEqual(restored.hydrant_type, "Wet Riser Landing Valve")

    def test_hose_reel_round_trip(self):

        hose_reel = HoseReel(id="HR1", name="HR1", floor_id="f1", zone_ids=("z1",))

        restored = HoseReel.from_dict(hose_reel.to_dict())

        self.assertEqual(restored.zone_ids, ("z1",))

    def test_floor_round_trip_all_four_types(self):

        floor = Floor(id="f1", name="F1")
        floor.add_sprinkler(Sprinkler(id="S1", name="S1", floor_id="f1"))
        floor.add_fire_extinguisher(FireExtinguisher(id="E1", name="E1", floor_id="f1"))
        floor.add_fire_hydrant(FireHydrant(id="H1", name="H1", floor_id="f1"))
        floor.add_hose_reel(HoseReel(id="HR1", name="HR1", floor_id="f1"))

        restored = Floor.from_dict(floor.to_dict())

        self.assertEqual(restored.sprinkler_count, 1)
        self.assertEqual(restored.fire_extinguisher_count, 1)
        self.assertEqual(restored.fire_hydrant_count, 1)
        self.assertEqual(restored.hose_reel_count, 1)

        self.assertEqual(restored.sprinklers[0].id, "S1")
        self.assertEqual(restored.fire_extinguishers[0].id, "E1")
        self.assertEqual(restored.fire_hydrants[0].id, "H1")
        self.assertEqual(restored.hose_reels[0].id, "HR1")

    def test_legacy_project_without_any_new_lists_loads(self):

        floor = Floor(id="f1", name="F1")
        data = floor.to_dict()

        del data["sprinklers"]
        del data["fire_extinguishers"]
        del data["fire_hydrants"]
        del data["hose_reels"]

        restored = Floor.from_dict(data)

        self.assertEqual(restored.sprinklers, [])
        self.assertEqual(restored.fire_extinguishers, [])
        self.assertEqual(restored.fire_hydrants, [])
        self.assertEqual(restored.hose_reels, [])

        self.assertEqual(restored.sprinkler_count, 0)
        self.assertEqual(restored.fire_extinguisher_count, 0)
        self.assertEqual(restored.fire_hydrant_count, 0)
        self.assertEqual(restored.hose_reel_count, 0)


if __name__ == "__main__":
    unittest.main()
