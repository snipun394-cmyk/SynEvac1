import unittest

from models.building import Building
from models.floor import Floor
from models.fire_extinguisher import FireExtinguisher
from models.fire_hydrant import FireHydrant
from models.hose_reel import HoseReel
from models.sensor_asset import HealthStatus
from models.sprinkler import Sprinkler

from fire_safety_manager.manager import FireSafetyAssetManager


# =====================================================
# Fire Suppression & Water-Based Safety Asset Digital Twin milestone --
# FireSafetyAssetManager discovery/lookup/status/snapshot tests,
# mirroring tests.test_emergency_light_manager's own structure.
# =====================================================


def _building_with(sprinklers=(), extinguishers=(), hydrants=(), hose_reels=()):

    floor = Floor(id="f1", name="F1")

    for sprinkler in sprinklers:
        floor.add_sprinkler(sprinkler)

    for extinguisher in extinguishers:
        floor.add_fire_extinguisher(extinguisher)

    for hydrant in hydrants:
        floor.add_fire_hydrant(hydrant)

    for hose_reel in hose_reels:
        floor.add_hose_reel(hose_reel)

    return Building(id="b1", name="B", floors=[floor])


class DiscoveryTests(unittest.TestCase):

    def test_discovers_every_asset_across_floors(self):

        floor1 = Floor(id="f1", name="F1")
        floor1.add_sprinkler(Sprinkler(id="S1", name="S1", floor_id="f1"))

        floor2 = Floor(id="f2", name="F2")
        floor2.add_fire_extinguisher(FireExtinguisher(id="E1", name="E1", floor_id="f2"))

        building = Building(id="b1", name="B", floors=[floor1, floor2])

        manager = FireSafetyAssetManager()
        manager.discover_assets(building)

        self.assertEqual(len(manager.all_assets()), 2)

    def test_no_building_returns_empty(self):

        manager = FireSafetyAssetManager()
        result = manager.discover_assets(None)

        self.assertEqual(result, ())
        self.assertEqual(manager.all_assets(), ())

    def test_rediscovery_drops_removed_asset(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1")
        building = _building_with(sprinklers=[sprinkler])

        manager = FireSafetyAssetManager()
        manager.discover_assets(building)
        self.assertEqual(len(manager.all_assets()), 1)

        building.floors[0].remove_sprinkler(sprinkler)
        manager.discover_assets(building)

        self.assertEqual(manager.all_assets(), ())


class LookupTests(unittest.TestCase):

    def test_get_asset_unknown_returns_none(self):

        manager = FireSafetyAssetManager()
        self.assertIsNone(manager.get_asset("nope"))

    def test_assets_in_zone(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1", zone_ids=("z1",))
        extinguisher = FireExtinguisher(id="E1", name="E1", floor_id="f1", zone_ids=("z2",))

        building = _building_with(sprinklers=[sprinkler], extinguishers=[extinguisher])

        manager = FireSafetyAssetManager()
        manager.discover_assets(building)

        self.assertEqual({a.id for a in manager.assets_in_zone("z1")}, {"S1"})
        self.assertEqual({a.id for a in manager.assets_in_zone("z2")}, {"E1"})

    def test_assets_on_floor(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1")
        building = _building_with(sprinklers=[sprinkler])

        manager = FireSafetyAssetManager()
        manager.discover_assets(building)

        self.assertEqual({a.id for a in manager.assets_on_floor("f1")}, {"S1"})
        self.assertEqual(manager.assets_on_floor("f2"), ())

    def test_typed_lookups(self):

        building = _building_with(
            sprinklers=[Sprinkler(id="S1", name="S1", floor_id="f1")],
            extinguishers=[FireExtinguisher(id="E1", name="E1", floor_id="f1")],
            hydrants=[FireHydrant(id="H1", name="H1", floor_id="f1")],
            hose_reels=[HoseReel(id="HR1", name="HR1", floor_id="f1")],
        )

        manager = FireSafetyAssetManager()
        manager.discover_assets(building)

        self.assertEqual({s.id for s in manager.sprinklers()}, {"S1"})
        self.assertEqual({e.id for e in manager.fire_extinguishers()}, {"E1"})
        self.assertEqual({h.id for h in manager.fire_hydrants()}, {"H1"})
        self.assertEqual({r.id for r in manager.hose_reels()}, {"HR1"})


class EnableDisableTests(unittest.TestCase):

    def test_disable_then_enable(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1")
        building = _building_with(sprinklers=[sprinkler])

        manager = FireSafetyAssetManager()
        manager.discover_assets(building)

        manager.disable_asset("S1")
        self.assertFalse(sprinkler.active)

        manager.enable_asset("S1")
        self.assertTrue(sprinkler.active)

    def test_disable_unknown_raises(self):

        manager = FireSafetyAssetManager()

        with self.assertRaises(KeyError):
            manager.disable_asset("nope")


class StatusAndSnapshotTests(unittest.TestCase):

    def test_status_of_unknown_raises(self):

        manager = FireSafetyAssetManager()

        with self.assertRaises(KeyError):
            manager.status_of("nope")

    def test_sprinkler_status_no_temperature_provider_is_normal(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1", zone_ids=("z1",))
        building = _building_with(sprinklers=[sprinkler])

        manager = FireSafetyAssetManager()
        manager.discover_assets(building)

        status = manager.status_of("S1")

        self.assertEqual(status.asset_type, "Sprinkler")
        self.assertEqual(status.zone_ids, ("z1",))
        self.assertEqual(status.state, "NORMAL")

    def test_sprinkler_status_with_supplied_temperature_is_activated(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1")
        building = _building_with(sprinklers=[sprinkler])

        manager = FireSafetyAssetManager()
        manager.discover_assets(building)

        status = manager.status_of("S1", sprinkler_temperatures={"S1": 90.0})

        self.assertEqual(status.state, "ACTIVATED")

    def test_sprinkler_fault_outranks_supplied_temperature(self):

        sprinkler = Sprinkler(id="S1", name="S1", floor_id="f1", health_status=HealthStatus.FAULT)
        building = _building_with(sprinklers=[sprinkler])

        manager = FireSafetyAssetManager()
        manager.discover_assets(building)

        status = manager.status_of("S1", sprinkler_temperatures={"S1": 90.0})

        self.assertEqual(status.state, "FAULT")

    def test_passive_asset_status_is_availability(self):

        extinguisher = FireExtinguisher(id="E1", name="E1", floor_id="f1")
        building = _building_with(extinguishers=[extinguisher])

        manager = FireSafetyAssetManager()
        manager.discover_assets(building)

        status = manager.status_of("E1")

        self.assertEqual(status.asset_type, "FireExtinguisher")
        self.assertEqual(status.state, "AVAILABLE")

    def test_snapshot_counts(self):

        building = _building_with(
            sprinklers=[
                Sprinkler(id="S1", name="S1", floor_id="f1"),
                Sprinkler(id="S2", name="S2", floor_id="f1", health_status=HealthStatus.FAULT),
            ],
            extinguishers=[
                FireExtinguisher(id="E1", name="E1", floor_id="f1"),
                FireExtinguisher(id="E2", name="E2", floor_id="f1", active=False),
            ],
            hydrants=[FireHydrant(id="H1", name="H1", floor_id="f1")],
            hose_reels=[HoseReel(id="HR1", name="HR1", floor_id="f1")],
        )

        manager = FireSafetyAssetManager()
        manager.discover_assets(building)

        snapshot = manager.snapshot(sprinkler_temperatures={"S1": 90.0})

        self.assertEqual(snapshot.sprinklers.total, 2)
        self.assertEqual(snapshot.sprinklers.activated, 1)
        self.assertEqual(snapshot.sprinklers.fault, 1)
        self.assertEqual(snapshot.sprinklers.normal, 0)

        self.assertEqual(snapshot.fire_extinguishers.total, 2)
        self.assertEqual(snapshot.fire_extinguishers.available, 1)
        self.assertEqual(snapshot.fire_extinguishers.unavailable, 1)

        self.assertEqual(snapshot.fire_hydrants.total, 1)
        self.assertEqual(snapshot.fire_hydrants.available, 1)

        self.assertEqual(snapshot.hose_reels.total, 1)
        self.assertEqual(snapshot.hose_reels.available, 1)

        self.assertEqual(len(snapshot.entries), 6)

    def test_empty_manager_snapshot_is_all_zero(self):

        manager = FireSafetyAssetManager()
        snapshot = manager.snapshot()

        self.assertEqual(snapshot.entries, ())
        self.assertEqual(snapshot.sprinklers.total, 0)
        self.assertEqual(snapshot.fire_extinguishers.total, 0)
        self.assertEqual(snapshot.fire_hydrants.total, 0)
        self.assertEqual(snapshot.hose_reels.total, 0)


if __name__ == "__main__":
    unittest.main()
