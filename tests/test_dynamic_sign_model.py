import unittest

from models.dynamic_sign import DynamicEvacuationSign, SignIndication
from models.floor import Floor

from sign_manager.manager import SignManager

from tests.dynamic_signage_fixtures import make_sign, make_signage_building


# =====================================================
# Live Dynamic Evacuation Signage milestone, Phase 27 -- test matrix
# items 1-6: sign asset round-trip, legacy load, discovery, lookup by
# floor/zone, enable/disable.
# =====================================================


class SignAssetSerializationTests(unittest.TestCase):

    def test_1_round_trip_to_dict_from_dict(self):

        sign = make_sign("SIGN-1", floor_id="f1", zone_ids=("z1", "z2"), position=(3.5, 4.5), orientation=45.0)
        sign.mount_height = 2.5
        sign.active = False

        restored = DynamicEvacuationSign.from_dict(sign.to_dict())

        self.assertEqual(restored.id, sign.id)
        self.assertEqual(restored.floor_id, "f1")
        self.assertEqual(restored.zone_ids, ("z1", "z2"))
        self.assertEqual(restored.position, (3.5, 4.5))
        self.assertEqual(restored.orientation, 45.0)
        self.assertFalse(restored.active)
        self.assertEqual(restored.supported_indications, SignIndication.DEFAULT_SUPPORTED)

    def test_2_floor_serialization_round_trip(self):

        floor = Floor(id="f1", name="Floor 1")
        floor.add_sign(make_sign("SIGN-1"))

        restored = Floor.from_dict(floor.to_dict())

        self.assertEqual(restored.sign_count, 1)
        self.assertEqual(restored.signs[0].id, "SIGN-1")

    def test_3_legacy_project_without_signs_loads(self):

        floor = Floor(id="f1", name="Floor 1")
        data = floor.to_dict()
        del data["signs"]

        restored = Floor.from_dict(data)

        self.assertEqual(restored.signs, [])
        self.assertEqual(restored.sign_count, 0)

    def test_4_narrowed_supported_indications_round_trip(self):

        sign = make_sign("SIGN-1", supported_indications=(SignIndication.LEFT, SignIndication.RIGHT))

        restored = DynamicEvacuationSign.from_dict(sign.to_dict())

        self.assertEqual(restored.supported_indications, (SignIndication.LEFT, SignIndication.RIGHT))


class SignManagerTests(unittest.TestCase):

    def setUp(self):

        self.building = make_signage_building()
        self.building.floors[0].add_sign(make_sign("SIGN-1", floor_id="f1", zone_ids=("z1",)))
        self.building.floors[0].add_sign(make_sign("SIGN-2", floor_id="f1", zone_ids=("z2",)))
        self.building.floors[1].add_sign(make_sign("SIGN-3", floor_id="f2", zone_ids=("z3",)))

    def test_5_discovery(self):

        manager = SignManager()
        signs = manager.discover_signs(self.building)

        self.assertEqual({s.id for s in signs}, {"SIGN-1", "SIGN-2", "SIGN-3"})

    def test_6_lookup_by_floor(self):

        manager = SignManager()
        manager.discover_signs(self.building)

        self.assertEqual({s.id for s in manager.signs_on_floor("f1")}, {"SIGN-1", "SIGN-2"})
        self.assertEqual({s.id for s in manager.signs_on_floor("f2")}, {"SIGN-3"})

    def test_7_lookup_by_zone(self):

        manager = SignManager()
        manager.discover_signs(self.building)

        self.assertEqual({s.id for s in manager.signs_in_zone("z2")}, {"SIGN-2"})

    def test_8_enable_disable(self):

        manager = SignManager()
        manager.discover_signs(self.building)

        manager.disable_sign("SIGN-1")
        self.assertFalse(manager.get_sign("SIGN-1").active)

        manager.enable_sign("SIGN-1")
        self.assertTrue(manager.get_sign("SIGN-1").active)

    def test_9_status_snapshot(self):

        manager = SignManager()
        manager.discover_signs(self.building)

        status = manager.sign_status("SIGN-2")

        self.assertEqual(status.sign_id, "SIGN-2")
        self.assertEqual(status.zone_ids, ("z2",))
        self.assertTrue(status.active)


if __name__ == "__main__":
    unittest.main()
