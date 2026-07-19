import unittest

from models.engineering_asset import ConnectionInfo, DeviceMode, EngineeringAsset


class ConnectionInfoTests(unittest.TestCase):

    def test_defaults_are_all_empty_placeholders(self):

        connection = ConnectionInfo()

        self.assertEqual(connection.rtsp_address, "")
        self.assertEqual(connection.ip_address, "")
        self.assertEqual(connection.username, "")
        self.assertEqual(connection.password, "")
        self.assertIsNone(connection.credential_ref)

    def test_to_dict_never_includes_the_password(self):

        # Credential Separation (Phase 10): password is deliberately
        # NOT part of to_dict()'s output, unconditionally -- only
        # credential_ref round-trips through project save/load. See
        # credential_store/project_credentials.py for what actually
        # captures a password into the CredentialStore.
        connection = ConnectionInfo(
            rtsp_address="rtsp://10.0.0.5/stream1",
            ip_address="10.0.0.5",
            username="admin",
            password="hunter2",
            credential_ref="CAM-1",
        )

        data = connection.to_dict()

        self.assertNotIn("password", data)
        self.assertEqual(
            data,
            {
                "rtsp_address": "rtsp://10.0.0.5/stream1",
                "ip_address": "10.0.0.5",
                "username": "admin",
                "credential_ref": "CAM-1",
            },
        )

    def test_non_secret_fields_and_credential_ref_round_trip_through_from_dict(self):

        connection = ConnectionInfo(
            rtsp_address="rtsp://10.0.0.5/stream1",
            ip_address="10.0.0.5",
            username="admin",
            password="hunter2",
            credential_ref="CAM-1",
        )

        restored = ConnectionInfo.from_dict(connection.to_dict())

        self.assertEqual(restored.rtsp_address, connection.rtsp_address)
        self.assertEqual(restored.ip_address, connection.ip_address)
        self.assertEqual(restored.username, connection.username)
        self.assertEqual(restored.credential_ref, connection.credential_ref)
        # The password itself is never carried by to_dict()'s output.
        self.assertEqual(restored.password, "")

    def test_from_dict_still_tolerates_a_legacy_plaintext_password_key(self):

        # Backward compatibility only -- an old project file saved
        # before Phase 10 still loads without crashing or losing the
        # password; credential_store/project_credentials.py is what
        # migrates it out afterward, not this method.
        legacy_data = {
            "rtsp_address": "",
            "ip_address": "10.0.0.5",
            "username": "admin",
            "password": "hunter2",
        }

        restored = ConnectionInfo.from_dict(legacy_data)

        self.assertEqual(restored.password, "hunter2")
        self.assertIsNone(restored.credential_ref)

    def test_from_dict_handles_missing_or_none_data(self):

        self.assertEqual(ConnectionInfo.from_dict(None), ConnectionInfo())
        self.assertEqual(ConnectionInfo.from_dict({}), ConnectionInfo())

    def test_repr_redacts_password_but_shows_other_fields(self):

        connection = ConnectionInfo(ip_address="10.0.0.5", password="hunter2")
        text = repr(connection)

        self.assertNotIn("hunter2", text)
        self.assertIn("10.0.0.5", text)

    def test_repr_of_an_unset_password_does_not_claim_one_is_set(self):

        connection = ConnectionInfo()
        self.assertIn("<unset>", repr(connection))


class EngineeringAssetTests(unittest.TestCase):

    # EngineeringAsset itself is never instantiated in production (Camera
    # is its one concrete subclass today), but it is a plain dataclass
    # with only default values, so exercising it directly is still a
    # valid, isolated test of the shared foundation independent of any
    # one device type.

    def test_defaults(self):

        asset = EngineeringAsset()

        self.assertEqual(asset.floor_id, "")
        self.assertEqual(asset.zone_ids, ())
        self.assertEqual(asset.position, (0.0, 0.0))
        self.assertEqual(asset.mount_height, 3.0)
        self.assertTrue(asset.active)
        self.assertEqual(asset.mode, DeviceMode.SIMULATION)
        self.assertEqual(asset.connection, ConnectionInfo())

    def test_asset_dict_includes_every_shared_field(self):

        asset = EngineeringAsset(
            floor_id="floor-1",
            zone_ids=("zone-a", "zone-b"),
            position=(1.5, 2.5),
            mount_height=2.8,
            active=False,
            mode=DeviceMode.LIVE,
            connection=ConnectionInfo(ip_address="192.168.1.10"),
        )

        data = asset._asset_dict()

        self.assertEqual(data["floor_id"], "floor-1")
        self.assertEqual(data["zone_ids"], ["zone-a", "zone-b"])
        self.assertEqual(data["position"], (1.5, 2.5))
        self.assertEqual(data["mount_height"], 2.8)
        self.assertEqual(data["active"], False)
        self.assertEqual(data["mode"], DeviceMode.LIVE)
        self.assertEqual(
            data["connection"],
            {"rtsp_address": "", "ip_address": "192.168.1.10", "username": "", "credential_ref": None},
        )

    def test_asset_kwargs_reconstructs_every_shared_field_from_a_dict(self):

        data = {
            "id": "abc-123",
            "name": "Some Device",
            "properties": {"note": "test"},
            "created_at": "2026-01-01T00:00:00",
            "modified_at": "2026-01-02T00:00:00",
            "floor_id": "floor-9",
            "zone_ids": ["zone-x"],
            "position": (3.0, 4.0),
            "mount_height": 4.2,
            "active": False,
            "mode": DeviceMode.REPLAY,
            "connection": {"rtsp_address": "rtsp://x", "ip_address": "", "username": "", "password": ""},
        }

        kwargs = EngineeringAsset._asset_kwargs(data)
        asset = EngineeringAsset(**kwargs)

        self.assertEqual(asset.id, "abc-123")
        self.assertEqual(asset.name, "Some Device")
        self.assertEqual(asset.properties, {"note": "test"})
        self.assertEqual(asset.floor_id, "floor-9")
        self.assertEqual(asset.zone_ids, ("zone-x",))
        self.assertEqual(asset.position, (3.0, 4.0))
        self.assertEqual(asset.mount_height, 4.2)
        self.assertFalse(asset.active)
        self.assertEqual(asset.mode, DeviceMode.REPLAY)
        self.assertEqual(asset.connection.rtsp_address, "rtsp://x")

    def test_asset_kwargs_defaults_every_shared_field_when_absent(self):

        # Simulates an old, pre-EngineeringAsset saved dict (e.g. a
        # legacy Camera dict) that has none of the newly-added keys.
        data = {"id": "legacy-1"}

        kwargs = EngineeringAsset._asset_kwargs(data)
        asset = EngineeringAsset(**kwargs)

        self.assertEqual(asset.floor_id, "")
        self.assertEqual(asset.zone_ids, ())
        self.assertEqual(asset.position, (0.0, 0.0))
        self.assertEqual(asset.mount_height, 3.0)
        self.assertTrue(asset.active)
        self.assertEqual(asset.mode, DeviceMode.SIMULATION)
        self.assertEqual(asset.connection, ConnectionInfo())


if __name__ == "__main__":
    unittest.main()
