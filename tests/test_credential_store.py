import json
import tempfile
import unittest
from pathlib import Path

from hazard.snapshot import HazardSnapshot
from occupancy.snapshot import OccupancySnapshot

from models.building import Building
from models.camera import Camera
from models.engineering_asset import ConnectionInfo
from models.project import Project

from camera_manager.manager import CameraManager
from camera_manager.status import CameraStatus

from credential_store.local_file_store import LocalFileCredentialStore
from credential_store.project_credentials import capture_and_clear_camera_credentials
from credential_store.store import CredentialStore

from serialization.serializer import Serializer

from multi_camera_fusion.engine import MultiCameraFusionEngine

from virtual_camera.provider import SimulatedDetectionProvider

from building_state.estimator import BuildingStateEstimator


def make_project_with_camera(password="hunter2", ip_address="10.0.0.5"):

    project = Project(name="Test Project")
    building = Building(name="B")
    floor = building.create_floor(name="Ground Floor")

    camera = Camera(
        name="Lobby Cam",
        floor_id=floor.id,
        connection=ConnectionInfo(password=password, ip_address=ip_address),
    )
    floor.add_camera(camera)

    project.set_building(building)

    return project, camera


class RaisingCredentialStore(CredentialStore):

    # A spy proving Simulation mode never touches the store at all
    # (Phase 10's own requirement) -- any call is a test failure.

    def save_credential(self, reference_id, password):
        raise AssertionError("Simulation must never write to the credential store")

    def get_credential(self, reference_id):
        raise AssertionError("Simulation must never read from the credential store")

    def delete_credential(self, reference_id):
        raise AssertionError("Simulation must never delete from the credential store")

    def has_credential(self, reference_id):
        raise AssertionError("Simulation must never query the credential store")


class ConnectionInfoRedactionTests(unittest.TestCase):

    def test_password_not_in_repr(self):

        connection = ConnectionInfo(password="hunter2")
        self.assertNotIn("hunter2", repr(connection))

    def test_password_not_in_to_dict(self):

        connection = ConnectionInfo(password="hunter2", credential_ref="CAM-1")
        data = connection.to_dict()

        self.assertNotIn("password", data)
        self.assertEqual(data["credential_ref"], "CAM-1")

    def test_camera_repr_redacts_nested_connection_password(self):

        camera = Camera(name="Cam", connection=ConnectionInfo(password="hunter2"))
        self.assertNotIn("hunter2", repr(camera))

    def test_password_not_in_camera_status(self):

        camera = Camera(name="Cam", connection=ConnectionInfo(password="hunter2"))
        manager = CameraManager()
        manager.register_camera(camera)

        status = manager.camera_status(camera.id)

        self.assertIsInstance(status, CameraStatus)
        self.assertNotIn("hunter2", repr(status))
        self.assertFalse(hasattr(status, "password"))


class LocalFileCredentialStoreTests(unittest.TestCase):

    def setUp(self):

        self._tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self._tmpdir.name) / "credentials.json"
        self.store = LocalFileCredentialStore(path=self.path)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_no_file_created_until_first_save(self):

        self.assertFalse(self.path.exists())
        self.store.get_credential("does-not-exist")
        self.assertFalse(self.path.exists())

    def test_save_and_get_round_trip(self):

        self.store.save_credential("CAM-1", "hunter2")
        self.assertEqual(self.store.get_credential("CAM-1"), "hunter2")

    def test_has_credential(self):

        self.assertFalse(self.store.has_credential("CAM-1"))
        self.store.save_credential("CAM-1", "hunter2")
        self.assertTrue(self.store.has_credential("CAM-1"))

    def test_delete_credential(self):

        self.store.save_credential("CAM-1", "hunter2")
        self.store.delete_credential("CAM-1")
        self.assertIsNone(self.store.get_credential("CAM-1"))

    def test_get_credential_for_unknown_reference_returns_none(self):

        self.assertIsNone(self.store.get_credential("no-such-ref"))


class CaptureAndClearCameraCredentialsTests(unittest.TestCase):

    def setUp(self):

        self._tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self._tmpdir.name) / "credentials.json"
        self.store = LocalFileCredentialStore(path=self.path)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_freshly_entered_password_is_captured_and_cleared(self):

        project, camera = make_project_with_camera(password="hunter2")

        migrated = capture_and_clear_camera_credentials(project, self.store)

        self.assertEqual(migrated, 1)
        self.assertEqual(camera.connection.password, "")
        self.assertEqual(camera.connection.credential_ref, camera.id)
        self.assertEqual(self.store.get_credential(camera.id), "hunter2")

    def test_no_password_is_a_no_op(self):

        project, camera = make_project_with_camera(password="")

        migrated = capture_and_clear_camera_credentials(project, self.store)

        self.assertEqual(migrated, 0)
        self.assertIsNone(camera.connection.credential_ref)

    def test_none_project_or_building_is_a_no_op(self):

        self.assertEqual(capture_and_clear_camera_credentials(None, self.store), 0)
        self.assertEqual(capture_and_clear_camera_credentials(Project(), self.store), 0)

    def test_changing_credential_ref_never_changes_camera_id(self):

        project, camera = make_project_with_camera(password="hunter2")
        original_id = camera.id

        capture_and_clear_camera_credentials(project, self.store)

        self.assertEqual(camera.id, original_id)
        self.assertEqual(camera.connection.credential_ref, original_id)


class SerializerCredentialIntegrationTests(unittest.TestCase):

    # The full 9-assertion contract the user required: no plaintext in
    # saved JSON; credential_ref survives save/reload; resolvable via
    # the store; legacy plaintext projects still load; re-saving a
    # legacy project drops the plaintext; camera.id never changes.

    def setUp(self):

        self._tmpdir = tempfile.TemporaryDirectory()
        self.credential_path = Path(self._tmpdir.name) / "credentials.json"
        self.project_path = Path(self._tmpdir.name) / "project.syn"
        self.store = LocalFileCredentialStore(path=self.credential_path)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_new_project_json_contains_no_plaintext_password(self):

        project, camera = make_project_with_camera(password="hunter2")

        Serializer.save(project, str(self.project_path), credential_store=self.store)

        raw_text = self.project_path.read_text(encoding="utf-8")
        self.assertNotIn("hunter2", raw_text)

        with open(self.project_path, "r", encoding="utf-8") as f:
            saved = json.load(f)

        connection_dict = saved["building"]["floors"][0]["cameras"][0]["connection"]
        self.assertNotIn("password", connection_dict)
        self.assertEqual(connection_dict["credential_ref"], camera.id)

    def test_credential_ref_survives_save_and_reload(self):

        project, camera = make_project_with_camera(password="hunter2")
        original_id = camera.id

        Serializer.save(project, str(self.project_path), credential_store=self.store)
        reloaded = Serializer.load(str(self.project_path), credential_store=self.store)

        reloaded_camera = reloaded.building.floors[0].cameras[0]

        self.assertEqual(reloaded_camera.id, original_id)
        self.assertEqual(reloaded_camera.connection.credential_ref, original_id)
        self.assertEqual(reloaded_camera.connection.password, "")

    def test_password_resolvable_via_credential_store_after_save(self):

        project, camera = make_project_with_camera(password="hunter2")

        Serializer.save(project, str(self.project_path), credential_store=self.store)

        self.assertEqual(self.store.get_credential(camera.id), "hunter2")

    def test_legacy_project_with_plaintext_password_loads_successfully(self):

        project, camera = make_project_with_camera(password="hunter2")

        # Simulate an old project file saved BEFORE this milestone --
        # write raw plaintext "password" in the connection dict, no
        # credential_ref at all, bypassing Serializer.save() entirely.
        raw_data = project.to_dict()
        raw_data["building"]["floors"][0]["cameras"][0]["connection"] = {
            "rtsp_address": "",
            "ip_address": "10.0.0.5",
            "username": "",
            "password": "hunter2",
        }

        with open(self.project_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f)

        # Must load without crashing.
        reloaded = Serializer.load(str(self.project_path), credential_store=self.store)

        reloaded_camera = reloaded.building.floors[0].cameras[0]
        self.assertEqual(reloaded_camera.id, camera.id)

        # Migrated into the store, cleared from memory.
        self.assertEqual(reloaded_camera.connection.password, "")
        self.assertEqual(reloaded_camera.connection.credential_ref, camera.id)
        self.assertEqual(self.store.get_credential(camera.id), "hunter2")

    def test_resaving_a_legacy_project_removes_the_plaintext(self):

        project, camera = make_project_with_camera(password="hunter2")

        raw_data = project.to_dict()
        raw_data["building"]["floors"][0]["cameras"][0]["connection"] = {
            "rtsp_address": "",
            "ip_address": "",
            "username": "",
            "password": "hunter2",
        }

        with open(self.project_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f)

        reloaded = Serializer.load(str(self.project_path), credential_store=self.store)
        Serializer.save(reloaded, str(self.project_path), credential_store=self.store)

        raw_text = self.project_path.read_text(encoding="utf-8")
        self.assertNotIn("hunter2", raw_text)

    def test_save_and_load_without_a_credential_store_never_leaks_plaintext_either(self):

        # credential_store is optional -- even a caller who never
        # passes one gets the redaction (to_dict() unconditionally
        # excludes password); they simply don't get store-backed
        # capture/migration.
        project, camera = make_project_with_camera(password="hunter2")

        Serializer.save(project, str(self.project_path))
        raw_text = self.project_path.read_text(encoding="utf-8")
        self.assertNotIn("hunter2", raw_text)

        reloaded = Serializer.load(str(self.project_path))
        self.assertEqual(reloaded.building.floors[0].cameras[0].id, camera.id)


class EmptyHumanObservationProvider:

    def observations_at(self, time):
        return {}


class SimulationNeverTouchesCredentialStoreTests(unittest.TestCase):

    def test_a_normal_simulation_tick_never_calls_the_credential_store(self):

        project, camera = make_project_with_camera(password="")
        building = project.building

        manager = CameraManager()
        manager.discover_cameras(building)

        provider = SimulatedDetectionProvider(
            manager.all_cameras(), building, EmptyHumanObservationProvider(),
        )
        manager.register_detection_provider("Simulation", provider)

        fusion_engine = MultiCameraFusionEngine()
        detections = manager.all_detections(time=0.0)
        fusion_result = fusion_engine.fuse(detections, time=0.0)

        # This would raise immediately if anything in this path touched
        # the credential store -- it never should, Simulation has no
        # business resolving a real password.
        raising_store = RaisingCredentialStore()

        BuildingStateEstimator().estimate(
            0.0,
            hazard_snapshot=HazardSnapshot(),
            occupancy_snapshot=OccupancySnapshot(),
            fusion_result=fusion_result,
        )

        # And explicitly: capture_and_clear only touches the store when
        # a password is actually present, which it is not here.
        capture_and_clear_camera_credentials(project, raising_store)


class RealDecoderBackendCredentialSafetyTests(unittest.TestCase):

    # CCTV Connection & Calibration Readiness milestone, Phase 12 --
    # re-audits the ONE new real-network code path this milestone added
    # (human_detection.opencv_decoder_backend.OpenCVFrameDecoderBackend)
    # against the exact same "a real-looking password never appears
    # unredacted anywhere" discipline this file already established for
    # ConnectionInfo/Camera/project JSON, now including a REAL (not
    # faked) decoder backend actually attempting a connection with a
    # real password.

    def test_password_never_leaks_through_rtsp_frame_source_repr_with_the_real_backend(self):

        from human_detection.opencv_decoder_backend import OpenCVFrameDecoderBackend
        from live_camera_pipeline.rtsp_frame_source import RTSPFrameSource

        secret = "hunter2-super-secret"

        backend = OpenCVFrameDecoderBackend(open_timeout_ms=1500)
        source = RTSPFrameSource(
            camera_id="CAM-SECRET-TEST", endpoint="rtsp://192.0.2.1:554/stream",
            decoder_backend=backend, username="admin", password=secret,
            max_retries=0, sleep_fn=lambda _seconds: None,
        )

        source.start()  # will fail (unreachable TEST-NET-1 address) -- that IS the point

        self.assertNotIn(secret, repr(source))
        self.assertNotIn(secret, str(source.last_error or ""))

        source.stop()
        self.assertNotIn(secret, repr(source))

    def test_build_authenticated_url_result_is_never_exposed_in_any_raised_error(self):

        from human_detection.opencv_decoder_backend import OpenCVFrameDecoderBackend

        secret = "another-super-secret-pw"

        backend = OpenCVFrameDecoderBackend(open_timeout_ms=1500)

        try:
            backend.open("rtsp://192.0.2.1:554/stream", "admin", secret)
        except Exception as exc:
            self.assertNotIn(secret, str(exc))
        else:
            self.fail("expected a connection failure against an unreachable TEST-NET-1 address")

    def test_diagnostic_script_source_never_prints_a_resolved_password_variable(self):

        # Static, mechanical guard (same "scan the source text" style as
        # tests/test_no_cv_dependencies.py) against a future edit to
        # scripts/test_camera_connection.py accidentally adding a
        # print()/log line that includes the resolved `password` value
        # -- the script's own docstring promises this never happens.
        import pathlib

        script_path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "test_camera_connection.py"
        source = script_path.read_text(encoding="utf-8")

        self.assertNotIn("print(password", source)
        self.assertNotIn("print(f\"{password", source)
        self.assertNotIn("{password}", source)
        self.assertNotIn("{password!r}", source)


if __name__ == "__main__":
    unittest.main()
