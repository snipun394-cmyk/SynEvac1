import unittest

from models.building import Building
from models.camera import Camera
from models.engineering_asset import DeviceMode
from models.floor import Floor
from models.zone import Zone

from camera_manager.manager import CameraManager
from camera_manager.status import CameraStatus


def make_zone(name, x, y, width, height, floor_id=""):

    return Zone(name=name, x=x, y=y, width=width, height=height, floor_id=floor_id)


def make_camera(name, floor_id, zone_ids=(), **overrides):

    fields = dict(name=name, floor_id=floor_id, zone_ids=tuple(zone_ids))
    fields.update(overrides)

    return Camera(**fields)


class DiscoveryTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor_1 = self.building.create_floor(name="Ground Floor")
        self.floor_2 = self.building.create_floor(name="Floor 1", height=3.0)

        self.camera_1 = make_camera("Cam 1", self.floor_1.id)
        self.camera_2 = make_camera("Cam 2", self.floor_2.id)

        self.floor_1.add_camera(self.camera_1)
        self.floor_2.add_camera(self.camera_2)

        self.manager = CameraManager()

    def test_discover_cameras_finds_every_camera_across_every_floor(self):

        # Camera is a plain (unhashable, mutable) dataclass -- compare
        # by id rather than putting instances in a set.
        discovered = self.manager.discover_cameras(self.building)

        self.assertEqual(
            {camera.id for camera in discovered}, {self.camera_1.id, self.camera_2.id},
        )
        self.assertEqual(len(self.manager.all_cameras()), 2)

    def test_discover_cameras_with_no_building_clears_the_registry(self):

        self.manager.discover_cameras(self.building)
        self.assertEqual(len(self.manager.all_cameras()), 2)

        result = self.manager.discover_cameras(None)

        self.assertEqual(result, ())
        self.assertEqual(self.manager.all_cameras(), ())

    def test_rediscovering_picks_up_a_newly_added_camera(self):

        self.manager.discover_cameras(self.building)
        self.assertEqual(len(self.manager.all_cameras()), 2)

        camera_3 = make_camera("Cam 3", self.floor_1.id)
        self.floor_1.add_camera(camera_3)

        self.manager.discover_cameras(self.building)

        self.assertEqual(len(self.manager.all_cameras()), 3)
        self.assertIs(self.manager.get_camera(camera_3.id), camera_3)

    def test_rediscovering_drops_a_removed_camera(self):

        self.manager.discover_cameras(self.building)

        self.floor_1.remove_camera(self.camera_1)

        self.manager.discover_cameras(self.building)

        self.assertIsNone(self.manager.get_camera(self.camera_1.id))
        self.assertEqual(len(self.manager.all_cameras()), 1)


class RegistrationTests(unittest.TestCase):

    def setUp(self):

        self.manager = CameraManager()
        self.camera = make_camera("Cam", "floor-1")

    def test_register_camera_makes_it_discoverable_by_id(self):

        self.manager.register_camera(self.camera)

        self.assertIs(self.manager.get_camera(self.camera.id), self.camera)

    def test_remove_camera_by_id(self):

        self.manager.register_camera(self.camera)
        self.manager.remove_camera(self.camera.id)

        self.assertIsNone(self.manager.get_camera(self.camera.id))

    def test_remove_unknown_camera_id_is_a_no_op(self):

        self.manager.remove_camera("does-not-exist")  # must not raise

    def test_get_camera_returns_none_for_unknown_id(self):

        self.assertIsNone(self.manager.get_camera("does-not-exist"))


class LookupTests(unittest.TestCase):

    def setUp(self):

        self.manager = CameraManager()

        self.camera_a = make_camera("A", "floor-1", zone_ids=("zone-1",))
        self.camera_b = make_camera("B", "floor-1", zone_ids=("zone-2",))
        self.camera_c = make_camera("C", "floor-2", zone_ids=("zone-1", "zone-3"))

        for camera in (self.camera_a, self.camera_b, self.camera_c):
            self.manager.register_camera(camera)

    def test_cameras_on_floor(self):

        ids_on_floor_1 = {camera.id for camera in self.manager.cameras_on_floor("floor-1")}
        self.assertEqual(ids_on_floor_1, {self.camera_a.id, self.camera_b.id})

        self.assertEqual(self.manager.cameras_on_floor("floor-2"), (self.camera_c,))
        self.assertEqual(self.manager.cameras_on_floor("no-such-floor"), ())

    def test_cameras_in_zone(self):

        ids_in_zone_1 = {camera.id for camera in self.manager.cameras_in_zone("zone-1")}
        self.assertEqual(ids_in_zone_1, {self.camera_a.id, self.camera_c.id})

        self.assertEqual(self.manager.cameras_in_zone("zone-2"), (self.camera_b,))
        self.assertEqual(self.manager.cameras_in_zone("no-such-zone"), ())


class EnableDisableTests(unittest.TestCase):

    def setUp(self):

        self.manager = CameraManager()
        self.camera = make_camera("Cam", "floor-1", active=True)
        self.manager.register_camera(self.camera)

    def test_disable_then_enable(self):

        self.manager.disable_camera(self.camera.id)
        self.assertFalse(self.camera.active)

        self.manager.enable_camera(self.camera.id)
        self.assertTrue(self.camera.active)

    def test_enable_unknown_camera_raises(self):

        with self.assertRaises(KeyError):
            self.manager.enable_camera("no-such-camera")

    def test_disable_unknown_camera_raises(self):

        with self.assertRaises(KeyError):
            self.manager.disable_camera("no-such-camera")


class ModeSwitchingTests(unittest.TestCase):

    def setUp(self):

        self.manager = CameraManager()
        self.camera = make_camera("Cam", "floor-1")
        self.manager.register_camera(self.camera)

    def test_default_mode_is_simulation(self):

        self.assertEqual(self.manager.camera_mode(self.camera.id), DeviceMode.SIMULATION)

    def test_set_camera_mode_to_each_valid_value(self):

        for mode in DeviceMode.ALL:

            self.manager.set_camera_mode(self.camera.id, mode)
            self.assertEqual(self.manager.camera_mode(self.camera.id), mode)

    def test_setting_mode_never_replaces_the_camera_asset(self):

        # Phase 4's own hard requirement: "Changing the mode must not
        # require replacing the Camera Asset."
        original_camera = self.manager.get_camera(self.camera.id)

        self.manager.set_camera_mode(self.camera.id, DeviceMode.LIVE)

        self.assertIs(self.manager.get_camera(self.camera.id), original_camera)
        self.assertIs(self.manager.get_camera(self.camera.id), self.camera)

    def test_invalid_mode_raises(self):

        with self.assertRaises(ValueError):
            self.manager.set_camera_mode(self.camera.id, "NotAMode")

    def test_set_mode_on_unknown_camera_raises(self):

        with self.assertRaises(KeyError):
            self.manager.set_camera_mode("no-such-camera", DeviceMode.LIVE)


class StatusTests(unittest.TestCase):

    def setUp(self):

        self.manager = CameraManager()
        self.camera = make_camera(
            "Lobby Cam", "floor-1", zone_ids=("zone-1", "zone-2"), active=True,
        )
        self.manager.register_camera(self.camera)

    def test_camera_status_reflects_current_camera_fields(self):

        status = self.manager.camera_status(self.camera.id)

        self.assertIsInstance(status, CameraStatus)
        self.assertEqual(status.camera_id, self.camera.id)
        self.assertEqual(status.name, "Lobby Cam")
        self.assertEqual(status.floor_id, "floor-1")
        self.assertEqual(status.zone_ids, ("zone-1", "zone-2"))
        self.assertTrue(status.active)
        self.assertEqual(status.mode, DeviceMode.SIMULATION)
        self.assertFalse(status.has_detection_provider)

    def test_status_reflects_a_registered_detection_provider(self):

        self.manager.register_detection_provider(DeviceMode.SIMULATION, object())

        status = self.manager.camera_status(self.camera.id)

        self.assertTrue(status.has_detection_provider)

    def test_status_for_unknown_camera_raises(self):

        with self.assertRaises(KeyError):
            self.manager.camera_status("no-such-camera")

    def test_all_statuses_covers_every_registered_camera(self):

        camera_2 = make_camera("Cam 2", "floor-1")
        self.manager.register_camera(camera_2)

        statuses = self.manager.all_statuses()

        self.assertEqual(len(statuses), 2)
        self.assertEqual(
            {status.camera_id for status in statuses}, {self.camera.id, camera_2.id},
        )


class DetectionProviderRegistrationTests(unittest.TestCase):

    def setUp(self):

        self.manager = CameraManager()

    def test_register_and_query_a_provider(self):

        provider = object()

        self.assertFalse(self.manager.has_detection_provider(DeviceMode.SIMULATION))

        self.manager.register_detection_provider(DeviceMode.SIMULATION, provider)

        self.assertTrue(self.manager.has_detection_provider(DeviceMode.SIMULATION))

    def test_unregister_a_provider(self):

        self.manager.register_detection_provider(DeviceMode.SIMULATION, object())
        self.manager.unregister_detection_provider(DeviceMode.SIMULATION)

        self.assertFalse(self.manager.has_detection_provider(DeviceMode.SIMULATION))

    def test_unregister_unknown_mode_is_a_no_op(self):

        self.manager.unregister_detection_provider(DeviceMode.LIVE)  # must not raise

    def test_register_provider_for_invalid_mode_raises(self):

        with self.assertRaises(ValueError):
            self.manager.register_detection_provider("NotAMode", object())

    def test_replay_and_live_have_no_provider_by_default(self):

        self.assertFalse(self.manager.has_detection_provider(DeviceMode.REPLAY))
        self.assertFalse(self.manager.has_detection_provider(DeviceMode.LIVE))


class FakeDetectionProvider:

    # A minimal stand-in satisfying the DetectionProvider duck type
    # (detections_at(camera_id, time) -> Tuple[Detection-shaped, ...]) --
    # CameraManager never imports virtual_camera.provider.DetectionProvider
    # at all (see this test module's own dependency-direction test
    # below), so nothing here needs to subclass it.

    def __init__(self, detections_by_camera_id):

        self._detections_by_camera_id = detections_by_camera_id

    def detections_at(self, camera_id, time):

        return self._detections_by_camera_id.get(camera_id, ())


class FakeDetection:

    def __init__(self, camera_id, floor_id, zone_id):

        self.camera_id = camera_id
        self.floor_id = floor_id
        self.zone_id = zone_id


class DetectionRoutingTests(unittest.TestCase):

    def setUp(self):

        self.manager = CameraManager()

        self.camera_1 = make_camera("Cam 1", "floor-1", zone_ids=("zone-1",))
        self.camera_2 = make_camera("Cam 2", "floor-1", zone_ids=("zone-2",))
        self.camera_3 = make_camera("Cam 3", "floor-2", zone_ids=("zone-3",), mode=DeviceMode.LIVE)

        for camera in (self.camera_1, self.camera_2, self.camera_3):
            self.manager.register_camera(camera)

        detection_1 = FakeDetection(self.camera_1.id, "floor-1", "zone-1")
        detection_2 = FakeDetection(self.camera_2.id, "floor-1", "zone-2")

        provider = FakeDetectionProvider({
            self.camera_1.id: (detection_1,),
            self.camera_2.id: (detection_2,),
        })

        self.manager.register_detection_provider(DeviceMode.SIMULATION, provider)

        self.detection_1 = detection_1
        self.detection_2 = detection_2

    def test_detections_for_camera_delegates_to_the_registered_provider(self):

        detections = self.manager.detections_for_camera(self.camera_1.id, time=0.0)

        self.assertEqual(detections, (self.detection_1,))

    def test_detections_for_camera_in_a_mode_with_no_provider_is_empty(self):

        # Camera 3 is in Live mode -- no adapter registered (Phase 4's
        # "architecture placeholder" requirement) -- never an error.
        detections = self.manager.detections_for_camera(self.camera_3.id, time=0.0)

        self.assertEqual(detections, ())

    def test_detections_for_unknown_camera_raises(self):

        with self.assertRaises(KeyError):
            self.manager.detections_for_camera("no-such-camera", time=0.0)

    def test_all_detections_aggregates_every_camera(self):

        detections = self.manager.all_detections(time=0.0)

        self.assertEqual(set(detections), {self.detection_1, self.detection_2})

    def test_detections_by_floor(self):

        grouped = self.manager.detections_by_floor(time=0.0)

        self.assertEqual(set(grouped.keys()), {"floor-1"})
        self.assertEqual(set(grouped["floor-1"]), {self.detection_1, self.detection_2})

    def test_detections_by_zone(self):

        grouped = self.manager.detections_by_zone(time=0.0)

        self.assertEqual(grouped, {"zone-1": (self.detection_1,), "zone-2": (self.detection_2,)})

    def test_detections_by_camera(self):

        grouped = self.manager.detections_by_camera(time=0.0)

        self.assertEqual(
            grouped, {self.camera_1.id: (self.detection_1,), self.camera_2.id: (self.detection_2,)},
        )

    def test_switching_a_camera_to_a_mode_with_a_provider_starts_producing_detections(self):

        self.assertEqual(self.manager.detections_for_camera(self.camera_3.id, time=0.0), ())

        self.manager.register_detection_provider(
            DeviceMode.LIVE, FakeDetectionProvider({
                self.camera_3.id: (FakeDetection(self.camera_3.id, "floor-2", "zone-3"),),
            }),
        )

        detections = self.manager.detections_for_camera(self.camera_3.id, time=0.0)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].camera_id, self.camera_3.id)


class BackwardCompatibilityTests(unittest.TestCase):

    # A CameraManager built against a Building loaded from an old,
    # pre-Camera-Management-System (indeed, pre-Camera-Coverage-Engine)
    # project file must work with no special-casing -- Camera.from_dict()
    # already defaults every new field honestly (see tests.test_camera.
    # CameraModelTests.test_loading_a_pre_framework_camera_dict_still_works),
    # so nothing here should need its own compatibility shim.

    def test_discovers_and_manages_a_legacy_shaped_camera(self):

        floor_data = {
            "id": "floor-legacy",
            "name": "Ground Floor",
            "display_order": 0,
            "height": 3.0,
            "floor_plan": "",
            "visible": True,
            "locked": False,
            "zones": [],
            "exits": [],
            "stairs": [],
            "elevators": [],
            "cameras": [
                {
                    "id": "cam-legacy",
                    "name": "Legacy Cam",
                    "object_type": "Camera",
                    "properties": {},
                    "created_at": "",
                    "modified_at": "",
                    "position": (1.0, 1.0),
                    "floor_id": "floor-legacy",
                    "rotation": 0.0,
                    "horizontal_fov": 90.0,
                    "max_range": 25.0,
                    "mount_height": 3.0,
                    "active": True,
                }
            ],
            "detectors": [],
            "assembly_points": [],
            "obstacles": [],
            "doors": [],
        }

        floor = Floor.from_dict(floor_data)

        building = Building(name="Legacy Building")
        building.floors.append(floor)

        manager = CameraManager()
        manager.discover_cameras(building)

        self.assertEqual(len(manager.all_cameras()), 1)

        status = manager.camera_status("cam-legacy")
        self.assertEqual(status.mode, DeviceMode.SIMULATION)
        self.assertEqual(status.zone_ids, ())
        self.assertTrue(status.active)

        manager.disable_camera("cam-legacy")
        manager.set_camera_mode("cam-legacy", DeviceMode.LIVE)

        self.assertFalse(manager.get_camera("cam-legacy").active)
        self.assertEqual(manager.camera_mode("cam-legacy"), DeviceMode.LIVE)


class CameraManagerPackageDependencyDirectionTests(unittest.TestCase):

    # Same regex-scan-the-source-files convention every other package
    # boundary in this codebase already enforces (see
    # tests.test_virtual_camera.VirtualCameraPackageDependencyDirectionTests,
    # tests.test_perception.PerceptionPackageDependencyDirectionTests).
    # camera_manager/ must reach detections only through whatever
    # duck-typed DetectionProvider object it is handed -- it does not
    # even need to import virtual_camera/visibility to do that (see
    # camera_manager/manager.py's own docstring), which this test
    # enforces structurally rather than by convention alone.

    def test_camera_manager_never_imports_simulation_or_detection_internals(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "camera_manager"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(simulator|ground_truth|behavior|behavior_library|behaviour_profile_resolver|"
            r"simulation_runtime|hazard_evolution|ai_training|rl_training|advisory_system|"
            r"command_center|designer|virtual_camera|visibility|cv2|torch|ultralytics|onvif)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"camera_manager/{path.name} imports a simulation/detection/decision-layer "
                f"module directly -- it must only manage Camera Assets and route to whatever "
                f"DetectionProvider-shaped object it is registered with",
            )


if __name__ == "__main__":
    unittest.main()
