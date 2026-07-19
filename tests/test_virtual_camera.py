import unittest

from models.building import Building
from models.camera import Camera
from models.door import Door
from models.exit import Exit
from models.zone import Zone

from navigation.graph_builder import NavigationGraphGenerator

from pathfinding.engine import PathfindingEngine

from perception.models.human_observation import HumanClassification, HumanObservation, HumanState
from perception.providers.human_observation_provider import HumanObservationProvider

from simulation_runtime.human_observation_bridge import GroundTruthHumanObservationProvider

from simulator.coordinator import MultiAgentSimulation

from virtual_camera.camera import VirtualCamera
from virtual_camera.detection import Detection
from virtual_camera.imperfections import DetectionImperfectionModel
from virtual_camera.provider import DetectionProvider, SimulatedDetectionProvider


def make_zone(name, x, y, width, height, floor_id=""):

    return Zone(name=name, x=x, y=y, width=width, height=height, floor_id=floor_id)


class FakeHumanObservationProvider(HumanObservationProvider):

    # A hand-built stand-in -- exactly the role a test fixture plays
    # for every other Provider-shaped interface in this codebase
    # (ManualHazardProvider, ManualOccupancyProvider, ...). Static
    # regardless of `time` unless a caller supplies a
    # by_time mapping (used only by the tracking-delay test, where the
    # whole point is that the answer legitimately differs by time).

    def __init__(self, observations=None, by_time=None):

        self._observations = observations or {}
        self._by_time = by_time or {}

    def observations_at(self, time):

        if time in self._by_time:
            return self._by_time[time]

        return dict(self._observations)


class VirtualCameraTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone_a = make_zone("Zone A", 0.0, 0.0, 10.0, 10.0, floor_id=self.floor.id)
        self.zone_b = make_zone("Zone B", 10.0, 0.0, 10.0, 10.0, floor_id=self.floor.id)

        self.floor.add_zone(self.zone_a)
        self.floor.add_zone(self.zone_b)

        self.camera = Camera(
            name="Cam", position=(5.0, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=20.0,
        )
        self.floor.add_camera(self.camera)

    def test_visible_occupant_generates_a_detection(self):

        observation = HumanObservation(
            person_id="occ-1", zone_id=self.zone_a.id, floor_id=self.floor.id,
            classification=HumanClassification.ADULT, state=HumanState.WALKING,
        )
        provider = FakeHumanObservationProvider({"occ-1": observation})

        virtual_camera = VirtualCamera(self.camera, self.building, provider)
        detections = virtual_camera.detections_at(time=10.0)

        self.assertEqual(len(detections), 1)

        detection = detections[0]
        self.assertIsInstance(detection, Detection)
        self.assertEqual(detection.camera_id, self.camera.id)
        self.assertEqual(detection.timestamp, 10.0)
        self.assertEqual(detection.occupant_id, "occ-1")
        self.assertEqual(detection.floor_id, self.floor.id)
        self.assertEqual(detection.zone_id, self.zone_a.id)
        self.assertEqual(detection.position, self.zone_a.center)
        self.assertEqual(detection.confidence, 1.0)
        self.assertEqual(detection.classification, HumanClassification.ADULT)
        self.assertEqual(detection.human_state, HumanState.WALKING)
        self.assertFalse(detection.is_false_positive)

    def test_hidden_occupant_generates_no_detection(self):

        # Camera faces away from the zone it's standing in.
        camera = Camera(
            name="Away", position=(5.0, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=10.0, max_range=20.0,
        )
        self.floor.add_camera(camera)

        observation = HumanObservation(
            person_id="occ-1", zone_id=self.zone_b.id, floor_id=self.floor.id,
        )
        provider = FakeHumanObservationProvider({"occ-1": observation})

        # Zone B is never reachable from Zone A (no door) regardless of
        # rotation, but here even the FOV/range fundamentals wouldn't
        # cover it: verifies plain "hidden" (as opposed to specifically
        # wall-occluded, covered separately below).
        virtual_camera = VirtualCamera(camera, self.building, provider)

        self.assertEqual(virtual_camera.detections_at(time=0.0), ())

    def test_occupant_behind_a_wall_is_ignored(self):

        # No door between Zone A and Zone B -- the shared wall is fully
        # opaque (see visibility/segments.py), so an occupant physically
        # in Zone B is invisible to a camera in Zone A no matter how
        # wide its FOV/range are.
        camera = Camera(
            name="Wide", position=(5.0, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=100.0,
        )
        self.floor.add_camera(camera)

        observation = HumanObservation(
            person_id="occ-1", zone_id=self.zone_b.id, floor_id=self.floor.id,
        )
        provider = FakeHumanObservationProvider({"occ-1": observation})

        virtual_camera = VirtualCamera(camera, self.building, provider)

        self.assertEqual(virtual_camera.detections_at(time=0.0), ())

    def test_occupant_becomes_visible_once_a_door_connects_the_zones(self):

        door = Door(
            name="D", start_point=(10.0, 4.0), end_point=(10.0, 6.0),
            floor_id=self.floor.id, width=2.0,
            zone_a_id=self.zone_a.id, zone_b_id=self.zone_b.id,
        )
        self.floor.add_door(door)

        camera = Camera(
            name="Wide", position=(5.0, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=100.0,
        )
        self.floor.add_camera(camera)

        observation = HumanObservation(
            person_id="occ-1", zone_id=self.zone_b.id, floor_id=self.floor.id,
        )
        provider = FakeHumanObservationProvider({"occ-1": observation})

        virtual_camera = VirtualCamera(camera, self.building, provider)

        detections = virtual_camera.detections_at(time=0.0)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].zone_id, self.zone_b.id)

    def test_occupant_on_a_different_floor_is_ignored(self):

        other_floor = self.building.create_floor(name="Floor 2", height=3.0)
        other_zone = make_zone("Other", 0.0, 0.0, 5.0, 5.0, floor_id=other_floor.id)
        other_floor.add_zone(other_zone)

        observation = HumanObservation(
            person_id="occ-1", zone_id=other_zone.id, floor_id=other_floor.id,
        )
        provider = FakeHumanObservationProvider({"occ-1": observation})

        virtual_camera = VirtualCamera(self.camera, self.building, provider)

        self.assertEqual(virtual_camera.detections_at(time=0.0), ())

    def test_occupant_with_unresolvable_zone_is_ignored(self):

        observation = HumanObservation(person_id="occ-1", zone_id=None, floor_id=None)
        provider = FakeHumanObservationProvider({"occ-1": observation})

        virtual_camera = VirtualCamera(self.camera, self.building, provider)

        self.assertEqual(virtual_camera.detections_at(time=0.0), ())

    def test_inactive_camera_produces_no_detections(self):

        inactive_camera = Camera(
            name="Off", position=(5.0, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=20.0, active=False,
        )
        self.floor.add_camera(inactive_camera)

        observation = HumanObservation(
            person_id="occ-1", zone_id=self.zone_a.id, floor_id=self.floor.id,
        )
        provider = FakeHumanObservationProvider({"occ-1": observation})

        virtual_camera = VirtualCamera(inactive_camera, self.building, provider)

        self.assertEqual(virtual_camera.detections_at(time=0.0), ())

    def test_multiple_cameras_observing_the_same_occupant_each_generate_a_detection(self):

        camera_2 = Camera(
            name="Cam 2", position=(6.0, 6.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=20.0,
        )
        self.floor.add_camera(camera_2)

        observation = HumanObservation(
            person_id="occ-1", zone_id=self.zone_a.id, floor_id=self.floor.id,
        )
        provider = FakeHumanObservationProvider({"occ-1": observation})

        provider_multi = SimulatedDetectionProvider(
            [self.camera, camera_2], self.building, provider,
        )

        detections = provider_multi.all_detections_at(time=0.0)

        camera_ids_detected = {d.camera_id for d in detections}
        self.assertEqual(camera_ids_detected, {self.camera.id, camera_2.id})
        self.assertEqual(len(detections), 2)

    def test_detection_stream_is_deterministic_when_imperfections_disabled(self):

        observation = HumanObservation(
            person_id="occ-1", zone_id=self.zone_a.id, floor_id=self.floor.id,
            classification=HumanClassification.CHILD, state=HumanState.STANDING,
        )
        provider = FakeHumanObservationProvider({"occ-1": observation})

        camera_1 = VirtualCamera(self.camera, self.building, provider)
        camera_2 = VirtualCamera(self.camera, self.building, provider)

        # Two independently-constructed instances (no shared RNG state,
        # no seed at all) must produce byte-identical output, repeatedly
        # -- proof that the perfect-detection path never touches
        # randomness.
        for _ in range(5):
            self.assertEqual(
                camera_1.detections_at(time=3.0), camera_2.detections_at(time=3.0),
            )


class VirtualCameraImperfectionTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone = make_zone("Zone", 0.0, 0.0, 10.0, 10.0, floor_id=self.floor.id)
        self.floor.add_zone(self.zone)

        self.camera = Camera(
            name="Cam", position=(5.0, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=20.0,
        )
        self.floor.add_camera(self.camera)

        self.observation = HumanObservation(
            person_id="occ-1", zone_id=self.zone.id, floor_id=self.floor.id,
        )
        self.provider = FakeHumanObservationProvider({"occ-1": self.observation})

    def test_zero_detection_probability_always_misses(self):

        model = DetectionImperfectionModel(detection_probability=0.0, seed=1)
        virtual_camera = VirtualCamera(self.camera, self.building, self.provider, imperfection_model=model)

        for t in range(10):
            self.assertEqual(virtual_camera.detections_at(time=float(t)), ())

    def test_full_detection_probability_never_misses(self):

        model = DetectionImperfectionModel(detection_probability=1.0)
        virtual_camera = VirtualCamera(self.camera, self.building, self.provider, imperfection_model=model)

        for t in range(10):
            self.assertEqual(len(virtual_camera.detections_at(time=float(t))), 1)

    def test_false_positive_rate_of_one_always_adds_a_synthetic_detection(self):

        model = DetectionImperfectionModel(false_positive_rate=1.0, seed=2)
        virtual_camera = VirtualCamera(self.camera, self.building, self.provider, imperfection_model=model)

        detections = virtual_camera.detections_at(time=0.0)

        false_positives = [d for d in detections if d.is_false_positive]
        real_detections = [d for d in detections if not d.is_false_positive]

        self.assertEqual(len(false_positives), 1)
        self.assertEqual(len(real_detections), 1)
        self.assertIsNone(false_positives[0].zone_id)
        self.assertEqual(false_positives[0].classification, HumanClassification.UNKNOWN)

    def test_confidence_variation_changes_confidence_from_the_perfect_default(self):

        model = DetectionImperfectionModel(confidence_variation=0.3, seed=3)
        virtual_camera = VirtualCamera(self.camera, self.building, self.provider, imperfection_model=model)

        detections = virtual_camera.detections_at(time=0.0)

        self.assertEqual(len(detections), 1)
        self.assertNotEqual(detections[0].confidence, 1.0)
        self.assertTrue(0.0 <= detections[0].confidence <= 1.0)

    def test_tracking_delay_resolves_ground_truth_at_an_earlier_time(self):

        past_observation = HumanObservation(
            person_id="occ-1", zone_id=self.zone.id, floor_id=self.floor.id, state=HumanState.STANDING,
        )
        current_observation = HumanObservation(
            person_id="occ-1", zone_id=self.zone.id, floor_id=self.floor.id, state=HumanState.WALKING,
        )

        provider = FakeHumanObservationProvider(
            by_time={
                5.0: {"occ-1": past_observation},
                10.0: {"occ-1": current_observation},
            }
        )

        model = DetectionImperfectionModel(tracking_delay=5.0)
        virtual_camera = VirtualCamera(self.camera, self.building, provider, imperfection_model=model)

        detections = virtual_camera.detections_at(time=10.0)

        self.assertEqual(len(detections), 1)
        # Reports the delayed (5.0s-old) state, not the current one --
        # the whole point of simulating detector latency.
        self.assertEqual(detections[0].human_state, HumanState.STANDING)
        # ...but the Detection's own timestamp is still the real,
        # current capture time, not the delayed one -- a consumer
        # needs to know *when this was reported*, separately from how
        # stale the reported content is.
        self.assertEqual(detections[0].timestamp, 10.0)

    def test_seeded_imperfections_are_reproducible_across_instances(self):

        model_a = DetectionImperfectionModel(
            detection_probability=0.5, false_positive_rate=0.5, confidence_variation=0.2, seed=42,
        )
        model_b = DetectionImperfectionModel(
            detection_probability=0.5, false_positive_rate=0.5, confidence_variation=0.2, seed=42,
        )

        camera_a = VirtualCamera(self.camera, self.building, self.provider, imperfection_model=model_a)
        camera_b = VirtualCamera(self.camera, self.building, self.provider, imperfection_model=model_b)

        for t in range(20):
            self.assertEqual(
                camera_a.detections_at(time=float(t)), camera_b.detections_at(time=float(t)),
            )

    def test_invalid_detection_probability_raises(self):

        with self.assertRaises(ValueError):
            DetectionImperfectionModel(detection_probability=1.5)

    def test_invalid_false_positive_rate_raises(self):

        with self.assertRaises(ValueError):
            DetectionImperfectionModel(false_positive_rate=-0.1)

    def test_invalid_confidence_variation_raises(self):

        with self.assertRaises(ValueError):
            DetectionImperfectionModel(confidence_variation=-1.0)

    def test_invalid_tracking_delay_raises(self):

        with self.assertRaises(ValueError):
            DetectionImperfectionModel(tracking_delay=-1.0)

    def test_default_model_is_perfect(self):

        self.assertTrue(DetectionImperfectionModel().is_perfect)

    def test_any_imperfection_makes_the_model_not_perfect(self):

        self.assertFalse(DetectionImperfectionModel(detection_probability=0.9).is_perfect)
        self.assertFalse(DetectionImperfectionModel(false_positive_rate=0.1).is_perfect)
        self.assertFalse(DetectionImperfectionModel(confidence_variation=0.1).is_perfect)
        self.assertFalse(DetectionImperfectionModel(tracking_delay=1.0).is_perfect)


class DetectionValidationTests(unittest.TestCase):

    def _make(self, **overrides):

        fields = dict(
            camera_id="cam-1", timestamp=0.0, occupant_id="occ-1",
            floor_id="floor-1", zone_id="zone-1", position=(1.0, 1.0),
            confidence=1.0, classification=HumanClassification.ADULT, human_state=HumanState.STANDING,
        )
        fields.update(overrides)

        return Detection(**fields)

    def test_valid_confidence_is_accepted(self):

        self._make(confidence=0.0)
        self._make(confidence=1.0)
        self._make(confidence=0.5)

    def test_out_of_range_confidence_raises(self):

        with self.assertRaises(ValueError):
            self._make(confidence=1.1)

        with self.assertRaises(ValueError):
            self._make(confidence=-0.1)


class SimulatedDetectionProviderTests(unittest.TestCase):

    def setUp(self):

        self.building = Building(name="B")
        self.floor = self.building.create_floor(name="Ground Floor")

        self.zone = make_zone("Zone", 0.0, 0.0, 10.0, 10.0, floor_id=self.floor.id)
        self.floor.add_zone(self.zone)

        self.camera_1 = Camera(
            name="Cam 1", position=(5.0, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=20.0,
        )
        self.camera_2 = Camera(
            name="Cam 2", position=(5.0, 5.0), floor_id=self.floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=20.0,
        )
        self.floor.add_camera(self.camera_1)
        self.floor.add_camera(self.camera_2)

        observation = HumanObservation(person_id="occ-1", zone_id=self.zone.id, floor_id=self.floor.id)
        self.provider = FakeHumanObservationProvider({"occ-1": observation})

        self.detection_provider = SimulatedDetectionProvider(
            [self.camera_1, self.camera_2], self.building, self.provider,
        )

    def test_is_a_detection_provider(self):

        self.assertIsInstance(self.detection_provider, DetectionProvider)

    def test_detections_at_delegates_to_the_requested_camera(self):

        detections = self.detection_provider.detections_at(self.camera_1.id, time=0.0)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].camera_id, self.camera_1.id)

    def test_unknown_camera_id_raises(self):

        with self.assertRaises(KeyError):
            self.detection_provider.detections_at("no-such-camera", time=0.0)

    def test_all_detections_at_aggregates_every_camera(self):

        detections = self.detection_provider.all_detections_at(time=0.0)

        self.assertEqual(len(detections), 2)
        self.assertEqual(
            {d.camera_id for d in detections}, {self.camera_1.id, self.camera_2.id},
        )


class RealHumanObservationProviderIntegrationTests(unittest.TestCase):

    # End-to-end through the ACTUAL production Human Observation
    # system (simulation_runtime.human_observation_bridge.
    # GroundTruthHumanObservationProvider, built on a real
    # MultiAgentSimulationResult) -- not the hand-built
    # FakeHumanObservationProvider every other test in this file uses.
    # Proves VirtualCamera genuinely reuses the existing Human
    # Behaviour/Human State systems end-to-end, not just against an
    # interface it happens to satisfy.

    def test_a_real_occupant_route_produces_detections_along_the_way(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        lobby = make_zone("Lobby", 0.0, 0.0, 10.0, 10.0, floor_id=floor.id)
        floor.add_zone(lobby)

        floor.add_exit(Exit(name="Ex", zone_id=lobby.id, floor_id=floor.id))

        camera = Camera(
            name="Lobby Cam", position=(5.0, 5.0), floor_id=floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=20.0,
        )
        floor.add_camera(camera)

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)

        simulation = MultiAgentSimulation(engine)
        simulation.add_occupant(lobby.id, occupant_id="occ-1", depart_time=0.0)

        result = simulation.run()

        # This occupant's only route is Lobby -> Outside via the Exit --
        # they start out standing in Lobby (observable to the camera)
        # before departing, exactly the moment this test checks.
        human_observation_provider = GroundTruthHumanObservationProvider(result)

        virtual_camera = VirtualCamera(camera, building, human_observation_provider)

        detections = virtual_camera.detections_at(time=0.0)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].occupant_id, "occ-1")
        self.assertEqual(detections[0].zone_id, lobby.id)
        # human_state/classification both flow straight through from
        # GroundTruthHumanObservationProvider's own resolution --
        # never recomputed or reinterpreted by VirtualCamera.
        self.assertIsNotNone(detections[0].human_state)
        self.assertEqual(detections[0].classification, HumanClassification.UNKNOWN)

    def test_a_fully_evacuated_occupant_is_no_longer_detected(self):

        building = Building(name="B")
        floor = building.create_floor(name="Ground Floor")

        lobby = make_zone("Lobby", 0.0, 0.0, 10.0, 10.0, floor_id=floor.id)
        floor.add_zone(lobby)
        floor.add_exit(Exit(name="Ex", zone_id=lobby.id, floor_id=floor.id))

        camera = Camera(
            name="Lobby Cam", position=(5.0, 5.0), floor_id=floor.id,
            rotation=0.0, horizontal_fov=360.0, max_range=20.0,
        )
        floor.add_camera(camera)

        graph = NavigationGraphGenerator().build(building)
        engine = PathfindingEngine(graph)

        simulation = MultiAgentSimulation(engine)
        simulation.add_occupant(lobby.id, occupant_id="occ-1", depart_time=0.0)

        result = simulation.run()

        human_observation_provider = GroundTruthHumanObservationProvider(result)
        virtual_camera = VirtualCamera(camera, building, human_observation_provider)

        arrival_time = result.occupants["occ-1"].arrival_time
        self.assertIsNotNone(arrival_time)

        detections_after_evacuation = virtual_camera.detections_at(time=arrival_time + 1.0)

        self.assertEqual(detections_after_evacuation, ())


class VirtualCameraPackageDependencyDirectionTests(unittest.TestCase):

    # Same regex-scan-the-source-files enforcement this codebase already
    # requires of every new package with an architectural boundary to
    # keep (see tests.test_perception.PerceptionPackageDependencyDirectionTests,
    # tests.test_sensors.SensorsPackageDependencyDirectionTests,
    # tests.test_ai_decision.AIDecisionPackageDependencyDirectionTests).
    # virtual_camera/ must reach classification/state/position only
    # through the HumanObservationProvider interface it's handed (see
    # virtual_camera/camera.py's own docstring) -- never by importing
    # ground_truth/behaviour_profile_resolver/behavior/simulator
    # directly, which is exactly what would make a future Live/Replay
    # DetectionProvider implementation require changes here.

    def test_virtual_camera_never_imports_simulation_internals_directly(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "virtual_camera"

        forbidden = (
            r"^\s*(from|import)\s+"
            r"(simulator|ground_truth|behavior|behavior_library|behaviour_profile_resolver|"
            r"simulation_runtime|hazard_evolution|ai_training|rl_training|advisory_system|"
            r"command_center|designer|cv2|torch|ultralytics|onvif)\b"
        )

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"virtual_camera/{path.name} imports a simulation/decision-layer module "
                f"directly -- it must only depend on perception/'s HumanObservationProvider "
                f"interface and visibility/, so a future Live/Replay DetectionProvider needs "
                f"no change here",
            )


if __name__ == "__main__":
    unittest.main()
