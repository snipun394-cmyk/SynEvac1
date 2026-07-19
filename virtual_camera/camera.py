import math
import random
import uuid

from typing import Optional, Tuple

from perception.models.human_observation import HumanClassification
from perception.providers.human_observation_provider import HumanObservationProvider

from visibility.engine import VisibilityEngine

from virtual_camera.detection import Detection
from virtual_camera.imperfections import DetectionImperfectionModel, PERFECT_DETECTION


class VirtualCamera:

    # "What would this real CCTV camera see, right now" -- the one
    # class this milestone adds that actually generates Detections.
    # Deliberately a thin composition of three already-existing,
    # already-tested pieces, nothing more:
    #
    #   Camera Asset (models/camera.py)         -- where/how it looks
    #   Visibility Engine (visibility/engine.py) -- what it can see
    #   HumanObservationProvider (perception/)   -- who/what state
    #
    # This class contains no classification logic, no fall/injury
    # derivation, no route/behavior logic of its own -- every one of
    # those already exists (ground_truth.human_behavior,
    # behaviour_profile_resolver, simulation_runtime.
    # human_observation_bridge.GroundTruthHumanObservationProvider) and
    # is reached only through the HumanObservationProvider interface
    # passed in, never imported directly here. That is also exactly
    # what makes this class Simulation/Replay/Live-agnostic (Phase 6):
    # a Replay-backed or Live-backed HumanObservationProvider (should
    # one ever exist) would need zero change here to keep working --
    # though a real Live CCTV deployment would more likely replace this
    # whole class with a real detector (see virtual_camera/provider.py
    # module docstring for the full replacement story).
    #
    # camera/building/human_observation_provider are fixed for this
    # instance's lifetime, same "constructed once per run, never
    # mutated" convention MultiAgentSimulation's own capacity_model/
    # congestion_model already follow. The camera's own visibility
    # polygon is computed ONCE, here, rather than recomputed on every
    # detections_at() call -- a Camera's geometry does not change
    # mid-simulation, only occupant positions do, so recomputing it
    # per tick would be pure waste (see visibility/engine.py's own
    # documented per-call cost). A caller that DOES change a Camera's
    # geometry mid-run should construct a new VirtualCamera, the same
    # "rebuild rather than mutate" pattern NavigationGraphGenerator's
    # own build() already establishes for Building edits.

    def __init__(
        self,
        camera,
        building,
        human_observation_provider: HumanObservationProvider,
        visibility_engine: Optional[VisibilityEngine] = None,
        imperfection_model: Optional[DetectionImperfectionModel] = None,
    ):

        self.camera = camera
        self.building = building
        self.human_observation_provider = human_observation_provider

        visibility_engine = visibility_engine or VisibilityEngine()
        self.imperfection_model = imperfection_model or PERFECT_DETECTION

        self._visibility = visibility_engine.compute_camera_visibility(camera, building)

        self._rng = (
            random.Random(self.imperfection_model.seed)
            if self.imperfection_model.seed is not None
            else random.Random()
        )

    # =====================================================

    def detections_at(self, time: float) -> Tuple[Detection, ...]:

        if not self.camera.active:
            return ()

        observation_time = time - self.imperfection_model.tracking_delay

        observations = self.human_observation_provider.observations_at(observation_time)

        detections = []

        for occupant_id, observation in observations.items():

            if not self._is_observation_visible(observation):
                continue

            if (
                self.imperfection_model.detection_probability < 1.0
                and self._rng.random() >= self.imperfection_model.detection_probability
            ):
                # A missed detection -- the occupant IS visible, this
                # tick simply failed to report them, exactly the way a
                # real detector occasionally drops a frame's worth of
                # a genuinely-visible person.
                continue

            detections.append(
                self._build_detection(occupant_id, observation, time)
            )

        if (
            self.imperfection_model.false_positive_rate > 0.0
            and self._rng.random() < self.imperfection_model.false_positive_rate
        ):
            detections.append(self._build_false_positive(time))

        return tuple(detections)

    # =====================================================

    def _is_observation_visible(self, observation) -> bool:

        # Camera range/FOV/occlusion are already fully baked into
        # visibility_polygon by the ray-sweep (visibility/engine.py) --
        # a zone counted as visible or partially visible here already
        # passed all three checks Phase 2 names. An occupant on a
        # different floor, or whose own location this run's Ground
        # Truth cannot honestly resolve (zone_id is None -- e.g.
        # UNREACHABLE with no route at all), is never visible; guessing
        # either would fabricate a sighting this camera could not
        # actually have.

        if observation.floor_id != self.camera.floor_id:
            return False

        if observation.zone_id is None:
            return False

        return (
            observation.zone_id in self._visibility.visible_zone_ids
            or observation.zone_id in self._visibility.partially_visible_zone_ids
        )

    # =====================================================

    def _zone_center(self, zone_id):

        floor = self.building.get_floor(self.camera.floor_id)

        if floor is None:
            return None

        for zone in floor.zones:

            if zone.id == zone_id:
                return zone.center

        return None

    # =====================================================

    def _build_detection(self, occupant_id, observation, time) -> Detection:

        confidence = 1.0

        if self.imperfection_model.confidence_variation > 0.0:

            # A one-sided draw (always <= 1.0), not a symmetric Gaussian
            # clipped at the ceiling -- a symmetric draw centered on
            # 1.0 would get clipped back to exactly 1.0 on every
            # positive-side sample (roughly half of them), silently
            # defeating "variation" for about half of all detections.
            # 1.0 is the ceiling every real detector confidence sits
            # at or below, never above, so variation should only ever
            # pull confidence down from it.
            confidence = max(
                0.0,
                1.0 - abs(
                    self._rng.gauss(0.0, self.imperfection_model.confidence_variation)
                ),
            )

        return Detection(
            camera_id=self.camera.id,
            timestamp=time,
            occupant_id=occupant_id,
            floor_id=observation.floor_id,
            zone_id=observation.zone_id,
            position=self._zone_center(observation.zone_id),
            confidence=confidence,
            classification=observation.classification,
            human_state=observation.state,
        )

    # =====================================================

    def _build_false_positive(self, time) -> Detection:

        cx, cy = self.camera.position

        half_fov = self.camera.horizontal_fov / 2
        angle_degrees = self.camera.rotation + self._rng.uniform(-half_fov, half_fov)
        angle_radians = math.radians(angle_degrees)

        distance = self._rng.uniform(0.0, self.camera.max_range)

        position = (
            cx + distance * math.cos(angle_radians),
            cy + distance * math.sin(angle_radians),
        )

        # A false positive is, by construction, less trustworthy than
        # a real detection -- sampled from a deliberately mediocre
        # band rather than either extreme, so a downstream confidence
        # threshold has something realistic to filter against.
        confidence = self._rng.uniform(0.3, 0.7)

        # Drawn from this instance's own (optionally seeded) RNG, not
        # uuid.uuid4() -- uuid4 always uses the OS's own non-
        # deterministic entropy source regardless of any seed passed to
        # DetectionImperfectionModel, which would silently break
        # Phase 5's reproducibility guarantee for the one field a
        # seeded run couldn't otherwise control.
        synthetic_id = uuid.UUID(int=self._rng.getrandbits(128))

        return Detection(
            camera_id=self.camera.id,
            timestamp=time,
            occupant_id=f"false-positive-{synthetic_id}",
            floor_id=self.camera.floor_id,
            zone_id=None,
            position=position,
            confidence=confidence,
            classification=HumanClassification.UNKNOWN,
            human_state=None,
            is_false_positive=True,
        )
