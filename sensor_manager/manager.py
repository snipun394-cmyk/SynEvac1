from typing import Dict, Mapping, Optional, Tuple

from models.detector_migration import adapt_legacy_detector
from models.sensor_asset import DetectorState

from sensor_manager.status import SensorStatus


class SensorManager:

    # The single place fire-protection sensors (Smoke Detector, Heat
    # Detector today; Flame/Gas/Manual Call Point once those asset
    # classes exist) are managed -- mirrors camera_manager.manager.
    # CameraManager's own design principles point for point (discover/
    # register/remove, lookup by id/floor/zone, enable/disable, group,
    # aggregate) WITHOUT importing it at all (enforced by this
    # package's own dependency-direction test, tests.test_sensor_manager.
    # SensorManagerPackageDependencyDirectionTests) -- Phase 5's own
    # explicit instruction. Two independent packages that happen to
    # share a shape, not one built on the other, the same "structurally
    # distinct concepts get structurally distinct types even when their
    # shape overlaps" principle perception/models/building_observation.py
    # already documents for HazardSnapshot/OccupancySnapshot.
    #
    # This class computes no hazard/physics of any kind -- it is pure
    # bookkeeping and routing, same "never performs computer vision"
    # discipline CameraManager holds for itself. aggregate_alarm_state()
    # below takes an already-computed Mapping[sensor_id, DetectorState]
    # (produced by calling each SmokeDetector/HeatDetector's own
    # compute_state() against whatever hazard reading a caller already
    # has -- see models/smoke_detector.py's own docstring on why that
    # stays a pure model method, never something SensorManager calls
    # for you) and only ever aggregates it; it never resolves a hazard
    # reading, a zone, or a sensor's own state itself.

    def __init__(self):

        self._sensors: Dict[str, object] = {}

    # =====================================================
    # Discovery / Registration
    # =====================================================

    def discover_sensors(self, building) -> Tuple[object, ...]:

        # Full, idempotent rescan -- same "rebuild rather than
        # incrementally patch" convention CameraManager.discover_
        # cameras() already establishes, for the same reason (simpler
        # and more robust than hooking every possible sensor-add/
        # sensor-remove call site in the Designer).
        #
        # Also discovers legacy generic Detector objects (floor.detectors,
        # the pre-canonical-asset model -- see models/detector.py) whose
        # detector_type is Smoke/Heat, adapting each one into the same
        # canonical SmokeDetector/HeatDetector shape via
        # models.detector_migration.adapt_legacy_detector(), preserving
        # its id exactly. This is what gives one physical detector one
        # stable identity across SensorManager/Perception/BuildingState
        # regardless of which Designer tool originally placed it -- the
        # adapted object is never written back onto Floor.detectors, so
        # a legacy detector never becomes two persisted assets, only one
        # asset visible through two Designer tools' worth of history.

        self._sensors = {}

        if building is None:
            return ()

        for floor in building.ordered_floors():

            for smoke_detector in floor.smoke_detectors:
                self.register_sensor(smoke_detector)

            for heat_detector in floor.heat_detectors:
                self.register_sensor(heat_detector)

            for legacy_detector in floor.detectors:

                adapted = adapt_legacy_detector(legacy_detector, floor.zones)

                if adapted is not None:
                    self.register_sensor(adapted)

        return self.all_sensors()

    # =====================================================

    def register_sensor(self, sensor) -> None:

        self._sensors[sensor.id] = sensor

    # =====================================================

    def remove_sensor(self, sensor_id: str) -> None:

        self._sensors.pop(sensor_id, None)

    # =====================================================

    def all_sensors(self) -> Tuple[object, ...]:

        return tuple(self._sensors.values())

    # =====================================================
    # Lookup
    # =====================================================

    def get_sensor(self, sensor_id: str) -> Optional[object]:

        return self._sensors.get(sensor_id)

    # =====================================================

    def sensors_on_floor(self, floor_id: str) -> Tuple[object, ...]:

        return tuple(
            sensor for sensor in self._sensors.values() if sensor.floor_id == floor_id
        )

    # =====================================================

    def sensors_in_zone(self, zone_id: str) -> Tuple[object, ...]:

        return tuple(
            sensor for sensor in self._sensors.values() if zone_id in sensor.zone_ids
        )

    # =====================================================
    # Enable / Disable
    # =====================================================

    def enable_sensor(self, sensor_id: str) -> None:

        self._require_sensor(sensor_id).active = True

    # =====================================================

    def disable_sensor(self, sensor_id: str) -> None:

        self._require_sensor(sensor_id).active = False

    # =====================================================
    # Grouping
    # =====================================================

    def sensors_by_type(self) -> Dict[str, Tuple[object, ...]]:

        grouped: Dict[str, list] = {}

        for sensor in self._sensors.values():
            grouped.setdefault(sensor.object_type, []).append(sensor)

        return {sensor_type: tuple(sensors) for sensor_type, sensors in grouped.items()}

    # =====================================================
    # Status
    # =====================================================

    def sensor_status(self, sensor_id: str, current_state=None) -> SensorStatus:

        sensor = self._require_sensor(sensor_id)

        return SensorStatus(
            sensor_id=sensor.id,
            sensor_type=sensor.object_type,
            name=sensor.name,
            floor_id=sensor.floor_id,
            zone_ids=sensor.zone_ids,
            active=sensor.active,
            mode=sensor.mode,
            health_status=sensor.health_status,
            current_state=current_state,
        )

    # =====================================================

    def all_statuses(self, states: Optional[Mapping[str, object]] = None) -> Tuple[SensorStatus, ...]:

        states = states or {}

        return tuple(
            self.sensor_status(sensor.id, current_state=states.get(sensor.id))
            for sensor in self.all_sensors()
        )

    # =====================================================
    # Aggregate Alarm State
    # =====================================================

    def aggregate_alarm_state(self, states: Mapping[str, object]):

        # Priority order: any ALARM anywhere wins outright (the most
        # urgent, safety-critical fact this building has); otherwise
        # any FAULT anywhere is still worth surfacing (a device that
        # cannot be trusted is not the same as "all clear"); otherwise
        # NORMAL. Only ever aggregates whatever DetectorState values
        # the caller already computed -- see this class's own docstring
        # on why SensorManager never computes one itself. Sensors with
        # no entry in `states` (not yet evaluated this cycle) are
        # simply not counted either way.

        relevant_states = [
            states[sensor_id] for sensor_id in self._sensors if sensor_id in states
        ]

        if any(state == DetectorState.ALARM for state in relevant_states):
            return DetectorState.ALARM

        if any(state == DetectorState.FAULT for state in relevant_states):
            return DetectorState.FAULT

        return DetectorState.NORMAL

    # =====================================================
    # Internals
    # =====================================================

    def _require_sensor(self, sensor_id: str):

        sensor = self._sensors.get(sensor_id)

        if sensor is None:

            raise KeyError(
                f"SensorManager has no sensor registered with id {sensor_id!r}."
            )

        return sensor
