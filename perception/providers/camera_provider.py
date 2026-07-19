from sensors.provider import CameraProvider as RawCameraProvider

from perception.models.camera_observation import CameraFrameObservation


class CameraProvider(RawCameraProvider):

    # Specializes sensors.provider.CameraProvider's raw, opaque-
    # payload contract into a typed, per-device "Sensor Observation"
    # interface -- still device-scoped, still unfused, still not
    # node-keyed (see CameraFrameObservation), but no longer an opaque
    # SensorReading.payload a caller has to interpret itself.
    # Resolving which Node.id(s) a given camera's frame corresponds
    # to, and turning a series of these into a fused, node-keyed
    # ObservedOccupancy, is a future Occupancy Estimation stage's job,
    # not this interface's -- this class adds exactly one method and
    # implements none of it.

    def frame_observation_at(self, camera_id: str, time: float) -> CameraFrameObservation:

        raise NotImplementedError
