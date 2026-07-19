from typing import Mapping

from perception.models.human_observation import HumanObservation


class HumanObservationProvider:

    # The one seam every future producer of per-person human-behavior
    # observations -- a YOLO-based detector, a pose-estimation stage,
    # an action-recognition stage, or any other future perception
    # model -- is meant to implement. Mirrors PerceptionProvider/
    # CameraProvider/HazardProvider's own established "one seam, no
    # concrete vision implementation" pattern exactly: this file
    # contains no detection logic, no model, and no networking, only
    # the contract. GroundTruthHumanObservationProvider (this
    # package's own sibling module) is, for now, the only concrete
    # implementation -- a simulation-backed stand-in, not a vision
    # model, kept honestly separate from this interface so a real
    # vision-backed implementation can be swapped in later with zero
    # change to any consumer (live_system.integration.
    # SensorFusionPerceptionGateway, in particular) that depends only
    # on this class.

    def observations_at(self, time: float) -> Mapping[str, HumanObservation]:

        raise NotImplementedError
