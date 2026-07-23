from models.sensor_asset import HealthStatus


class PassiveFireSafetyAvailability:

    # Shared vocabulary for FireExtinguisher / FireHydrant / HoseReel --
    # three genuinely manual, passive fire-safety resources introduced
    # together in this milestone (Fire Suppression & Water-Based Safety
    # Asset Digital Twin). Reuses the exact three-value shape
    # models.emergency_light.EmergencyLightAvailability already
    # establishes for the same underlying question ("is this device
    # currently usable"), restated here as its own type rather than
    # imported from EmergencyLight -- EmergencyLight belongs to a
    # different, already-committed milestone and this family must not
    # couple to it; see compute_passive_availability() below for the
    # one shared computation these three siblings actually share.
    #
    # Deliberately NOT reused by Sprinkler -- a Sprinkler has a genuine
    # ACTIVATED condition (see models.sprinkler.SprinklerActivationState),
    # which is a structurally different question from "is this passive
    # resource currently usable".

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    FAULT = "FAULT"

    ALL = (AVAILABLE, UNAVAILABLE, FAULT)


def compute_passive_availability(active: bool, health_status: str) -> str:

    # The one computation FireExtinguisher/FireHydrant/HoseReel's own
    # compute_availability() methods each call rather than restating --
    # identical logic to EmergencyLight.compute_availability() (FAULT
    # outranks inactive/offline, mirroring every other "device fault
    # outranks anything else" priority in this codebase), factored out
    # here because these three siblings share it exactly, unlike
    # EmergencyLight which stands alone in its own milestone.

    if health_status == HealthStatus.FAULT:
        return PassiveFireSafetyAvailability.FAULT

    if not active or health_status == HealthStatus.OFFLINE:
        return PassiveFireSafetyAvailability.UNAVAILABLE

    return PassiveFireSafetyAvailability.AVAILABLE
