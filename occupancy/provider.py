from occupancy.snapshot import OccupancySnapshot


class OccupancyProvider:

    # The one interface every future producer of people-location data
    # (CCTV, vision models, entry counters, BLE/UWB positioning, WiFi
    # localization, or a simulated replay) is meant to let consumers
    # reach it through -- mirrors HazardProvider's role exactly, one
    # level over on the Occupancy side. A future AI Decision Engine /
    # Firefighter Decision Support system depends only on this method,
    # never on how a given OccupancySnapshot was produced.

    def snapshot_at(self, time: float) -> OccupancySnapshot:

        raise NotImplementedError


class ManualOccupancyProvider(OccupancyProvider):

    # V1's whole provider: one fixed, hand-authored (or empty)
    # OccupancySnapshot, returned regardless of the requested time --
    # mirrors ManualHazardProvider's role for hand-built hazard
    # scenarios, here for occupancy instead. What a future time-
    # varying provider (backed by live sensors, or by an Occupancy
    # Fusion Engine analogous to HazardEvolutionEngine) replaces
    # without any change to whatever consumes snapshot_at().

    def __init__(self, snapshot: OccupancySnapshot = None):

        self.snapshot = snapshot or OccupancySnapshot()

    # =====================================================

    def snapshot_at(self, time: float) -> OccupancySnapshot:

        return self.snapshot
