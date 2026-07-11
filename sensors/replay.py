from typing import List, Mapping, Optional

from hazard_evolution.contribution import HazardContribution
from hazard_evolution.source import HazardSource

from occupancy.provider import OccupancyProvider
from occupancy.snapshot import OccupancySnapshot

from sensors.provider import SensorProvider
from sensors.reading import SensorReading


class SimulatedReplaySensorProvider(SensorProvider, HazardSource, OccupancyProvider):

    # The one deliberate bridge in this package -- everywhere else in
    # sensors/ stays independent of Hazard Evolution and Occupancy
    # (see sensors/provider.py), but this class's entire purpose is
    # proving that a live-sensor-shaped implementation is a drop-in
    # replacement for a simulated one, through the interfaces that
    # already exist today rather than new ones invented for it: it
    # satisfies SensorProvider (so it IS a sensor, structurally), and
    # simultaneously replays pre-authored HazardContribution/
    # OccupancySnapshot objects through the exact same HazardSource/
    # OccupancyProvider interfaces FireGrowthModel/SmokePropagation
    # Model/TenabilityModel and a future AI Decision Engine already
    # use. HazardEvolutionEngine and any future OccupancyProvider
    # consumer require zero changes to accept this class -- and a real
    # sensor integration built the same way later would require none
    # either.
    #
    # Deliberately not computer vision, not a detector protocol, not a
    # network client of any kind -- every schedule here is a plain,
    # hand-authored (or test-authored) mapping, copied once at
    # construction and never mutated. A requested time/step with
    # nothing scheduled simply yields "no opinion" (an empty
    # HazardContribution) or "no reading" (an OccupancySnapshot with
    # no observations) -- the same absent-means-nothing convention
    # HazardContribution/OccupancySnapshot already use elsewhere, so
    # nothing here needs a special case for gaps in the schedule.

    def __init__(
        self,
        readings_schedule: Optional[Mapping[float, List[SensorReading]]] = None,
        hazard_schedule: Optional[Mapping[float, HazardContribution]] = None,
        occupancy_schedule: Optional[Mapping[float, OccupancySnapshot]] = None,
    ):

        self._readings_schedule = dict(readings_schedule or {})
        self._hazard_schedule = dict(hazard_schedule or {})
        self._occupancy_schedule = dict(occupancy_schedule or {})

    # =====================================================
    # SensorProvider -- replays a predefined list of SensorReadings
    # keyed by the exact requested time.

    def readings_at(self, time: float) -> List[SensorReading]:

        return list(self._readings_schedule.get(time, []))

    # =====================================================
    # HazardSource -- replays a predefined HazardContribution keyed by
    # the step's end time, consulted by HazardEvolutionEngine exactly
    # like FireGrowthModel/SmokePropagationModel/TenabilityModel are.

    def propose(self, previous_snapshot, time: float, dt: float) -> HazardContribution:

        return self._hazard_schedule.get(time + dt, HazardContribution())

    # =====================================================
    # OccupancyProvider -- replays a predefined OccupancySnapshot keyed
    # by the requested time, the same single-lookup shape
    # ManualOccupancyProvider uses, just schedule-backed instead of
    # one fixed snapshot.

    def snapshot_at(self, time: float) -> OccupancySnapshot:

        return self._occupancy_schedule.get(time, OccupancySnapshot(timestamp=time))
