"""Sensor Fusion Engine milestone, Phase 10 -- offline demo.

Camera + Smoke detector + Heat detector -> SensorFusionEngine -> prints
inputs, confidence, and fused result per (location, kind) group.

No CCTV, no network -- every provider here is fed synthetic,
hand-built data.

Not a pytest test: run manually --
    python scripts/demo_sensor_fusion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sensor_fusion.engine import SensorFusionEngine
from sensor_fusion.observation import ObservationKind
from sensor_fusion.provider import CameraObservationProvider, HeatObservationProvider, SmokeObservationProvider


class FakeOccupant:

    def __init__(self, occupant_id, current_zone_id, behavior, confidence):
        self.occupant_id = occupant_id
        self.current_zone_id = current_zone_id
        self.behavior = behavior
        self.confidence = confidence


class FakeReading:

    def __init__(self, detector_id, alarm_active, confidence, timestamp=0.0):
        self.detector_id = detector_id
        self.timestamp = timestamp
        self.alarm_active = alarm_active
        self.confidence = confidence


def run_demo() -> None:

    # Camera: two occupants seen in zone-1.
    camera_provider = CameraObservationProvider()
    camera_provider.set_occupants([
        FakeOccupant("OCC-1", "zone-1", None, 0.9),
        FakeOccupant("OCC-2", "zone-1", None, 0.85),
    ])

    # Smoke detector in zone-1: alarming.
    smoke_provider = SmokeObservationProvider(zone_by_detector_id={"SD-1": "zone-1"})
    smoke_provider.set_readings([FakeReading("SD-1", alarm_active=True, confidence=0.9)])

    # Heat detector in zone-1: NOT alarming (disagrees with smoke on
    # the underlying fire condition, but this is two SEPARATE kinds --
    # SMOKE and HEAT each fuse and report independently, never
    # compared against each other).
    heat_provider = HeatObservationProvider(zone_by_detector_id={"HD-1": "zone-1"})
    heat_provider.set_readings([FakeReading("HD-1", alarm_active=False, confidence=0.9)])

    engine = SensorFusionEngine()

    print("=== Sensor Fusion Engine -- Offline Demo ===")
    print()
    print("Inputs:")
    print("  camera: 2 occupants in zone-1")
    print("  smoke detector SD-1 (zone-1): alarm_active=True, confidence=0.9")
    print("  heat detector HD-1 (zone-1): alarm_active=False, confidence=0.9")
    print()

    observations = engine.collect([camera_provider, smoke_provider, heat_provider], time=0.0)
    fused = engine.fuse(observations, time=0.0)

    print(f"{'location':<10} {'kind':<12} {'measurement':<14} {'confidence':>10} {'conflict':>9} sources")

    for f in fused:
        print(
            f"{f.location:<10} {f.kind.name:<12} {str(f.measurement):<14} "
            f"{f.confidence:>10.3f} {str(f.conflict):>9} {', '.join(f.contributing_sources)}"
        )

    print()
    print("Network access performed: NO")
    print("Physical CCTV accessed: NO")


if __name__ == "__main__":
    run_demo()
