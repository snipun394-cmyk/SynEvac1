"""Manual Call Point -> Live Emergency Response Integration milestone,
Phase 15 -- performance benchmark for localized FACP/MCP alarm-source
evidence, at the milestone's own named scale (50 zones, 100 detector/
MCP alarm sources).

Not a pytest test -- run manually:
    python scripts/benchmark_manual_call_point_emergency_response.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.building import Building
from models.zone import Zone
from models.manual_call_point import ManualCallPoint
from models.smoke_detector import SmokeDetector

from sensor_manager.manager import SensorManager

from facp.engine import SimulatedFACP
from facp.models import DetectorConditionReport

from models.sensor_asset import DetectorState

from building_state.estimator import BuildingStateEstimator

from emergency_response.engine import EmergencyResponseIntelligenceEngine

from live_occupants.manager import LiveOccupantManager

from hazard.snapshot import HazardSnapshot
from occupancy.snapshot import OccupancySnapshot

ZONE_COUNT = 50
ALARM_SOURCE_COUNT = 100
ITERATIONS = 20


def _build_building():

    building = Building(name="Benchmark Building")
    floor = building.create_floor(name="Ground")

    for i in range(ZONE_COUNT):
        floor.add_zone(Zone(id=f"Z{i}", name=f"Zone {i}", floor_id=floor.id, x=float(i * 12), y=0.0, width=10.0, height=10.0))

    for i in range(ALARM_SOURCE_COUNT):

        zone_id = f"Z{i % ZONE_COUNT}"

        if i % 2 == 0:
            floor.smoke_detectors.append(SmokeDetector(id=f"SD{i}", name=f"SD{i}", floor_id=floor.id, zone_ids=(zone_id,)))
        else:
            floor.manual_call_points.append(ManualCallPoint(id=f"MCP{i}", name=f"MCP{i}", floor_id=floor.id, zone_ids=(zone_id,)))

    return building, floor


def _percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def main():

    building, floor = _build_building()

    sensor_manager = SensorManager()
    sensor_manager.discover_sensors(building)

    facp = SimulatedFACP()

    reports = {}
    for i in range(ALARM_SOURCE_COUNT):

        zone_id = f"Z{i % ZONE_COUNT}"

        if i % 2 == 0:
            reports[f"SD{i}"] = DetectorConditionReport(
                asset_id=f"SD{i}", asset_type="SmokeDetector", state=DetectorState.ALARM, floor_id=floor.id, zone_ids=(zone_id,),
            )
        else:
            mcp = sensor_manager.get_sensor(f"MCP{i}")
            mcp.activate()
            reports[f"MCP{i}"] = DetectorConditionReport(
                asset_id=f"MCP{i}", asset_type="ManualCallPoint", state=mcp.compute_state(0.0), floor_id=floor.id, zone_ids=(zone_id,),
            )

    facp.evaluate(reports, 0.0)
    facp_snapshot = facp.current_snapshot(0.0)

    smoke_statuses = tuple(s for s in sensor_manager.all_statuses() if s.sensor_type == "SmokeDetector")
    mcp_statuses = tuple(s for s in sensor_manager.all_statuses() if s.sensor_type == "ManualCallPoint")

    building_state = BuildingStateEstimator().estimate(
        0.0, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
        smoke_detector_statuses=smoke_statuses, manual_call_point_statuses=mcp_statuses,
        facp_snapshot=facp_snapshot,
    )

    occupant_manager = LiveOccupantManager()
    for i in range(ZONE_COUNT):
        occupant_manager.update(f"OCC-{i}", None, None, f"Z{i}", floor.id, None, None, None, 0.9, 0.0)

    response_engine = EmergencyResponseIntelligenceEngine(building, occupant_manager)

    samples_ms = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        response_engine.compute(0.0, building_state)
        samples_ms.append((time.perf_counter() - start) * 1000)

    print(f"EmergencyResponseIntelligenceEngine.compute() -- {ZONE_COUNT} zones, {ALARM_SOURCE_COUNT} alarm sources (mixed Smoke/MCP):")
    print(
        f"  mean {statistics.mean(samples_ms):.4f} ms, "
        f"p95 {_percentile(samples_ms, 0.95):.4f} ms, "
        f"max {max(samples_ms):.4f} ms"
    )

    # Baseline: same building, zero alarm sources, for direct comparison.
    baseline_building_state = BuildingStateEstimator().estimate(
        0.0, hazard_snapshot=HazardSnapshot(), occupancy_snapshot=OccupancySnapshot(),
    )
    baseline_samples_ms = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        response_engine.compute(0.0, baseline_building_state)
        baseline_samples_ms.append((time.perf_counter() - start) * 1000)

    print()
    print(f"Baseline (no FACP status at all):")
    print(
        f"  mean {statistics.mean(baseline_samples_ms):.4f} ms, "
        f"p95 {_percentile(baseline_samples_ms, 0.95):.4f} ms, "
        f"max {max(baseline_samples_ms):.4f} ms"
    )

    overhead_ms = statistics.mean(samples_ms) - statistics.mean(baseline_samples_ms)
    print()
    print(f"Incremental cost of localized FACP/MCP evidence ({ALARM_SOURCE_COUNT} sources): {overhead_ms:.4f} ms/cycle")


if __name__ == "__main__":
    main()
