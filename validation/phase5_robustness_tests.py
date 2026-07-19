"""
Phase 5 -- Robustness Testing.

Stress-tests the real production pipeline against edge-case buildings/
scenarios: tiny buildings, large buildings, single/multiple exits,
multiple floors, heavy occupancy, detector/camera failures, blocked
exits, locked doors, and stair failures. Does not modify any existing
package -- only calls its public API and records pass/fail + timing.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.building import Building
from models.camera import Camera
from models.detector import Detector
from models.door import Door
from models.exit import Exit
from models.floor import Floor
from models.staircase import Staircase
from models.zone import Zone

from scenario import (
    Scenario, ScenarioCameraState, ScenarioDetectorState, ScenarioDoorState,
    ScenarioExitState, ScenarioFire, ScenarioMetadata, ScenarioOccupant, ScenarioStairState,
    DoorState, StairAvailability, DeviceAvailability,
)
from scenario_runner import run as run_scenario
from behaviour_profile_resolver import register_occupants

from ai_decision.engine import AIDecisionEngine

from simulation_runtime import SimulationRuntime

from ground_truth.analyzer import SimulationArtifacts, analyze

CHECKS = []


def check(name):
    def decorator(fn):
        CHECKS.append((name, fn))
        return fn
    return decorator


def _metadata(scenario_id):
    return ScenarioMetadata(
        scenario_id=scenario_id, definition_id="robustness-def", definition_content_hash="hash",
        generation_version="validation/1", seed=1, created_at="2026-07-15T00:00:00",
    )


def _run_full_pipeline(building, scenario, dt=5.0, end_time=None):

    context = run_scenario(scenario, building)
    register_occupants(context)
    engine = AIDecisionEngine(base_engine=context.engine)
    runtime = SimulationRuntime(context, engine, dt=dt, end_time=end_time)
    tick_results = runtime.run()

    gt = analyze(SimulationArtifacts(
        scenario=scenario, building=building, movement_result=runtime.movement_result,
        tick_results=tick_results,
    ))

    return runtime, gt


def _occupants(zone_id, count, floor_id="floor-1", profile="Staff_Default", prefix="o"):
    return tuple(
        ScenarioOccupant(occupant_id=f"{prefix}{i}", zone_id=zone_id, floor_id=floor_id,
                          position=(0.0, 0.0), behaviour_profile_id=profile)
        for i in range(count)
    )


# =====================================================


@check("very_small_building_single_zone_single_exit")
def check_tiny_building():

    floor1 = Floor(name="Only", id="floor-1",
                    zones=[Zone(id="zone-1", name="Room", x=0.0, y=0.0, width=2.0, height=2.0)],
                    exits=[Exit(id="exit-1", zone_id="zone-1")])
    building = Building(name="Tiny", id="b-tiny", floors=[floor1])

    scenario = Scenario(
        metadata=_metadata("tiny"), occupants=_occupants("zone-1", 1),
        fire=ScenarioFire(ignition_zone_id="zone-1", ignition_floor_id="floor-1",
                           fire_profile="Electrical", growth_parameters={"growth_time": 200.0}),
        events=(),
    )

    t0 = time.perf_counter()
    runtime, gt = _run_full_pipeline(building, scenario)
    elapsed = time.perf_counter() - t0

    ok = gt.building_cleared and gt.people_evacuated == 1
    return ok, {"elapsed": elapsed, "building_cleared": gt.building_cleared, "people_evacuated": gt.people_evacuated}


@check("very_large_building_many_zones_and_floors")
def check_large_building():

    floors = []
    n_floors = 5
    zones_per_floor = 10

    for f in range(n_floors):
        floor_id = f"floor-{f}"
        zones = [
            Zone(id=f"zone-{f}-{z}", name=f"Zone {f}-{z}", x=float(z * 10), y=0.0, width=2.0, height=2.0)
            for z in range(zones_per_floor)
        ]
        doors = [
            Door(id=f"door-{f}-{z}", normally_open=True, zone_a_id=f"zone-{f}-{z}", zone_b_id=f"zone-{f}-{z+1}")
            for z in range(zones_per_floor - 1)
        ]
        exits = [Exit(id=f"exit-{f}", zone_id=f"zone-{f}-{zones_per_floor - 1}")] if f == 0 else []
        stairs = []
        if f < n_floors - 1:
            stairs.append(Staircase(id=f"stair-{f}", from_zone_id=f"zone-{f}-0",
                                     to_zone_id=f"zone-{f+1}-0", to_floor_id=f"floor-{f+1}"))
        floors.append(Floor(name=floor_id, id=floor_id, zones=zones, doors=doors, exits=exits, stairs=stairs))

    building = Building(name="Large", id="b-large", floors=floors)

    occupants = tuple(
        ScenarioOccupant(occupant_id=f"o{f}", zone_id=f"zone-{f}-0", floor_id=f"floor-{f}",
                          position=(0.0, 0.0), behaviour_profile_id="Staff_Default")
        for f in range(n_floors) for _ in range(10)
    )

    scenario = Scenario(
        metadata=_metadata("large"), occupants=occupants,
        fire=ScenarioFire(ignition_zone_id="zone-0-0", ignition_floor_id="floor-0",
                           fire_profile="Electrical", growth_parameters={"growth_time": 400.0}),
        events=(),
    )

    t0 = time.perf_counter()
    runtime, gt = _run_full_pipeline(building, scenario, dt=5.0, end_time=2000.0)
    elapsed = time.perf_counter() - t0

    ok = gt.reachable_occupants + gt.unreachable_occupants == len(occupants)
    return ok, {"elapsed": elapsed, "total_occupants": len(occupants),
                "reachable": gt.reachable_occupants, "unreachable": gt.unreachable_occupants}


@check("single_exit_funnels_all_zones")
def check_single_exit():

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[Zone(id=f"zone-{i}", name=f"Z{i}", x=float(i * 5), y=0.0, width=2.0, height=2.0) for i in range(4)],
        doors=[Door(id=f"door-{i}", normally_open=True, zone_a_id=f"zone-{i}", zone_b_id=f"zone-{i+1}")
               for i in range(3)],
        exits=[Exit(id="exit-1", zone_id="zone-3")],
    )
    building = Building(name="SingleExit", id="b-single", floors=[floor1])

    occupants = tuple(
        ScenarioOccupant(occupant_id=f"o{i}", zone_id=f"zone-{i % 4}", floor_id="floor-1",
                          position=(0.0, 0.0), behaviour_profile_id="Staff_Default")
        for i in range(20)
    )

    scenario = Scenario(
        metadata=_metadata("single-exit"), occupants=occupants,
        fire=ScenarioFire(ignition_zone_id="zone-0", ignition_floor_id="floor-1",
                           fire_profile="Electrical", growth_parameters={"growth_time": 300.0}),
        events=(),
    )

    runtime, gt = _run_full_pipeline(building, scenario, dt=5.0, end_time=1000.0)
    ok = gt.reachable_occupants == len(occupants)
    return ok, {"reachable": gt.reachable_occupants, "evacuated": gt.people_evacuated}


@check("multiple_exits_load_distribution")
def check_multiple_exits():

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[Zone(id=f"zone-{i}", name=f"Z{i}", x=float(i * 5), y=0.0, width=2.0, height=2.0) for i in range(5)],
        exits=[Exit(id=f"exit-{i}", zone_id=f"zone-{i}") for i in range(5)],
    )
    building = Building(name="MultiExit", id="b-multi", floors=[floor1])

    occupants = tuple(
        ScenarioOccupant(occupant_id=f"o{i}", zone_id=f"zone-{i % 5}", floor_id="floor-1",
                          position=(0.0, 0.0), behaviour_profile_id="Staff_Default")
        for i in range(25)
    )

    scenario = Scenario(
        metadata=_metadata("multi-exit"), occupants=occupants,
        fire=ScenarioFire(ignition_zone_id="zone-0", ignition_floor_id="floor-1",
                           fire_profile="Electrical", growth_parameters={"growth_time": 300.0}),
        events=(),
    )

    runtime, gt = _run_full_pipeline(building, scenario)
    ok = gt.building_cleared
    return ok, {"building_cleared": gt.building_cleared}


@check("heavy_occupancy_200_occupants")
def check_heavy_occupancy():

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[Zone(id="zone-1", name="Hall", x=0.0, y=0.0, width=20.0, height=20.0),
               Zone(id="zone-2", name="Egress", x=30.0, y=0.0, width=4.0, height=4.0)],
        doors=[Door(id="door-1", normally_open=True, width=2.0, zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[Exit(id="exit-1", zone_id="zone-2", width=2.0)],
    )
    building = Building(name="Heavy", id="b-heavy", floors=[floor1])

    scenario = Scenario(
        metadata=_metadata("heavy"), occupants=_occupants("zone-1", 200),
        fire=ScenarioFire(ignition_zone_id="zone-2", ignition_floor_id="floor-1",
                           fire_profile="Electrical", growth_parameters={"growth_time": 400.0}),
        events=(),
    )

    t0 = time.perf_counter()
    runtime, gt = _run_full_pipeline(building, scenario, dt=5.0, end_time=3000.0)
    elapsed = time.perf_counter() - t0

    ok = gt.reachable_occupants == 200
    return ok, {"elapsed": elapsed, "reachable": gt.reachable_occupants, "evacuated": gt.people_evacuated,
                "total_evacuation_time": gt.total_evacuation_time}


def _detector_camera_building():
    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[Zone(id="zone-1", name="Room", x=0.0, y=0.0, width=4.0, height=4.0)],
        exits=[Exit(id="exit-1", zone_id="zone-1")],
        detectors=[Detector(id="det-1", active=True)],
        cameras=[Camera(id="cam-1", active=True)],
    )
    return Building(name="DetCam", id="b-detcam", floors=[floor1])


@check("detector_failure_does_not_crash_pipeline")
def check_detector_failure():

    building = _detector_camera_building()
    scenario = Scenario(
        metadata=_metadata("det-fail"), occupants=_occupants("zone-1", 2),
        fire=ScenarioFire(ignition_zone_id="zone-1", ignition_floor_id="floor-1",
                           fire_profile="Electrical", growth_parameters={"growth_time": 200.0}),
        detector_states=(ScenarioDetectorState(detector_id="det-1", availability=DeviceAvailability.FAILED),),
        events=(),
    )
    context = run_scenario(scenario, building)
    detector = next(d for f in context.building.floors for d in f.detectors)
    ok = detector.active is False
    return ok, {"detector_active": detector.active}


@check("camera_failure_does_not_crash_pipeline")
def check_camera_failure():

    building = _detector_camera_building()
    scenario = Scenario(
        metadata=_metadata("cam-fail"), occupants=_occupants("zone-1", 2),
        fire=ScenarioFire(ignition_zone_id="zone-1", ignition_floor_id="floor-1",
                           fire_profile="Electrical", growth_parameters={"growth_time": 200.0}),
        camera_states=(ScenarioCameraState(camera_id="cam-1", availability=DeviceAvailability.FAILED),),
        events=(),
    )
    context = run_scenario(scenario, building)
    camera = next(c for f in context.building.floors for c in f.cameras)
    ok = camera.active is False
    return ok, {"camera_active": camera.active}


@check("blocked_exit_forces_alternate_route_or_unreachable")
def check_blocked_exit():

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[Zone(id="zone-1", name="Room", x=0.0, y=0.0, width=2.0, height=2.0)],
        exits=[Exit(id="exit-1", zone_id="zone-1"), Exit(id="exit-2", zone_id="zone-1")],
    )
    building = Building(name="BlockedExit", id="b-blocked", floors=[floor1])
    scenario = Scenario(
        metadata=_metadata("blocked-exit"), occupants=_occupants("zone-1", 3),
        fire=ScenarioFire(ignition_zone_id="zone-1", ignition_floor_id="floor-1",
                           fire_profile="Electrical", growth_parameters={"growth_time": 200.0}),
        exit_states=(ScenarioExitState(exit_id="exit-1", is_open=False),),
        events=(),
    )
    runtime, gt = _run_full_pipeline(building, scenario)
    ok = gt.building_cleared and gt.reachable_occupants == 3
    return ok, {"building_cleared": gt.building_cleared, "reachable": gt.reachable_occupants}


@check("locked_door_blocks_route_no_crash")
def check_locked_door():

    # DISCOVERED FINDING (Phase 5), CORRECTED in the platform-refinement
    # phase that followed: a zone with no route to any exit under the
    # default ShortestRouteChoiceStrategy used to surface as
    # OccupantState.STATIONARY, not UNREACHABLE (PathfindingEngine.
    # nearest_exit() correctly returned None, but
    # MultiAgentSimulation.submit_decision() couldn't tell that apart
    # from a deliberate WAIT/IGNORE, since both look like
    # goal_id=None/route=None). BehaviorDecision now carries an explicit
    # route_unavailable flag (set by HumanBehaviorLayer.register(), the
    # one place that still knows whether movement was required once
    # goal_id/route have already collapsed to None/None) so
    # submit_decision() registers these occupants the same way
    # add_occupant() registers any other unreachable occupant. This
    # check now asserts the corrected UNREACHABLE outcome.

    floor1 = Floor(
        name="Ground", id="floor-1",
        zones=[Zone(id="zone-1", name="Trapped", x=0.0, y=0.0, width=2.0, height=2.0),
               Zone(id="zone-2", name="Egress", x=10.0, y=0.0, width=2.0, height=2.0)],
        doors=[Door(id="door-1", normally_open=True, zone_a_id="zone-1", zone_b_id="zone-2")],
        exits=[Exit(id="exit-1", zone_id="zone-2")],
    )
    building = Building(name="LockedDoor", id="b-locked", floors=[floor1])
    scenario = Scenario(
        metadata=_metadata("locked-door"), occupants=_occupants("zone-1", 3),
        fire=ScenarioFire(ignition_zone_id="zone-2", ignition_floor_id="floor-1",
                           fire_profile="Electrical", growth_parameters={"growth_time": 200.0}),
        door_states=(ScenarioDoorState(door_id="door-1", state=DoorState.LOCKED),),
        events=(),
    )
    runtime, gt = _run_full_pipeline(building, scenario)

    occupant_states = {o: t.state.name for o, t in runtime.movement_result.occupants.items()}
    all_unreachable = all(state == "UNREACHABLE" for state in occupant_states.values())

    ok = (
        (not gt.building_cleared) and all_unreachable and gt.people_evacuated == 0
        and gt.unreachable_occupants == 3 and gt.reachable_occupants == 0
    )
    return ok, {
        "unreachable_per_ground_truth": gt.unreachable_occupants,
        "reachable_per_ground_truth": gt.reachable_occupants,
        "actual_occupant_states": occupant_states,
        "building_cleared": gt.building_cleared,
        "finding": "no-route occupants are correctly classified UNREACHABLE (fixed)",
    }


@check("stair_failure_blocks_route_no_crash")
def check_stair_failure():

    # Same discovered-and-corrected finding as
    # locked_door_blocks_route_no_crash above, exercised via a stair
    # closure instead of a locked door.

    floor1 = Floor(name="Ground", id="floor-1",
                    zones=[Zone(id="zone-1", name="Landing", x=0.0, y=0.0, width=2.0, height=2.0)],
                    exits=[Exit(id="exit-1", zone_id="zone-1")],
                    stairs=[Staircase(id="stair-1", from_zone_id="zone-1", to_zone_id="zone-2", to_floor_id="floor-2")])
    floor2 = Floor(name="Upper", id="floor-2",
                    zones=[Zone(id="zone-2", name="Upper Room", x=0.0, y=0.0, width=2.0, height=2.0)])
    building = Building(name="StairFail", id="b-stairfail", floors=[floor1, floor2])

    scenario = Scenario(
        metadata=_metadata("stair-fail"), occupants=_occupants("zone-2", 2, floor_id="floor-2"),
        fire=ScenarioFire(ignition_zone_id="zone-1", ignition_floor_id="floor-1",
                           fire_profile="Electrical", growth_parameters={"growth_time": 200.0}),
        stair_states=(ScenarioStairState(stair_id="stair-1", availability=StairAvailability.CLOSED),),
        events=(),
    )
    runtime, gt = _run_full_pipeline(building, scenario)

    occupant_states = {o: t.state.name for o, t in runtime.movement_result.occupants.items()}
    all_unreachable = all(state == "UNREACHABLE" for state in occupant_states.values())

    ok = (
        (not gt.building_cleared) and all_unreachable and gt.people_evacuated == 0
        and gt.unreachable_occupants == 2 and gt.reachable_occupants == 0
    )
    return ok, {
        "unreachable_per_ground_truth": gt.unreachable_occupants,
        "reachable_per_ground_truth": gt.reachable_occupants,
        "actual_occupant_states": occupant_states,
        "building_cleared": gt.building_cleared,
        "finding": "no-route occupants are correctly classified UNREACHABLE (fixed)",
    }


def main():

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)

    report = []

    for name, fn in CHECKS:
        try:
            ok, details = fn()
        except Exception as exc:
            ok, details = False, {"exception": repr(exc)}

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {json.dumps(details, default=str)}")
        report.append({"check": name, "status": status, "details": details})

    with open(os.path.join(output_dir, "phase5_robustness_tests.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    failed = [r for r in report if r["status"] == "FAIL"]
    print(f"\n{len(report) - len(failed)}/{len(report)} checks passed.")


if __name__ == "__main__":
    main()
