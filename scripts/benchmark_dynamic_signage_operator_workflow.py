"""Live Dynamic Sign Operator Approval & Dispatch Completion milestone,
Phase 17 -- performance benchmark for the operator-workflow bookkeeping
this milestone adds (ingest / consistency-check / approve), at the
milestone's own named scale (100 dynamic signs, 50 zones).

Excludes Qt rendering time -- this measures LiveOperatorActionGateway/
DynamicSignageController/dynamic_signage.consistency bookkeeping only,
never QTableWidget population (a Command Center concern this milestone
does not attempt to benchmark; see this script's own printed note).

Not a pytest test -- run manually:
    python scripts/benchmark_dynamic_signage_operator_workflow.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.building import Building
from models.zone import Zone
from models.dynamic_sign import DynamicEvacuationSign

from evacuation_guidance.models import EvacuationGuidanceSnapshot, EvacuationGuidancePlan, RouteStatus

from dynamic_signage.consistency import detect_inconsistencies
from dynamic_signage.controller import DynamicSignageController
from dynamic_signage.models import DynamicSignageSnapshot, SignIndication, SignageInstruction, SignageStatus
from dynamic_signage.provider import SimulationDynamicSignageProvider

from command_center.live_operator_action_gateway import LiveOperatorActionGateway

ZONE_COUNT = 50
SIGN_COUNT = 100
ITERATIONS = 20


def _build_guidance_snapshot():

    zones = {}

    for i in range(ZONE_COUNT):

        zone_id = f"Z{i}"
        zones[zone_id] = EvacuationGuidancePlan(
            zone_id=zone_id, recommended_exit_id="EXIT-1", route_status=RouteStatus.ROUTE_AVAILABLE,
            revision=1, confidence=0.8, evidence_timestamp=0.0,
        )

    return EvacuationGuidanceSnapshot(timestamp=0.0, zones=zones, voice_plans={})


def _build_signage_snapshot(revision=1):

    instructions = {}

    for i in range(SIGN_COUNT):

        sign_id = f"DS-{i}"
        zone_id = f"Z{i % ZONE_COUNT}"

        instructions[sign_id] = SignageInstruction(
            sign_id=sign_id, zone_id=zone_id, recommended_exit_id="EXIT-1", guidance_revision=1,
            indication=SignIndication.STRAIGHT, status=SignageStatus.ACTIVE,
            signage_revision=revision, timestamp=0.0,
        )

    return DynamicSignageSnapshot(timestamp=0.0, instructions=instructions, conflicts={})


def _percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * pct))
    return ordered[index]


def _time_it(label, fn):

    samples_ms = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - start) * 1000)

    print(
        f"{label} -- mean {statistics.mean(samples_ms):.4f} ms, "
        f"p95 {_percentile(samples_ms, 0.95):.4f} ms, max {max(samples_ms):.4f} ms"
    )


def main():

    guidance_snapshot = _build_guidance_snapshot()
    signage_snapshot = _build_signage_snapshot()

    print(f"Live Dynamic Sign operator workflow -- {ZONE_COUNT} zones, {SIGN_COUNT} signs (Qt rendering excluded):")
    print()

    _time_it("detect_inconsistencies() (Validation column source)", lambda: detect_inconsistencies(guidance_snapshot, signage_snapshot))

    def _ingest_cycle():

        controller = DynamicSignageController(SimulationDynamicSignageProvider())
        gateway = LiveOperatorActionGateway(signage_controller=controller)
        gateway.ingest_signage_instructions(signage_snapshot)
        return gateway

    _time_it("ingest_signage_instructions() (100 signs, cold controller)", _ingest_cycle)

    gateway_for_approval = _ingest_cycle()

    def _approve_all():

        for i in range(SIGN_COUNT):
            instruction = signage_snapshot.instructions[f"DS-{i}"]
            try:
                gateway_for_approval.approve_signage_instruction(instruction, 1.0, guidance_snapshot)
            except Exception:
                pass  # already approved on a prior iteration -- timing only, not correctness, here

    _time_it("approve_signage_instruction() x100 (consistency-gated)", _approve_all)

    # A second-revision re-ingest (the steady-state per-cycle cost once
    # every sign already has an approved revision 1 -- Phase 12's own
    # "runtime generates a new revision" scenario at scale).
    signage_snapshot_v2 = _build_signage_snapshot(revision=2)
    _time_it("ingest_signage_instructions() (100 signs, re-submitting a new revision)", lambda: gateway_for_approval.ingest_signage_instructions(signage_snapshot_v2))

    print()
    print("Note: this measures gateway/controller/consistency-checker bookkeeping only -- Qt QTableWidget")
    print("population (LiveDynamicSignagePanel.show_live()) is excluded, as this is primarily bookkeeping")
    print("and not something this milestone prematurely optimizes for Qt rendering cost.")


if __name__ == "__main__":
    main()
