# Live Dynamic Evacuation Signage

Status as of this milestone: SynEvac can, for the first time, turn an already-computed `EvacuationGuidancePlan` ("how to reach the recommended exit") into individual, per-sign structured instructions ("Sign SIGN-2A → RIGHT → toward Door D2") — via a new `dynamic_signage/` package, a new `models.dynamic_sign.DynamicEvacuationSign` Building Digital Twin asset, a `DynamicSignageController`/`SimulationDynamicSignageProvider` pair reached only through explicit operator approval, and a Command Center panel. This is a digital-twin signage framework and offline simulation only — no physical signage hardware, no vendor protocol, no automatic execution, no AI-generated directions.

## 1. Investigation findings (Phase 1)

1. No `Sign`/`ExitSign`/`DynamicSign` asset existed anywhere in the codebase. `building_control/advisory_adapter.py` explicitly documents (and this milestone's own architecture guard locks) that the pre-existing `"Update Dynamic Exit Signs"` `BuildingRecommendation` (produced by `advisory_system/advisory_engine.py`) was investigated and deliberately never translated into a `ControlRequest`, precisely because no digital signage abstraction existed — this milestone fills exactly that gap, and `building_control` continues to leave it untranslated (never a second, competing dispatch path).
2. `Camera`/`Speaker`/`SmokeDetector` all establish the `EngineeringAsset` → (`SensorAsset` →) concrete-asset pattern (`floor_id`/`zone_ids`/`position`/`active`/`mode`/`connection`, plus a concrete class's own few extra fields). `Camera.rotation` is the exact orientation precedent a sign needs (0° = +x, increasing clockwise, matching Qt's own scene rotation convention — confirmed against `designer.items.camera_item.CameraItem`).
3. `Floor` already lists every asset type as a plain `list[...]` field with `add_*`/`remove_*`/`*_count` plus `to_dict()`/`from_dict()` entries — `signs: list[DynamicEvacuationSign]` was added the same way; a project file with no `"signs"` key simply defaults to an empty list (`data.get("signs", [])`), so every pre-existing `.syn` file keeps loading unchanged.
4. Designer click-to-place assets are wired in exactly three places: `designer/widgets/toolbar.py` (a `QAction`), `designer/scene/graphics_scene.py` (`mousePressEvent`'s `if self.current_tool == "..."` branch + `rebuild_scene()`'s load loop + the delete/isinstance lists), and `designer/windows/main_window.py` (action → `change_tool()`, item-select → `property_panel.show_*()`). The Sign tool follows this identically.
5. `EvacuationGuidancePlan.ordered_navigation_steps` (from `evacuation_guidance/models.py`) already carries typed steps (`PASS_THROUGH_DOOR`/`USE_STAIR`/`CONTINUE_TO_EXIT`/...) with real Door/Stair/Exit ids in exact route order — everything Dynamic Signage needs to find "the next meaningful decision after a sign's zone" without touching Crowd Intelligence/AI/Decision Policy.
6. `Door`/`Exit` expose `.center`; `Staircase` exposes `.from_position`/`.to_position` (one per floor it spans) — real, sufficient geometry to derive a target position for every edge type honestly.
7. LEFT/RIGHT/STRAIGHT can be honestly derived from the signed angular difference between a sign's own facing direction and the bearing to its target, given the SAME coordinate convention `Camera` already uses — verified computationally (not assumed) in `tests/test_dynamic_signage_direction.py` against concrete worked positions; see `dynamic_signage/direction.py`'s own module docstring for the full derivation.
8. `command_center.live_operator_action_gateway.LiveOperatorActionGateway` already established the exact "explicit operator click → controller → provider" shape for voice/building-control; Dynamic Signage adds a parallel, identically-shaped `ingest_signage_instructions()`/`approve_signage_instruction()`/`reject_signage_instruction()` triple.

## 2. Architecture — the four concepts, kept strictly separate

```
Evacuation Recommendation   -- WHICH EXIT
        |
        v
Evacuation Guidance         -- HOW TO GET THERE (a graph-valid Route)
        |
        v
dynamic_signage/                                  <- THIS milestone
  route_mapping.py   -- "next meaningful decision after a sign's zone"
                         (Door/Stair/Exit + real position), including
                         zones with NO recommendation of their own that
                         are merely passed through by another zone's
                         route (a corridor sign)
  direction.py        -- pure geometry: LEFT/RIGHT/STRAIGHT from sign
                         position+orientation vs. target position
  planner.py          -- DynamicSignagePlanner: WHAT EACH SIGN SHOULD
                         INDICATE. Zero dispatch authority. Recomputed
                         FRESH every cycle from the current guidance
                         snapshot only -- nothing carries forward except
                         a revision counter used purely to detect change.
  consistency.py       -- independent Voice/Signage/Guidance cross-check
  history.py           -- append-only instruction history
  controller.py        -- DynamicSignageController: PENDING_APPROVAL /
                         APPROVED / REJECTED / SUPERSEDED / DISPATCHED /
                         CONFIRMED / FAILED lifecycle, never auto-approves
  provider.py           -- DynamicSignageProvider (seam) /
                         SimulationDynamicSignageProvider (ACTUAL DISPLAY
                         STATE, in-memory only, no network/hardware)
        |
        v
DynamicSignageSnapshot ("what each sign should indicate", planning only)
        |
        +--------------------------------------------+
        v                                             v
Advisory ("Dynamic signage guidance          Command Center (Live Dynamic
 available" -- display only, chooses          Signage panel: per-sign
 no direction of its own)                     instruction + Approve/Reject)
                                                        |
                                                        v
                                          Operator's own explicit click
                                                        |
                                                        v
                          command_center.live_operator_action_gateway.
                          LiveOperatorActionGateway.approve_signage_instruction()
                                                        |
                                                        v
                          dynamic_signage.controller.DynamicSignageController
                                                        |
                                                        v
                          dynamic_signage.provider.SimulationDynamicSignageProvider
                          (or a future LIVE_HARDWARE provider -- not implemented)
```

**Recommendation = WHICH EXIT. Guidance = HOW TO GET THERE. Signage Planning = WHAT EACH SIGN SHOULD INDICATE. Provider = ACTUAL DISPLAY STATE.** Each layer only ever reads the one directly above it; none re-derives or second-guesses it.

## 3. Sign asset (`models/dynamic_sign.py`)

`DynamicEvacuationSign(EngineeringAsset)` adds exactly two fields: `orientation` (degrees, `Camera.rotation`'s own convention) and `supported_indications` (a tuple narrowing which of `SignIndication.ALL` this specific physical sign can display — default is every indication). No IP address, no Modbus/BACnet registers, no serial port, no firmware assumptions — digital twin only.

## 4. Indication vocabulary and its honest triggers (`models/dynamic_sign.SignIndication`)

| Indication | When produced |
|---|---|
| `STRAIGHT` / `LEFT` / `RIGHT` | The next meaningful decision is a **Door** — a directional arrow computed by `direction.relative_direction()`. |
| `USE_STAIRS` | The next meaningful decision is a **Stair** — a placard, not an arrow (real-world stairwell signage says "STAIRS", not "turn left"), per this milestone's own documented, deliberate design choice. |
| `EXIT_HERE` | The next meaningful decision is the recommended **Exit** itself, or the zone is already `ALREADY_AT_EXIT`. |
| `NO_SAFE_DIRECTION` | The zone's own recommendation genuinely found no safe exit (`RouteStatus.NO_SAFE_EXIT`). |
| `DO_NOT_USE` | The previously-valid route through this zone specifically became unsafe (`GuidanceInconsistency.ROUTE_BECAME_UNSAFE`) — distinct from simply having no information. |
| `UNAVAILABLE` | Sign inactive; no guidance at all for any of the sign's zones; target geometry missing; sign hardware doesn't support the computed indication; or (Phase 7) multiple zones/routes disagree — see below. |

## 5. Conflict handling (Phase 7/22)

A sign may be assigned to more than one zone, or (for a zone with no recommendation of its own) may sit on more than one other zone's currently-valid route. Every candidate zone/route is resolved independently; if they do not all agree on the exact same `(indication, target_asset_id, recommended_exit_id)`, the planner reports a `SignageConflict` (deterministic — sorted zone ids, never "whichever iterated first") and the sign's own instruction becomes `SignageStatus.CONFLICT` / `UNAVAILABLE`. **No arbitrary pick ever happens.**

## 6. Revision & supersession (Phase 12/16)

`DynamicSignagePlanner` recomputes every sign fresh every cycle; a per-sign fingerprint `(indication, status, target_asset_id, recommended_exit_id, zone_id)` — deliberately excluding the raw guidance revision — determines whether `signage_revision` increments. `DynamicSignageController.submit()` supersedes an old **pending** instruction the moment a newer revision arrives for the same sign; an already-`CONFIRMED`/`DISPATCHED` instruction is never rewritten, only left behind in history.

## 7. Authority ladder (Phase 11) — mechanically guarded

`Decision Policy > Evacuation Recommendation > Evacuation Guidance > Dynamic Signage`. The Planner never imports `dynamic_signage.provider`/`dynamic_signage.controller` (zero dispatch authority); AI/Advisory/Decision Policy/FACP never import `dynamic_signage.controller`/`dynamic_signage.provider`/`command_center.live_operator_action_gateway` (zero execution authority) — both mechanically enforced by `tests/test_dynamic_signage_architecture_guards.py`.

## 8. BuildingState relationship (Phase 24)

Dynamic signage is an OUTPUT/control-planning state, not a state-estimation input — it is **not** folded into canonical `BuildingState`. It follows the exact sibling-snapshot pattern `EvacuationGuidanceSnapshot` already established: `LiveBuildingSnapshot.dynamic_signage`, maintained by `StateManager.update_dynamic_signage()`/`latest_dynamic_signage()`, populated by `LiveOrchestrator` via `evacuation_signage_gateway` immediately after the guidance stage.

## 9. Voice/Signage consistency (Phase 21)

`dynamic_signage/consistency.py::detect_inconsistencies()` independently cross-checks each active `SignageInstruction` against the SAME `EvacuationGuidanceSnapshot` it (and voice) were built from — `SIGNAGE_GUIDANCE_MISMATCH`, `VOICE_SIGNAGE_EXIT_MISMATCH`, `STALE_SIGNAGE_REVISION`. Detected, never auto-corrected.

## 10. Provider capability (Phase 15)

`NO_PROVIDER` / `SIMULATION` implemented; `LIVE_HARDWARE` exists only as a capability label (`LiveOperatorActionGateway.signage_capability`), mirroring voice/building-control's own identical convention — no real provider implemented.

## 11. Answers to the milestone's own required questions

- **A.** Yes — `DynamicSignagePlanner.compute()` converts any graph-valid `EvacuationGuidancePlan` into per-sign `SignageInstruction`s.
- **B.** Yes — `dynamic_signage/direction.py`, verified against worked positions, not assumed.
- **C.** Yes — Door/Stair/Exit sequence, via `route_mapping.py` walking `ordered_navigation_steps`.
- **D.** No — the Planner never imports the Provider/Controller and never reads Crowd Intelligence/AI directly; it only ever reads the current `EvacuationGuidancePlan`.
- **E.** No — every cycle recomputes fresh from the current guidance snapshot; nothing but a change-detecting revision counter persists.
- **F.** Yes — `SignageConflict`, deterministic, never arbitrarily resolved.
- **G.** Yes — `SignageStatus.UNAVAILABLE` (no information) vs. route-status-derived `NO_SAFE_DIRECTION`/`DO_NOT_USE` (route genuinely invalid) are kept distinct.
- **H.** Yes — both read the same `EvacuationGuidanceSnapshot.recommended_exit_id`; `consistency.py` detects any drift.
- **I.** Yes — a changed `signage_revision` is a new, separately-keyed request requiring its own `approve()`.
- **J.** Yes — `SignageHistory`/`DynamicSignageController.all_instructions()` never rewrite a prior entry.
- **K.** No — mechanically guarded (see §7).
- **L.** No — `SimulationDynamicSignageProvider` is in-memory bookkeeping only; no `socket`/`serial`/`requests`/hardware import anywhere in `dynamic_signage/`.
