# Designer Product Boundary

This document records the outcome of the SynEvac Designer Simplification & Product-Boundary Cleanup milestone, built directly on `docs/architecture/designer_asset_connectivity_audit.md` (the authoritative connectivity trace this milestone treats as its evidence base — read that document first for the code-cited *why* behind every decision here).

## 1. SynEvac's core product purpose

SynEvac is **an AI-enabled, human-behavior-centric dynamic fire evacuation system.** Its research and product value is in perceiving where people actually are, reasoning about hazard and route safety, and producing/guiding an evacuation recommendation a human operator can act on. It is explicitly **not** a fire-protection BIM/asset-management platform — modeling every piece of fire-safety inventory a real building might contain is not a goal in itself, only a means when (and only when) that inventory genuinely changes evacuation reasoning.

## 2. The distinction this document turns on

Two questions look similar but are not the same, and this milestone's own audit found they had been getting conflated:

- **SUPPORTED BY DIGITAL TWIN** — does a `models/` class exist, can it be placed in the Designer, does it serialize, does it have a runtime manager and a Command Center status display? Every asset in this codebase, including all nine fire-safety/water-infrastructure assets, answers yes to this.
- **PARTICIPATES IN EVACUATION INTELLIGENCE** — does the asset's state genuinely change what `evacuation_recommendation`/`evacuation_guidance`/`advisory_system` output, traced through real code, not assumption? Only Zone, Exit, Door, Stair, Occupant, and Camera answer yes to this today (Smoke Detector partially, through Emergency Response; Heat Detector architecturally but not yet in practice — see §5).

Being "supported by the Digital Twin" is necessary but not sufficient for prominent placement in the default Designer authoring workflow. This milestone's whole job was to stop conflating the two.

## 3. Default authoring tools (main toolbar)

Grouped, separated, and ordered around the assets that actually participate in evacuation intelligence:

| Group | Tools |
|---|---|
| File | New, Open, Save |
| Edit | Undo, Redo *(both disabled — no command stack exists yet, unchanged by this milestone)* |
| Navigation | Select |
| **Building** | Zone, Door, Exit, Stair, Obstacle, Assembly Point |
| **Perception & Alarm** | Camera, Smoke Detector, Heat Detector, Manual Call Point |
| **Guidance & Output** | Speaker, Dynamic Sign |
| **Simulation** | Occupant, Simulation |
| View | Zoom +, Zoom -, Reset, Coverage |

Implemented in `designer/widgets/toolbar.py` (`MainToolbar`'s own layout section) — every `QAction` still constructed exactly as before; only which ones are added to the visible toolbar (`self.addAction(...)`) changed.

## 4. Advanced Fire-Safety Tools

The nine fire-safety/water-infrastructure assets — audited and confirmed to terminate at a Command Center status table with no influence on hazard, pathfinding, Decision Policy, Advisory, AI, or Evacuation Recommendation/Guidance — moved to an explicitly secondary surface: **Insert menu → Advanced Fire-Safety Tools submenu**, containing, in order: Emergency Light, Sprinkler, Fire Extinguisher, Fire Hydrant, Hose Reel, Fire Water Tank, Fire Pump, Jockey Pump, Fire Service Inlet.

This reuses the exact same `QAction` objects `MainToolbar` already constructs and `MainWindow.connect_toolbar()` already wires to `change_tool()` — a `QAction` is not tied to one widget, so adding it to a menu instead of a toolbar changes nothing about how it is triggered, what it does, or how the resulting asset is placed, edited, serialized, or displayed. Nothing was deleted: every model, manager (`EmergencyLightManager`, `FireSafetyAssetManager`, `FireWaterInfrastructureManager`), snapshot type, and Command Center panel is completely unchanged.

Why "advanced," not "deleted": every one of these nine assets is well-built, well-tested, and honestly self-documented (each model's own docstring already disclaims exactly the physical effects it doesn't model — "operational state != hydraulic performance," "isolated from route safety," etc.). The finding was about *prominence*, not correctness. An advanced user who genuinely wants to model a building's full fire-suppression inventory can still do so completely; a new user learning the tool is no longer presented with nine fire-protection-inventory buttons before ever seeing Camera or Occupant.

## 5. Legacy compatibility policy

**Generic "Detector" (`models/detector.py`)** is no longer offered anywhere in the UI for new authoring (not the main toolbar, not the Advanced menu, not any menu at all) — but:

- The model, `models/detector.py`, is completely unchanged.
- `models/detector_migration.py::adapt_legacy_detector()` — the function that transparently converts a legacy `Detector(detector_type="Smoke"/"Heat")` into the canonical `SmokeDetector`/`HeatDetector` shape, same id preserved — is completely unchanged.
- An old `.syn` project containing any legacy `Detector` (Smoke, Heat, **or** Flame/Gas) still loads, still renders on the canvas (`GraphicsScene.rebuild_scene()`'s own `for detector_obj in self.current_floor.detectors:` loop is untouched), still opens correctly in the Property Panel (`show_detector()` untouched), and still saves/reloads identically.
- `Floor.detectors`, `Floor.add_detector()`/`remove_detector()`, and every serialization path are completely unchanged.

**Why generic Detector is hidden:** Smoke/Heat-typed legacy Detectors are transparently superseded by the dedicated Smoke Detector/Heat Detector tools (identical real behavior, clearer authoring intent). Flame/Gas-typed legacy Detectors have **zero** real behavior anywhere in this codebase — `adapt_legacy_detector()` returns `None` for both, so neither is ever registered with `SensorManager`, reaches perception/FACP/hazard, or appears in any intelligence engine or Command Center table. Offering a tool that authors an asset with no behavior at all serves no one; the model stays for the projects that already have one.

**Why Elevator remains hidden:** Elevator's toolbar action was already constructed disabled, with an explanatory tooltip, before this milestone — no drawing tool exists in `GraphicsScene.set_tool()` to back it, it has no representation in the navigation graph, and it has zero simulation/decision connectivity anywhere in the codebase (confirmed by exhaustive audit, not assumption). This milestone changed nothing about it: the model (`models/elevator.py`) and `Floor.elevators` serialization remain fully intact for any hand-authored or test project that already places one; no elevator routing was implemented, and none was attempted.

**Why fire-water infrastructure is advanced rather than deleted:** see §4 — every asset is correct, tested code answering a real question ("does this piece of fire-protection inventory exist and is it currently usable"), just not a question the evacuation-intelligence pipeline currently needs answered to produce a recommendation. Deleting tested code to solve a UI-prominence problem would be a disproportionate response; hiding it behind an explicitly-labeled secondary menu solves the same problem without discarding real engineering work.

## 6. Obstacle — a deliberate exception

The audit classified Obstacle as currently lacking meaningful runtime decision connectivity (its `traversal_cost` field is unused outside its own model; it affects only camera-visibility geometry and a dataset-generation column). Unlike the nine fire-safety assets, Obstacle is **kept in the Building group** rather than hidden — it conceptually belongs to evacuation geometry (a real obstruction in a corridor is exactly the kind of thing that should eventually affect pathfinding), and `navigation/cost.py` already documents a named, unimplemented `CostModel` extension point for exactly this. This is a deliberate exception, not an oversight, and is recorded here as a candidate future-connectivity target (see `docs/architecture/designer_asset_connectivity_audit.md` §13) — not something to quietly wire up as part of this cleanup.

## 7. What this milestone explicitly did not do

Per its own scope boundary, this milestone did not: fix Manual Call Point's Emergency Response gap, wire Dynamic Sign's operator-approval UI into Command Center, add any Obstacle→pathfinding connectivity, add any new fire-safety system, add any new simulation physics, redesign any architecture, delete any tested asset model, or break any existing `.syn` project. Every one of those remains a named, separate candidate milestone (`docs/architecture/designer_asset_connectivity_audit.md` §13/Final Answer 18).

## 8. What this milestone additionally fixed

While proving backward compatibility (Phase 8's own explicit requirement — old projects must still *render*), a genuine, pre-existing bug was found and fixed: `GraphicsScene.rebuild_scene()` only ever reconstructed `EmergencyLight` graphics items among the fire-safety assets on project load/floor switch — Sprinkler, Fire Extinguisher, Fire Hydrant, Hose Reel, Fire Water Tank, Fire Pump, Jockey Pump, and Fire Service Inlet were never reconstructed at all, meaning a loaded project containing any of them would silently show an empty canvas for that asset despite the model itself being completely intact (still editable via Property Panel if you knew to look, still present in `Floor.sprinklers` etc.). Fixed additively, following the exact same per-item pattern every other asset type already used — no new behavior, no new asset type, only restoring rendering these assets should always have had.

## 9. Test coverage

`tests/test_designer_simplification.py` — 30 tests covering: the exact default toolbar contents/order, the Advanced Fire-Safety Tools submenu contents and functionality, generic Detector's removal from all new-authoring surfaces alongside full legacy round-trip/render/edit proof, backward-compatibility round-trips for all nine fire-safety assets (load/render/edit/save/reload), a full core-workflow end-to-end authoring test (2 zones, 1 door, 2 exits, 1 stair across 2 floors, 1 camera, 1 smoke detector, 1 heat detector, 1 MCP, 1 speaker, 1 dynamic sign, 2 occupants — all through the real toolbar + `GraphicsScene.mousePressEvent`), an advanced-authoring end-to-end test (5 fire-safety assets placed via the Advanced menu's own actions), and architecture guards confirming no fire-safety model/manager/serialization support was removed and that `decision_policy`/`ai_features`/`evacuation_recommendation`/`evacuation_guidance`/`simulator`/`live_camera_pipeline`/`camera_calibration`/`human_detection` all still import cleanly.
