# SynEvac Builder V1 — Standalone Digital Twin Authoring Tool

Status: **IMPLEMENTED.** This document describes the shipped `builder/` application. It supersedes
`docs/architecture/synevac_builder_feasibility_investigation.md` as the authoritative description of what
exists — that document remains valuable as the *evidence trail* for why each reuse/new-code decision below
was made, but this document is the current architecture.

---

## 1. What Builder is

SynEvac Builder is a second, completely separate executable (`builder_main.py`) whose only responsibility is
authoring engineering-grade digital twin building models and saving them as `.syn` project files that open
directly inside SynEvac Studio (`main.py`) with zero modification. Builder contains no Simulation, AI,
Perception, Live Runtime, Recommendation/Guidance, or Campaign/Dataset code — not merely disabled at
runtime, but structurally absent from its import graph.

```
builder_main.py
    -> builder.app.BuilderApp
        -> builder.windows.builder_main_window.BuilderMainWindow
```

This mirrors `main.py -> core.app.SynEvacApp -> designer.windows.main_window.MainWindow` exactly, but the
two trees share no window class — `BuilderMainWindow` is a new, independently small file, not a stripped
Studio `MainWindow`.

---

## 2. Architecture

### 2.1 Shared components (reused unchanged, zero duplication)

These are the same modules Studio imports — literally the same files, not copies. A change to any of them
affects both applications identically, and both are protected by the same test suite.

| Package | Role in Builder |
|---|---|
| `models/` | `Building`, `Floor`, `Zone`, `Door`, `Exit`, `Staircase`, `Camera`, `SmokeDetector`, `HeatDetector`, `Speaker`, `Obstacle`, `Project` — the entire data model. |
| `navigation/` | `NavigationGraphGenerator`, `NavigationGraph`, `ValidationReport` — used both by the Validation Panel (graph-level checks) and the Navigation Preview Panel. |
| `serialization/` | `Serializer.save()` / `Serializer.load()` — the actual `.syn` file mechanism. Builder's "Save" *is* the export step; there is no separate export action. |
| `designer/scene/graphics_scene.py`, `graphics_view.py` | The entire drawing canvas, every tool's mouse-dispatch logic, floor-plan rendering. One small, guarded addition (§4) was made here — see below. |
| `designer/items/*.py` | Every `QGraphicsItem` subclass Builder places (`ZoneRectangle`, `DoorItem`, `ExitItem`, `StairItem`, `CameraItem`, `SmokeDetectorItem`, `HeatDetectorItem`, `SpeakerItem`, `ObstacleItem`). |
| `designer/validation.py` | `validate_building_authoring()` — Door/Exit/Stair zone-wiring completeness. |
| `designer/widgets/floor_list.py`, `project_tree.py`, `bottom_info_bar.py` | Reused directly as Builder dock widgets. |

### 2.2 Builder-only components (new)

| Module | Purpose |
|---|---|
| `builder/app.py` | `BuilderApp` — thin `QApplication` + `BuilderMainWindow` bootstrap, mirrors `core/app.py`. |
| `builder/windows/builder_main_window.py` | The main window: project management, floor wiring, selection routing, scale calibration flow, validation gating on save. |
| `builder/widgets/builder_toolbar.py` | A new, small action registry scoped to exactly the milestone's asset palette — not a trimmed copy of `designer/widgets/toolbar.py`. |
| `builder/widgets/builder_property_panel.py` | A new property editor covering Zone + the nine required asset types + Floor. Zero `camera_calibration` import (Studio's `PropertyPanel` is 7800+ lines and imports it at module level — a live-camera/Perception dependency Builder must not carry even dormant). |
| `builder/widgets/validation_panel.py` | Dedicated, continuously-refreshed panel merging three report sources (§5). |
| `builder/widgets/project_summary_panel.py` | Counts + total area + scale + validation status. |
| `builder/widgets/navigation_preview_panel.py` | Per-floor graph visualization for authoring verification (§6). |
| `builder/scale_calibration.py` | `ScaleCalibrationController` (two-click capture) + `compute_scale_pixels_per_meter()` (pure math). |
| `builder/validation_extras.py` | Three checks neither existing validator covers: overlapping zones, missing names, missing scale calibration (§5). |

### 2.3 Import boundary (verified, not assumed)

`grep -rEn "^import |^from " builder/` contains exactly: Python stdlib (`math`, `sys`, `ast`), `PyQt6.*`,
`models.*`, `navigation.*`, `serialization.serializer`, `designer.items.*`, `designer.scene.*`,
`designer.widgets.{floor_list,project_tree,bottom_info_bar}`, `designer.validation`, and `builder.*` itself.
No `simulation*`, `ai_*`, `*perception*`, `live_*`, `camera_calibration`, `camera_manager`,
`speaker_manager`, `evacuation_*`, `campaign*`, or `dataset*` import appears anywhere in the package.

---

## 3. Why a new small property panel and toolbar instead of reusing Studio's

`designer/widgets/property_panel.py` is 7866 lines and imports `camera_calibration` at module level (three
times) for its live-camera calibration section. `designer/widgets/toolbar.py` constructs actions for the
full ~25-item Studio asset palette, including Occupant and Simulation buttons. Reusing either wholesale
would mean either (a) Builder carries a Perception import even though it never exercises that code path —
violating the brief's "no Simulation/AI/Perception" requirement as a *structural* guarantee, or (b) editing
a Studio-owned, 7866-line shared file for a Builder-only reason, risking regressions in Studio's own UI.

`BuilderPropertyPanel` and `BuilderToolbar` are new files scoped to exactly the milestone's explicit
asset list (Zone, Door, Exit, Stair, Camera, SmokeDetector, HeatDetector, Speaker, Obstacle) plus Floor.
A `QAction` declaration and a `QFormLayout` field carry no logic to "duplicate" — this is a scoped surface
serving a narrower, explicitly-defined product, not a fork of Studio's.

---

## 4. Scale Calibration

**The only genuinely new feature** (confirmed absent repo-wide by the feasibility investigation).

### 4.1 Design

Every Zone/Door/Exit/Stair in this codebase is already authored in a fixed meter-based scene coordinate
system: `GraphicsScene.GRID_SIZE = 50` scene-pixels per meter, unconditionally. An imported floor plan
image, however, is just a raster of unknown real-world scale. Scale Calibration's job is **not** to
introduce a second coordinate system — it aligns the floor plan *backdrop image* to the grid that already
exists, so drawn geometry visually lines up with the imported drawing.

```
Floor.floor_plan_scale            -- pixels-per-meter of the RAW image (0.0 = not calibrated)
Floor.floor_plan_calibration_point_a / _b   -- the two clicked points, in raw image pixel space
Floor.floor_plan_calibration_distance_m     -- the real-world distance the user entered
```

`GraphicsScene._display_floor_plan()` (shared with Studio) gained one guarded addition:

```python
if self.current_floor.floor_plan_scale:
    self.floor_plan_item.setScale(self.GRID_SIZE / self.current_floor.floor_plan_scale)
```

When `floor_plan_scale == 0.0` (the default — every pre-existing `.syn` file, and every floor before its
first calibration) this is a no-op: the floor plan renders exactly as it always has. This is why the change
is safe to make in the shared file rather than forking it — verified by running the full pre-existing test
suite before and after (§8).

### 4.2 Calibration flow

1. User clicks **Calibrate Scale** (toolbar or Tools menu).
2. `ScaleCalibrationController` installs a Qt event filter on the canvas viewport (`builder/
   scale_calibration.py`) and captures exactly two left-clicks — **not** a new branch in
   `GraphicsScene.mousePressEvent`'s existing 2700+-line tool-dispatch chain. Consuming events at the
   viewport level means whatever drawing tool happens to be selected is irrelevant during calibration.
3. Each click's scene position is mapped through `floor_plan_item.mapFromScene()` — which correctly reverses
   *whatever scale is currently applied* (zero on a first calibration, the previous scale on a
   recalibration) — yielding the raw image's own pixel coordinates every time.
4. `BuilderMainWindow` prompts for the real-world distance (pre-filled with the previous value, if any).
5. `compute_scale_pixels_per_meter()` (pure function, no Qt dependency) computes pixels-per-meter; raises
   `ScaleCalibrationError` for coincident points or a non-positive distance.
6. `Floor.set_scale_calibration()` stores the result; `rebuild_scene()` re-applies it; the status bar's
   existing (previously unused) `Scale : Not Calibrated` label updates to `NN.N px = 1 m`.

Recalibration is just running the flow again — the previous distance is prefilled, and the two new points
overwrite the stored ones.

---

## 5. Validation Panel

Combines three sources, all speaking the same `navigation.validation.ValidationReport` /
`ValidationIssue` value type:

1. **`designer.validation.validate_building_authoring()`** (reused, unmodified) — Door/Exit/Stair
   zone-wiring completeness (ERROR), zone-assignment completeness for the rest of the palette (WARNING).
2. **`navigation.graph_builder.NavigationGraphGenerator().build(building).validate()`** (reused,
   unmodified) — structural graph validity, isolated zones, disconnected floors.
3. **`builder.validation_extras.validate_builder_extras()`** (new, additive) — the three checks the
   milestone brief names that neither existing validator covers:
   - `overlapping_zones` — pairwise axis-aligned bounding-box overlap between Zones on the same floor.
   - `*_missing_name` — any placed object (including Floor itself) with a blank name.
   - `floor_missing_scale_calibration` — a floor with an imported plan but no calibration.
   - `floor_missing_floor_plan` — **informational only** (a floor legitimately may never need one), the
     one place a third severity tier (`INFO`) is used, fulfilling the brief's "Errors / Warnings /
     Information" three-way display.

Refreshed after every authoring edit (`BuilderPropertyPanel.item_changed_callback`,
`FloorList.floors_changed_callback`), not only from an explicit button — continuous validation, per the
feasibility investigation's own UX recommendation (its Phase 7). **`_save_to()` refuses to write a `.syn`
file while any ERROR-severity issue exists** — "Save" is "Export", so this is the export gate the brief
requires.

`designer/validation.py` and `navigation/validation.py` are both left completely unmodified.

---

## 6. Navigation Preview

Per-floor, built from `navigation.graph_builder.NavigationGraphGenerator` (reused, unmodified) — the exact
same derived graph Studio's own tooling builds. Zone/AssemblyPoint nodes are drawn at their real
`center`/`position`; edges whose both ends are on the current floor are drawn as lines; an edge whose other
end is off-floor (an Exit leading to the single shared Outside node, or a Stair to a different floor) is
drawn as a short labelled stub rather than a line to nowhere — mirroring `Staircase.from_position`/
`to_position`'s own "no shared coordinate system between ends" design. Reachability from Outside is a plain
BFS over `Edge.traversable`, restated locally (the equivalent check inside `NavigationGraph.validate()` is
private to that method) — never a pathfinder, never a simulated occupant.

This is authoring verification only, not a preview of evacuation behavior — Builder never imports
`pathfinding`'s route-cost logic or any simulation package.

---

## 7. User Workflow

```
New Project
  -> Import Floor Plan (per floor)
  -> Calibrate Scale (per floor plan)
  -> Add Floor(s)                              [Floors dock, reused designer.widgets.floor_list.FloorList]
  -> Draw Structure (Zones, Doors, Exits, Stairs)
  -> Place Assets (Cameras, Smoke/Heat Detectors, Speakers, Obstacles)
  -> Validate (continuous -- Validation panel updates after every edit)
  -> Save (.syn -- this IS "Export .syn", gated on zero ERROR-severity issues)
  -> Open directly in SynEvac Studio
```

---

## 8. Disclosed gaps (matching Studio's own current state, not silently overreached)

- **Undo/Redo**: disabled with a tooltip ("not implemented yet"), identical to `designer/widgets/
  toolbar.py`'s own Undo/Redo. No undo/redo command stack exists anywhere in the reused code
  (`GraphicsScene`, `designer/items/*`); building one is a substantial new subsystem outside this
  milestone's scope (the feasibility investigation never budgeted it).
- **Multi-select**: `GraphicsScene.selected_item` is a single-item field throughout the reused scene/item
  code — Studio itself has no multi-select today. Builder matches that, rather than inventing a
  Builder-only selection model the shared `GraphicsScene` doesn't support.
- **Obstacle position**: intentionally not an editable Property Panel field, matching `models/obstacle.py`'s
  own documented scope ("placement/repositioning happens via the Tool and Move") — the same choice Studio
  already made.
- **Camera live-calibration/connection fields** (RTSP/IP/credentials): deliberately absent from
  `BuilderPropertyPanel` — configuring a *real* camera's live feed is a Studio/live-deployment concern, not
  digital-twin authoring (feasibility investigation, Phase 2). A `CameraItem`'s authored position/FOV/range
  is fully editable; its `connection` field is simply never populated from Builder, so no plaintext
  credential is ever produced by this application (`Serializer.save()` is called without a
  `credential_store`, since Builder never captures one).
- **Fire-safety/water-infrastructure asset palette** (Sprinkler, Hydrant, Hose Reel, Fire Water Tank/Pump/
  Jockey Pump/Inlet, Manual Call Point, Emergency Light, Dynamic Sign, Assembly Point): not included in V1.
  The feasibility investigation's own Phase 3 recommended including the full palette since it costs nothing
  dependency-wise; this implementation instead followed the milestone brief's explicit, narrower SHALL list
  literally. All of these model/item classes remain independent and reusable — adding their toolbar actions
  and property-panel sections to Builder in a future pass is pure assembly, not new architecture (see
  Roadmap).

---

## 9. Roadmap

- Extend `BuilderPropertyPanel`/`BuilderToolbar` to the full fire-safety/water-infrastructure/signage asset
  palette (Assembly Point, Sign, Manual Call Point, Emergency Light, Sprinkler, Fire Extinguisher, Fire
  Hydrant, Hose Reel, Fire Water Tank/Pump/Jockey Pump/Inlet) — every model/item class already exists and is
  confirmed independent; this is assembly work, not new subsystems.
  Camera visibility statistics (visible/partial/hidden zone counts) via `visibility.engine.VisibilityEngine`
  — confirmed dependency-clean (Layer 1.5), simply not wired into `BuilderPropertyPanel`'s Camera section
  yet.
- A real undo/redo command stack, shared between Builder and Studio (currently neither has one).
- Multi-select, if `GraphicsScene`'s selection model is ever generalized beyond a single `selected_item`.
- The Phase 5 "shared package" open question from the feasibility investigation (leave `designer.items`/
  `designer.scene` where they are vs. extract to a neutrally-named `twin_authoring/` package) remains
  unresolved and unaffected by this implementation — Builder imports `designer.items.*`/`designer.scene.*`
  directly, exactly as that investigation described as the zero-effort option.
