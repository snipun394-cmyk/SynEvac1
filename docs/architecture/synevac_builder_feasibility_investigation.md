# SynEvac Builder Feasibility & Architecture Investigation

Status: **INVESTIGATION ONLY.** No code changed. No Builder application created. This document is the
sole deliverable. Every claim below is cited to an actual import statement or line number found by direct
`Grep`/`Read` of the repository — none is assumed. Full per-file evidence tables are preserved in this
document rather than only summarized, since "do not assume, verify from the actual code" is this
milestone's own explicit instruction.

---

## Phase 1 — Current Designer Audit: dependency map

`designer/` contains 67 `.py` files (7 are empty stub/no-op files: `designer/__init__.py`, `designer/
controllers/__init__.py`, `designer/items/__init__.py`, `designer/tools/pan_tool.py`, `designer/tools/
polygon_tool.py`, `designer/tools/select_tools.py`, plus `config/app_config.py` outside the package).
`designer/controllers/selection_manager.py` has zero imports at all.

### The layered dependency structure, as it actually exists

```
Layer 0 — pure data model (zero framework/simulation/AI dependency)
  models/            35 files, verified: EVERY file imports only sibling models.* + stdlib (math/typing/
                     dataclasses). Zero simulation/AI/perception import anywhere in the package.
  navigation/         9 files, verified: imports only models.connectable_space + visibility.geometry
                     (math-only). NavigationGraphGenerator lives in navigation/graph_builder.py.
  scenario/           self-contained simulation-SCENARIO data model (fire/occupant/firefighter/event) --
                     structurally parallel to models/ but a DIFFERENT object graph (Scenario, not
                     Building). Zero import of models/ or any heavy package. Used by the simulation
                     stack, NOT by Designer's Building/Project editing at all.

Layer 1 — persistence (depends only on Layer 0)
  serialization/      4 files. Serializer.save()/.load() operate on models.Project only. Zero
                     scenario/simulation import. THIS is the confirmed real mechanism behind SynEvac's
                     .syn file format (see Phase 2, Serialization/Project Save-Load).
  scenario_storage/   4 files. Persists scenario.Scenario objects (not models.Project) for Campaign
                     Studio. Built on serialization/json_writer/json_reader (not Serializer). A
                     DIFFERENT persistence mechanism than .syn project files -- confirmed NOT used by
                     designer/windows/main_window.py's save_project()/open_project() at all.
  designer/validation.py  1 file, 1 import: navigation.validation.ValidationReport. Designer's own
                     authoring-completeness checker (Door/Exit/Stair zone wiring, dangling references).
                     Zero relation to the top-level validation/ package (see below) despite the name
                     collision.
  credential_store/   independent, stdlib/JSON-only local credential storage utility used by Serializer.

Layer 1.5 — light geometry/editing-support packages, also independent
  sandbox/            2 files (occupant.py, manager.py). Design-time occupant PLACEHOLDER model (for
                     Designer's own occupant-generation preview), not a simulation engine. Imports only
                     navigation.* + pathfinding.*.
  pathfinding/        A*-style route/heuristic utility. No simulation/AI import.
  visibility/          4 files. Pure line-of-sight/coverage geometry (math/dataclasses/typing only) for
                     camera-placement preview. Zero heavy import. MUST NOT be confused with the
                     similarly-named but entirely separate, heavy camera_coverage/ package (see below).

Layer 2 — the Designer GUI itself (depends on Layers 0-1.5 + PyQt6)
  designer/items/*.py         35 files, 34 fully independent, 1 (occupant_item.py) uses navigation+sandbox
                              (still independent per above).
  designer/scene/*.py         graphics_scene.py, graphics_view.py, floor_plan_item.py -- independent
                              (models + items + sandbox + deferred visibility only).
  designer/tools/*.py         3 files, ALL EMPTY (0 bytes) -- dead/stub code today (see note below).
  designer/controllers/*.py   independent, trivial Qt-only state holders.
  designer/widgets/*.py       17 files: 9 fully independent (toolbar, status_bar, floor_list,
                              project_tree, bottom_info_bar, fire_water_system_list, simulation_panel
                              [name only -- zero heavy imports despite the name], occupant_generation_
                              dialog, ...), 8 heavy-coupled (see Layer 3 below).
  designer/validation.py      independent (Layer 1, listed here for completeness of the package).

Layer 3 — heavy-coupled Designer files (the actual, narrow set that pulls in Simulation/AI/Perception)
  designer/building_state_debug_runner.py   Simulation (hazard.*, facp.*) + Perception (occupancy.*,
                                             multi_camera_fusion, perception.models) + building_state
                                             (itself a large fan-out -- see below)
  designer/perception_debug_runner.py       Simulation (hazard.*) + Perception (occupancy.*, perception.
                                             fusion.*, perception.models.*, perception.providers.*)
  designer/live_runtime_controller.py       Runtime (live_runtime_launcher.session.LiveRuntimeSession)
  designer/widgets/building_state_debug_panel.py  wraps building_state_debug_runner (MIXED, transitive)
  designer/widgets/perception_debug_panel.py      wraps perception_debug_runner (Perception)
  designer/widgets/camera_manager_panel.py        Perception (camera_calibration.*, camera_manager.*)
  designer/widgets/camera_validation_panel.py     Perception (camera_validation.validator)
  designer/widgets/speaker_manager_panel.py       Recommendation/guidance-adjacent (speaker_manager.manager)
  designer/widgets/live_runtime_panel.py          Runtime (live_runtime_launcher.modes.ApplicationMode)
  designer/widgets/property_panel.py              Perception (camera_calibration.* -- 3 imports; deferred
                                                   visibility.engine is light, unrelated)
  designer/campaign/campaign_worker.py            Simulation + AI + Dataset + Recommendation (13 distinct
                                                   heavy modules: scenario_definition, scenario_generator,
                                                   scenario_pipeline, scenario_validator, scenario_runner,
                                                   behaviour_profile_resolver, ai_decision, simulation_
                                                   runtime, human_decision_engine, dataset_builder, ground_
                                                   truth, decision_policy)
  designer/campaign/campaign_window.py            scenario_definition
  designer/campaign/{__init__,campaign_controller}.py   transitively heavy (re-export/import campaign_worker)

Layer 4 — the aggregation root
  designer/windows/main_window.py   Imports NO heavy package directly, but unconditionally instantiates
                                     all 8 Layer-3 panel/controller classes in __init__ (verified: line
                                     159 PerceptionDebugPanel(), 173 BuildingStateDebugPanel(), 184
                                     CameraManagerPanel(...), 197 SpeakerManagerPanel(), 209
                                     CameraValidationPanel(), 240-242 LiveRuntimePanel()/
                                     LiveRuntimeController(...) -- all eager, all docked via
                                     addDockWidget()). CampaignController/CampaignWindow are the one
                                     exception -- lazily constructed only inside a menu-action handler
                                     (line ~1599), not in __init__.
```

### Two important name-collision traps, confirmed by direct inspection (not assumed)

1. **`designer/validation.py` vs. top-level `validation/`**: completely unrelated packages that happen to
   share a name fragment. `designer/validation.py` is Designer's own authoring-completeness checker
   (1 import: `navigation.validation.ValidationReport`). The top-level `validation/` package
   (`phase1_campaign_benchmark.py` through `phase6_profiling.py`) is a heavy AI/RL/simulation research
   test-harness importing `ai_decision`, `ai_training`, `rl_training`, `simulation_runtime`, `scenario_
   runner`, `ground_truth`, `dataset_builder`, `decision_policy`, `sklearn`, `stable_baselines3`. **A grep
   for any `designer/` file importing the top-level `validation/` package returns zero hits** — no
   accidental coupling exists today, but the name collision is a real risk for future contributors and
   should be renamed if this investigation's recommendations are ever acted on (Phase 4).
2. **`visibility/` vs. `camera_coverage/`**: `designer/scene/graphics_scene.py` and `designer/widgets/
   property_panel.py` both use `visibility.coverage`/`visibility.engine` (pure geometry, light) for
   camera-placement preview — a textually similar but functionally and dependency-wise entirely different
   package from `camera_coverage/` (a heavy package reached only via `building_state.py`, part of the
   live perception/crowd-intelligence stack). No `designer/` file imports `camera_coverage/` directly.

### `scenario_storage/` is not what `.syn` files use — traced precisely

`designer/windows/main_window.py`'s actual Save/Open Project implementation (confirmed by reading the
functions directly):
- `save_project()` (main_window.py:1354-1378): `Serializer.save(self.canvas.scene_obj.get_project(),
  filename, credential_store=self._credential_store)`
- `open_project()` (main_window.py:1420-1435): `Serializer.load(filename, credential_store=self.
  _credential_store)`

`serialization.Serializer` — operating purely on `models.Project`/`models.Building` — is the confirmed,
sole `.syn` file mechanism. `scenario_storage/` (used only by `designer/campaign/campaign_worker.py` for
Campaign Studio's own scenario-acceptance bookkeeping) plays no role in `.syn` project files at all,
despite the superficially similar name.

### The "tool" architecture is not what the milestone brief's phrasing implies

`designer/tools/pan_tool.py`, `polygon_tool.py`, `select_tools.py` are **empty files (0 bytes) today** —
dead/stub code. There are no `DoorTool`/`ExitTool`/`StairTool` classes anywhere in the repository. The
actual drawing-mode logic is a single `self.current_tool` string switched by toolbar `QAction`s
(`designer/widgets/toolbar.py`, itself fully independent — pure `QAction` definitions, e.g. `door_action`,
`exit_action`, `stair_action`, `camera_action`, `obstacle_action`) and dispatched through one large
`if self.current_tool == "...":` chain inside `graphics_scene.py`'s `mousePressEvent` (confirmed: 20+
`if self.current_tool == "<name>"` branches spanning lines 368-1931 of that one file). "Door Tool" /
"Exit Tool" / "Stair Tool" / "Camera Tool" / "Obstacle Tool", as named in this milestone's brief, are
therefore not separable files today — they are one monolithic (but fully independent — Layer 2, above)
`graphics_scene.py`, plus the corresponding `designer/items/*.py` class each mode creates. This is a real,
disclosed architectural observation (Phase 6 risk), not a blocker to reuse.

---

## Phase 2 — Reusability Analysis

| Subsystem | Files / location | Verdict | Why |
|---|---|---|---|
| **Main Window** | `designer/windows/main_window.py` | **Studio-exclusive as written; the pattern is reusable** | Imports no heavy package directly but unconditionally constructs all 8 Layer-3 panels in `__init__` (Phase 1). A Builder cannot reuse this file unmodified — but every non-heavy piece it wires (menus, project tree, toolbar, canvas, status bar) is independent, so a parallel, much smaller `BuilderMainWindow` reusing those same pieces is the correct pattern, not a fork of this file. |
| **Graphics Scene** | `designer/scene/graphics_scene.py` | **Reusable unchanged** | Confirmed independent (models + items + sandbox + deferred light `visibility`). Contains ALL drawing-mode logic for every tool, including fire-safety-asset tools (sprinkler, hydrant, etc.) Builder likely won't need — see Phase 3 on whether to trim these, which is a scope decision, not a dependency blocker. |
| **Graphics Items** | `designer/items/*.py` (35 files) | **Reusable unchanged** | 34 of 35 have zero non-Qt imports; the 1 exception (`occupant_item.py`) only adds `navigation`+`sandbox`, both confirmed light. |
| **Building Models** | `models/` (35 files) | **Reusable unchanged — the cleanest layer in the whole audit** | Exhaustive repo-wide grep across every file for every simulation/AI/perception package name returned zero matches. This is the correct, and already-proven-clean, foundation for both apps. |
| **Zone Editor** | `zone_item.py`, `zone_rectangle.py`, `controllers/zone_controller.py`, plus the `"zone"` branch in `graphics_scene.py` | **Reusable unchanged** | All independent; `zone_controller.py` is a 2-import (Qt-only) state holder. |
| **Door / Exit / Stair / Camera / Obstacle "Tools"** | toolbar actions + `graphics_scene.py` mode branches + corresponding `items/*.py` | **Reusable unchanged, but not as separable units** | See Phase 1's "tool architecture" finding — these aren't standalone files to extract, they're inseparable parts of the one large (but independent) `graphics_scene.py`. Reuse means reusing that whole file, not cherry-picking. |
| **Validation** | `designer/validation.py` (authoring-completeness) | **Reusable unchanged** | 1 import, `navigation.validation` only. Distinct from and unrelated to the heavy top-level `validation/` package (Phase 1). |
| **Serialization** | `serialization/` (4 files) | **Reusable unchanged — this IS the `.syn` format** | Confirmed the actual mechanism `main_window.py`'s save/open calls. Zero scenario/simulation coupling. |
| **Project Save/Load** | `main_window.py`'s `save_project()`/`open_project()` methods, calling `Serializer` | **Reusable with trivial extraction** | The methods themselves are ~2 lines of glue around `Serializer` + a `QFileDialog` — not a subsystem with its own coupling, just needs to be copied (or better, factored into a small shared helper — see Phase 5) into whatever hosts Builder's own main window. |
| **Camera / Detector / Sensor "manager" panels** (not explicitly asked, but named in Phase 1's toolbar list) | `camera_manager_panel.py`, `camera_validation_panel.py`, `property_panel.py`'s camera-calibration section | **Studio-exclusive** | These configure LIVE camera calibration/connection state (`camera_calibration`, `camera_manager`) — a live-deployment concern, not a digital-twin-authoring concern. Placing a `CameraItem` on the canvas (authoring) is independent; calibrating/connecting to a REAL camera feed (live operation) is not, and correctly should not be in Builder. |
| **Debug panels** (Building State Debug, Perception Debug) | `building_state_debug_*.py`, `perception_debug_*.py` | **Studio-exclusive, unambiguously** | Exist purely to visualize simulation/perception internals — meaningless without a running simulation or live camera feed. No authoring purpose whatsoever. |
| **Campaign Studio** | `designer/campaign/*.py` | **Studio-exclusive, unambiguously** | The single most heavily-coupled subtree found (13 distinct heavy modules in `campaign_worker.py` alone: scenario generation, AI decision engine, simulation runtime, dataset export, ground truth analysis, decision policy). |
| **Speaker Manager panel** | `speaker_manager_panel.py` | **Studio-exclusive** | Configures live PA/voice-evacuation routing (`speaker_manager.manager`) — an operational-deployment concern, not authoring. (Placing a `SpeakerItem` on the canvas is independent and belongs in Builder; managing it live does not.) |
| **Live Runtime panel/controller** | `live_runtime_panel.py`, `live_runtime_controller.py` | **Studio-exclusive, unambiguously** | Directly imports `live_runtime_launcher` — the live-deployment entry point. |

---

## Phase 3 — Builder Scope

Verifying the brief's proposed checklist against the repository, item by item:

| Proposed feature | Status in repo | Verdict for Builder |
|---|---|---|
| Import floor plan | **Exists** — `main_window.py:1339` `import_floor_plan()` → `canvas.load_floor_plan(filename)`, backed by `designer/scene/floor_plan_item.py` (independent) | ✓ Include — direct reuse |
| Scale calibration | **Does NOT exist anywhere in the repository.** Repo-wide search for `scale_calibrat`/`floor_plan_scale`/`real_world_scale`/`set_scale` found zero hits. (Not to be confused with `camera_calibration/`, which is an unrelated, heavy, live-camera concept.) | ✓ Include, but flag as **genuinely new code** — not a reuse item, a real gap this investigation found. Likely the single largest true "new development" item in Builder's entire scope. |
| Draw zones / Doors / Exits / Stairs / Cameras / Detectors / Obstacles | All exist and are independent (Phase 1/2) | ✓ Include — direct reuse of `graphics_scene.py` + `items/*.py` |
| Save/Open project | Exists, independent (`Serializer`) | ✓ Include — direct reuse |
| Validation | Exists, independent (`designer/validation.py`) | ✓ Include — direct reuse |
| Export `.syn` | This is not a separate action — **`Serializer.save()` already produces the exact `.syn` file Studio reads.** There is no separate "export" format; save and export are the same operation, because Builder and Studio would share the identical `models.Project`/`serialization.Serializer` pair. | ✓ Include, but reframe: "Save" already IS "Export .syn" — no extra work needed, and no risk of a format drift between the two apps as long as `serialization/` and `models/` stay shared (Phase 5). |

**Should anything else be included?**
- **Sensor/fire-safety asset tools already exist and are independent** (Smoke/Heat Detector, Speaker, Sign, Manual Call Point, Emergency Light, Sprinkler, Fire Extinguisher, Fire Hydrant, Hose Reel, Fire Water Tank/Pump/Jockey Pump/Service Inlet, Assembly Point) — all confirmed zero-heavy-import item classes (Phase 1 §4 of the designer-package audit). If Builder's purpose is genuinely "engineering-grade digital twins," these are load-bearing for that grade of twin (a fire-safety engineer building a real twin needs sprinklers/hydrants placed, not just Zones/Doors/Exits) and cost nothing extra to include, since they're already proven independent. Recommend including the full asset palette, not just the brief's shorter example list.
- **`designer/validation.py`'s authoring-completeness checks** should run automatically before every save, not just be available — this is cheap (already independent, already exists) and directly serves Builder's "engineering-grade" goal.

**Should anything be removed?**
- **Occupant placement** (`OccupantItem`, `sandbox.occupant`) is independent per the dependency map, but its *purpose* — pre-placing occupants for a simulation to consume — is a Studio/simulation concern, not a digital-twin-authoring concern. It's a genuine judgment call: technically includable at zero dependency cost, but arguably out of scope for a tool whose stated purpose is "ONLY digital twin creation." Recommend **excluding it from Builder's default toolset**, not because it's coupled (it isn't), but because it's scenario/simulation-adjacent by PURPOSE, consistent with the brief's own "NO Simulation" instruction interpreted by intent rather than by import graph alone.
- **Camera calibration/connection features** (`camera_manager_panel.py`, `camera_validation_panel.py`, `property_panel.py`'s calibration section) should be excluded — these configure a LIVE camera's real-world mapping, not the digital twin's authored `CameraItem` placement (which stays, per Phase 2).

---

## Phase 4 — Explicit Exclusions

**Must never be in Builder** (verified via Phase 1's exhaustive import audit, not assumed):

- Simulation: `simulator`, `simulation_runtime`, `simulation_interactive`, `scenario_runner`, `scenario_generator`, `scenario_pipeline`, `scenario_event_executor`, `ai_decision`, `hazard`, `hazard_evolution`, `fire_growth`, `smoke_propagation`, `tenability`, `behavior_library`, `human_decision_engine`, `behaviour_profile_resolver`
- AI / prediction: `ai_registry`, `ai_features`, `ai_training`, `ai_inference`, `ai_explainability`, `predictive_dataset`, `predictive_model`, `prediction_evaluation`, `model_benchmark`, `rl`, `rl_training`
- Perception / live: `human_detection`, `human_evidence`, `live_occupants`, `live_perception`, `live_camera_pipeline`, `crowd_intelligence`, `camera_calibration`, `camera_coverage`, `camera_validation`, `camera_manager`, `sensor_manager`, `multi_camera_fusion`, `cross_camera_identity`, `tracking`, `sensor_fusion`, `virtual_camera`, `occupancy`, `trajectory_intelligence`, `stair_flow`, `building_state`, `facp`
- Runtime/live-system: `live_runtime`, `live_runtime_launcher`, `live_system`, `command_center`, `dashboard`
- Recommendation/guidance/response: `evacuation_recommendation`, `evacuation_guidance`, `advisory_system`, `emergency_response`, `decision_policy`, `voice_evacuation`, `dynamic_signage`, `sign_manager`, `speaker_manager`
- Dataset/campaign generation: `scenario_definition`, `dataset_builder`, `training_dataset`, `campaign_analytics`, `ground_truth`, `scenario_storage`, `scenario_validator`, `scenario` (the simulation-Scenario data model itself — distinct from `scenario_storage`, but equally out of scope for a Building/Project-only app)
- The top-level `validation/` package (Phase 1's name-collision trap) — never `designer/validation.py`, which is fine.

**Hidden-dependency check — the actual finding, not a hypothetical**: Phase 1's exhaustive cross-check grep
(searching all of `designer/` for `from ai_`, `from simulat`, `from live_`, `from human_detection`, `from
crowd_intelligence`, `from evacuation_`, `from predictive_`, `from rl`, `from prediction_evaluation`, `from
model_benchmark`) found **zero hits outside the already-identified Layer-3 files** (`building_state_debug_
runner.py`, `perception_debug_runner.py`, `live_runtime_controller.py`, `campaign_worker.py`, `campaign_
window.py`, and the widgets/panels wrapping them). **No hidden/accidental dependency exists in any of the
Layer 0-2 files this investigation classified as reusable.** The exclusion boundary is real and already
enforced by the current import graph — Builder does not need to defensively strip anything from `models/`,
`navigation/`, `serialization/`, `designer/items/`, `designer/scene/`, or `designer/validation.py`; they
are already clean.

**One disclosed residual risk, not a current violation**: `designer/scene/graphics_scene.py` is a single
2700+-line file handling every tool's mouse-event logic in one class (Phase 1). It is independent TODAY,
but its size and monolithic structure make it the single highest-risk file for a FUTURE accidental
coupling — e.g., if a future Studio-only feature (say, a live-occupant overlay) were added by extending
this same file rather than a separate one, Builder would inherit it silently. This is a process risk
(code-review discipline), not an architecture defect today, and worth naming explicitly (Phase 6).

---

## Phase 5 — Shared Library Strategy

**Can Builder and Studio safely share `models/`, `serialization/`, Graphics Items, `designer/validation.py`,
geometry (`navigation/`, `visibility/`, `utils/geometry.py`), and Navigation Objects without duplicating
code? Yes — confirmed, not assumed**, because every one of these is already:
1. Free of any Studio-only (simulation/AI/perception) dependency (Phase 1).
2. Not itself dependent on `designer/` (the GUI layer) — `models/`, `navigation/`, `serialization/`,
   `sandbox/`, `pathfinding/`, `visibility/` are all pure logic/data packages with no PyQt6 import at all
   (confirmed: none of the import lists gathered in Phase 1 include `PyQt6` for any file in these
   packages), so they are trivially importable from two separate GUI applications without either one
   depending on the other's window/widget code.

**Recommended architecture**: no new "shared library" package needs to be CREATED — it already exists, as
the exact set of packages just named. The correct action (a future implementation milestone, not this one)
is:
- **Builder imports from**: `models/`, `navigation/`, `serialization/`, `sandbox/`, `pathfinding/`,
  `visibility/`, `credential_store/`, `designer/items/*.py`, `designer/scene/*.py`, `designer/
  controllers/*.py`, `designer/validation.py`, `designer/widgets/{toolbar,status_bar,floor_list,
  project_tree,bottom_info_bar}.py` — all already independent, all already exist, zero duplication needed.
- **Studio continues to own**: `designer/windows/main_window.py` (as today, or a renamed/refactored
  Studio-specific main window built the same way), all Layer-3 panels, `designer/campaign/`, and every
  heavy package in Phase 4's exclusion list.
- **One genuine open design question, not resolved by this investigation**: whether `designer/items/`,
  `designer/scene/`, etc. should physically MOVE into a new top-level package (e.g. `twin_authoring/`) that
  both `designer/` (Studio) and a new `builder/` package import from, versus Builder importing directly
  from `designer.items`/`designer.scene` as they sit today. The latter is zero-effort but leaves a
  slightly confusing package name (`designer` implying "the full Studio" while Builder also depends on
  parts of it); the former is cleaner long-term but is itself a real refactor (moving ~45 files, updating
  every import site in both apps) with a non-zero regression risk on Studio, which the current 5138-test
  suite provides only partial protection for regarding pure import-path changes across such a wide surface. This is a genuine trade-off, not a
  decided recommendation — flagged for Phase 6/8 to weigh in on, not resolved here.

---

## Phase 6 — Development Effort

**Percentage of Builder already exists** (measured against the verified-in-scope feature list from Phase
3): the overwhelming majority. Every drawing tool, every asset item, save/open, and authoring validation
already exist as independent, working code (Phase 1/2). Rough breakdown by the Phase 3 checklist:
- Import floor plan: 100% exists.
- Draw zones/doors/exits/stairs/cameras/detectors/obstacles/full asset palette: 100% exists.
- Save/Open/.syn export: 100% exists (same operation, as established in Phase 3).
- Validation: 100% exists.
- **Scale calibration: 0% exists — confirmed genuinely absent repo-wide (Phase 3).**

**Percentage requiring new code**: small in absolute file count, but concentrated almost entirely in two
places: (1) scale calibration (a real, non-trivial UI+math feature: user clicks two points on the imported
floor-plan image, enters a real-world distance, the system derives and stores a pixels-per-meter or
meters-per-pixel factor against the `Building`/`Floor` model — this does not exist on the model side
either, so it may also need a small, additive `models/` field, which would need to be designed carefully
to stay backward-compatible with Studio's own `.syn` reading), and (2) a new, deliberately small
`BuilderMainWindow` (or equivalent) that wires together the ALREADY-independent pieces (toolbar, canvas,
project tree, status bar, Save/Open, Validation) without any of the 8 heavy panels — this is assembly work
reusing proven pieces, not new logic, but it is real new code (a new file, not a modification of
`main_window.py`, per Phase 2's own recommendation against forking that file).

**Biggest technical risks**:
1. **Scale calibration's model-schema question** — if it needs even one new field on `Floor`/`Building`,
   that field must round-trip through `Serializer`/`.syn` in a way Studio's own (currently unaware of
   scale) code doesn't choke on. Needs explicit design, not assumed to be "just a UI feature."
2. **`graphics_scene.py`'s monolithic size** (Phase 4's disclosed residual risk) — not a blocker, but the
   single file most likely to accumulate future accidental Studio-only coupling if not actively guarded.
3. **The `models/project.py` field-set is currently minimal** (`id, name, author, version, created_at,
   modified_at, building` — confirmed via direct read) — no version-compatibility or migration mechanism
   was found; if Builder and Studio are ever released on different cadences and one adds a model field
   the other doesn't understand yet, `.syn` compatibility could silently break. Not a problem today (single
   app, single version), but a real risk the MOMENT two independently-versioned applications both write
   `.syn` files — worth an explicit compatibility-testing discipline in any future implementation milestone.

**Biggest architectural risks**:
1. **The shared-package question from Phase 5** (leave-in-place vs. extract-to-`twin_authoring/`) is
   unresolved and has real trade-offs either way — deferring this decision without deciding is itself a
   risk if implementation starts before it's settled, since the two paths have very different effort
   profiles and file layouts.
2. **No existing "headless"/GUI-independent test seam** was found for `designer/items/`/`designer/scene/`
   — they are PyQt6 `QGraphicsItem`/`QGraphicsScene` subclasses, meaning any Builder-specific testing still
   requires a Qt application context, same as Studio's own tests already do (not a NEW risk Builder
   introduces, but not a risk Builder resolves either).

**Expected implementation effort** (qualitative, since no implementation was performed this milestone):
**Low-to-moderate for the "assembly" 90% of the scope** (reusing already-independent, already-tested code
behind a new, small window/wiring file) — **moderate, with real design work, for the scale-calibration 10%**
(the one genuinely new feature, touching UI, math, and potentially the shared model schema). Overall,
substantially cheaper than a from-scratch application, and cheaper than the brief's own framing might
suggest is needed, precisely because Phase 1's audit found the exclusion boundary already clean rather
than needing to be carved out.

---

## Phase 7 — User Experience

**Critique of the brief's proposed workflow** (Create Project → Import Floor Plan → Set Scale → Draw
Building → Validate → Save → Open in Studio):

- **Structurally sound and matches what already exists**, with one necessary reordering: **Validate should
  not be a single step immediately before Save — it should run continuously/automatically**, since
  `designer/validation.py`'s checks (Door/Exit/Stair zone wiring, dangling references) are cheap and exist
  today; gating them to one late step invites a user to draw an entire building then discover 20 wiring
  errors at once, when incremental feedback (e.g., a live-updating validation panel, or at minimum
  re-validate after every zone/asset placement) is both cheap (feature already exists, just needs to be
  invoked more often) and a materially better experience.
- **"Set Scale" must come before "Draw Building", not after import alone** — the brief's own ordering
  already gets this right (Scale is step 3, before Draw at step 4), and this investigation confirms it's
  the RIGHT order: every zone/door/wall drawn is a set of coordinates in whatever the current pixel space
  is, and retrofitting a scale factor onto already-drawn geometry after the fact is a strictly harder,
  more error-prone operation than fixing the scale before drawing starts. No change recommended here,
  just confirmation.
- **Missing step: asset placement should be explicitly split from structural drawing** in the workflow
  description, not folded into "Draw Building" — Zones/Doors/Exits/Stairs (structural) and
  Cameras/Detectors/Sprinklers/etc. (asset overlay) are different mental modes for a first-time user, even
  though they're technically the same `graphics_scene.py` tool-dispatch mechanism (Phase 1). Recommend the
  UX flow name them as two visually/conceptually distinct phases even though the underlying implementation
  is unified.
- **Missing step: no "New Floor" is named anywhere in the brief's workflow**, but `floor_list.py` (a
  confirmed-independent widget) exists specifically for multi-floor projects — a real building with
  stairs needs 2+ floors defined before Stairs are meaningful. Should be inserted between "Set Scale" and
  "Draw Building" (or made an explicit sub-step of "Draw Building").

**Recommended revised workflow**:
```
Create Project
  ↓
Import Floor Plan (per floor)
  ↓
Set Scale (per floor plan)
  ↓
Add Floor(s) [new step -- floor_list.py already supports this]
  ↓
Draw Structure (Zones, Doors, Exits, Stairs) [renamed from "Draw Building" for clarity]
  ↓
Place Assets (Cameras, Detectors, Sprinklers, and the rest of the already-independent asset palette)
  ↓
Validate (continuous/incremental, not a single gate -- re-run designer/validation.py's checks after
  every structural change, not just once before save)
  ↓
Save (.syn -- this step already IS "Export .syn", per Phase 3 -- no separate export action needed)
  ↓
Open in SynEvac Studio
```

---

## Phase 8 — Final Recommendation

**1. Is Builder technically feasible without duplicating large amounts of code?**
**Yes, confirmed by direct code inspection, not assumption.** `models/`, `serialization/`, `navigation/`,
`designer/items/*.py` (34 of 35 files), `designer/scene/*.py`, `designer/controllers/*.py`, `designer/
validation.py`, and roughly half of `designer/widgets/*.py` are ALL already free of any simulation/AI/
perception dependency (Phase 1's exhaustive audit). Builder would import these directly — zero
duplication required for the vast majority of its scope. Only scale calibration is genuinely new.

**2. Should Builder be a separate executable or simply a different startup mode?**
**Separate executable, backed by a shared codebase — not merely a startup flag inside the current
`main_window.py`.** Reasoning: `main_window.py` unconditionally constructs all 8 heavy panels in `__init__`
(Phase 1) — a runtime "lite mode" flag would still need to import every one of those heavy modules just to
conditionally skip instantiating them, defeating the entire purpose (Builder is supposed to have ZERO
dependency on Simulation/AI, not merely hide it at runtime). A genuinely separate entry point (its own
small main-window file, its own `main.py`-equivalent launcher) importing only the confirmed-light packages
is the only way to make "NO AI, NO Simulation" a structural guarantee rather than a runtime convention —
consistent with `core/app.py`'s own existing pattern of being a thin bootstrap over `MainWindow`
(`main.py:1: from core.app import SynEvacApp`), which a `BuilderApp` could mirror exactly.

**3. Can Builder be developed independently while Studio continues evolving?**
**Yes, with one caveat.** Since Builder's entire dependency surface (`models/`, `navigation/`,
`serialization/`, `designer/items/`, `designer/scene/`, `designer/validation.py`) is code Studio ALSO
depends on and does not own exclusively, changes to those shared files by Studio-focused work could affect
Builder and vice versa — this is a normal shared-library concern, not a blocker, but means the two cannot
be developed in total isolation; a change to, say, `models/building.py` for a Studio feature needs the same
scrutiny for Builder-compatibility it already gets for Studio (the existing 5138-test suite already covers
`models/`/`navigation/`/`serialization/` reasonably, per this session's own repeated full-suite runs, but
was never exercised against a Builder-shaped consumer specifically).

**4. What functionality MUST remain inside Studio?**
Everything in Phase 4's exclusion list, structurally guaranteed by the current import graph: Simulation,
AI/Prediction, Perception/Live camera-calibration-and-connection, Live Runtime, Recommendation/Guidance/
Response, Campaign/Dataset Generation, and the debug panels that visualize any of the above. Concretely,
in `designer/` terms: `campaign/`, `live_runtime_controller.py`, `building_state_debug_*.py`, `perception_
debug_*.py`, `camera_manager_panel.py`, `camera_validation_panel.py`, `speaker_manager_panel.py`,
`live_runtime_panel.py`, and `property_panel.py`'s camera-calibration section specifically (the rest of
`property_panel.py` is asset-property editing and could plausibly be split — not attempted in this
investigation).

**5. What functionality MUST move into shared libraries?**
**Nothing needs to MOVE — it already lives in independently-importable packages** (Phase 5). The only real
decision is whether to leave `designer/items/`/`designer/scene/`/etc. where they are (Builder imports
`designer.items.*` directly — zero-effort, slightly confusing naming) or extract them into a new,
neutrally-named package both apps import from (cleaner, non-trivial refactor with real regression
surface). This investigation does not resolve that choice — it is a legitimate implementation-time
decision for whichever future milestone builds Builder, not an architectural fact this investigation can
determine from the current code alone.

**6. Is now the correct time in the project to build Builder?**
**Plausibly yes, but this is the one question this investigation cannot answer purely from dependency
evidence — it depends on team/product priorities outside this repository's code.** What THIS investigation
can say with evidence: the TECHNICAL precondition for building Builder cheaply — a clean, already-proven,
already-independent authoring layer separable from Simulation/AI — already exists today, and did not need
to be created for this recommendation to be true. Waiting longer does not make the technical foundation
any more ready than it already is; if anything, every new Studio-only feature added to `designer/windows/
main_window.py` (per Phase 1's own pattern of eager, entangled panel construction) makes the eventual
`BuilderMainWindow` extraction marginally more work by increasing the gap between "everything main_window.py
does" and "everything Builder needs," since Builder should be a NEW small file, not a stripped-down copy of
the ever-growing existing one (Phase 2). The one substantive missing piece (scale calibration) is
independent of team timing and would need to be built whenever Builder is built, regardless of when that is.
