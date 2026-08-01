# Execution Layer V1

## 1. Overview — Orchestration, Not a Replacement Execution Engine

**The Execution Layer is an orchestration/coordinating layer over `voice_evacuation`, `building_control`, `dynamic_signage`, and the new `warden_notification` controllers. It never calls a provider itself, never bypasses the operator-approval gate, and never becomes a second execution authority — those three (now four) controllers remain the sole place a command is actually dispatched.**

This is a load-bearing distinction, not a stylistic one. A pre-implementation architectural review found a working, tested, production-reachable execution mechanism already exists: `command_center/live_operator_action_gateway.py::LiveOperatorActionGateway`, backing `VoiceEvacuationController`, `BuildingControlController`, and `DynamicSignageController` — each following an identical shape (submit → `PENDING_APPROVAL` → explicit operator click → `approve()`/`reject()` → `_dispatch()` → `provider.execute()/apply()/send()` → `CONFIRMED`/`FAILED`, with an append-only history). This milestone adds exactly one genuinely new controller (`warden_notification/`, mirroring that shape verbatim) and a read-side coordinator (`execution_layer/`) that unifies all four into one cross-category audit view — it duplicates none of their logic and reimplements no execution.

## 2. Public API

```python
from execution_layer.layer import ExecutionLayer

layer = ExecutionLayer(
    voice_controller=voice_evacuation_controller,      # voice_evacuation.VoiceEvacuationController or None
    control_controller=building_control_controller,    # building_control.BuildingControlController or None
    signage_controller=dynamic_signage_controller,      # dynamic_signage.DynamicSignageController or None
    warden_controller=warden_notification_controller,   # warden_notification.WardenNotificationController or None
)

execution_set = layer.compute(time)   # read-only -- reads what the controllers already recorded

execution_set.requests                        # Tuple[ExecutionRequest, ...]
execution_set.by_category(category)           # filter by ExecutionCategory
execution_set.for_recommendation(recommendation_id)   # every ExecutionRequest tracing back to one Recommendation
execution_set.to_dict()

layer.latest    # the most recent ExecutionSet, or None before the first compute()
```

`compute()` never raises and always returns an `ExecutionSet` (possibly empty) — a bug in one category's adapter never blanks the other three.

**Submitting a new Warden Notification** (the one category this layer has a submit-side for) is NOT done through `ExecutionLayer` — it goes through the same gateway every other category already uses, for consistency and because approval belongs there:

```python
from command_center.live_operator_action_gateway import LiveOperatorActionGateway

gateway.ingest_warden_recommendations(recommendation_set, time)   # submit-side, mirrors ingest_control_recommendations()
gateway.approve_warden_notification(request_id)                    # operator action
gateway.reject_warden_notification(request_id)                     # operator action
```

## 3. The Four-Timestamp Traceability Model

Every `ExecutionRequest` preserves a complete audit chain:

| Field | Meaning |
|---|---|
| `execution_request_id` | Mirrors the underlying request/instruction-key/message id |
| `category` | One of `ExecutionCategory` (`VOICE_EVACUATION`/`BUILDING_CONTROL`/`DYNAMIC_SIGNAGE`/`WARDEN_NOTIFICATION`) |
| `status` | One of `ExecutionStatus` — normalized across all four categories' own (slightly different) status vocabularies |
| `provider_source` | The real provider's own class name (e.g. `"SimulationControlProvider"`) — audit-facing, never a live object reference |
| `originating_recommendation_id` | The upstream recommendation id, when known |
| `recommendation_id_provenance` | **Honest disclosure** of which id space `originating_recommendation_id` came from — see §4 |
| `created_at` / `approved_at` / `dispatched_at` / `completed_at` | The four moments in the request's lifecycle, derived by reading the controller's own append-only history |
| `result_message` / `result_confirmed` | The provider's own reported outcome |

### Worked example: Recommendation → Warden Notification → Approval → Dispatch → Completion

```python
recommendation = Recommendation(recommendation_id="rec-e2e-001", type=RecommendationType.WARDEN_DISPATCH, ...)
recommendation_set = RecommendationSet(timestamp=1.0, recommendations=(recommendation,))

gateway.ingest_warden_recommendations(recommendation_set, 1.0)   # -> PENDING_APPROVAL, created_at=1.0
gateway.approve_warden_notification(request_id)                   # -> APPROVED -> DISPATCHED -> CONFIRMED

execution_set = runtime.tick_execution_layer(2.0)
request = execution_set.for_recommendation("rec-e2e-001")[0]
# request.status == "CONFIRMED"
# request.recommendation_id_provenance == "recommendation_layer"
# request.created_at, .approved_at, .dispatched_at, .completed_at all populated
```

This is the one category built correctly from day one, since it is new — the real `recommendation_layer.Recommendation.recommendation_id` flows through unmodified from creation to completion.

## 4. Disclosed Gap — `recommendation_id_provenance` for the Three Pre-Existing Categories

`ControlRequest.source_recommendation_id`/`VoiceMessage.source_recommendation_id` today only ever carry `building_control.advisory_adapter`'s own **synthesized** id (a content-derived hash, `"rec-" + sha1(...)`, confirmed in `_synthesize_recommendation_id()`) — never a real `recommendation_layer.Recommendation.recommendation_id`. `SignageInstruction` carries no recommendation-traceability field at all. The adapters tag these honestly:

| Category | `recommendation_id_provenance` today |
|---|---|
| Occupant Routing / Hazard Avoidance / etc. via Building Control, Voice | `"advisory_system"` when a synthesized id is present, else `"unavailable"` |
| Dynamic Signage | always `"unavailable"` — no field exists |
| Warden Notification | `"recommendation_layer"` — the real id, always |

**Closing this gap fully** (making `evacuation_guidance`/`advisory_adapter` carry the real `recommendation_layer` id through to `ControlRequest`/`VoiceMessage`) is future work, out of scope for V1. This mirrors `recommendation_layer`'s own disclosed "`advisory_report` is usually `None` live" gap — an honest limitation, never silently glossed over.

## 5. Integration Points

- **`live_runtime/factory.py`** — constructs `WardenNotificationController`/`SimulationWardenNotificationProvider` under `build_offline_demo_runtime()` only (never defaulted under `build_live_runtime()`, matching every sibling provider's NO_PROVIDER-under-LIVE convention). Constructs `ExecutionLayer` unconditionally, given all four controllers (degrades gracefully with any subset `None`). Threads both into `LiveOperatorActionGateway` and `LiveRuntime`.
- **`live_runtime/runtime.py`** — new `tick_execution_layer(time)` method, **deliberately separate from `run_cycle()`**. `LiveOrchestrator` is mechanically forbidden from importing `voice_evacuation`/`building_control.controller`/`dynamic_signage.controller` (architecture guard `LiveOrchestratorCannotDirectlyCallControllersTests`) — the four controllers live on `LiveRuntime`, one layer above the orchestrator, so `ExecutionLayer` (which reads them) cannot be ticked from inside `run_cycle()` without violating that guard. `run_cycle()` itself is untouched, still a pure forward to `orchestrator.run_cycle()`.
- **`command_center/live_operator_action_gateway.py`** — the one additive extension to the established execution seam: `warden_controller` param, `ingest_warden_recommendations()`, `approve_warden_notification()`, `reject_warden_notification()`, mirroring the Building Control workflow's own methods exactly.
- **`live_system/live_command_center_gateway.py` + `command_center/data_source.py`** — `recommendation_set` forwarded into `CommandCenterSnapshot`, mirroring how `dynamic_signage`/`advisory_report` are already forwarded — the bridge that lets Command Center's own render loop ingest Warden recommendations.
- **Studio** — new `designer/widgets/execution_panel.py`, a read-only, informational-only dock (no Approve/Reject affordance at all — that stays exclusively in Command Center), refreshed on the same `LiveRuntimeController.on_cycle_callback` tick as the Recommendation panel.
- **Command Center** — new `command_center/warden_notifications_panel.py`, mirroring `building_controls_panel.py`'s own real Approve/Reject workflow.

## 6. Files That Remain Frozen / Untouched

`evacuation_recommendation/`, `evacuation_guidance/`, `recommendation_layer/`, `advisory_system/`, `emergency_response/`, `crowd_intelligence/`, `voice_evacuation/controller.py`, `building_control/controller.py`, `dynamic_signage/controller.py` (called, never modified), `live_system/orchestrator.py` (correctly never touched — the architecture guard is respected, not worked around).

## 7. Architecture Guards

- `execution_layer/`/`warden_notification/` forbidden from importing AI/decision_policy/hardware-protocol modules.
- `execution_layer/`'s adapters for Voice/BuildingControl/Signage are mechanically proven never to call `.execute(`/`.apply(`/`.send(`/`.notify(`/`.dispatch(` — read-only, always.
- `advisory_system/`, `live_ai_gateway.py`, `live_advisory_gateway.py`, `decision_policy/`, `live_system/orchestrator.py` are mechanically proven never to import `execution_layer`/`warden_notification`/any controller — the AI/Advisory side of the platform can never reach execution authority directly.
