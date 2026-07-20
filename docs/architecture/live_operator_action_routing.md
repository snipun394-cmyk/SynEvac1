# Live Operator Action Routing — Human-in-the-Loop

Status: **implemented, offline-tested, no physical PA/BMS/fire-system hardware involved.** This milestone connects the live `AdvisoryReport` already reaching Command Center (`docs/architecture/live_system_integration_audit.md` §18) to the already-existing Voice Evacuation and Building Control frameworks, through an explicit, human-in-the-loop operator workflow. AI/Advisory generation remains recommendation-only; only an explicit operator action can trigger execution.

## 1. The architecture

```
BuildingState
    -> Live AI
    -> AI-Augmented Advisory System
    -> AdvisoryReport
    -> Live Command Center (Recommendation)
    -> Operator reviews recommendation
    -> Operator explicitly approves/selects action
    -> VoiceEvacuationController OR BuildingControlController
    -> Configured output provider (Simulation today; Live Hardware architecturally, not implemented)
```

| Layer | Role |
|---|---|
| AI (`ai_registry`/`live_ai_gateway`) | Supporting evidence only |
| Advisory (`advisory_system`/`live_advisory_gateway`) | Recommendation only |
| Operator (Command Center UI) | Authorization — the only trigger |
| Controller (`VoiceEvacuationController`/`BuildingControlController`) | Validation, deduplication, lifecycle, history |
| Provider (`VoiceOutputProvider`/`BuildingControlProvider`) | Execution boundary |

**The one new piece of code:** `command_center/live_operator_action_gateway.py` — `LiveOperatorActionGateway`. It is the *only* seam Command Center is allowed to route explicit operator intent through to a real controller. AdvisoryReport generation, AI inference, and Decision Policy never call it — see §5.

## 2. Voice Evacuation path

```
CivilianAnnouncement (advisory_system)
    -> VoiceEvacuationPanel.show_live() renders it, status = RECOMMENDED
    -> Operator clicks Approve / Send
    -> LiveOperatorActionGateway.approve_voice_message(announcement, time)
        -> voice_evacuation.adapter.civilian_announcement_to_voice_message()
           (message_text carried through VERBATIM — never regenerated)
        -> VoiceEvacuationController.broadcast(message, time)
           (existing priority/supersession/per-zone routing, unchanged)
    -> BroadcastInstruction(s) recorded in the real BroadcastLog
```

Status vocabulary shown in the UI (`command_center/live_operator_action_gateway.py`): `RECOMMENDED` → `APPROVED` → `SENT` / `FAILED`, or `REJECTED` / `SUPERSEDED`. `SENT`/`FAILED`/`SUPERSEDED`/`REJECTED` are derived directly from `voice_evacuation.models.BroadcastStatus` (`BROADCAST`/`NO_SPEAKERS_AVAILABLE`/`SUPERSEDED`/`CANCELLED` respectively) — no competing state machine was created; only `RECOMMENDED`/`APPROVED` are genuinely new labels, because nothing before this milestone tracked "has an operator reviewed this yet" at all.

**Rejecting never requires a provider** — `reject_voice_message()` never touches `VoiceEvacuationController`; it is purely a gateway-side decision (Phase 5: "operator can always review/reject").

**Idempotency:** a decision is keyed on `zone_id + announcement text`, not time — a duplicate Approve click for the identical, already-approved recommendation returns the prior result rather than re-broadcasting; a *new* recommendation for a zone that previously had a decided-on message always starts fresh as `RECOMMENDED`.

**Audit:** `voice_evacuation.models.BroadcastInstruction` carries no actor field, and this milestone does not modify that frozen, widely-consumed dataclass. `OperatorActionRecord` (gateway-side, additive) is the audit trail for voice decisions instead — `action`, `zone_id`, `actor` (always `COMMAND_CENTER_OPERATOR` — no authentication/user-management system exists or was built), `time`.

## 3. Building Control path

```
BuildingRecommendation (advisory_system)
    -> LiveOperatorActionGateway.ingest_control_recommendations(report)
        -> building_control.advisory_adapter.translate_report() (reused, unchanged)
        -> BuildingControlController.submit() per request (dedup/validation, unchanged)
    -> ControlRequest, PENDING_APPROVAL, rendered in BuildingControlsPanel
    -> Operator clicks Approve / Reject
    -> LiveOperatorActionGateway.approve_control_request(request_id) / reject_control_request(request_id)
        -> BuildingControlController.approve()/reject(actor="COMMAND_CENTER_OPERATOR")
    -> On approve: BuildingControlController dispatches -> BuildingControlProvider.execute()
    -> ControlResult -> CONFIRMED or FAILED (never CONFIRMED unless the provider actually reported it)
```

`BuildingControlController`'s own validation, deduplication, approval/rejection/cancellation lifecycle, and append-only `ControlEvent` history (`building_control/history.py`, already carries an `actor` field) are reused **completely unchanged** — the gateway never bypasses it, only routes an explicit operator call into it.

**Rejection persistence (a gap this milestone found and fixed):** `REJECTED` is a terminal `RequestStatus`, so `BuildingControlController`'s own live-duplicate guard (`_LIVE_STATUSES`) no longer protects a rejected recommendation from being silently resubmitted the next time `ingest_control_recommendations()` re-ingests the same `AdvisoryReport` (which happens on every panel re-render, including the one immediately after the reject itself). `LiveOperatorActionGateway` tracks `source_recommendation_id` values an operator has explicitly rejected and skips resubmitting exactly that recommendation — a genuinely different recommendation (different system/target/action, hence a different synthesized id) is never blocked. See `tests/test_live_operator_action_gateway.py::ControlApprovalTests.test_rejecting_a_recommendation_prevents_it_from_silently_resubmitting`.

## 4. Provider capability

`is_simulation_only` (already established on `building_control.providers.BuildingControlProvider`) was mirrored onto `voice_evacuation.provider.VoiceOutputProvider` (default `False`) and `SimulationVoiceOutputProvider` (`True`) — both controllers gained a trivial, read-only `provider` property so the gateway can derive capability without importing or guessing about any specific provider implementation:

- `NO_PROVIDER` — no controller configured at all. Operator can still review every recommendation (Building Control: the raw `BuildingRecommendation` list is shown, nothing ever submitted; Voice: `voice_recommendation_status()` still reports `RECOMMENDED`/`REJECTED` normally, since rejecting needs no provider). The Approve/Send button is disabled (voice) or the row simply never becomes a submittable `ControlRequest` (building control) — never a silent failure, never a fabricated success.
- `SIMULATION` — `SimulationVoiceOutputProvider`/`SimulationControlProvider`. Full operator workflow works end-to-end (offline-tested).
- `LIVE_HARDWARE` — architectural placeholder only. No real hardware provider exists anywhere in this codebase; the capability label exists so the UI can one day distinguish it, and is proven correct against a minimal test-only non-simulation provider stand-in (`tests/test_live_operator_action_gateway.py::CapabilityTests`), never against anything that actually communicates with hardware.

## 5. AI Authority Guard

`tests/test_live_operator_action_routing.py::AIAuthorityGuardTests` mechanically proves, by source-text regex scan (the same convention every other package-boundary guard in this codebase already uses):

- `advisory_system/*.py`, `live_system/live_ai_gateway.py`, `live_system/live_advisory_gateway.py`, `live_system/orchestrator.py`, and `decision_policy/*.py` never import `voice_evacuation`, `speaker_manager`, `building_control.controller`, `building_control.providers`, or **`command_center.live_operator_action_gateway` itself**.
- No file anywhere in `advisory_system/`, `live_system/`, `decision_policy/`, `ai_registry/`, or `ai_inference/` even mentions `live_operator_action_gateway` in its source text.

Structurally: `LiveOperatorActionGateway` lives in `command_center/`, not `live_system/` — `live_system` is *already* mechanically forbidden (`tests/test_live_system.py::LiveSystemPackageDependencyDirectionTests`, pre-existing) from importing `voice_evacuation`/`building_control.controller`/`building_control.providers` at all, for exactly the reason this milestone depends on: the AI-inference/advisory-generation side of the platform can never reach execution authority, because there is no import path from that side to the controllers, and now none to the gateway either.

## 6. Command Center safety UX

`VoiceEvacuationPanel`/`BuildingControlsPanel` (`command_center/recommendation_center.py`, `command_center/building_controls_panel.py`) never import `voice_evacuation.controller`/`voice_evacuation.provider`/`speaker_manager`/`building_control.controller`/`building_control.providers` themselves — the pre-existing `tests/test_live_command_center.py::CommandCenterLiveIntegrationGuardTests` guard (unmodified) still passes. Every action routes through an injected `LiveOperatorActionGateway`, threaded from `MainWindow.enable_live_mode(data_source, operator_action_gateway=None)` → `Dashboard.set_operator_action_gateway()` → `RecommendationCenter.show_live(report, gateway)` → the two panels. `Dashboard`/`MainWindow` hold the gateway as an opaque reference and never call a method on it themselves.

`Dashboard.apply_snapshot()`'s pre-existing staleness discipline (Phase 14 of the prior milestone) is untouched: a `STALE` cycle withholds the `AdvisoryReport` entirely (`advisory_for_display = None`), so a stale report never reaches either panel as something an operator could act on.

**RECOMMENDATION vs EXECUTED ACTION** is visually explicit at every stage: `RECOMMENDED` (plain data) → `APPROVED` (gateway decision recorded, dispatch attempted) → `SENT`/`CONFIRMED` (provider actually reported success) or `FAILED` (provider reported failure — never displayed as `SENT`/`CONFIRMED`). Building Controls additionally keeps its three pre-existing tables (Pending / Active-Confirmed / History) — `active_table` (confirmed controls) is populated exclusively from `BuildingControlController.snapshot().entries`, which itself only ever contains `CONFIRMED` requests (unchanged, pre-existing `BuildingControlController.snapshot()` behavior).

## 7. FACP separation (re-confirmed, not re-litigated)

FACP (`facp/engine.SimulatedFACP`) remains what it already was: a read-only aggregation/coordination layer over detector conditions, with `BuildingState.facp_status` as its one output. This milestone adds **zero** code paths from FACP to voice or building-control execution — `tests/test_live_operator_action_routing.py::FACPSeparationTests` drives a real cycle with `SimulatedFACP` forced into `ALARM` (reusing `tests/test_live_command_center.py::_LiveChain`'s own existing FACP-alarm fixture) and confirms both a freshly-constructed `VoiceEvacuationController` and `BuildingControlController` remain completely untouched — zero broadcasts, zero requests, purely because nothing was ever wired to call them automatically.

## 8. Offline End-to-End results

- **Voice** (`tests/test_live_operator_action_routing.py::VoiceOperatorOfflineE2ETests`): a real two-zone `AdvisoryReport` (built via the real `AdvisoryOrchestrator`, matching `tests/test_advisory_system.py`'s own fixtures) → zero broadcasts before any operator action → operator approves zone A only → zone A receives exactly the advisory's own verbatim text, zone B is untouched → `BroadcastLog`/gateway audit log both record it → provider identity (`SIMULATION`) is visible → a zone with no registered speaker reports `FAILED`, never a silent drop or a fabricated `SENT`.
- **Building Control** (`ControlOperatorOfflineE2ETests`): zero execution before approval → Approve dispatches through the real controller → Reject never dispatches → a provider with no `ActionExecutor` reports `FAILED`, never `CONFIRMED` → re-ingesting the same recommendation never creates a duplicate live request → a contradictory recommendation (Close after Open) gets its own independent history entry → a state-only system (Smoke Exhaust) confirms honestly without ever claiming a physical hazard effect → a Door action, given a real `ActionExecutor`-shaped stand-in, dispatches through it exactly as `SimulationControlProvider` already does.

## 9. Failure modes (Phase 11, all 12 scenarios)

No voice/control provider → `OperatorActionUnavailable` on approve (Reject still succeeds); voice/control provider failure → honest `FAILED`, never `CONFIRMED`; a `STALE` `CommandCenterSnapshot` never offers a report an operator could act on; a recommendation superseded by a new one before approval starts fresh, never inherits a stale decision; a duplicate operator click never re-executes; an operator rejection is honored even across re-renders (§3); the live runtime stopping never corrupts an independently-held gateway; a recommendation simply absent from the next report is safe to query status for; an FACP alarm alone changes nothing; changing AI/advisory content across cycles still requires its own fresh operator decision. See `tests/test_live_operator_action_routing.py::FailureCaseTests`.

## 10. Replay stays read-only

Replay mode's own pre-existing interactive Approve/Reject buttons (`BuildingControlsPanel._on_approve()`/`._on_reject()`, driven by `self._incident.control_controller` — a `BuildingControlController` scoped entirely to that one loaded incident's own reconstructed history) are untouched and remain the only execution affordance in Replay. `Dashboard.show_frame()`/`set_frame_index()` (Replay's render path) never reference `self._operator_action_gateway` at all — only `Dashboard.apply_snapshot()` (the Live-only render call) does. `command_center/incident_data.py` (Replay's own reconstruction) does not import `live_operator_action_gateway`. See `tests/test_live_operator_action_routing.py::ReplayReadOnlyTests`.

## 11. Answers to the milestone's own explicit questions

- **Can AI automatically broadcast a voice message?** No.
- **Can Advisory System automatically execute a building control?** No.
- **Can an operator explicitly approve a recommended voice message in Live Command Center?** Yes, when an appropriate provider is configured (`SIMULATION` today).
- **Can an operator explicitly approve a building-control recommendation?** Yes, when an appropriate provider is configured.
- **Does this milestone communicate with real speakers, FACP hardware, BMS, or fire-system hardware?** No.
