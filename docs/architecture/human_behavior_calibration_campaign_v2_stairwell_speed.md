# Human Behavior Calibration Campaign V2 — Direct Behavioral Calibration: Walking Speed Against NIST TN 1675

Research milestone. Investigates whether `Adult_Default.walking_speed` can be calibrated against a
genuinely behavior-specific, LOCAL observable (NIST TN 1675's own measured stairwell descent speed)
rather than aggregate whole-building evacuation time (the objective
`scripts/run_automatic_calibration_walking_speed_campaign.py` already uses). Full investigation,
results, and decision are recorded in the companion chat report; this document is the durable,
in-repo record.

No `simulator/`, `behaviour_profile_resolver/`, or `calibration_benchmark/` source file is modified.
One additive extension was made to `automatic_calibration/objectives.py` (see below). No SynEvac
production default is changed by this milestone.

## Source

Peacock, R.D., Hoskins, B.L., Kuligowski, E.D. (2010), NIST Technical Note 1675, *"Overall and Local
Movement Speeds During Fire Drill Evacuations in Buildings up to 31 Stories"* — also published as
Peacock, Hoskins & Kuligowski (2012), *Safety Science* 50, 1655–1664 (the citation
`scripts/run_nist_*_validation.py` already use for their own Table 2 geometry/occupant counts and
Table 2 evacuation times). This is the SAME four buildings (10/18/24/31-story) and the SAME source
paper already partially operationalized in this repo — only Table 3 (per-building stairwell descent
speed) had never been used as a calibration target before this milestone.

## Phase 1 — Mapping validity (confirmed, indirect)

`BehaviorProfileTemplate.walking_speed` is not a stair-specific constant: `simulator/coordinator.py`
applies `effective_speed = occupant.walking_speed * congestion_model.speed_factor(...)` and
`duration = edge.traversal_cost / effective_speed` identically for Door/Exit/Stair edges. For a Stair
edge, `traversal_cost` is `Staircase.travel_distance(building)` (`models/staircase.py`) — the real
along-incline path length (`vertical_height / sin(35°)`), not a horizontal projection — so
`walking_speed` **is** directly comparable to TN1675's own along-incline speed, but only in the
free-flow (uncongested) case, where it is true by construction that `effective_speed == walking_speed`.

TN1675's own reported speeds (Table 3) are observed under real, escalating fire-drill crowd density —
not free-flow. Setting `walking_speed :=` TN1675's raw mean directly would double-count crowding
(SynEvac's own congestion model would degrade an already-congestion-inclusive number a second time).
The valid comparison is therefore **simulated (congestion-inclusive) local stair speed, measured from
real occupant timelines, against TN1675's own reported mean** — never `walking_speed` against
TN1675's mean directly.

## Phase 2 — TN1675 data actually available

Directly usable (Table 3, this campaign's target): per-building mean ± SD "Overall Movement Speed" —
10-story 0.44±0.19 m/s, 18-story 0.44±0.15 m/s, 24-story 0.56±0.12 m/s, 31-story 0.52±0.10 m/s
(combined across all 4: 0.48±0.16 m/s, the paper's own headline figure).

Summary-only, used for context/cross-check, not as a fit target: local speed range 0.056–1.7 m/s;
majority of local speeds between 0.3–0.7 m/s (from the paper's own CDF figure, no table); a linear
regression (Table 4, R²=0.21) whose intercept (~0.625 m/s for the reference stairwell/occupant
categories) is a weak, extrapolated density→0 estimate, not a direct free-flow measurement.

Not reproducible from this investigation: raw per-occupant speed observations (hosted at
`http://fire.nist.gov/egress/`, not fetched or downloaded by this campaign).

Independent literature cross-check (TN1675's own Table 1, cited from Fruin): unimpeded stair descent
speed — "optimum" 0.48 m/s, "moderate" 0.61 m/s, "maximum" 0.76 m/s.

## Phase 3 — Calibration target design

Primary metric: **movement-only stairwell descent speed** — per occupant, sum of Stair-edge
`walking_distance` divided by sum of that edge's own `(end_time - start_time)`, averaged across
occupants who used a stair in that scenario. Deliberately excludes admission-control
`queue_wait_time` (a separate, already-instrumented SynEvac concept) so the metric isolates
`walking_speed x congestion_model.speed_factor` — the exact mechanism under test.

Diagnostic-only metric: **wall-clock span stairwell speed** — same distance, but wall-clock elapsed
time from first Stair-edge entry to last Stair-edge exit (including inter-segment admission-control
queueing). Empirically found to be dominated in absolute magnitude by admission-control queueing
(roughly 8–9x slower than movement-only at every tested walking_speed) — never used as the objective.

Statistical comparison: mean (with 95% CI across n=4 paired seeded scenarios, via the existing
`research_framework.statistics` machinery) vs. TN1675's own published per-building mean — NOT a
Kolmogorov–Smirnov or other raw-distribution test, since TN1675 provides no raw per-occupant data to
this investigation, only summary statistics.

## Phase 4 — Infrastructure reuse and the one extension made

Reused unmodified: `WalkingSpeedCandidate`, `AutoCalibrationEngine`, `GridSearchStrategy`,
`CalibrationStudio.run_published_benchmark()`, `calibration_benchmark.harness.run_calibration_benchmark()`
(already threads `additional_metrics` end-to-end into `CalibrationBenchmarkResult.additional_comparisons`).

New, script-local (no `calibration_benchmark/` file touched): two `AdditionalMetric` implementations
(`StairwellDescentSpeedMetric`, `StairwellSpanSpeedMetric`), defined in the campaign script itself —
the same "extension point, not touching harness.py" discipline `calibration_benchmark/optional_metrics.py`'s
own docstring already establishes.

Gap found and fixed: `automatic_calibration/objectives.py`'s `PublishedValueObjective.score()` only
ever read `session.result.comparisons` (the five built-in `METRIC_FIELDS`), never
`session.result.additional_comparisons` — so it could not rank grid points by a custom metric at all.
Fixed with a 3-line additive fallback (checks `comparisons` first, `additional_comparisons` second) —
verified against the full existing `automatic_calibration`/`calibration_benchmark`/`calibration_studio`
test suite (132 tests, all passing, zero regressions) before use.

## Phase 5 — Calibration results (behavioral fit)

Grid: `{0.5, 0.6, 0.7, 0.8, 0.9, 1.2}` m/s — brackets Fruin's own literature-cited unimpeded stair
range (0.48–0.76 m/s), deliberately NOT centered on TN1675's raw congested means; 1.2 m/s (today's
production default) retained as the established diagnostic reference point. n=4 paired scenarios per
grid point per building; `dt=1.0`.

| Building | TN1675 target (mean±SD) | Best-fit walking_speed (grid) | Simulated movement-only speed | Distance |
|---|---|---|---|---|
| 10-story | 0.44 ± 0.19 m/s | 0.5 m/s | 0.4417 m/s | 0.0017 |
| 18-story | 0.44 ± 0.15 m/s | 0.5 m/s | 0.4409 m/s | 0.0009 |
| 24-story | 0.56 ± 0.12 m/s | 0.6 m/s (interp. optimum ≈0.63) | 0.5319 m/s (0.6207 at 0.7) | 0.0281 |
| 31-story | 0.52 ± 0.10 m/s | 0.6 m/s | 0.5293 m/s | 0.0093 |

At today's production default (1.2 m/s), simulated free-flow-equivalent stairwell speed is ≈1.06 m/s
in every building — roughly double every building's real TN1675 stairwell speed. Every grid point away
from its own building's best fit differs from the TN1675 target with p < 1e-6 (huge, consistent effect
sizes) — the calibration surface is sharply resolved.

Independent cross-check: the best-fit range (0.5–0.63 m/s) sits almost exactly between TN1675's own
cited Fruin "optimum" (0.48 m/s) and "moderate" (0.61 m/s) unimpeded stair speeds — two independent
sources converge.

Full per-candidate results (all 4 buildings, movement-only AND wall-clock-span speed, whole-building
evacuation time, CIs, p-values, effect sizes): `docs/architecture/human_behavior_calibration_campaign_v2_stairwell_speed_raw_results.json`.

Note on the `recommendation`/`recommendation_summary` fields in that JSON: these come from
`calibration_benchmark.recommend()`'s own pre-existing, unrelated 5-metric adoption check (candidate
vs. today's 1.2 m/s production default on evacuation_time/congestion_duration/queue_length/
peak_occupancy_ratio/exit_utilization_balance) — REJECT there means "this candidate is slower on
whole-building evacuation time than today's default," not "this candidate fits TN1675 poorly." The two
are deliberately NOT the same judgment (see Phase 6/7).

## Phase 6 — Out-of-target cross-validation (whole-building evacuation time)

At the SAME sessions' built-in `evacuation_time` metric, against Table 2's published whole-building
evacuation times:

| Building | Published | @ 1.2 m/s (today's default) | @ TN1675-best-fit speed |
|---|---|---|---|
| 10-story | 1022 s | 3118 s (3.05x) | 7470 s @ 0.5 (7.3x) / 6226 s @ 0.6 (6.1x) |
| 18-story | 1192 s | 19616 s (16.5x) | 47067 s @ 0.5 (39.5x) / 39224 s @ 0.6 (32.9x) |
| 24-story | 1090 s | 10228 s (9.4x) | 20443 s @ 0.6 (18.8x) / 17524 s @ 0.7 (16.1x) |
| 31-story | 1002 s | 11813 s (11.8x) | 23614 s @ 0.6 (23.6x) / 28335 s @ 0.5 (28.3x) |

Whole-building evacuation time is already massively over-predicted at TODAY'S 1.2 m/s default
(3–16.5x), well before this campaign changes anything. Adopting the TN1675-consistent stairwell speed
makes this materially WORSE (roughly another 2–4x on top). This is consistent with — and independently
corroborates — the Phase 3 finding that wall-clock-span stairwell speed is dominated by
admission-control queueing, not by walking_speed: the dominant driver of SynEvac's whole-building
evacuation-time overprediction is very likely the capacity/admission-control architecture, not
`walking_speed`, and is out of scope for this milestone.

## Phase 7 — Decision

**Behavioral validity (TN1675 local stairwell speed): ADOPT the finding.** A walking_speed in the
0.5–0.6 m/s range (building-specific optimum 0.5–0.63 m/s) reproduces every tested building's real
TN1675 stairwell descent speed to within 0.001–0.03 m/s, independently corroborated by Fruin's
literature range. This is genuinely stronger, more direct evidence than the whole-building
evacuation-time objective the existing walking-speed campaign uses.

**Direct production-parameter change: NOT adopted — genuinely blocked by Phase 1's own architecture
finding, not by the data.** `walking_speed` is a single scalar shared identically between stair AND
horizontal (Door/Exit) movement. Setting it to ~0.55 m/s to fit TN1675's stair data would also force
horizontal corridor movement down to stair-descent pace — a value well below typical unimpeded adult
walking speed. SynEvac has no edge-type-specific speed mechanism today; adding one is a real,
identifiable next step, but is out of scope here ("do NOT redesign the simulator").

**Whole-building predictive validity: REJECT.** The TN1675-consistent walking_speed value materially
worsens an ALREADY very poor (3–16.5x over-predicted, even at today's default) whole-building
evacuation-time fit. Per the brief's own instruction, this does NOT override the behavioral finding —
it identifies a separate, larger, pre-existing problem (very likely admission-control/capacity, not
walking_speed) as the actual dominant driver of evacuation-time overprediction, and as the more
consequential next investigation.

## Phase 8 — Extensions (not executed)

Not run this milestone, per the brief's own "Adult_Default first" instruction. `Elderly_Default`/
`Wheelchair_Default` extension is a real candidate for a future campaign IF a genuinely comparable
local-speed dataset (not the pre-movement-delay literature already used in Campaign V1) is identified
and its own mapping validity is verified the same way Phase 1 verified this one — not assumed.

## Files

- `scripts/run_human_behavior_calibration_campaign_v2_stairwell_speed.py` — the campaign
- `automatic_calibration/objectives.py` — `PublishedValueObjective.score()` additional_comparisons fallback (additive, tested)
- `docs/architecture/human_behavior_calibration_campaign_v2_stairwell_speed_raw_results.json` — full raw results
- this document
