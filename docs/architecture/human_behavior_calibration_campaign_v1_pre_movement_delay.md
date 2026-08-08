# Human Behavior Calibration Campaign V1 — Pre-Movement Delay

Implementation milestone. Calibrates `ProbabilisticPreMovementDelay` (the existing, active, default
pre-movement mechanism for 5 of 7 civilian/staff profiles) using the existing, unmodified
Calibration Studio / Automatic Calibration Engine / Grid Search / statistics / reporting
infrastructure — mirroring `scripts/run_automatic_calibration_walking_speed_campaign.py` exactly,
substituting `PreMovementDelayCandidate` for `WalkingSpeedCandidate`.

No file under `simulator/`, `calibration_studio/`, `automatic_calibration/`, `behavior_library/`,
or `behaviour_profile_resolver/` is modified. Only new campaign scripts and this document were
added:

- `scripts/run_human_behavior_calibration_campaign_v1_pre_movement_delay.py`
- `scripts/run_human_behavior_calibration_campaign_v1_configuration_comparison.py`
- raw JSON results in `docs/architecture/`
- this document

## Phase 1 — Literature survey and calibration ranges

Searched directly (web search, this milestone; no prior in-repo citation existed for pre-movement
delay — confirmed by grep across `docs/architecture/*.md` and `scripts/*.py` before searching).

### Adult — LITERATURE-SUPPORTED (strongest evidence, direct building-type match)

**Source**: NIST Technical Note 1664, "Occupant Behavior in a High-rise Office Building Fire"
(referenced via NIST's own evacuation-behavior publication series). 19 observed occupants,
individual pre-evacuation delay times ranging **10–55 seconds**, mean **28s**, standard deviation
**11s**.

This is the single most directly comparable figure available: it is itself an office/high-rise
fire-evacuation study — the same building class as all four NIST validation buildings this
campaign runs against (10/18/24/31-story office buildings, Peacock/Hoskins/Kuligowski 2012).
SynEvac's current `Adult_Default.median_delay = 30s` is already very close to this figure's own
mean (28s).

**Grid used**: `{15, 20, 25, 30, 35, 45, 55}` seconds — brackets the reported 10–55s range with the
current production default (30s) deliberately included as a grid point, exactly as `1.2 m/s` was
for the walking-speed campaign.

**Caveat, also disclosed**: a wider literature review (Proulx & Bénichou, 13-story office building,
1191 occupants) found individual delay times ranging from 30 seconds to over 24 minutes — pre-movement
time has a much heavier real-world tail than a 19-occupant sample or a lognormal(median=30, spread=0.5)
model captures. This campaign calibrates the **median**, not the tail; the tail-shape question is
explicitly out of scope here (see Phase 6 of the companion roadmap report).

### Visitor (unfamiliar occupant) — MODERATELY LITERATURE-SUPPORTED

**Sources**: hotel/unfamiliar-occupancy evacuation literature reports self-estimated
pre-movement time of **25 seconds to 5 minutes, most common answer 1 minute (60s)**; a broader
review states pre-movement phases "average 2–6 minutes in office settings but extend to over 10
minutes in residential or unfamiliar structures when alarms are ambiguous."

**Caveat, disclosed**: this literature's hotel-guest data includes sleep/intoxication factors that
do not apply to a daytime office visitor (SynEvac's `Visitor_Default` context). The grid below
deliberately excludes the source range's upper tail (5+ minutes) as non-representative of that
context.

**Grid**: `{25, 40, 60, 90, 120}` seconds — current default (40s) included, spanning the
self-estimated low end through the "most common answer" (60s).

### Child — LITERATURE-SUPPORTED BUT HETEROGENEOUS (wide inter-study spread; different population)

**Sources**: school fire-drill studies. Ages 3–6: pre-evacuation time range 10–545s, mean **114s**.
Ages 6–16 (five drills, Spanish schools): range 4–166s. A separate study: 57–164s. Another: 3–59s.

**Caveat, prominently disclosed**: these are *drilled schoolchildren with a trained routine* —
SynEvac's `Child_Default` represents a child in a general building population, not necessarily
one with repeated fire-drill training, so these figures may understate real delay for an
unfamiliar/untrained child occupant. Central tendencies across studies cluster loosely around
60–120s, well above SynEvac's current 45s default.

**Grid used (diagnostic only — see Phase 3)**: `{45, 90, 120}` seconds — current default plus two
literature-informed points.

### Elderly — LITERATURE-INFORMED BUT CROSS-CONTEXT (residential, not office/high-rise)

**Source**: apartment-building evacuation drill, average pre-evacuation delay **6 minutes (360s)**,
range 0–25 minutes (0–1500s). This study **explicitly excluded wheelchair users** — ambulatory
elderly only.

**Caveat, prominently disclosed**: this is residential/apartment data, not office/high-rise data
(a different building-occupancy expectation than the NIST benchmark buildings this campaign
validates against). SynEvac's current `Elderly_Default.median_delay = 50s` is roughly 7x lower
than this literature's own reported mean — a large, real discrepancy, but not a like-for-like
comparison; treat as directional evidence that 50s likely understates real elderly pre-movement
delay, not as a directly-transplantable target value.

**Proposed grid (not executed against these benchmarks — see Phase 3)**: `{50, 90, 150, 240, 360}`
seconds.

### Wheelchair — EXPLORATORY (no direct published figure found)

No pre-movement-delay study specific to wheelchair users was found in this search; the one
elderly/disability-adjacent study located explicitly **excluded** wheelchair users. General
qualitative literature on occupants with disabilities describes delay driven by "information
barriers, movement restrictions, and psychological factors," without a quantified median.

**This range is therefore EXPLORATORY, not literature-supported** — it is not derived from any
citation and should not be reported as such. Included only as a plausible sensitivity-search range
for a future dedicated investigation once real data is located.

**Proposed grid (not executed — see Phase 3)**: `{60, 90, 150, 240}` seconds.

### Summary table

| Profile | Confidence | Current default | Literature figure(s) | Grid used |
|---|---|---|---|---|
| Adult | **Literature-supported** (direct building-type match) | 30s | NIST TN1664: mean 28s, SD 11s, range 10–55s | `{15,20,25,30,35,45,55}` — **executed, all 4 buildings** |
| Visitor | Moderately literature-supported | 40s | Hotel/unfamiliar-occupant: 25s–5min, mode 60s | `{25,40,60,90,120}` — proposed, not executed |
| Child | Literature-supported, heterogeneous, different population (drilled schoolchildren) | 45s | School drills: mean 114s (3-6y), 4-166s (6-16y) | `{45,90,120}` — **executed as diagnostic only, 10-story** |
| Elderly | Literature-informed, cross-context (residential, not office) | 50s | Apartment drill: mean 360s, range 0-1500s | `{50,90,150,240,360}` — proposed, not executed |
| Wheelchair | **Exploratory — no citation found** | 60s | none found; disability literature qualitative only | `{60,90,150,240}` — proposed, not executed |

## Phase 3 — Why only Adult_Default was executed against the NIST benchmarks

Verified by direct source inspection (`grep -n 'FixedValue("' scripts/run_nist_*_validation.py`):
**all four NIST validation buildings populate 100% of occupants as `FixedValue("Adult_Default")`.**
No Child/Elderly/Wheelchair/Visitor occupant exists in any of the four benchmark scenarios.

Calibrating any of those four profiles' pre-movement delay against these specific benchmarks is
therefore a **structural no-op**: changing a profile's template that zero occupants in the scenario
resolve to cannot change any simulated outcome. This is not a hypothesis — it follows directly from
`behaviour_profile_resolver.resolver.resolve_profile()` only ever being called with the
`behaviour_profile_id` strings actually present in the scenario's own `OccupantDefinition`.

Rather than run a full 4-building × multi-point grid sweep for Child/Elderly/Wheelchair/Visitor
whose result is analytically predictable (every candidate mean identical to baseline, p-value
undefined/NaN, verdict INCONCLUSIVE — exactly matching the walking-speed campaign's own behavior at
its 1.2 m/s "no-op" grid point), this campaign runs **one confirmatory diagnostic**
(`Child_Default`, 3-point grid, 10-story building only, reduced sample) to empirically verify that
prediction rather than merely assert it. See the raw results JSON's
`Child_Default_diagnostic_only` key for the measured confirmation.
