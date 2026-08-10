# Empirically Parameterized Proportion-Conditioned Herding

Implementation milestone. Adds `EmpiricalProportionHerdingRouteChoiceStrategy` (`behavior_library/route_choice_strategies.py`) and a dedicated evidence data module (`behavior_library/kinateder_warren_2021_herding_evidence.py`). Additive only: `StaticHerdingRouteChoiceStrategy` is completely unmodified, no production default changed, not registered anywhere as a default strategy.

## 1. Evidence

Kinateder, M. & Warren, W.H. (2021), "Exit choice during evacuation is influenced by both the size and proportion of the egressing crowd," *Physica A* 569, 125746. DOI: 10.1016/j.physa.2021.125746. Three immersive VR experiments, N=45 (15/experiment), crowd proportion (60–100%, tested in 10% steps) and crowd size (10 or 20) experimentally manipulated; a binary logistic mixed model (Table 4) was fit pooled across all three experiments.

## 2. Published Coefficients (transcribed exactly, never re-derived or re-fit)

Reference category: crowd proportion = 60%. Intercept = −0.036 (SE 0.240, p=.882).

| Proportion | Coefficient | p |
|---|---|---|
| 70% | +0.063 | .753 (n.s.) |
| 80% | +0.551 | <.001 |
| 90% | +0.740 | <.001 |
| 100% | +1.190 | <.001 |

Crowd-size=20 interaction (relative to size 10):

| Proportion | Interaction | p |
|---|---|---|
| 80% | −0.593 | <.05 |
| 90% | −0.751 | <.001 |
| 100% | −0.269 | .354 (n.s.) |

No interaction term was reported at 60% or 70% — treated as exactly 0.0, not guessed at any other value.

## 3. Logistic Transformation

`logit(p) = intercept + proportion_coefficient[level] + (size_20_interaction[level] if crowd_size == 20 else 0)`, then `p = 1 / (1 + exp(-logit))`. This is a **fixed-effects-only** approximation — the published mixed model also carries a participant-level random-effects term this implementation does not, and cannot, reproduce. The resulting `p` is a **model-derived probability**, never presented or stored as a raw observed participant frequency.

## 4. Supported Domain and Out-of-Domain Policy

Directly supported: crowd size ∈ {10, 20}, proportion ∈ {60, 70, 80, 90, 100}%. Table 4 codes proportion as a **categorical** factor, not a continuous predictor — there is no published functional form to interpolate between the five tested levels, so none is invented.

- Crowd size ∉ {10, 20} → falls back to the legacy constant `follow_probability` (no evidence about the size dimension anywhere else — the size×proportion interaction is demonstrably non-monotonic, so no snapping between 10 and 20 is attempted).
- Proportion < 60% → falls back to legacy constant (the reference category is 60%; nothing brackets values below it).
- Proportion between two tested levels → snapped to the **nearest** tested level, never interpolated; exact ties break toward the **lower** (more conservative) category.
- Proportion > 100% → structurally impossible given how `majority_count / total_observed_decisions` is constructed, but fails safe (falls back) rather than raising, consistent with every `RouteChoiceStrategy` in this codebase always returning a `RouteChoice`.

## 5. SynEvac Mapping and Its Limitation

`observed crowd proportion := majority_count / total_observed_decisions`, `observed crowd size := total_observed_decisions`, both derived from `context.decisions_so_far` — the exact same construction `StaticHerdingRouteChoiceStrategy` already uses for its own exit tally.

**This is a structural/functional analogue of Kinateder & Warren's experimental variables, not a perceptual equivalent, and this must not be treated as settled.** Three concrete gaps: (1) `decisions_so_far` has no spatial/visibility gating — an occupant registered earlier anywhere in the building counts identically to one standing next to the observer, unlike K&W's simultaneously-visible virtual crowd; (2) `total_observed_decisions` grows with registration order across the whole registration pass, unlike K&W's fixed 10-or-20 per-trial stimulus; (3) `HumanBehaviorLayer.register()` resolves every decision before any simulated movement occurs, so "already resolved" is not "already visibly moving" in wall-clock time. This implementation is therefore **empirically parameterized**, not **perceptually validated**.

## 6. Why This Is Parameterization, Not Calibration

The coefficients above are already published — SynEvac does not need to discover them via its own simulated outcomes. The correct workflow is literature coefficients → implement empirical model (this milestone) → reproduce the published pattern under conditions resembling the study's own structure (the mechanism-fidelity tests, done this milestone) → **only then** consider independent validation. `AutoCalibrationEngine`/`GridSearchStrategy`/`PublishedValueObjective`/Calibration Studio were deliberately not used — running a search against the same data that already produced these coefficients would be circular.

## 7. Why This Is Not Yet Behavioral Validation

Mechanism fidelity testing (this milestone) proves the *code* correctly represents the *published numbers*. It does not prove the *numbers* correctly predict *real building evacuation behavior* — that requires an independent dataset Kinateder & Warren cannot supply about themselves, and a whole-building test this milestone deliberately does not run.

## 8. Independent Validation Requirement

Before any claim of behavioral validation: a genuinely independent dataset (different population, environment, or elicitation method) is needed — Haghani & Sarvi's own coefficients (not fully accessible at the time of the underlying investigation) are a natural candidate. Before any production adoption: resolution of the semantic gap in Section 5, and a whole-building cross-validation this milestone explicitly does not perform.

## Status

MECHANISM IMPLEMENTED: yes. EMPIRICALLY PARAMETERIZED: yes. MECHANISM FIDELITY-TESTED: yes. BEHAVIORALLY VALIDATED: no. WHOLE-BUILDING VALIDATED: no.
