# SynEvac Scenario Campaign — Feasibility Work: Project Summary

**Purpose of this document:** a self-contained explanation of what this piece of work is, why it started, what's been done, what was found, and where it stands right now — written so someone with zero prior context (including another AI assistant being asked for a second opinion) can follow it without needing to read the codebase or any prior report first.

---

## 1. What is SynEvac, and what's a "Scenario Campaign"?

SynEvac is a building-evacuation simulation and AI system. Part of it — the **Scenario Engine** — can generate large batches ("campaigns") of randomized fire-evacuation scenarios for a given building, so the rest of the system (simulation, AI models, etc.) has realistic test/training data to run against.

A campaign works like this:
1. A user (via a GUI called **Campaign Studio**) designs a **Scenario Definition** — rules like "occupants might be in these rooms," "the fire might start in one of these rooms," "this door is normally open but might be sampled locked 10% of the time," etc. These rules are expressed as **probability distributions**, not fixed facts.
2. The user asks for, say, 10 scenarios.
3. For each of the 10 "slots," the system randomly samples a candidate scenario from those distributions, then checks whether it's actually valid (e.g., "can everyone in this scenario actually reach an exit?"). If a candidate fails that check, it's thrown away and re-sampled, up to a configurable retry limit (**max_attempts**, default 10 tries per slot).

Two separate pieces of code are involved, and — this matters a lot for everything below — they are **deliberately kept separate**:
- The **Generator**: samples a candidate scenario from the rules. It has no idea whether the result is "sensible" — it just draws numbers from the configured distributions.
- The **Validator**: takes one already-built candidate and checks it, including whether every occupant can actually reach an exit (a graph-reachability check over the building's rooms/doors/exits, called `navigation_validation`).

## 2. The problem that started this whole thread

A user ran a campaign and got: **10 requested scenarios, 0 accepted, 100 wasted generation attempts** (10 slots × 10 retries each), all rejected for the same reason: an occupied room's only path to an exit ran straight through wherever the fire happened to start.

That's not a crash or a bug in the strict sense — the Validator was correctly rejecting genuinely-unsafe scenarios. The real problem was **architectural**: nothing in the pipeline ever asked "is this configuration even *capable* of producing a valid scenario?" *before* burning through all 100 random attempts. The Generator and Validator, by design, never talk to each other ahead of time.

## 3. Root-cause investigation (before any code was touched)

We traced the exact code path and confirmed:
- The Generator samples fire location, occupant placement, and every door/exit/stair state **completely independently of each other and of the building's layout** — on purpose, per the existing architecture (it's not supposed to "know" about reachability).
- The Validator is the *only* place reachability is ever checked, and only *after* a full candidate is built.
- The existing "pre-flight" checks (run once before a campaign starts) only checked structural things like "do these door IDs actually exist" — never "can anyone actually escape."

**Root cause, in one sentence:** the user's configuration described a building/fire-rule combination with an extremely small (in this case, provably *zero*) chance of ever producing a passable scenario, and nothing checked for that ahead of time — so the system had no choice but to discover it the hard way, 100 times over.

This was written up in two investigation reports (no code changes yet):
- `scenario_campaign_zero_generation_investigation.txt`
- `scenario_campaign_feasibility_preflight_investigation.txt` (this one designed the fix, still without writing code)

## 4. The fix we designed: a new "feasibility preflight"

Rather than changing the Generator (which would break the deliberate separation of concerns) or weakening the Validator (which would let genuinely unsafe scenarios through), we designed a **third, new, read-only analysis step** that runs once, before a campaign starts generating anything:

> Given the building's layout and the Scenario Definition's rules, can we prove — *before* sampling a single candidate — whether this configuration has a realistic chance of ever producing a valid scenario?

Key ground rules we set for ourselves and stuck to throughout:
- Never make the Generator "reachability-aware."
- Never weaken or bypass what the Validator considers a valid scenario.
- Never silently change a user's configured probabilities.
- Reuse existing graph/reachability code rather than building a second, possibly-inconsistent copy of it.

## 5. Phase 1: "prove zero feasibility cheaply" — implemented and verified

We built a new package, `campaign_feasibility/`, wired into the existing "pre-flight" step in Campaign Studio. It can currently **prove** (not guess) two specific ways a campaign is guaranteed to fail:

1. **A room is unreachable even in the best possible case** — even if every door/exit/stair happened to be open, that room still couldn't reach an exit. No amount of random luck can fix this.
2. **The fire is guaranteed to block the only escape route** — every single room the fire is allowed to start in would cut off some occupied room's only path out. There's no "safe" fire location at all.

Either of these now **blocks the campaign immediately** with a clear explanation, instead of silently wasting attempts.

We implemented it, wrote 21 new focused tests plus ran 443 existing tests (0 failures), and then did a **separate, independent verification pass** specifically to check our own work wasn't just "claimed" to work — re-derived every claim from scratch (git diffs, line-by-line code comparisons, monkey-patched proof that generation genuinely never starts when blocked, etc.). No defects found.

Reports: `scenario_campaign_feasibility_preflight_phase1_implementation_report.txt`, `scenario_campaign_feasibility_phase1_final_verification.txt`.

## 6. A "wait, it's still broken?" scare — investigated, resolved

After Phase 1 was in place, a report came in that the *exact same* symptom (0 accepted, 100 wasted attempts) happened again, with the pre-flight screen saying **"Pre-flight checks passed."** That looked like a contradiction — either Phase 1 wasn't really running, or it had a bug.

We investigated this rigorously (not assuming either way) and found:
- The real configuration had roughly a **99.667%** chance (not 100%) that the sampled fire location would block the occupied room's escape route. Since it wasn't *literally* 100%, Phase 1 correctly did **not** block the campaign — it only issued a non-fatal **warning**.
- We calculated the actual math: with a 99.667% per-attempt failure chance and only 10 retries per slot, the probability that **all 10 requested slots would fail completely** was about **71.6%** — i.e., total failure was the *more likely* outcome, not a fluke or a bug.
- We reproduced the *exact* reported numbers (0 accepted, 100/100 rejected, same failure code) purely by constructing this 99.667% scenario and running it through the real, unmodified Phase 1 code. Match confirmed, field for field.
- We also found a real, separate contributing issue: the pre-flight screen's warning message is real and does get shown, but it's appended *after* a line that says "Pre-flight checks passed," which is easy to skim past and misread as "nothing to worry about."
- As due diligence, we also checked for boring-but-real explanations (wrong code running, stale cached files, etc.) and found two old, unrelated leftover project copies on disk that *could* theoretically produce the old behavior if someone accidentally launched the app from the wrong folder — flagged as an open, unconfirmed possibility, not a proven cause.

**Verdict: Phase 1 is working exactly as designed.** It correctly distinguishes "provably impossible" (which it blocks) from "extremely unlikely but not impossible" (which it currently only warns about). The warning-worthy case just isn't *quantified* yet — which is exactly what Phase 2 is for.

Report: `scenario_campaign_phase1_runtime_contradiction_investigation.txt`.

## 7. Phase 2: quantifying the "uncertain middle" — investigation just started, NOT finished

Phase 1 deliberately only handles the *provable* extremes. It does **not** currently calculate an actual risk percentage for the vast middle ground — e.g., a door that's 99% likely to be locked, or a fire zone that's merely *very* likely (not certain) to be lethal. Those cases just get a generic "this might be risky" warning today, with no number attached.

The goal of Phase 2 is to close that gap: given the *exact* configured probabilities (not guesses), calculate things like:
- What's the actual chance a single generated candidate will pass?
- What's the chance a whole 10-scenario campaign produces zero usable scenarios?
- How many scenarios can the user realistically expect to get out of this configuration?
- Should this be a hard block, a "are you sure?" warning, a plain warning, or just proceed — depending on how bad the number is?

**Current status: I was in the middle of the investigation phase for this (no code written, nothing implemented) when this summary was requested.** So far I had only:
- Confirmed that stair-state rules are fully supported by the UI (so they need to be covered too).
- Confirmed, with a live test, that a *purely door-lock-driven* high-risk case (no fire risk at all) currently gets only a vague warning from Phase 1, with **no probability number at all** — directly confirming that fire-origin risk isn't the only thing that needs this treatment; door/exit/stair risk needs it too.

Nothing else has been decided or written yet. The next steps (not yet done) are to work out exactly how to calculate these probabilities correctly (reusing the real sampling rules, not approximating them), decide when exact math is feasible versus when an estimate is needed, decide the exact user-facing behavior/wording, and write it all up as a plan *before* touching any code.

## 8. Also worth knowing: the ML/RL work is on hold

Before any of this began, there was a separate, larger thread of work investigating how to connect SynEvac's machine-learning prediction models to its reinforcement-learning decision agent. That work was **explicitly paused** by request to deal with this campaign-generation reliability problem first, since it's foundational (data generation needs to work before it's worth building ML/RL on top of it). It hasn't been touched since; nothing there has changed.

## 9. TL;DR for a quick skim

- **Problem:** Scenario Campaigns could silently burn their whole retry budget and produce nothing, with no warning ahead of time.
- **Fix so far (Phase 1, done and verified):** a new, additive "feasibility preflight" that *proves* and blocks the worst (0%-chance) cases before wasting any attempts, without touching the Generator or weakening the Validator.
- **A scare that turned out fine:** a repeat failure looked like Phase 1 was broken; investigation proved it was working exactly as intended — it just doesn't yet *quantify* risk that's severe-but-not-certain.
- **Next (Phase 2, investigation started, not finished, nothing implemented):** teach the preflight to calculate an actual risk percentage for that in-between zone (covering fire *and* door/exit/stair rules together) and translate it into a clear, campaign-level answer for the user, before they hit "Start."
- **Unrelated, paused:** the ML/RL integration work is on hold until this is settled.
