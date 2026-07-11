from typing import Optional

from hazard.node_state import HazardNodeState

from hazard_evolution.contribution import HazardContribution
from hazard_evolution.source import HazardSource

from tenability.scoring import SmokeVisibilityTenabilityScorer, TenabilityScorer


class TenabilityModel(HazardSource):

    # Unlike FireGrowthModel/SmokePropagationModel -- primary producers
    # that simulate their own physics from absolute elapsed time and
    # deliberately never read previous_snapshot -- TenabilityModel has
    # no physics of its own to simulate. It is a pure *derived* source:
    # it reads whatever hazard readings (smoke_level, visibility)
    # another source already wrote into previous_snapshot, and
    # produces a life-safety hazard_score reflecting how survivable
    # each already-touched node currently is. This is exactly the use
    # of previous_snapshot the HazardSource interface was designed to
    # support (see hazard_evolution/source.py) but that neither
    # existing model exercises.
    #
    # A practical consequence of reading previous_snapshot: this
    # model's opinion for step N reflects conditions as merged at the
    # end of step N-1, one step behind whatever fresh values Fire/
    # Smoke compute for step N itself. This needs no special handling
    # -- DefaultHazardMergeStrategy takes max(hazard_score) across
    # every source that touches a node in the same step, so this
    # model's derived score simply joins that same worst-case pool
    # alongside Fire/Smoke's own fresh hazard_score.
    #
    # Implements HazardSource exactly like FireGrowthModel/
    # SmokePropagationModel, so it plugs into HazardEvolutionEngine
    # with zero engine changes. Produces HazardContribution objects
    # only, and never node ids previous_snapshot doesn't already know
    # about -- this model invents no new nodes, it only re-scores ones
    # some other source already reported on. Never touches
    # edge_states. Never modifies previous_snapshot -- only ever reads
    # it via HazardSnapshot's own total accessors.
    #
    # No CO, CO2, FED, radiant heat, toxicity, oxygen depletion, or
    # evacuation-fatality modeling -- all deliberately absent; see
    # tenability/scoring.py for the one, intentionally simple V1
    # scoring policy this delegates to.

    def __init__(self, scorer: Optional[TenabilityScorer] = None):

        self.scorer = scorer or SmokeVisibilityTenabilityScorer()

    # =====================================================

    def propose(self, previous_snapshot, time, dt) -> HazardContribution:

        node_states = {}

        for node_id, state in previous_snapshot.node_states.items():

            score = self.scorer.score(state)

            if score <= 0.0:
                continue

            node_states[node_id] = HazardNodeState(hazard_score=score)

        return HazardContribution(node_states=node_states)
