from typing import List, Optional

from hazard.snapshot import HazardSnapshot

from hazard_evolution.merge_strategy import DefaultHazardMergeStrategy, HazardMergeStrategy
from hazard_evolution.source import HazardSource


class HazardEvolutionEngine:

    # The one interface that transforms HazardSnapshot(t) into
    # HazardSnapshot(t + dt) -- HazardSnapshot stays immutable
    # throughout: evolve() never modifies the snapshot it's given, it
    # only ever returns a brand-new one. Every registered HazardSource
    # is consulted for its opinion; every conflict between sources
    # about the same node/edge is resolved entirely through the
    # injected HazardMergeStrategy -- this class embeds no merge policy
    # and no hazard-specific physics of its own, only orchestration.
    # That is what "independent of any specific hazard source" means
    # in practice: this file has zero knowledge of fire, smoke,
    # detectors, cameras, or any other producer, and needs none to add
    # one -- see hazard_evolution.source.HazardSource for how a future
    # producer plugs in without any change here.

    def __init__(
        self,
        sources: Optional[List[HazardSource]] = None,
        merge_strategy: Optional[HazardMergeStrategy] = None,
    ):

        self.sources = sources or []
        self.merge_strategy = merge_strategy or DefaultHazardMergeStrategy()

    # =====================================================

    def evolve(self, snapshot: HazardSnapshot, time: float, dt: float) -> HazardSnapshot:

        contributions = [
            source.propose(snapshot, time, dt)
            for source in self.sources
        ]

        node_states = dict(snapshot.node_states)
        edge_states = dict(snapshot.edge_states)

        for node_id in self._touched_ids(contributions, "node_states"):

            opinions = [
                contribution.node_state(node_id)
                for contribution in contributions
                if contribution.node_state(node_id) is not None
            ]

            node_states[node_id] = self.merge_strategy.merge_node_states(opinions)

        for edge_id in self._touched_ids(contributions, "edge_states"):

            opinions = [
                contribution.edge_state(edge_id)
                for contribution in contributions
                if contribution.edge_state(edge_id) is not None
            ]

            edge_states[edge_id] = self.merge_strategy.merge_edge_states(opinions)

        return HazardSnapshot(
            timestamp=time + dt,
            node_states=node_states,
            edge_states=edge_states,
        )

    # =====================================================

    def _touched_ids(self, contributions, attr_name):

        # Every id at least one source proposed something for this
        # step -- ids absent from every contribution are left as-is in
        # node_states/edge_states above (carry-forward), never visited
        # here at all.

        touched = set()

        for contribution in contributions:
            touched.update(getattr(contribution, attr_name).keys())

        return touched
