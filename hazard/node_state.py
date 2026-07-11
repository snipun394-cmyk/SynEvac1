from dataclasses import dataclass

from hazard.severity import HazardSeverity


@dataclass(frozen=True)
class HazardNodeState:

    # One node's environmental conditions at the moment a HazardSnapshot
    # was taken -- keyed externally by Node.id (see HazardSnapshot),
    # never stored on Node itself, for the same reason Node/Edge already
    # document: the Navigation Graph is rebuilt from the Building on
    # every change, so nothing on a Node instance would survive that.
    #
    # hazard_score is the ground truth a producer (Fire, Smoke,
    # Detectors, RL) writes -- normalized to [0.0, 1.0], 0.0 clear,
    # 1.0 untenable. severity is deliberately NOT a sibling field a
    # producer sets independently; it is derived from hazard_score
    # through HazardSeverity.from_score() below, the same "derived,
    # never stored redundantly" convention Node/Edge's own engineering
    # properties already follow. This keeps hazard_score lossless for
    # future producers/consumers while giving Human Behavior/reporting/
    # UI a single, shared classification instead of each reinventing
    # its own thresholds.

    smoke_level: float = None
    visibility: float = None
    temperature: float = None

    hazard_score: float = 0.0

    # =====================================================

    @property
    def severity(self) -> HazardSeverity:

        return HazardSeverity.from_score(self.hazard_score)
