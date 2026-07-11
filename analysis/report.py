from dataclasses import dataclass, field
from typing import Dict, List, Optional

from navigation.validation import ValidationReport


@dataclass
class ExitServiceArea:

    # Every ZONE node whose route to Outside leaves through this
    # specific Exit edge, per exit_service_areas(). served_node_ids
    # is derived from a single distances_from(OUTSIDE) pass, never
    # from a per-zone query -- see BuildingAnalysisEngine.

    exit_edge_id: str
    served_node_ids: List[str]
    distances: Dict[str, float]


@dataclass
class ExitServiceAreaReport:

    by_exit: Dict[str, ExitServiceArea]

    # ZONE nodes with no route to Outside at all -- distinct from
    # graph.validate()'s "isolated_zone"/"zone_without_connections",
    # which this report does not reimplement, only cross-references
    # via ConnectivityReport.
    unserved_node_ids: List[str] = field(default_factory=list)


@dataclass
class TravelDistanceReport:

    # zone_id -> distance to its nearest exit.
    distances: Dict[str, float]

    max_distance: Optional[float]
    max_distance_node_id: Optional[str]

    unreachable_node_ids: List[str] = field(default_factory=list)

    # Only populated when a threshold was passed to
    # max_travel_distance() -- empty, not None, when no threshold was
    # given, so callers never need to branch on None vs [].
    exceeding_threshold_node_ids: List[str] = field(default_factory=list)


@dataclass
class CriticalConnectorFinding:

    # An edge whose removal disconnects one or more currently-
    # reachable-to-Outside nodes. stranded_node_ids is always
    # non-empty -- critical_connectors()/stair_dependency() only ever
    # return findings that qualify, they don't report every edge.

    edge_id: str
    edge_type: str
    stranded_node_ids: List[str]


@dataclass
class ConnectivityReport:

    # Pure composition of the other four analyses plus the Navigation
    # Graph's own validate() -- no connectivity logic is reimplemented
    # here.

    validation: ValidationReport
    travel_distance: TravelDistanceReport
    critical_connectors: List[CriticalConnectorFinding]
    stair_dependencies: List[CriticalConnectorFinding]
