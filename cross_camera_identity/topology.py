from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple


DEFAULT_WALKING_SPEED_M_PER_S = 1.2  # a typical unhurried adult walking pace
DEFAULT_TRANSITION_TIME_SLACK_FACTOR = 2.0  # min/max window = expected / factor .. expected * factor


@dataclass(frozen=True)
class CameraTransition:

    # Cross-Camera Identity Resolution (ReID Framework) milestone,
    # Phase 4 -- one directed, plausible movement between two cameras.
    # min/max_transition_time bound how long a real person could
    # honestly take to walk that path -- never a single fixed number,
    # since real walking pace varies.
    #
    # expected_transition_time defaults to the midpoint of min/max (the
    # only honest "best guess" available to a caller who only supplied
    # a min/max window, e.g. add_transition() below) -- but a builder
    # with a more precise, genuinely-derived estimate (e.g.
    # build_topology_from_navigation_graph()'s own real
    # distance/walking-speed calculation) may supply it explicitly, so
    # that more precise value is never silently discarded/distorted by
    # re-averaging an asymmetric min/max window back down to a lossier
    # number.

    from_camera_id: str
    to_camera_id: str
    min_transition_time: float
    max_transition_time: float
    expected_transition_time: Optional[float] = None

    def __post_init__(self):

        if self.expected_transition_time is None:

            object.__setattr__(
                self, "expected_transition_time",
                (self.min_transition_time + self.max_transition_time) / 2.0,
            )


class CameraTopology:

    # A lightweight, explicitly-built movement graph over camera_ids --
    # Phase 4's own "lightweight topology model" fallback. Deliberately
    # never hardcodes a building layout itself: a caller populates it
    # from whatever ITS OWN Digital Twin/deployment actually looks like
    # (directly, or via build_topology_from_navigation_graph() below,
    # which derives one honestly from an existing Building's
    # NavigationGraph + Camera.zone_ids when that data is available --
    # Phase 4's own "if camera connectivity information exists: reuse
    # it").

    def __init__(self):

        self._cameras: Set[str] = set()
        self._transitions: Dict[Tuple[str, str], CameraTransition] = {}

    # =====================================================
    # Construction
    # =====================================================

    def add_camera(self, camera_id: str) -> None:

        self._cameras.add(camera_id)

    # =====================================================

    def add_transition(
        self,
        from_camera_id: str,
        to_camera_id: str,
        min_transition_time: float,
        max_transition_time: float,
        bidirectional: bool = True,
        expected_transition_time: Optional[float] = None,
    ) -> None:

        self.add_camera(from_camera_id)
        self.add_camera(to_camera_id)

        self._transitions[(from_camera_id, to_camera_id)] = CameraTransition(
            from_camera_id, to_camera_id, min_transition_time, max_transition_time,
            expected_transition_time=expected_transition_time,
        )

        if bidirectional:

            self._transitions[(to_camera_id, from_camera_id)] = CameraTransition(
                to_camera_id, from_camera_id, min_transition_time, max_transition_time,
                expected_transition_time=expected_transition_time,
            )

    # =====================================================
    # Queries
    # =====================================================

    def is_known_camera(self, camera_id: str) -> bool:

        return camera_id in self._cameras

    def transition(self, from_camera_id: str, to_camera_id: str) -> Optional[CameraTransition]:

        return self._transitions.get((from_camera_id, to_camera_id))

    def possible_destinations(self, camera_id: str) -> Tuple[str, ...]:

        return tuple(sorted(
            to_id for (from_id, to_id) in self._transitions if from_id == camera_id
        ))

    # =====================================================

    def is_plausible_transition(
        self,
        from_camera_id: str,
        to_camera_id: str,
        elapsed_seconds: float,
        default_min_transition_time: float,
        default_max_transition_time: float,
    ) -> Tuple[bool, Optional[float]]:

        # Returns (plausible, expected_transition_time). Three cases,
        # deliberately distinct:
        #
        #   1. This exact (from, to) pair is a registered transition --
        #      the strongest case: plausibility is judged against that
        #      transition's own min/max window.
        #   2. Neither camera is registered in this topology at all --
        #      genuinely open-world (this topology has no information
        #      to rule anything out with), so the CALLER's own default
        #      window is used instead of silently refusing every match.
        #   3. At least one of the two cameras IS registered, but this
        #      specific pair is not a registered transition -- the
        #      topology explicitly does have information here, and it
        #      says "no direct path" -- treated as implausible (closed-
        #      world for cameras the topology actually knows about).

        registered = self.transition(from_camera_id, to_camera_id)

        if registered is not None:

            plausible = registered.min_transition_time <= elapsed_seconds <= registered.max_transition_time
            return plausible, registered.expected_transition_time

        if self.is_known_camera(from_camera_id) or self.is_known_camera(to_camera_id):
            return False, None

        plausible = default_min_transition_time <= elapsed_seconds <= default_max_transition_time
        return plausible, (default_min_transition_time + default_max_transition_time) / 2.0


# =====================================================
# Digital Twin reuse (Phase 4) -- OPTIONAL, additive derivation from an
# existing Building's NavigationGraph + each Camera's own zone_ids.
# Never hardcodes a building layout: every number here is derived from
# the caller's own real navigation_graph/camera_zone_ids, except the
# walking-speed/slack-factor CONVERSION parameters, which are
# configurable assumptions about how fast a person walks, not the
# shape of the building itself.
# =====================================================


def build_topology_from_navigation_graph(
    navigation_graph,
    camera_zone_ids: Mapping[str, Sequence[str]],
    walking_speed_m_per_s: float = DEFAULT_WALKING_SPEED_M_PER_S,
    transition_time_slack_factor: float = DEFAULT_TRANSITION_TIME_SLACK_FACTOR,
) -> CameraTopology:

    # navigation_graph: a navigation.graph.NavigationGraph already built
    # from a real Building (this function only ever calls its public
    # find_neighbors(zone_id) -- never imports the navigation package
    # at module scope, so constructing a plain CameraTopology by hand
    # never requires navigation/models to be involved at all).
    #
    # camera_zone_ids: camera_id -> the zones models.camera.Camera.
    # zone_ids already assigns it (the caller's own Digital Twin data;
    # this function never inspects a Camera object itself, only the
    # plain zone-id strings it's given, keeping this function testable
    # without constructing a real Camera).
    #
    # Two cameras are connected here if any zone of one is either the
    # SAME zone as, or one real NavigationGraph edge away from, any
    # zone of the other -- using that edge's own genuine
    # Edge.walking_distance (meters) to derive an honest transition-time
    # estimate, never a fabricated one.

    topology = CameraTopology()

    camera_ids = list(camera_zone_ids)

    for camera_id in camera_ids:
        topology.add_camera(camera_id)

    for i, camera_a in enumerate(camera_ids):

        zones_a = set(camera_zone_ids[camera_a])

        for camera_b in camera_ids[i + 1:]:

            zones_b = set(camera_zone_ids[camera_b])

            best_distance = _best_zone_distance(navigation_graph, zones_a, zones_b)

            if best_distance is None:
                continue

            expected_seconds = best_distance / walking_speed_m_per_s if best_distance > 0 else 0.0

            min_time = expected_seconds / transition_time_slack_factor
            max_time = max(expected_seconds * transition_time_slack_factor, min_time)

            topology.add_transition(
                camera_a, camera_b, min_time, max_time, bidirectional=True,
                expected_transition_time=expected_seconds,
            )

    return topology


def _best_zone_distance(navigation_graph, zones_a: Set[str], zones_b: Set[str]) -> Optional[float]:

    if zones_a & zones_b:
        # Directly sharing a zone -- no honest walking distance to
        # measure, treated as effectively adjacent (0.0).
        return 0.0

    best: Optional[float] = None

    for zone_a in zones_a:

        for neighbor_node, edge in navigation_graph.find_neighbors(zone_a):

            if neighbor_node.id in zones_b and edge.walking_distance is not None:

                if best is None or edge.walking_distance < best:
                    best = edge.walking_distance

    return best
