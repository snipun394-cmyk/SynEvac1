from typing import Callable, List, Optional

from live_system.state_manager import LiveBuildingSnapshot


CycleFn = Callable[[float], LiveBuildingSnapshot]


class UpdateLoop:

    # The continuous execution loop -- Phase 4's own scope: drive
    # cycle_fn (live_system.orchestrator.LiveOrchestrator.run_cycle)
    # repeatedly, advancing time by interval_seconds each call. Step-
    # driven, not thread/asyncio-based, mirroring
    # simulation_runtime.SimulationRuntime.tick()/run() and
    # simulation_interactive.InteractiveSimulation.step()'s own
    # established "one method advances exactly one step; a thin loop
    # calls it repeatedly" pattern -- this is what keeps a driven
    # UpdateLoop fully deterministic under mocked sensor inputs (this
    # phase's own hard requirement): no wall-clock sleeping, no
    # scheduler, no background thread anywhere in this class.
    #
    # interval_seconds is "the loop frequency should be configurable"
    # (Phase 4): it is read by run_for()/next_time() to compute each
    # successive cycle's timestamp, and by run_realtime() (for a real
    # deployment that wants actual wall-clock pacing) as how long to
    # sleep between cycles -- this class never calls time.sleep()
    # itself, only the sleep_fn a real caller injects, so importing
    # this module never pulls in real timing/threading machinery.

    def __init__(self, cycle_fn: CycleFn, interval_seconds: float = 1.0):

        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be > 0, got {interval_seconds!r}")

        self.cycle_fn = cycle_fn
        self.interval_seconds = interval_seconds

        self.cycle_count = 0
        self.last_run_time: Optional[float] = None

    # =====================================================

    def tick(self, time: float) -> LiveBuildingSnapshot:

        # Exactly one cycle, at a caller-supplied time -- the primitive
        # every other method here is built from.

        snapshot = self.cycle_fn(time)

        self.cycle_count += 1
        self.last_run_time = time

        return snapshot

    # =====================================================

    def next_time(self) -> float:

        # The time the *next* tick() would run at if driven by
        # run_for()/run_realtime() rather than an explicitly chosen
        # time -- last_run_time + interval_seconds, or 0.0 for the very
        # first cycle.

        if self.last_run_time is None:
            return 0.0

        return self.last_run_time + self.interval_seconds

    # =====================================================

    def run_for(self, iterations: int, start_time: Optional[float] = None) -> List[LiveBuildingSnapshot]:

        # Deterministic, sleep-free: runs `iterations` cycles back to
        # back, each interval_seconds after the last, starting from
        # start_time (defaulting to wherever next_time() already is).
        # This is the method tests use to exercise "continuous update
        # loop" behavior without any real elapsed wall-clock time.

        if iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {iterations!r}")

        time = start_time if start_time is not None else self.next_time()
        snapshots = []

        for _ in range(iterations):

            snapshots.append(self.tick(time))
            time += self.interval_seconds

        return snapshots

    # =====================================================

    def run_realtime(
        self,
        sleep_fn: Callable[[float], None],
        time_fn: Callable[[], float],
        stop_condition: Callable[[], bool],
    ) -> None:

        # The one real-deployment driver: sleep_fn/time_fn are injected
        # (e.g. time.sleep/time.monotonic in production code that wires
        # this up), never imported here directly -- keeping this class
        # itself free of any real timing dependency, per this phase's
        # "interfaces and dependency injection only" rule, and keeping
        # every test of this loop's actual cycling logic free to use
        # run_for() instead, with zero real elapsed time.

        while not stop_condition():

            self.tick(time_fn())
            sleep_fn(self.interval_seconds)
