from typing import Iterable, Optional, Protocol

from dynamic_signage.models import DynamicSignageSnapshot
from dynamic_signage.planner import DynamicSignagePlanner


# =====================================================
# Live Dynamic Evacuation Signage milestone -- the seam LiveOrchestrator
# uses to reach dynamic_signage/, mirroring live_system.evacuation_
# guidance_gateway's own "thin Protocol + real adapter" shape exactly.
# =====================================================


class EvacuationSignageGateway(Protocol):

    def compute(
        self, time: float, guidance_snapshot=None, signs: Iterable[object] = (),
    ) -> Optional[DynamicSignageSnapshot]: ...


# =====================================================


class EngineEvacuationSignageGateway:

    # The real adapter -- never allowed to raise out of compute() and
    # crash the live cycle, exactly the same discipline every sibling
    # Engine*Gateway in this module already established. sign_manager
    # is forwarded through unchanged, duck-typed (only .all_signs() is
    # ever called) -- this gateway never imports sign_manager/ itself
    # either.

    def __init__(self, planner: DynamicSignagePlanner, sign_manager=None):

        self._planner = planner
        self._sign_manager = sign_manager

    # =====================================================

    def compute(
        self, time: float, guidance_snapshot=None, signs: Optional[Iterable[object]] = None,
    ) -> Optional[DynamicSignageSnapshot]:

        try:

            if signs is None:
                signs = self._sign_manager.all_signs() if self._sign_manager is not None else ()

            return self._planner.compute(time, guidance_snapshot, signs)

        except Exception:  # noqa: BLE001 -- an unexpected signage failure must never crash the live cycle

            return None
