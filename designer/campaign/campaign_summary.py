from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CampaignSummary:

    # The one result CampaignWorker ever produces at the end of a run
    # (whether it ran to completion or was cancelled) -- mirrors
    # PipelineResult/BatchPipelineResult's own "always return a
    # structured result describing what happened" convention
    # (scenario_pipeline/result.py) rather than a bare boolean.

    total_requested: int
    total_generated: int
    accepted: int
    rejected: int

    average_evacuation_time: Optional[float]
    average_simulation_duration: float

    output_directory: str

    cancelled: bool = False

    # Requirement 7 -- populated only when accepted == 0 and Diagnostics
    # Mode was enabled (designer/campaign/diagnostics.py::
    # explain_total_rejection()); None otherwise, never a fabricated
    # explanation when there is nothing to explain or no diagnostics
    # data was collected to explain it from.
    rejection_explanation: Optional[str] = None
