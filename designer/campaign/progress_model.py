from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressModel:

    # One immutable snapshot of a running campaign's live progress --
    # CampaignWorker emits a fresh instance after every scenario index
    # it finishes processing; CampaignWindow only ever displays
    # whatever it is handed, the same "dumb widget, pushed updates"
    # convention SimulationPanel/PerceptionDebugPanel already follow
    # (designer/widgets/simulation_panel.py). Never mutated in place --
    # a new ProgressModel is built for every update, so a caller that
    # kept a reference to a prior snapshot never sees it change under
    # it.

    total: int
    processed: int
    accepted: int
    rejected: int
    remaining: int

    current_scenario_label: str

    average_simulation_time: float
    generation_speed: float
    eta_seconds: float
    elapsed_seconds: float
