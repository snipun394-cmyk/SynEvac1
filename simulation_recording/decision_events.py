from typing import Any, Dict, Tuple

from serialization.json_reader import JsonReader
from serialization.json_writer import JsonWriter


# =====================================================
# Persists human_decision_engine.events.DecisionEvent.to_dict()'s own
# plain-dict shape -- that shape already exists, already computed, at
# every campaign scenario's export time (designer/campaign/
# campaign_worker.py, research_framework/runner.py); before this
# milestone it was folded into Dataset Builder's *_Count columns and
# then discarded, never written to disk as its own ordered, replayable
# event log. This module adds no new event vocabulary and performs no
# decision-making of its own -- it is a save/load pair only, following
# the same JsonWriter/JsonReader convention scenario_storage/storage.py
# and simulation_recording.occupant_routes already use.
# =====================================================


def save_decision_events(events: Tuple[Dict[str, Any], ...], path: str) -> None:

    JsonWriter.write(path, list(events))


def load_decision_events(path: str) -> Tuple[Dict[str, Any], ...]:

    return tuple(JsonReader.read(path))
