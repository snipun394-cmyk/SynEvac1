from typing import Callable, Optional, Protocol, Tuple

from perception.models.building_observation import BuildingObservation

from ai_inference.recommendation import Recommendation

from decision_policy.policy import DecisionPolicy

from live_system.sensor_registry import SensorReadings
from live_system.state_manager import LiveBuildingSnapshot


# =====================================================
# Live Runtime Architecture Cleanup milestone -- LEGACY / NOT PART OF
# THE CURRENT PRODUCTION COMPOSITION PATH.
#
# This module was originally Phase 7's "Integration Layer": four thin
# Protocols plus four concrete adapters wiring perception/ ->
# ai_inference -> decision_policy -> command_center into
# live_system.orchestrator.LiveOrchestrator. That generation has been
# fully superseded by the newer, actually-used composition root:
#
#   live_runtime.factory.build_live_runtime()
#       -> live_system.building_state_gateway.EstimatorBuildingStateGateway
#          (perception, via live_perception/ + sensor_fusion/, not
#          perception.fusion.sensor_fusion.SensorFusion)
#       -> live_system.live_ai_gateway.LiveAIInferenceGateway
#          (AI inference, not ai_inference.predictor.Predictor wired
#          directly)
#       -> live_system.live_advisory_gateway.LiveAdvisoryGateway
#          (advisory generation -- a live incident has no live
#          DecisionPolicy equivalent at all; this is not a like-for-like
#          replacement, it is a different, deliberately live-shaped
#          concept)
#       -> live_system.live_command_center_gateway.LiveCommandCenterDataSource
#          (Command Center reads StateManager.current() on its own
#          timer -- a PULL model, never pushed a frame by the
#          orchestrator the way DashboardCommandCenterGateway once was)
#
# See docs/architecture/live_runtime_architecture_cleanup.md for the
# full before/after graph and the usage audit that confirmed every
# concrete adapter that once lived in this module had zero production
# callers (only tests/test_live_system.py exercised them directly).
# Those four concrete adapters (SensorFusionPerceptionGateway,
# PredictorAIInferenceGateway, GeneratePolicyDecisionPolicyGateway,
# DashboardCommandCenterGateway) and their two adapter-specific type
# aliases (FeatureRowBuilder, DecisionInputsBuilder) were deleted by
# that milestone -- not merely deprecated -- because deletion's own
# preconditions were all genuinely met: zero legitimate callers, no
# serialization dependency, and no test documented an intentional
# compatibility contract for the ADAPTERS themselves (only for
# LiveOrchestrator's own generic dispatch behavior, which uses
# hand-written stubs in tests/test_live_system.py, not these classes).
#
# What remains below -- the three Protocols and RecommendationBuilder --
# is kept ONLY because live_system.orchestrator.LiveOrchestrator's own
# constructor signature still type-annotates its perception_gateway/
# decision_policy_gateway/command_center_gateway/recommendation_builder
# parameters against them (LiveOrchestrator itself is explicitly out of
# scope for this milestone -- "do not redesign LiveOrchestrator"). Every
# one of those four parameters defaults to None and is never populated
# by build_live_runtime() in production -- they remain a generic,
# unused-today extension point on LiveOrchestrator, not a second
# production composition path. Do not write new production code against
# this module; compose against live_runtime/factory.py instead.
#
# Shadow-Mode Predictive AI Integration milestone, Phase 1 -- the
# Freeze Review's own second cleanup item: AIInferenceGateway (the
# generation-1 Phase-7 inference seam, `predict(snapshot) -> Mapping[str,
# Prediction]`) has been RETIRED (deleted, not merely deprecated) here,
# and its live_system.orchestrator.LiveOrchestrator parameter/attribute/
# dispatch removed too -- "ensure only one inference gateway remains"
# now that live_system.live_ai_gateway.LiveAIInferenceGateway is the
# actually-used, actually-wired inference seam (see that module's own
# docstring). Unlike PerceptionGateway/DecisionPolicyGateway/
# CommandCenterGateway/RecommendationBuilder above, AIInferenceGateway
# had a genuine successor to converge on, not just an absence of a
# production caller -- keeping both indefinitely would have meant two
# gateway-shaped things answering "how does LiveOrchestrator get an AI
# prediction," a real "which one do I use" risk the Freeze Review
# explicitly flagged. state_manager.StateManager.update_ai_predictions()/
# LiveBuildingSnapshot.ai_predictions (the OLDER Mapping[str, Prediction]
# storage this gateway used to populate) are left untouched -- they are
# passive data storage, not a gateway, and removing a snapshot field is
# a materially different, larger change than retiring an unused seam;
# see docs/architecture/shadow_mode_prediction.md.
# =====================================================


class PerceptionGateway(Protocol):

    def collect(self, readings: SensorReadings, time: float) -> BuildingObservation: ...


class DecisionPolicyGateway(Protocol):

    def evaluate(self, snapshot: LiveBuildingSnapshot) -> Optional[DecisionPolicy]: ...


class CommandCenterGateway(Protocol):

    def notify(self, snapshot: LiveBuildingSnapshot) -> None: ...


RecommendationBuilder = Callable[[LiveBuildingSnapshot], Tuple[Recommendation, ...]]
