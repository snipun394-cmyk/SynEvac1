# =====================================================
# AI Research Framework -- a controlled-experiment layer over the
# already-frozen SynEvac pipeline (scenario_definition/scenario_
# generator/scenario_pipeline/scenario_runner/behaviour_profile_
# resolver/simulation_runtime/dataset_builder/ground_truth/decision_
# policy/ai_training/rl_training/campaign_analytics). Nothing here
# modifies the simulator or any existing package -- every module below
# only calls already-public APIs of those packages, the same "restate
# an orchestration sequence, never reimplement the logic it calls"
# discipline designer.campaign.campaign_worker.CampaignWorker itself
# already established (its own module docstring explains exactly this
# pattern; research_framework.runner mirrors that worker's own
# generate -> simulate -> export sequence, adding only the one seam
# (an injectable registration strategy) that worker has no hook for).
#
# This package answers research QUESTIONS -- it deliberately builds no
# new simulation capability. Every phase (Experiment Definition,
# Campaign Runner, Statistical Analysis, Ablation Studies, Sensitivity
# Analysis, Publication Figures, Research Report) is either a thin
# orchestration wrapper over an existing package, or a self-contained
# statistics/plotting utility with no simulation logic of its own.
# =====================================================
