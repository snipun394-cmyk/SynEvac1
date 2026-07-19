import os
import tempfile
import unittest

from models.building import Building
from models.exit import Exit
from models.floor import Floor
from models.zone import Zone

from scenario_definition import (
    EngineeringConstraints,
    FireDefinition,
    FixedValue,
    OccupantDefinition,
    ScenarioDefinition,
    UniformRange,
    WeightedOptions,
)
from scenario_definition.firefighter_definition import FirefighterDeploymentDefinition

from research_framework.experiment_spec import (
    ExperimentSpec,
    apply_overrides,
    make_arm,
    override_detector_failure_rate,
    override_fire_growth_time,
    override_firefighter_team_size,
    override_helping_probability,
    override_occupant_population,
    override_profile_mix,
)
from research_framework.runner import ArmRunResult, run_arm, run_experiment, run_scenario_artifacts, train_and_evaluate_rl_for_arm
from research_framework.statistics import (
    compare_arms,
    confidence_interval,
    distribution_shift,
    effect_size_cohens_d,
    feature_sensitivity,
    one_way_anova,
    paired_comparison,
)
from research_framework.ablation import (
    DEFAULT_CAPABILITIES,
    NO_PERCEPTION_NOTE,
    build_ablation_arms,
    run_ablation_study,
    run_no_rl_ablation,
)
from research_framework.sensitivity import run_parameter_sweep
from research_framework.figures import (
    save_confusion_matrix,
    save_evacuation_time_distribution,
    save_feature_importance,
    save_pr_curve,
    save_rl_reward_curve,
    save_roc_curve,
    save_training_curve,
)
from research_framework.report import generate_research_report, write_research_report

from ai_training.experiment import ExperimentConfig
from ai_training.models.base import build_classifier
from rl_training import TrainerConfig


# =====================================================
# Shared fixtures -- a single-zone-with-one-exit Building/Definition
# pair, mirroring tests.test_scenario_pipeline's own known-good,
# validator-accepted fixture shape exactly (allowed_fire_profiles and
# exit_state_distribution/min_open_exits are both required for
# scenario_validator to accept a candidate -- an otherwise-reasonable
# Definition missing either is rejected every time, MAX_ATTEMPTS_
# EXCEEDED).
# =====================================================


def make_building():

    floor = Floor(
        name="Ground", id="floor-1",
        zones=[Zone(id="zone-1", name="Lobby", x=0.0, y=0.0, width=10.0, height=8.0)],
        doors=[], exits=[Exit(id="exit-1", zone_id="zone-1")],
    )

    return Building(name="Test Building", id="building-1", floors=[floor])


def make_definition(**overrides):

    defaults = dict(
        fire=FireDefinition(
            growth_parameter_distribution=UniformRange(100.0, 400.0),
            allowed_ignition_zone_ids={"zone-1"}, allowed_fire_profiles={"Electrical"},
        ),
        engineering=EngineeringConstraints(
            exit_state_distribution={"exit-1": FixedValue(True)}, min_open_exits=1,
        ),
        occupant=OccupantDefinition(
            occupancy_distribution={"zone-1": FixedValue(4)},
            behaviour_profile_distribution={
                "zone-1": WeightedOptions(
                    {"Adult_Default": 0.5, "Wheelchair_Default": 0.25, "Elderly_Default": 0.25},
                ),
            },
            assistance_pairing_probability=0.5,
        ),
        firefighter=FirefighterDeploymentDefinition(
            team_count_distribution=FixedValue(1), team_size_distribution=FixedValue(1),
            entry_zone_ids=("zone-1",), rescue_assignment_probability=0.5,
        ),
    )
    defaults.update(overrides)

    return ScenarioDefinition(**defaults)


# =====================================================
# Phase 1
# =====================================================


class ExperimentSpecTests(unittest.TestCase):

    def test_override_helping_probability(self):

        definition = make_definition()
        updated = apply_overrides(definition, (override_helping_probability(0.0),))

        self.assertEqual(updated.occupant.assistance_pairing_probability, 0.0)
        self.assertEqual(definition.occupant.assistance_pairing_probability, 0.5)  # never mutated

    def test_override_firefighter_team_size(self):

        definition = make_definition()
        updated = apply_overrides(definition, (override_firefighter_team_size(team_count=0),))

        self.assertEqual(updated.firefighter.team_count_distribution, FixedValue(0))

    def test_override_profile_mix(self):

        definition = make_definition()
        updated = apply_overrides(
            definition, (override_profile_mix(["zone-1"], {"Adult_Default": 1.0}),),
        )

        distribution = updated.occupant.behaviour_profile_distribution["zone-1"]
        self.assertEqual(dict(distribution.weights), {"Adult_Default": 1.0})

    def test_override_fire_growth_time(self):

        definition = make_definition()
        updated = apply_overrides(definition, (override_fire_growth_time(250.0),))

        self.assertEqual(updated.fire.growth_parameter_distribution, FixedValue(250.0))

    def test_override_detector_failure_rate(self):

        definition = make_definition()
        updated = apply_overrides(definition, (override_detector_failure_rate(["det-1"], 0.3),))

        distribution = updated.engineering.detector_state_distribution["det-1"]
        self.assertAlmostEqual(dict(distribution.weights)["FAILED"], 0.3)

    def test_override_occupant_population(self):

        definition = make_definition()
        updated = apply_overrides(
            definition, (override_occupant_population(occupancy_distribution={"zone-1": FixedValue(10)}),),
        )

        self.assertEqual(updated.occupant.occupancy_distribution["zone-1"], FixedValue(10))

    def test_experiment_spec_rejects_duplicate_arm_names(self):

        with self.assertRaises(ValueError):
            ExperimentSpec(
                name="dup", base_definition=make_definition(), base_definition_id="def-1",
                base_building=make_building(), arms=(make_arm("a"), make_arm("a")),
                samples_per_arm=1, master_seed=1, output_root=".",
            )

    def test_experiment_spec_rejects_no_arms(self):

        with self.assertRaises(ValueError):
            ExperimentSpec(
                name="empty", base_definition=make_definition(), base_definition_id="def-1",
                base_building=make_building(), arms=(), samples_per_arm=1, master_seed=1, output_root=".",
            )

    def test_arm_building_falls_back_to_experiment_base_building(self):

        building = make_building()
        spec = ExperimentSpec(
            name="s", base_definition=make_definition(), base_definition_id="def-1",
            base_building=building, arms=(make_arm("a"),), samples_per_arm=1, master_seed=1,
            output_root=".",
        )

        self.assertIs(spec.building_for(spec.arms[0]), building)


# =====================================================
# Phase 2
# =====================================================


class RunnerTests(unittest.TestCase):

    def test_run_scenario_artifacts_writes_the_training_dataset_layout(self):

        from scenario_pipeline import run_pipeline

        building = make_building()
        definition = make_definition()

        with tempfile.TemporaryDirectory() as tmp:

            result = run_pipeline(definition, "def-1", building, seed=123)
            self.assertTrue(result.accepted)

            scenario_id = result.scenario.metadata.scenario_id
            run_scenario_artifacts(result.scenario, building, tmp, dt=5.0, registration="population")

            self.assertTrue(os.path.exists(os.path.join(tmp, "datasets", scenario_id, "scenario_features.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "datasets", scenario_id, "simulation_outcomes.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "timelines", scenario_id, "timeline.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "ground_truth", scenario_id, "ground_truth.json")))
            self.assertTrue(
                os.path.exists(os.path.join(tmp, "decision_policy", scenario_id, "decision_policy.json")),
            )

    def test_run_scenario_artifacts_rejects_unknown_registration(self):

        from scenario_pipeline import run_pipeline

        building = make_building()
        definition = make_definition()
        result = run_pipeline(definition, "def-1", building, seed=1)

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_scenario_artifacts(result.scenario, building, tmp, registration="not_a_strategy")

    def test_run_arm_produces_a_loadable_campaign(self):

        building = make_building()
        definition = make_definition()

        with tempfile.TemporaryDirectory() as tmp:

            spec = ExperimentSpec(
                name="exp", base_definition=definition, base_definition_id="def-1",
                base_building=building, arms=(make_arm("baseline"),),
                samples_per_arm=3, master_seed=42, output_root=tmp,
            )
            arm_result = run_arm(spec, spec.arms[0])

            self.assertEqual(arm_result.accepted_count, 3)
            self.assertEqual(len(arm_result.evacuation_times()), 3)

            dataset = arm_result.load_dataset()
            self.assertEqual(len(dataset), 3)

    def test_run_experiment_across_two_arms_is_deterministic(self):

        building = make_building()
        definition = make_definition()

        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:

            def _build_spec(root):
                return ExperimentSpec(
                    name="exp", base_definition=definition, base_definition_id="def-1",
                    base_building=building,
                    arms=(make_arm("baseline"), make_arm("no_help", override_helping_probability(0.0))),
                    samples_per_arm=2, master_seed=99, output_root=root,
                )

            first = run_experiment(_build_spec(tmp1))
            second = run_experiment(_build_spec(tmp2))

            first_times = first.arm_result("baseline").evacuation_times()
            second_times = second.arm_result("baseline").evacuation_times()

            self.assertEqual(first_times, second_times)

    def test_run_experiment_trains_ai_models_when_configured(self):

        building = make_building()
        definition = make_definition()

        with tempfile.TemporaryDirectory() as tmp:

            spec = ExperimentSpec(
                name="exp", base_definition=definition, base_definition_id="def-1",
                base_building=building, arms=(make_arm("baseline"),),
                samples_per_arm=8, master_seed=5, output_root=tmp,
            )
            configs = (ExperimentConfig(name="evac", model_name="evacuation_time", test_size=0.25),)

            result = run_experiment(spec, ai_experiment_configs=configs)
            arm_result = result.arm_result("baseline")

            self.assertIn("evac", arm_result.ai_results)
            self.assertIn("mae", arm_result.ai_results["evac"].metrics)

    def test_train_and_evaluate_rl_for_arm_returns_an_evaluation_report(self):

        building = make_building()
        definition = make_definition(
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": FixedValue(2)},
                behaviour_profile_distribution={"zone-1": FixedValue("Adult_Default")},
            ),
            firefighter=FirefighterDeploymentDefinition(),
        )

        spec = ExperimentSpec(
            name="rl", base_definition=definition, base_definition_id="def-1",
            base_building=building, arms=(make_arm("baseline"),),
            samples_per_arm=1, master_seed=1, output_root=".",
        )

        report = train_and_evaluate_rl_for_arm(
            spec, spec.arms[0], total_timesteps=100, eval_episode_count=2, max_steps=50,
        )

        self.assertEqual(len(report.episodes), 2)
        self.assertIsInstance(report.average_reward, float)


# =====================================================
# Phase 3
# =====================================================


class StatisticsTests(unittest.TestCase):

    def test_confidence_interval_brackets_the_mean(self):

        ci = confidence_interval([10.0, 12.0, 11.0, 13.0, 9.0])

        self.assertAlmostEqual(ci.mean, 11.0)
        self.assertLess(ci.lower, ci.mean)
        self.assertGreater(ci.upper, ci.mean)

    def test_confidence_interval_degenerate_for_fewer_than_two_samples(self):

        ci = confidence_interval([5.0])
        self.assertIsNone(ci.lower)
        self.assertEqual(ci.mean, 5.0)

    def test_effect_size_zero_for_identical_groups(self):

        effect = effect_size_cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(effect.cohens_d, 0.0)

    def test_effect_size_large_for_clearly_separated_groups(self):

        effect = effect_size_cohens_d([10.0, 11.0, 9.0], [1.0, 2.0, 0.0])
        self.assertGreater(effect.cohens_d, 2.0)

    def test_one_way_anova_significant_for_clearly_different_groups(self):

        result = one_way_anova([1.0, 1.1, 0.9], [10.0, 10.1, 9.9], [20.0, 20.1, 19.9])
        self.assertIsNotNone(result.p_value)
        self.assertLess(result.p_value, 0.01)

    def test_one_way_anova_degenerate_with_too_few_groups(self):

        result = one_way_anova([1.0, 2.0])
        self.assertIsNone(result.f_statistic)

    def test_paired_comparison_requires_equal_length(self):

        with self.assertRaises(ValueError):
            paired_comparison([1.0, 2.0], [1.0])

    def test_paired_comparison_detects_a_consistent_shift(self):

        a = [10.0, 11.0, 12.0, 13.0]
        b = [8.0, 9.0, 10.0, 11.0]

        result = paired_comparison(a, b)
        self.assertAlmostEqual(result.mean_difference, 2.0)
        self.assertLess(result.p_value, 0.05)

    def test_feature_sensitivity_detects_positive_correlation(self):

        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]

        result = feature_sensitivity(x, y)
        self.assertAlmostEqual(result.pearson_r, 1.0, places=6)
        self.assertAlmostEqual(result.slope, 2.0, places=6)

    def test_feature_sensitivity_degenerate_for_constant_x(self):

        result = feature_sensitivity([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
        self.assertIsNone(result.pearson_r)

    def test_distribution_shift_detects_clearly_different_distributions(self):

        a = [0.0, 0.1, 0.2, 0.1, 0.0]
        b = [10.0, 10.1, 10.2, 10.1, 10.0]

        result = distribution_shift(a, b)
        self.assertGreater(result.ks_statistic, 0.9)

    def test_compare_arms_computes_anova_and_effect_sizes_vs_baseline(self):

        report = compare_arms(
            {"baseline": [10.0, 11.0, 9.0], "treatment": [20.0, 21.0, 19.0]},
            metric_name="evacuation_time", baseline_arm="baseline",
        )

        self.assertEqual(report.baseline_arm, "baseline")
        self.assertIn("treatment", report.effect_sizes_vs_baseline)
        self.assertLess(report.anova.p_value, 0.05)


# =====================================================
# Phase 4
# =====================================================


class AblationTests(unittest.TestCase):

    def test_build_ablation_arms_includes_full_plus_one_per_capability(self):

        arms = build_ablation_arms(make_definition(), DEFAULT_CAPABILITIES)

        self.assertEqual(len(arms), len(DEFAULT_CAPABILITIES) + 1)
        self.assertEqual(arms[0].name, "full")

    def test_no_wheelchairs_removes_the_wheelchair_weight(self):

        arms = build_ablation_arms(make_definition(), ["no_wheelchairs"])
        no_wheelchairs_arm = arms[1]

        resolved = no_wheelchairs_arm.resolve_definition(make_definition())
        weights = dict(resolved.occupant.behaviour_profile_distribution["zone-1"].weights)

        self.assertNotIn("Wheelchair_Default", weights)
        self.assertIn("Adult_Default", weights)

    def test_run_ablation_study_produces_a_result_per_capability_and_metric(self):

        with tempfile.TemporaryDirectory() as tmp:

            result = run_ablation_study(
                make_definition(), "def-1", make_building(),
                samples_per_arm=2, master_seed=7, output_root=tmp,
                capabilities=("no_firefighters", "no_helping"),
            )

            capabilities_seen = {r.capability for r in result.results}
            self.assertEqual(capabilities_seen, {"no_firefighters", "no_helping"})

            for r in result.results:
                self.assertEqual(r.metric_name, "total_evacuation_time")

    def test_no_perception_note_is_documented_not_fabricated(self):

        self.assertIn("no PerceptionProvider", NO_PERCEPTION_NOTE)

    def test_run_no_rl_ablation_compares_with_and_without_policy(self):

        building = make_building()
        definition = make_definition(
            occupant=OccupantDefinition(
                occupancy_distribution={"zone-1": FixedValue(2)},
                behaviour_profile_distribution={"zone-1": FixedValue("Adult_Default")},
            ),
            firefighter=FirefighterDeploymentDefinition(),
        )
        spec = ExperimentSpec(
            name="no_rl", base_definition=definition, base_definition_id="def-1",
            base_building=building, arms=(make_arm("baseline"),),
            samples_per_arm=1, master_seed=3, output_root=".",
        )

        result = run_no_rl_ablation(
            spec, spec.arms[0], total_timesteps=100, eval_episode_count=2, max_steps=50,
        )

        self.assertEqual(len(result.with_rl.episodes), 2)
        self.assertEqual(len(result.without_rl.episodes), 2)


# =====================================================
# Phase 5
# =====================================================


class SensitivityTests(unittest.TestCase):

    def test_run_parameter_sweep_produces_one_sweep_per_metric(self):

        with tempfile.TemporaryDirectory() as tmp:

            experiment_result, sweeps = run_parameter_sweep(
                make_definition(), "def-1", make_building(),
                parameter_name="helping_probability", values=(0.0, 1.0),
                override_factory=lambda value: override_helping_probability(value),
                samples_per_arm=2, master_seed=11, output_root=tmp,
            )

            self.assertEqual(len(experiment_result.arm_results), 2)
            metric_names = {sweep.metric_name for sweep in sweeps}
            self.assertEqual(metric_names, {"evacuation_time", "congestion", "trapped_occupants"})

            for sweep in sweeps:
                self.assertEqual(sweep.values, (0.0, 1.0))


# =====================================================
# Phase 6
# =====================================================


class FiguresTests(unittest.TestCase):

    def test_save_confusion_matrix_writes_a_file(self):

        with tempfile.TemporaryDirectory() as tmp:

            path = save_confusion_matrix(
                ["A", "B", "A", "B"], ["A", "A", "A", "B"], os.path.join(tmp, "cm.png"),
            )
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_save_roc_and_pr_curves_write_files(self):

        y_true = [0, 0, 1, 1, 1, 0]
        y_score = [0.1, 0.3, 0.8, 0.6, 0.9, 0.4]

        with tempfile.TemporaryDirectory() as tmp:

            roc_path = save_roc_curve(y_true, y_score, os.path.join(tmp, "roc.png"))
            pr_path = save_pr_curve(y_true, y_score, os.path.join(tmp, "pr.png"))

            self.assertTrue(os.path.exists(roc_path))
            self.assertTrue(os.path.exists(pr_path))

    def test_save_feature_importance_writes_a_file(self):

        estimator = build_classifier("random_forest")
        import numpy as np
        X = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
        y = np.array([0, 1, 0, 1])
        estimator.fit(X, y)

        with tempfile.TemporaryDirectory() as tmp:

            path = save_feature_importance(estimator, ["feature_a", "feature_b"], os.path.join(tmp, "fi.png"))
            self.assertTrue(os.path.exists(path))

    def test_save_feature_importance_rejects_mismatched_names(self):

        estimator = build_classifier("random_forest")
        import numpy as np
        estimator.fit(np.array([[1, 2], [2, 1]]), np.array([0, 1]))

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_feature_importance(estimator, ["only_one"], os.path.join(tmp, "fi.png"))

    def test_save_training_curve_writes_a_file(self):

        with tempfile.TemporaryDirectory() as tmp:

            path = save_training_curve([0.9, 0.7, 0.5, 0.4], os.path.join(tmp, "tc.png"), val_scores=[0.95, 0.8, 0.6, 0.5])
            self.assertTrue(os.path.exists(path))

    def test_save_rl_reward_curve_writes_a_file(self):

        with tempfile.TemporaryDirectory() as tmp:

            path = save_rl_reward_curve([-5.0, -3.0, -1.0, 0.5], os.path.join(tmp, "rl.png"))
            self.assertTrue(os.path.exists(path))

    def test_save_evacuation_time_distribution_writes_a_file(self):

        with tempfile.TemporaryDirectory() as tmp:

            path = save_evacuation_time_distribution(
                {"baseline": [10.0, 20.0, 30.0], "treatment": [15.0, 25.0]}, os.path.join(tmp, "dist.png"),
            )
            self.assertTrue(os.path.exists(path))


# =====================================================
# Phase 7
# =====================================================


class ReportTests(unittest.TestCase):

    def test_report_contains_expected_sections(self):

        with tempfile.TemporaryDirectory() as tmp:

            result = run_ablation_study(
                make_definition(), "def-1", make_building(),
                samples_per_arm=2, master_seed=13, output_root=tmp,
                capabilities=("no_firefighters",),
            )
            comparison = compare_arms(
                {ar.arm.name: ar.evacuation_times() for ar in result.experiment_result.arm_results},
                metric_name="total_evacuation_time", baseline_arm="full",
            )

            report_text = generate_research_report(
                result.experiment_result, arm_comparisons=[comparison], ablation_result=result,
            )

            self.assertIn("# Research Report", report_text)
            self.assertIn("## Experiment Summary", report_text)
            self.assertIn("## Per-Arm KPIs", report_text)
            self.assertIn("## Statistical Analysis", report_text)
            self.assertIn("## Ablation Studies", report_text)
            self.assertIn("## Conclusions", report_text)
            self.assertIn(NO_PERCEPTION_NOTE, report_text)

    def test_write_research_report_writes_to_disk(self):

        with tempfile.TemporaryDirectory() as tmp:

            result = run_ablation_study(
                make_definition(), "def-1", make_building(),
                samples_per_arm=1, master_seed=17, output_root=os.path.join(tmp, "campaigns"),
                capabilities=("no_helping",),
            )

            report_path = write_research_report(result.experiment_result, file_path=os.path.join(tmp, "report.md"))

            self.assertTrue(os.path.exists(report_path))

            with open(report_path, "r", encoding="utf-8") as handle:
                content = handle.read()

            self.assertIn("Research Report", content)


if __name__ == "__main__":
    unittest.main()
