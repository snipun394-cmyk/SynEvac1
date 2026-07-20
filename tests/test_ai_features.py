import unittest

from hazard.snapshot import HazardSnapshot

from occupancy.snapshot import OccupancySnapshot

from models.building import Building
from models.floor import Floor
from models.zone import Zone
from models.camera import Camera
from models.smoke_detector import SmokeDetector
from models.sensor_asset import DetectorState, HealthStatus

from camera_manager.manager import CameraManager
from camera_manager.status import CameraStatus

from sensor_manager.manager import SensorManager
from sensor_manager.status import SensorStatus

from perception.models.smoke_detector_observation import SmokeDetectorReading

from live_camera_pipeline.replay_frame_source import ReplayFrameSource
from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.identity_resolver import MappingIdentityResolver
from live_camera_pipeline.pipeline import LiveCameraPipeline

from tests.live_camera_pipeline_fixtures import MockHumanDetector

from multi_camera_fusion.engine import MultiCameraFusionEngine

from facp.engine import SimulatedFACP
from facp.models import DetectorConditionReport

from building_state.estimator import BuildingStateEstimator
from building_state.models import BuildingState

from ai_training.experiment import ExperimentConfig, ExperimentRunner
from ai_inference.loader import LoadedModel, ModelProvenance

import ai_features as af
from ai_features.feature_schema import CANONICAL_LIVE_FEATURE_NAMES, SCHEMA_VERSION


# =====================================================
# Simulation-to-Live AI Feature Parity Framework -- Phase 13 tests.
# No LiveOrchestrator AI wiring, no Advisory System behavior change,
# and no model given control authority appears anywhere in this file
# (test 18 makes this explicit and mechanical).
# =====================================================


def _make_building(smoke_detectors=True):

    zone_a = Zone(id="zone-a", name="Zone A", floor_id="floor-1", x=0, y=0, width=4, height=4)
    zone_b = Zone(id="zone-b", name="Zone B", floor_id="floor-1", x=10, y=0, width=4, height=4)

    smoke = [
        SmokeDetector(id="SMOKE-A", name="Smoke A", floor_id="floor-1", zone_ids=("zone-a",)),
        SmokeDetector(id="SMOKE-B", name="Smoke B", floor_id="floor-1", zone_ids=("zone-b",)),
    ] if smoke_detectors else []

    floor = Floor(
        id="floor-1", name="Ground", zones=[zone_a, zone_b],
        cameras=[
            Camera(id="CAM-A", name="A", floor_id="floor-1", zone_ids=("zone-a",)),
            Camera(id="CAM-B", name="B", floor_id="floor-1", zone_ids=("zone-b",)),
        ],
        smoke_detectors=smoke,
    )

    return Building(id="b1", name="Test", floors=[floor])


# =====================================================
# 1/2 -- Canonical schema and feature ordering are deterministic.
# =====================================================


class CanonicalSchemaDeterminismTests(unittest.TestCase):

    def test_schema_names_are_deterministic_across_imports(self):

        import importlib

        import ai_features.feature_schema as fs

        importlib.reload(fs)

        self.assertEqual(fs.CANONICAL_LIVE_FEATURE_NAMES, CANONICAL_LIVE_FEATURE_NAMES)

    def test_feature_ordering_matches_schema_order_every_call(self):

        state = BuildingState()

        first = tuple(af.extract_canonical_features(state).keys())
        second = tuple(af.extract_canonical_features(state).keys())

        self.assertEqual(first, CANONICAL_LIVE_FEATURE_NAMES)
        self.assertEqual(second, CANONICAL_LIVE_FEATURE_NAMES)


# =====================================================
# 3/4 -- BuildingState extraction and simulation-side extraction both
# produce the canonical schema.
# =====================================================


class ExtractorSchemaComplianceTests(unittest.TestCase):

    def test_building_state_extraction_produces_the_canonical_schema(self):

        row = af.extract_canonical_features(BuildingState())

        self.assertEqual(tuple(row.keys()), CANONICAL_LIVE_FEATURE_NAMES)

    def test_simulation_side_extraction_produces_the_same_schema(self):

        building = _make_building()

        row = af.extract_canonical_training_row(building, total_occupants=5, ignition_zone_id="zone-a")

        self.assertEqual(tuple(row.keys()), CANONICAL_LIVE_FEATURE_NAMES)


# =====================================================
# 5/6 -- Simulation-only fields and outcome leakage are rejected.
# =====================================================


class SimulationOnlyAndLeakageRejectionTests(unittest.TestCase):

    def test_known_simulation_only_names_are_absent_from_canonical_schema(self):

        for name in ("ignition_zone", "fire_profile", "growth_time", "Adult_Count",
                     "Mean_Walking_Speed_Multiplier", "Group_Count", "Zone_1_Occupancy"):

            self.assertNotIn(name, CANONICAL_LIVE_FEATURE_NAMES)

    def test_a_model_requiring_a_simulation_only_column_is_flagged_as_such(self):

        loaded = _fake_loaded_model(numeric_columns=("Adult_Count", "Mean_Group_Size"))

        report = af.check_model_compatibility(loaded)

        self.assertFalse(report.compatible)
        kinds = {issue.kind for issue in report.issues}
        self.assertIn("simulation_only_dependency", kinds)

    def test_a_model_requiring_an_outcome_column_is_flagged_as_leakage(self):

        loaded = _fake_loaded_model(numeric_columns=("total_evacuation_time",))

        report = af.check_model_compatibility(loaded)

        self.assertFalse(report.compatible)
        kinds = {issue.kind for issue in report.issues}
        self.assertIn("outcome_leakage", kinds)


# =====================================================
# 7/8 -- Missing cameras/unknown status are never fabricated as safe.
# =====================================================


class HonestMissingDataTests(unittest.TestCase):

    def test_no_cameras_configured_leaves_occupancy_unknown_not_zero(self):

        row = af.extract_canonical_features(BuildingState())

        self.assertFalse(row["occupancy_observed"])
        self.assertIsNone(row["total_occupant_count"])

    def test_no_facp_configured_leaves_panel_state_unknown_not_normal(self):

        row = af.extract_canonical_features(BuildingState())

        self.assertFalse(row["facp_available"])
        self.assertIsNone(row["facp_panel_state"])
        self.assertNotEqual(row["facp_panel_state"], "NORMAL")


# =====================================================
# 9/10/11 -- Fault-vs-normal, FACP absence, and offline cameras.
# =====================================================


class DistinctDeviceStateTests(unittest.TestCase):

    def test_detector_fault_is_distinct_from_alarm_and_from_normal(self):

        status = SensorStatus(
            sensor_id="SMOKE-A", sensor_type="SmokeDetector", name="Smoke A", floor_id="floor-1",
            zone_ids=("zone-a",), active=True, mode="Simulation", health_status=HealthStatus.FAULT,
        )
        reading = SmokeDetectorReading(detector_id="SMOKE-A", timestamp=0.0, alarm_active=False)

        state = BuildingStateEstimator().estimate(
            0.0,
            hazard_snapshot=HazardSnapshot(),
            occupancy_snapshot=OccupancySnapshot(),
            smoke_detector_statuses=(status,), smoke_detector_readings=(reading,),
        )

        row = af.extract_canonical_features(state)

        self.assertEqual(row["smoke_detector_fault_count"], 1)
        self.assertEqual(row["smoke_detector_alarm_count"], 0)
        self.assertEqual(row["building_alarm_status"], DetectorState.FAULT.name)

    def test_facp_absence_is_represented_honestly_across_every_facp_field(self):

        row = af.extract_canonical_features(BuildingState())

        self.assertFalse(row["facp_available"])
        for field_name in (
            "facp_panel_state", "facp_active_alarm_source_count",
            "facp_active_fault_source_count", "facp_acknowledged", "facp_silenced",
        ):
            self.assertIsNone(row[field_name])

    def test_offline_cameras_are_represented(self):

        state = BuildingStateEstimator().estimate(
            0.0,
            hazard_snapshot=HazardSnapshot(),
            occupancy_snapshot=OccupancySnapshot(),
            camera_statuses=(
                CameraStatus(camera_id="CAM-A", name="A", floor_id="floor-1", zone_ids=(),
                              active=True, mode="Simulation", has_detection_provider=True),
                CameraStatus(camera_id="CAM-B", name="B", floor_id="floor-1", zone_ids=(),
                              active=False, mode="Simulation", has_detection_provider=True),
            ),
        )

        row = af.extract_canonical_features(state)

        self.assertEqual(row["camera_total_count"], 2)
        self.assertEqual(row["camera_active_count"], 1)
        self.assertEqual(row["camera_offline_count"], 1)


# =====================================================
# 12/13/14 -- Model compatibility checker.
# =====================================================


def _fake_loaded_model(numeric_columns=(), categorical_columns=(), metrics=None):

    from ai_training.preprocessing import FeatureSchema

    class _StubModel:
        feature_schema = FeatureSchema(numeric_columns=tuple(numeric_columns), categorical_columns=tuple(categorical_columns))

    return LoadedModel(
        model=_StubModel(),
        provenance=ModelProvenance(
            experiment_name="exp", model_name="stub-model", algorithm="dummy",
            model_version="v1", generated_at="2026-01-01T00:00:00",
            train_size=1, test_size=1, metrics=metrics or {},
        ),
        directory="in-memory",
    )


class ModelCompatibilityCheckerTests(unittest.TestCase):

    def test_rejects_missing_required_feature(self):

        loaded = _fake_loaded_model(numeric_columns=("some_column_not_in_canonical_schema",))

        report = af.check_model_compatibility(loaded)

        self.assertFalse(report.compatible)
        self.assertTrue(any(issue.kind == "missing_required_feature" for issue in report.issues))

    def test_rejects_schema_version_mismatch(self):

        loaded = _fake_loaded_model(
            numeric_columns=("total_occupant_count",),
            metrics={"feature_schema_version": "0.9-does-not-exist"},
        )

        report = af.check_model_compatibility(loaded)

        self.assertFalse(report.compatible)
        self.assertTrue(any(issue.kind == "schema_version_mismatch" for issue in report.issues))

    def test_accepts_a_model_trained_entirely_on_canonical_columns(self):

        loaded = _fake_loaded_model(
            numeric_columns=("total_occupant_count", "camera_active_count"),
            categorical_columns=("building_alarm_status",),
            metrics={"feature_schema_version": SCHEMA_VERSION},
        )

        report = af.check_model_compatibility(loaded)

        self.assertTrue(report.compatible)
        self.assertEqual(report.issues, ())

    def test_validate_feature_row_raises_on_missing_column_rather_than_silently_filling(self):

        loaded = _fake_loaded_model(numeric_columns=("total_occupant_count", "camera_active_count"))

        with self.assertRaises(af.IncompatibleFeatureRowError):
            af.validate_feature_row({"total_occupant_count": 3}, loaded)

    def test_validate_feature_row_raises_on_unexpected_column(self):

        loaded = _fake_loaded_model(numeric_columns=("total_occupant_count",))

        with self.assertRaises(af.IncompatibleFeatureRowError):
            af.validate_feature_row({"total_occupant_count": 3, "extra_unknown_key": 1}, loaded)


# =====================================================
# 15/16/17 -- Existing pipeline stays unmodified and working.
# =====================================================


class ExistingPipelineBackwardCompatibilityTests(unittest.TestCase):

    def test_dataset_builder_feature_extractor_is_unmodified(self):

        from types import SimpleNamespace

        from dataset_builder.feature_extractor import extract_scenario_features

        building = _make_building()

        metadata = SimpleNamespace(scenario_id="s1", definition_id="d1", seed=1)
        scenario = SimpleNamespace(
            metadata=metadata, fire=None, occupants=(), firefighters=(),
            door_states=(), exit_states=(), stair_states=(), obstacle_states=(),
            detector_states=(), camera_states=(),
        )
        run = SimpleNamespace(scenario=scenario, building=building)

        row = extract_scenario_features(run)

        # Every existing scenario-feature column this milestone's audit
        # found (ignition_zone/total_occupants/Camera_N_State/...) must
        # still be produced -- proving this milestone changed nothing
        # about the existing, richer research dataset path.
        self.assertIn("ignition_zone", row)
        self.assertIn("total_occupants", row)
        self.assertIn("Adult_Count", row)
        self.assertIn("Camera_1_State", row)

    def test_existing_trained_model_still_saves_and_loads(self):

        import os
        import shutil
        import tempfile

        from tests.training_dataset_fixtures import make_campaign
        import ai_training as at

        tmp = tempfile.mkdtemp()

        try:

            make_campaign(tmp, count=8, master_seed=1)
            dataset = at.load_campaign_dataset(tmp)

            runner = ExperimentRunner()
            result = runner.run(dataset, ExperimentConfig(name="exp-compat", model_name="evacuation_time"))

            directory = os.path.join(tmp, "saved")
            runner.save_result(result, directory)
            loaded = runner.load_result(directory)

            self.assertEqual(loaded.metrics, result.metrics)

        finally:

            shutil.rmtree(tmp, ignore_errors=True)

    def test_existing_ai_inference_predictor_remains_unaffected(self):

        from ai_inference.predictor import Predictor

        class _FakeRegressionModel:
            is_classifier = False
            def predict(self, X_rows):
                return [42.0 for _ in X_rows]

        loaded = LoadedModel(
            model=_FakeRegressionModel(),
            provenance=ModelProvenance(
                experiment_name="e", model_name="evacuation_time", algorithm="dummy",
                model_version="v1", generated_at="2026-01-01T00:00:00",
                train_size=1, test_size=1, metrics={},
            ),
            directory="in-memory",
        )

        predictions = Predictor([loaded]).predict_all({"anything": 1})

        self.assertEqual(predictions["evacuation_time"].value, 42.0)


# =====================================================
# 18 -- No LiveOrchestrator AI wiring introduced.
# =====================================================


class NoLiveAIWiringGuardTests(unittest.TestCase):

    def test_live_system_never_imports_ai_features(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "live_system"

        forbidden = r"^\s*(from|import)\s+ai_features\b"

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"live_system/{path.name} imports ai_features -- this milestone must not wire "
                f"AI feature extraction into LiveOrchestrator/run_cycle() yet.",
            )

    def test_ai_features_never_imports_live_system_or_advisory_system(self):

        import pathlib
        import re

        package_dir = pathlib.Path(__file__).resolve().parent.parent / "ai_features"

        forbidden = r"^\s*(from|import)\s+(live_system|advisory_system)\b"

        for path in package_dir.glob("*.py"):

            text = path.read_text()

            self.assertIsNone(
                re.search(forbidden, text, re.MULTILINE),
                f"ai_features/{path.name} imports live_system/advisory_system -- ai_features "
                f"must stay a standalone feature-contract library this milestone, wired to "
                f"nothing downstream yet.",
            )


# =====================================================
# 19/20 -- Deterministic Simulation->BuildingState->features, and
# Replay->BuildingState->features, produce compatible canonical rows.
# =====================================================


class SimulationReplayParityTests(unittest.TestCase):

    def test_simulation_side_extraction_is_deterministic(self):

        building = _make_building()

        first = af.extract_canonical_training_row(building, total_occupants=4, ignition_zone_id="zone-a")
        second = af.extract_canonical_training_row(building, total_occupants=4, ignition_zone_id="zone-a")

        self.assertEqual(first, second)

    def test_simulation_and_replay_paths_produce_the_same_schema_and_compatible_semantics(self):

        # Simulation-observation path: 3 occupants, ignition at zone-a
        # (mirrors this module's own build_building_state_at_alarm_
        # activation()).
        building = _make_building()

        simulation_row = af.extract_canonical_training_row(
            building, total_occupants=3, ignition_zone_id="zone-a",
        )

        # Replay path: the exact offline-proven CCTV chain (CameraManager
        # -> ReplayFrameSource -> LiveCameraPipeline -> MultiCameraFusion
        # Engine), engineered to observe the SAME ground truth (3 unique
        # people, one detector alarmed at zone-a) so exact parity can be
        # demonstrated under "perfect sensing" per this phase's own
        # instruction.
        camera_manager = CameraManager()
        camera_manager.discover_cameras(building)

        source_a = ReplayFrameSource(
            camera_id="CAM-A", frames=[(0.0, [{"local_track_id": "a1"}, {"local_track_id": "shared"}])],
        )
        source_b = ReplayFrameSource(
            camera_id="CAM-B", frames=[(0.0, [{"local_track_id": "b1"}])],
        )
        source_a.start()
        source_b.start()

        resolver = MappingIdentityResolver({
            ("CAM-A", "a1"): "P1", ("CAM-B", "b1"): "P2", ("CAM-A", "shared"): "P3",
        })
        detection_provider = LiveCameraPipelineDetectionProvider()
        pipeline = LiveCameraPipeline(
            frame_sources={"CAM-A": source_a, "CAM-B": source_b},
            human_detector=MockHumanDetector(), identity_resolver=resolver,
            detection_provider=detection_provider,
        )
        pipeline.run_cycle(0.0)

        detections = []
        for camera_id in ("CAM-A", "CAM-B"):
            detections.extend(detection_provider.detections_at(camera_id, 0.0))

        fusion_result = MultiCameraFusionEngine().fuse(detections, 0.0)
        self.assertEqual(len(fusion_result.tracks), 3, "replay path must also see 3 unique people")

        sensor_manager = SensorManager()
        sensor_manager.discover_sensors(building)

        facp = SimulatedFACP(panel_id="FACP-REPLAY-TEST")
        conditions = {
            "SMOKE-A": DetectorConditionReport(
                asset_id="SMOKE-A", asset_type="SmokeDetector", state=DetectorState.ALARM,
                floor_id="floor-1", zone_ids=("zone-a",),
            ),
            "SMOKE-B": DetectorConditionReport(
                asset_id="SMOKE-B", asset_type="SmokeDetector", state=DetectorState.NORMAL,
                floor_id="floor-1", zone_ids=("zone-b",),
            ),
        }
        facp.evaluate(conditions, 0.0)

        smoke_a_status = sensor_manager.sensor_status("SMOKE-A")
        smoke_b_status = sensor_manager.sensor_status("SMOKE-B")

        replay_state = BuildingStateEstimator().estimate(
            0.0,
            hazard_snapshot=HazardSnapshot(),
            occupancy_snapshot=OccupancySnapshot(),
            camera_statuses=camera_manager.all_statuses(),
            fusion_result=fusion_result,
            smoke_detector_statuses=(smoke_a_status, smoke_b_status),
            smoke_detector_readings=(
                SmokeDetectorReading(detector_id="SMOKE-A", timestamp=0.0, alarm_active=True),
                SmokeDetectorReading(detector_id="SMOKE-B", timestamp=0.0, alarm_active=False),
            ),
            facp_snapshot=facp.current_snapshot(0.0),
        )

        replay_row = af.extract_canonical_features(replay_state)

        # Same canonical schema (Phase 9's own core requirement).
        self.assertEqual(tuple(simulation_row.keys()), tuple(replay_row.keys()))

        # Compatible/deterministic semantics under matched, "perfect
        # sensing" ground truth -- exact value parity where expected.
        self.assertEqual(simulation_row["total_occupant_count"], replay_row["total_occupant_count"])
        self.assertEqual(simulation_row["building_alarm_status"], replay_row["building_alarm_status"])
        self.assertEqual(simulation_row["facp_panel_state"], replay_row["facp_panel_state"])
        self.assertEqual(
            simulation_row["smoke_detector_alarm_count"], replay_row["smoke_detector_alarm_count"],
        )


if __name__ == "__main__":
    unittest.main()
