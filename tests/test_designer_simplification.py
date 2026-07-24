import sys
import unittest

from PyQt6.QtCore import QPointF
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

from designer.windows.main_window import MainWindow

from models.building import Building
from models.project import Project
from models.zone import Zone
from models.detector import Detector
from models.emergency_light import EmergencyLight
from models.sprinkler import Sprinkler
from models.fire_extinguisher import FireExtinguisher
from models.fire_hydrant import FireHydrant
from models.hose_reel import HoseReel
from models.fire_water_tank import FireWaterTank
from models.fire_pump import FirePump
from models.jockey_pump import JockeyPump
from models.fire_service_inlet import FireServiceInlet

from sandbox.occupant import SandboxDistribution


# =====================================================
# SynEvac Designer Simplification & Product-Boundary Cleanup milestone.
#
# This file is the one place all of Phases 3/8/9/10/11 are proven
# together: the default toolbar now groups only the assets that
# actually participate in evacuation intelligence (docs/architecture/
# designer_asset_connectivity_audit.md is the authoritative source),
# the nine fire-safety/water-infrastructure assets moved to an
# "Advanced Fire-Safety Tools" submenu (Insert menu) rather than being
# deleted, the generic legacy Detector tool is no longer offered for
# new authoring, and Elevator remains exactly as disabled as it already
# was. Nothing about any model, manager, snapshot, serialization
# format, decision/AI/simulation/CCTV logic changed -- see
# ArchitectureGuardTests at the bottom.
# =====================================================


class _FakeSceneMouseEvent:

    def __init__(self, x, y):
        self._pos = QPointF(x, y)

    def scenePos(self):
        return self._pos


DEFAULT_TOOLBAR_ORDER = (
    "New", "Open", "Save",
    "Undo", "Redo",
    "Select",
    "Zone", "Door", "Exit", "Stair", "Obstacle", "Assembly Point",
    "Camera", "Smoke Detector", "Heat Detector", "Manual Call Point",
    "Speaker", "Dynamic Sign",
    "Occupant", "Simulation",
    "Zoom +", "Zoom -", "Reset",
    "Coverage",
)

ADVANCED_FIRE_SAFETY_ACTION_NAMES = (
    "emergency_light_action", "sprinkler_action", "fire_extinguisher_action",
    "fire_hydrant_action", "hose_reel_action", "fire_water_tank_action",
    "fire_pump_action", "jockey_pump_action", "fire_service_inlet_action",
)


class DefaultToolbarLayoutTests(unittest.TestCase):

    # Phase 2/7 -- the main toolbar now shows exactly the assets that
    # participate in evacuation intelligence (plus general editing/view
    # controls), grouped and separated, never one long undifferentiated
    # list.

    def test_toolbar_shows_exactly_the_expected_actions_in_order(self):

        window = MainWindow()

        visible_texts = tuple(a.text() for a in window.toolbar.actions() if a.text())
        self.assertEqual(visible_texts, DEFAULT_TOOLBAR_ORDER)

    def test_generic_detector_is_not_in_the_main_toolbar(self):

        window = MainWindow()

        texts = {a.text() for a in window.toolbar.actions()}
        self.assertNotIn("Detector", texts)

    def test_elevator_is_not_in_the_main_toolbar(self):

        window = MainWindow()

        texts = {a.text() for a in window.toolbar.actions()}
        self.assertNotIn("Elevator", texts)

    def test_none_of_the_nine_fire_safety_actions_are_in_the_main_toolbar(self):

        window = MainWindow()

        texts = {a.text() for a in window.toolbar.actions()}

        for action_name in ADVANCED_FIRE_SAFETY_ACTION_NAMES:
            action = getattr(window.toolbar, action_name)
            self.assertNotIn(action.text(), texts)

    def test_obstacle_stays_in_the_main_toolbar_building_group(self):

        # Phase 6 -- Obstacle is a deliberate exception: despite lacking
        # meaningful runtime decision connectivity today (per the
        # audit), it conceptually belongs to evacuation geometry and is
        # kept visible rather than hidden.
        window = MainWindow()

        texts = tuple(a.text() for a in window.toolbar.actions() if a.text())
        self.assertIn("Obstacle", texts)

        # Sits in the Building group, between Stair and Assembly Point.
        self.assertEqual(texts[texts.index("Stair") + 1], "Obstacle")
        self.assertEqual(texts[texts.index("Obstacle") + 1], "Assembly Point")


class AdvancedFireSafetyMenuTests(unittest.TestCase):

    # Phase 5/10 -- the nine assets remain fully authorable through an
    # explicitly secondary surface, never deleted.

    def _advanced_menu(self, window):

        for menu_action in window.menuBar().actions():
            menu = menu_action.menu()
            if menu is not None and menu.title() == "Insert":
                for insert_action in menu.actions():
                    submenu = insert_action.menu()
                    if submenu is not None and submenu.title() == "Advanced Fire-Safety Tools":
                        return submenu
        return None

    def test_advanced_fire_safety_submenu_exists_under_insert(self):

        window = MainWindow()
        self.assertIsNotNone(self._advanced_menu(window))

    def test_advanced_menu_contains_exactly_the_nine_actions(self):

        window = MainWindow()
        submenu = self._advanced_menu(window)

        expected = tuple(
            getattr(window.toolbar, name).text() for name in ADVANCED_FIRE_SAFETY_ACTION_NAMES
        )
        actual = tuple(a.text() for a in submenu.actions())

        self.assertEqual(actual, expected)

    def test_every_advanced_action_is_still_connected(self):

        window = MainWindow()

        for action_name in ADVANCED_FIRE_SAFETY_ACTION_NAMES:
            action = getattr(window.toolbar, action_name)
            self.assertGreater(action.receivers(action.triggered), 0)

    def test_triggering_an_advanced_action_still_sets_the_correct_tool(self):

        window = MainWindow()

        window.toolbar.sprinkler_action.trigger()
        self.assertEqual(window.canvas.scene_obj.current_tool, "sprinkler")

        window.toolbar.fire_water_tank_action.trigger()
        self.assertEqual(window.canvas.scene_obj.current_tool, "fire_water_tank")


class LegacyDetectorHiddenFromNewAuthoringTests(unittest.TestCase):

    # Phase 3 -- generic Detector no longer appears anywhere in the UI
    # for NEW authoring, but the model, migration logic, and existing-
    # object editing/rendering are all completely untouched.

    def test_detector_action_still_exists_and_is_connected(self):

        window = MainWindow()

        self.assertTrue(hasattr(window.toolbar, "detector_action"))
        action = window.toolbar.detector_action
        self.assertGreater(action.receivers(action.triggered), 0)

    def test_detector_action_is_not_in_the_toolbar_or_any_menu(self):

        window = MainWindow()

        toolbar_texts = {a.text() for a in window.toolbar.actions()}
        self.assertNotIn("Detector", toolbar_texts)

        def _walk(menu):
            for action in menu.actions():
                if action.text() == "Detector":
                    return True
                submenu = action.menu()
                if submenu is not None and _walk(submenu):
                    return True
            return False

        for menu_action in window.menuBar().actions():
            menu = menu_action.menu()
            if menu is not None:
                self.assertFalse(_walk(menu), f"'Detector' unexpectedly found in menu {menu.title()!r}")

    def test_legacy_smoke_typed_detector_still_round_trips(self):

        building = Building(name="B")
        project = Project(name="P", building=building)
        floor = building.create_floor(name="Ground")
        floor.detectors.append(
            Detector(id="LEGACY-1", name="Legacy Smoke", detector_type="Smoke", floor_id=floor.id, position=(1.0, 1.0))
        )

        data = project.to_dict()
        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(floor.id)

        self.assertEqual(restored_floor.detector_count, 1)
        self.assertEqual(restored_floor.detectors[0].detector_type, "Smoke")

    def test_legacy_flame_typed_detector_still_round_trips_even_though_inert(self):

        # Flame/Gas legacy detectors have zero real behavior anywhere
        # (see docs/architecture/designer_asset_connectivity_audit.md
        # §6) but must still load without error -- an old project
        # containing one must never fail to open.
        building = Building(name="B")
        project = Project(name="P", building=building)
        floor = building.create_floor(name="Ground")
        floor.detectors.append(
            Detector(id="LEGACY-FLAME-1", name="Legacy Flame", detector_type="Flame", floor_id=floor.id, position=(2.0, 2.0))
        )

        data = project.to_dict()
        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(floor.id)

        self.assertEqual(restored_floor.detector_count, 1)
        self.assertEqual(restored_floor.detectors[0].detector_type, "Flame")

    def test_legacy_detector_still_renders_after_rebuild_scene(self):

        window = MainWindow()
        floor = window.canvas.scene_obj.current_floor
        floor.detectors.append(Detector(id="LEGACY-1", name="Legacy", detector_type="Smoke", floor_id=floor.id, position=(1.0, 1.0)))

        window.canvas.scene_obj.rebuild_scene()

        items = [i for i in window.canvas.scene_obj.items() if getattr(i, "model", None) is not None and i.model.object_type == "Detector"]
        self.assertEqual(len(items), 1)

    def test_legacy_detector_still_editable_via_property_panel(self):

        window = MainWindow()
        floor = window.canvas.scene_obj.current_floor
        floor.detectors.append(Detector(id="LEGACY-1", name="Legacy", detector_type="Smoke", floor_id=floor.id, position=(1.0, 1.0)))
        window.canvas.scene_obj.rebuild_scene()

        item, = [i for i in window.canvas.scene_obj.items() if getattr(i, "model", None) is not None and i.model.object_type == "Detector"]

        # Must not raise -- Property Panel's own show_detector() is
        # completely untouched by this milestone.
        window.property_panel.show_detector(item)
        self.assertEqual(window.property_panel.object_name.text(), "Legacy")


ADVANCED_ASSET_SPECS = (
    ("emergency_lights", EmergencyLight, "EmergencyLight", "show_emergency_light"),
    ("sprinklers", Sprinkler, "Sprinkler", "show_sprinkler"),
    ("fire_extinguishers", FireExtinguisher, "FireExtinguisher", "show_fire_extinguisher"),
    ("fire_hydrants", FireHydrant, "FireHydrant", "show_fire_hydrant"),
    ("hose_reels", HoseReel, "HoseReel", "show_hose_reel"),
    ("fire_water_tanks", FireWaterTank, "FireWaterTank", "show_fire_water_tank"),
    ("fire_pumps", FirePump, "FirePump", "show_fire_pump"),
    ("jockey_pumps", JockeyPump, "JockeyPump", "show_jockey_pump"),
    ("fire_service_inlets", FireServiceInlet, "FireServiceInlet", "show_fire_service_inlet"),
)


class BackwardCompatibilityRoundTripTests(unittest.TestCase):

    # Phase 8 -- an old project containing every one of these nine
    # asset types must still load, render, appear in the Property
    # Panel, be editable, and save/reload identically. This test also
    # covers the real, pre-existing rendering gap this milestone found
    # and fixed: rebuild_scene() previously never reconstructed eight
    # of these nine asset types as graphics items at all (only
    # EmergencyLight was handled) -- a loaded project containing a
    # Sprinkler (etc.) would silently vanish from the canvas despite
    # the model itself being completely intact. Fixed additively in
    # designer/scene/graphics_scene.py, same per-item pattern every
    # other asset type already used.

    def setUp(self):

        self.window = MainWindow()
        self.floor = self.window.canvas.scene_obj.current_floor

        self.models_by_type = {}

        for list_attr, cls, _object_type, _show_method in ADVANCED_ASSET_SPECS:

            instance = cls(id=f"{cls.__name__}-1", name=cls.__name__, floor_id=self.floor.id, position=(1.0, 1.0))
            getattr(self.floor, list_attr).append(instance)
            self.models_by_type[cls.__name__] = instance

    def test_all_nine_assets_load_with_correct_counts(self):

        data = self.window.canvas.scene_obj.project.to_dict()
        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(self.floor.id)

        for list_attr, cls, _object_type, _show_method in ADVANCED_ASSET_SPECS:
            self.assertEqual(len(getattr(restored_floor, list_attr)), 1, f"{cls.__name__} did not round-trip")

    def test_all_nine_assets_render_after_rebuild_scene(self):

        self.window.canvas.scene_obj.rebuild_scene()

        rendered_types = {
            i.model.object_type for i in self.window.canvas.scene_obj.items() if getattr(i, "model", None) is not None
        }

        for _list_attr, _cls, object_type, _show_method in ADVANCED_ASSET_SPECS:
            self.assertIn(object_type, rendered_types, f"{object_type} did not render after rebuild_scene()")

    def test_all_nine_assets_are_editable_via_property_panel(self):

        self.window.canvas.scene_obj.rebuild_scene()

        for _list_attr, cls, object_type, show_method_name in ADVANCED_ASSET_SPECS:

            item, = [
                i for i in self.window.canvas.scene_obj.items()
                if getattr(i, "model", None) is not None and i.model.object_type == object_type
            ]

            show_method = getattr(self.window.property_panel, show_method_name)
            show_method(item)  # must not raise

            self.assertEqual(self.window.property_panel.object_name.text(), cls.__name__)

    def test_save_reload_preserves_identity_and_position(self):

        data = self.window.canvas.scene_obj.project.to_dict()
        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(self.floor.id)

        for list_attr, cls, _object_type, _show_method in ADVANCED_ASSET_SPECS:

            original = self.models_by_type[cls.__name__]
            restored_instance, = getattr(restored_floor, list_attr)

            self.assertEqual(restored_instance.id, original.id)
            self.assertEqual(restored_instance.position, original.position)
            self.assertEqual(restored_instance.floor_id, original.floor_id)


class CoreWorkflowE2ETests(unittest.TestCase):

    # Phase 9 -- through the real Designer UI (toolbar triggers + real
    # GraphicsScene.mousePressEvent, real Property Panel), prove a user
    # can still author the canonical evacuation-intelligence building
    # blocks after the toolbar cleanup, and that they save/reload
    # identically.

    def setUp(self):

        self.window = MainWindow()
        self.scene = self.window.canvas.scene_obj
        self.floor = self.scene.current_floor
        self.window.property_panel.building = self.scene.project.building

        # ---- 2 zones ----
        self.window.toolbar.zone_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(0, 0))
        self.scene.mousePressEvent(_FakeSceneMouseEvent(500, 500))  # Zone A: (0,0)-(10,10)m

        self.window.toolbar.zone_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(1000, 0))
        self.scene.mousePressEvent(_FakeSceneMouseEvent(1500, 500))  # Zone B: (20,0)-(30,10)m

        self.zone_a, self.zone_b = self.floor.zones
        self.zone_a.id, self.zone_b.id = "ZONE-A", "ZONE-B"

        # ---- 1 door (between the two zones) ----
        self.window.toolbar.door_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(500, 250))
        self.scene.mousePressEvent(_FakeSceneMouseEvent(1000, 250))

        # ---- 2 exits ----
        self.window.toolbar.exit_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(0, 0))
        self.scene.mousePressEvent(_FakeSceneMouseEvent(0, 500))

        self.window.toolbar.exit_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(1500, 0))
        self.scene.mousePressEvent(_FakeSceneMouseEvent(1500, 500))

        # ---- 1 stair (Zone A on this floor -> a zone on a second floor) ----
        self.second_floor = self.scene.project.building.create_floor(name="Floor 2", height=3.0)
        self.second_floor.add_zone(Zone(id="ZONE-C", name="Landing", floor_id=self.second_floor.id, x=0.0, y=0.0, width=10.0, height=10.0))

        self.scene.floor_picker_callback = lambda floors: self.second_floor
        self.scene.floor_switch_requested_callback = self.scene.set_current_floor

        self.window.toolbar.stair_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(250, 250))  # inside Zone A on floor 1 -> switches to floor 2
        self.scene.mousePressEvent(_FakeSceneMouseEvent(250, 250))  # inside Zone C (landing) on floor 2

        # Switch back to floor 1 for the remaining device placements.
        self.scene.set_current_floor(self.floor)

        # ---- 1 camera ----
        self.window.toolbar.camera_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(260, 260))

        # ---- 1 smoke detector, 1 heat detector, 1 MCP (auto zone assign) ----
        self.window.toolbar.smoke_detector_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(270, 270))

        self.window.toolbar.heat_detector_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(280, 280))

        self.window.toolbar.manual_call_point_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(290, 290))

        # ---- 1 speaker, 1 dynamic sign ----
        self.window.toolbar.speaker_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(300, 300))

        self.window.toolbar.sign_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(310, 310))

        # ---- occupants (drag-rectangle + generation callback) ----
        self.scene.occupant_generation_callback = lambda: (2, SandboxDistribution.UNIFORM)

        self.window.toolbar.occupant_action.trigger()
        self.scene.mousePressEvent(_FakeSceneMouseEvent(0, 0))
        self.scene.mousePressEvent(_FakeSceneMouseEvent(500, 500))

    def test_every_canonical_asset_was_authored(self):

        self.assertEqual(self.floor.zone_count, 2)
        self.assertEqual(self.floor.door_count, 1)
        self.assertEqual(self.floor.exit_count, 2)
        self.assertEqual(self.floor.camera_count, 1)
        self.assertEqual(self.floor.smoke_detector_count, 1)
        self.assertEqual(self.floor.heat_detector_count, 1)
        self.assertEqual(self.floor.manual_call_point_count, 1)
        self.assertEqual(self.floor.speaker_count, 1)
        self.assertEqual(self.floor.sign_count, 1)
        self.assertEqual(self.floor.stair_count, 1)
        self.assertEqual(len(self.scene.sandbox_manager.occupants_on_floor(self.floor.id)), 2)

    def test_stair_connects_the_two_floors(self):

        stair = self.floor.stairs[0]
        self.assertEqual(stair.from_floor_id, self.floor.id)
        self.assertEqual(stair.to_floor_id, self.second_floor.id)
        self.assertEqual(stair.from_zone_id, "ZONE-A")
        self.assertEqual(stair.to_zone_id, "ZONE-C")

    def test_smoke_heat_mcp_auto_assigned_to_zone_a(self):

        self.assertEqual(self.floor.smoke_detectors[0].zone_ids, ("ZONE-A",))
        self.assertEqual(self.floor.heat_detectors[0].zone_ids, ("ZONE-A",))
        self.assertEqual(self.floor.manual_call_points[0].zone_ids, ("ZONE-A",))

    def test_save_reload_preserves_every_canonical_relationship(self):

        data = self.scene.project.to_dict()
        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(self.floor.id)
        restored_second_floor = restored.building.get_floor(self.second_floor.id)

        self.assertEqual(restored_floor.zone_count, 2)
        self.assertEqual(restored_floor.door_count, 1)
        self.assertEqual(restored_floor.exit_count, 2)
        self.assertEqual(restored_floor.camera_count, 1)
        self.assertEqual(restored_floor.smoke_detector_count, 1)
        self.assertEqual(restored_floor.heat_detector_count, 1)
        self.assertEqual(restored_floor.manual_call_point_count, 1)
        self.assertEqual(restored_floor.speaker_count, 1)
        self.assertEqual(restored_floor.sign_count, 1)
        self.assertEqual(restored_floor.stair_count, 1)

        restored_stair = restored_floor.stairs[0]
        self.assertEqual(restored_stair.to_floor_id, restored_second_floor.id)
        self.assertEqual(restored_stair.from_zone_id, "ZONE-A")
        self.assertEqual(restored_stair.to_zone_id, "ZONE-C")

        self.assertEqual(restored_floor.smoke_detectors[0].zone_ids, ("ZONE-A",))


class AdvancedAuthoringE2ETests(unittest.TestCase):

    # Phase 10 -- an advanced user can intentionally reach Advanced
    # Fire-Safety Tools and place at least five of the nine assets,
    # then save/reload successfully. These are being DE-EMPHASIZED
    # only -- placement, editing, and persistence all still work
    # exactly as before.

    def setUp(self):

        self.window = MainWindow()
        self.scene = self.window.canvas.scene_obj
        self.floor = self.scene.current_floor

    def test_advanced_user_places_five_fire_safety_assets_via_the_menu_actions(self):

        for action_name, x, y in (
            ("sprinkler_action", 100, 100),
            ("fire_extinguisher_action", 150, 150),
            ("fire_hydrant_action", 200, 200),
            ("fire_water_tank_action", 250, 250),
            ("fire_pump_action", 300, 300),
        ):
            # Same QAction the Insert > Advanced Fire-Safety Tools menu
            # item triggers -- an advanced user reaches this exact
            # action via that menu; .trigger() here proves the same
            # code path a real menu click would run.
            getattr(self.window.toolbar, action_name).trigger()
            self.scene.mousePressEvent(_FakeSceneMouseEvent(x, y))

        self.assertEqual(self.floor.sprinkler_count, 1)
        self.assertEqual(len(self.floor.fire_extinguishers), 1)
        self.assertEqual(len(self.floor.fire_hydrants), 1)
        self.assertEqual(len(self.floor.fire_water_tanks), 1)
        self.assertEqual(len(self.floor.fire_pumps), 1)

    def test_placed_advanced_assets_save_and_reload_successfully(self):

        for action_name, x, y in (
            ("sprinkler_action", 100, 100),
            ("fire_extinguisher_action", 150, 150),
            ("fire_hydrant_action", 200, 200),
            ("fire_water_tank_action", 250, 250),
            ("fire_pump_action", 300, 300),
        ):
            getattr(self.window.toolbar, action_name).trigger()
            self.scene.mousePressEvent(_FakeSceneMouseEvent(x, y))

        data = self.scene.project.to_dict()
        restored = Project.from_dict(data)
        restored_floor = restored.building.get_floor(self.floor.id)

        self.assertEqual(restored_floor.sprinkler_count, 1)
        self.assertEqual(len(restored_floor.fire_extinguishers), 1)
        self.assertEqual(len(restored_floor.fire_hydrants), 1)
        self.assertEqual(len(restored_floor.fire_water_tanks), 1)
        self.assertEqual(len(restored_floor.fire_pumps), 1)


class ArchitectureGuardTests(unittest.TestCase):

    # Phase 11 -- proves the UI simplification did not touch model
    # architecture, runtime managers, serialization support, or any
    # decision/AI/simulation/CCTV logic. Combined with the full
    # regression suite passing at zero regressions (the practical proof
    # that nothing else's *behavior* changed), these checks prove
    # nothing else's *existence* changed either.

    def test_no_fire_safety_model_was_deleted(self):

        import models.emergency_light
        import models.sprinkler
        import models.fire_extinguisher
        import models.fire_hydrant
        import models.hose_reel
        import models.fire_water_tank
        import models.fire_pump
        import models.jockey_pump
        import models.fire_service_inlet
        import models.detector
        import models.elevator

        for module in (
            models.emergency_light, models.sprinkler, models.fire_extinguisher, models.fire_hydrant,
            models.hose_reel, models.fire_water_tank, models.fire_pump, models.jockey_pump,
            models.fire_service_inlet, models.detector, models.elevator,
        ):
            self.assertIsNotNone(module)

    def test_no_runtime_manager_was_deleted(self):

        from emergency_light_manager.manager import EmergencyLightManager
        from fire_safety_manager.manager import FireSafetyAssetManager
        from fire_water_manager.manager import FireWaterInfrastructureManager

        self.assertTrue(EmergencyLightManager)
        self.assertTrue(FireSafetyAssetManager)
        self.assertTrue(FireWaterInfrastructureManager)

    def test_no_serialization_support_was_removed(self):

        from models.floor import Floor

        floor = Floor(name="F")

        for method_name in (
            "add_sprinkler", "add_fire_extinguisher", "add_fire_hydrant", "add_hose_reel",
            "add_fire_water_tank", "add_fire_pump", "add_jockey_pump", "add_fire_service_inlet",
            "add_emergency_light",
        ):
            self.assertTrue(hasattr(floor, method_name), f"Floor.{method_name} was removed")

    def test_detector_migration_logic_is_untouched(self):

        from models.detector_migration import adapt_legacy_detector
        from models.detector import Detector

        smoke = Detector(id="D-1", detector_type="Smoke", floor_id="F1")
        adapted = adapt_legacy_detector(smoke, zones=())
        self.assertIsNotNone(adapted)
        self.assertEqual(adapted.object_type, "SmokeDetector")

        flame = Detector(id="D-2", detector_type="Flame", floor_id="F1")
        self.assertIsNone(adapt_legacy_detector(flame, zones=()))

    def test_decision_ai_simulation_cctv_packages_still_import_cleanly(self):

        # Not a behavior check (the full suite is) -- a cheap, fast
        # confirmation this milestone's UI-only changes did not
        # accidentally break an unrelated import graph.
        import decision_policy.zone_policy
        import decision_policy.exit_policy
        import decision_policy.stair_policy
        import ai_features.building_state_extractor
        import evacuation_recommendation.engine
        import evacuation_guidance.engine
        import simulator.engine
        import live_camera_pipeline.pipeline
        import camera_calibration.projection
        import human_detection.opencv_decoder_backend

        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
