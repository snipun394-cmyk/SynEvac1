import pathlib
import re
import unittest


# =====================================================
# Execution Layer V1 milestone -- mechanical dependency-direction
# guards for warden_notification/, mirroring tests/test_building_
# control.py's own guards exactly (warden_notification/ is intentionally
# a near-verbatim structural mirror of building_control/).
# =====================================================


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
WARDEN_NOTIFICATION_PACKAGE = REPO_ROOT / "warden_notification"

_FORBIDDEN_FOR_WARDEN_NOTIFICATION = (
    r"^\s*(from|import)\s+("
    r"ai_engine|ai_inference|ai_registry|ai_training|ai_explainability|reinforcement_learning|rl_training|"
    r"decision_policy|advisory_system|"
    r"command_center|voice_evacuation|speaker_manager|dynamic_signage|building_control|"
    r"modbus|bacnet|mqtt|paho|pymodbus|bacpypes|opcua|socket|serial|"
    r"cv2|torch|ultralytics"
    r")\b"
)


class WardenNotificationArchitectureGuardTests(unittest.TestCase):

    def test_warden_notification_never_imports_forbidden_modules(self):

        for path in sorted(WARDEN_NOTIFICATION_PACKAGE.glob("*.py")):

            text = path.read_text(encoding="utf-8")
            match = re.search(_FORBIDDEN_FOR_WARDEN_NOTIFICATION, text, re.MULTILINE)

            self.assertIsNone(
                match,
                f"{path.relative_to(REPO_ROOT)} imports {match.group(0).strip() if match else ''!r} -- "
                f"warden_notification/ must never depend on AI/decision_policy/advisory_system/execution "
                f"modules or hardware/vendor protocol libraries.",
            )

    def test_warden_notification_not_nested_inside_another_package(self):

        self.assertTrue(WARDEN_NOTIFICATION_PACKAGE.is_dir())
        self.assertEqual(WARDEN_NOTIFICATION_PACKAGE.parent, REPO_ROOT)

    def test_dispatch_is_only_reachable_through_approve(self):

        # WardenNotificationController has no auto-approve/AI-triggered
        # path at all in V1 (unlike BuildingControlController's own
        # ApprovalMode.AUTO_APPROVE_SIMULATION) -- submit() only ever
        # reaches PENDING_APPROVAL, never DISPATCHED, confirmed here by
        # reading the controller's own source for the single call site
        # of _dispatch().
        text = (WARDEN_NOTIFICATION_PACKAGE / "controller.py").read_text(encoding="utf-8")

        dispatch_call_sites = re.findall(r"self\._dispatch\(", text)

        self.assertEqual(
            len(dispatch_call_sites), 1,
            "WardenNotificationController must call _dispatch() from exactly one place -- approve().",
        )


if __name__ == "__main__":
    unittest.main()
