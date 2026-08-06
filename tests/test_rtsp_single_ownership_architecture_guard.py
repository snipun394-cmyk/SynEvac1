import pathlib
import re
import unittest


# =====================================================
# Multi-Camera Streaming Architecture milestone -- the investigation
# behind this milestone confirmed (by repo-wide grep, re-verified here
# as a standing regression guard, same "regex-scan-the-source-files"
# convention tests/test_no_cv_dependencies.py already establishes) that
# exactly ONE production call site constructs an RTSPFrameSource:
# live_runtime_launcher/rtsp_camera_sources.py::build_rtsp_frame_sources().
# One configured Camera -> one RTSPFrameSource, never a second/duplicate
# connection anywhere else in the application. `scripts/` (standalone
# diagnostic/demo tools, never imported by the shipped app) and `tests/`
# are legitimately exempt -- each opens its own throwaway source for a
# one-off manual run, never sharing the real LiveRuntime's own
# connections.
# =====================================================

FORBIDDEN = r"\bRTSPFrameSource\s*\("

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

ALLOWED_FILE = REPO_ROOT / "live_runtime_launcher" / "rtsp_camera_sources.py"

EXCLUDED_DIR_NAMES = {
    "tests", "scripts", ".git", ".claude", "__pycache__",
    "venv", ".venv", "env", "node_modules",
}


def _production_python_files():

    for path in REPO_ROOT.rglob("*.py"):

        if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
            continue

        yield path


class RTSPFrameSourceSingleOwnershipTests(unittest.TestCase):

    def test_only_rtsp_camera_sources_constructs_an_rtsp_frame_source(self):

        offending_files = []

        for path in _production_python_files():

            if path == ALLOWED_FILE:
                continue

            text = path.read_text(encoding="utf-8")

            if re.search(FORBIDDEN, text):
                offending_files.append(path.relative_to(REPO_ROOT))

        self.assertEqual(
            offending_files, [],
            f"RTSPFrameSource must be constructed in exactly one production "
            f"file ({ALLOWED_FILE.relative_to(REPO_ROOT)}) -- found it also "
            f"referenced in {offending_files}. Every camera must have exactly "
            f"one RTSP connection; a second construction site is how a "
            f"duplicate/competing connection would silently creep in.",
        )

    def test_the_allowed_file_still_actually_constructs_it(self):

        # A guard against the guard itself silently going stale -- if
        # ALLOWED_FILE ever stops constructing RTSPFrameSource at all
        # (e.g. the function were renamed/removed), the test above would
        # trivially pass with zero offending files, which would be a
        # false "all clear". This confirms the one real construction
        # site is genuinely still there.

        text = ALLOWED_FILE.read_text(encoding="utf-8")
        self.assertRegex(text, FORBIDDEN)


if __name__ == "__main__":
    unittest.main()
