import subprocess
from dataclasses import dataclass
from typing import Optional


# =====================================================
# Calibration Studio Phase 1 -- reproducibility metadata capture.
#
# A CalibrationSession's own reproducibility claim (docs/architecture
# for the approved persistent data model, Phase 3 "CalibrationSession")
# is incomplete without knowing WHICH CODE produced its result --
# something no file in this repository tracked before Calibration
# Studio (confirmed by repository-wide search during Phase 0). This is
# the one place that fact is captured, via a read-only `git` inspection
# -- never a simulation run, never a mutation, so calling it carries
# none of "no execution logic yet"'s risk.
#
# Every failure mode degrades honestly to (None, None) -- git not
# installed, this checkout not being a git repository, or any other
# subprocess failure -- rather than fabricating a plausible-looking
# commit hash or guessing a dirty state. Matches this codebase's own
# "never fabricate a value that wasn't actually observed" rule (see
# PROJECT_STATE.md's own Architectural Rules).
# =====================================================


@dataclass(frozen=True)
class GitProvenance:

    commit_hash: Optional[str]
    dirty: Optional[bool]

    def to_dict(self) -> dict:

        return {"commit_hash": self.commit_hash, "dirty": self.dirty}


def capture_git_provenance(cwd: Optional[str] = None) -> GitProvenance:

    commit_hash = _run_git(["git", "rev-parse", "HEAD"], cwd)

    if commit_hash is None:
        return GitProvenance(commit_hash=None, dirty=None)

    porcelain_status = _run_git(["git", "status", "--porcelain"], cwd)

    if porcelain_status is None:
        # Commit hash was resolvable but status wasn't (unexpected, but
        # possible under a partially-broken git invocation) -- an
        # unknown dirty state is honestly None, not assumed clean.
        return GitProvenance(commit_hash=commit_hash, dirty=None)

    return GitProvenance(commit_hash=commit_hash, dirty=len(porcelain_status) > 0)


def _run_git(args, cwd: Optional[str]) -> Optional[str]:

    try:
        completed = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    return completed.stdout.strip()
