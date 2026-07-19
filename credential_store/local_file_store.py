import json
from pathlib import Path
from typing import Optional

from credential_store.store import CredentialStore


DEFAULT_CREDENTIAL_PATH = Path.home() / ".synevac" / "credentials.json"


class LocalFileCredentialStore(CredentialStore):

    # Smallest-safe-architecture implementation for this milestone: a
    # single JSON file, keyed by reference_id -> password, living
    # outside the repository entirely (default: the user's home
    # directory, not anywhere a `git add` could ever reach) rather
    # than relying on .gitignore correctness. Local to the machine,
    # never synced, never part of a saved SynEvac project file.
    #
    # No eager I/O in __init__ -- the file is only created on the
    # first save_credential() call, so constructing a store (as
    # main_window.py now does unconditionally at startup) never
    # touches disk, and every existing test that saves/loads a
    # Project without passing a credential_store stays untouched.

    def __init__(self, path: Optional[Path] = None):

        self.path = Path(path) if path is not None else DEFAULT_CREDENTIAL_PATH

    # =====================================================

    def save_credential(self, reference_id: str, password: str) -> None:

        data = self._read()
        data[reference_id] = password
        self._write(data)

    def get_credential(self, reference_id: str) -> Optional[str]:

        return self._read().get(reference_id)

    def delete_credential(self, reference_id: str) -> None:

        data = self._read()

        if reference_id in data:
            del data[reference_id]
            self._write(data)

    def has_credential(self, reference_id: str) -> bool:

        return reference_id in self._read()

    # =====================================================

    def _read(self) -> dict:

        if not self.path.exists():
            return {}

        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: dict) -> None:

        self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
