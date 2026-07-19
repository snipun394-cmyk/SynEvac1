import json
from typing import Optional

from credential_store.project_credentials import capture_and_clear_camera_credentials
from credential_store.store import CredentialStore
from models.project import Project


class Serializer:

    @staticmethod
    def save(project: Project, filename: str, credential_store: Optional[CredentialStore] = None):

        # credential_store is optional and defaults to None so every
        # existing 2-arg caller keeps working unchanged -- to_dict()
        # never emits a plaintext password either way (see
        # models.engineering_asset.ConnectionInfo.to_dict()); passing
        # a store additionally captures any freshly typed password
        # into it before it would otherwise just be dropped.
        if credential_store is not None:
            capture_and_clear_camera_credentials(project, credential_store)

        with open(filename, "w", encoding="utf-8") as f:

            json.dump(
                project.to_dict(),
                f,
                indent=4,
            )

    @staticmethod
    def load(filename: str, credential_store: Optional[CredentialStore] = None):

        with open(filename, "r", encoding="utf-8") as f:

            data = json.load(f)

        project = Project.from_dict(data)

        # Migrates any legacy plaintext password from an old project
        # file into the store and clears it from memory, so the very
        # next save() no longer carries it forward.
        if credential_store is not None:
            capture_and_clear_camera_credentials(project, credential_store)

        return project