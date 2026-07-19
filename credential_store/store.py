from abc import ABC, abstractmethod
from typing import Optional


class CredentialStore(ABC):

    # The seam between "a Camera Asset knows it needs a password" and
    # "something actually holds that password." Nothing in models/,
    # camera_manager/, virtual_camera/, or multi_camera_fusion/ ever
    # imports this -- only the project save/load boundary
    # (credential_store/project_credentials.py, serialization/
    # serializer.py) and a future Live adapter (an RTSPFrameSource
    # resolving a camera's real password at connect time) should.
    #
    # Concrete implementations today: LocalFileCredentialStore only.
    # Deliberately swappable later for Windows Credential Manager, an
    # OS keychain, or an environment-based secret store -- none of
    # that is this milestone's job, only this interface is.

    @abstractmethod
    def save_credential(self, reference_id: str, password: str) -> None:
        ...

    @abstractmethod
    def get_credential(self, reference_id: str) -> Optional[str]:
        ...

    @abstractmethod
    def delete_credential(self, reference_id: str) -> None:
        ...

    @abstractmethod
    def has_credential(self, reference_id: str) -> bool:
        ...
