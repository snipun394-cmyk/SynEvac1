from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass
class BaseObject:
    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    object_type: str = ""
    properties: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "object_type": self.object_type,
            "properties": self.properties,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }