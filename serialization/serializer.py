import json

from models.project import Project


class Serializer:

    @staticmethod
    def save(project: Project, filename: str):

        with open(filename, "w", encoding="utf-8") as f:

            json.dump(
                project.to_dict(),
                f,
                indent=4,
            )

    @staticmethod
    def load(filename: str):

        with open(filename, "r", encoding="utf-8") as f:

            data = json.load(f)

        return Project.from_dict(data)