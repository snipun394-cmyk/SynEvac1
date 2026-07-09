import json


class JsonWriter:

    @staticmethod
    def write(filename, data):

        with open(filename, "w", encoding="utf-8") as file:

            json.dump(
                data,
                file,
                indent=4,
            )