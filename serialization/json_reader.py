import json


class JsonReader:

    @staticmethod
    def read(filename):

        with open(filename, "r", encoding="utf-8") as file:

            return json.load(file)