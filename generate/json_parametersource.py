from glob import glob

from utils.utils import JsonSerialization as json


class JsonParams:
    """
    Parameter source from json files in directory
    """

    def __init__(self, callback):
        self.callback = callback

    def read_and_start(self):
        files = glob("params/*.json")

        for file in files:
            print(file)
            with open(file) as f:
                param = json.loads(f.read())
            self.callback([param])
