"""Config class"""
from typing import Dict

import json

class Config:
    """Config class which contains data, train, model, and util hyperparameters"""

    def __init__(self, data, train, model, util):
        self.data = data
        self.train = train
        self.model = model
        self.util = util

    @classmethod
    def from_json(cls, cfg: Dict) -> object:
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.data, params.train, params.model, params.util)

class HelperObject(object):
    """Helper class to convert json into Python object"""
    def __init__(self, dict_):
        self.__dict__.update(dict_)