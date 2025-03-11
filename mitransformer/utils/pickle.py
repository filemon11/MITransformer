"""Necessary to deal with old module names.
"""
from pickle import *  # noqa: F403, F401   # type: ignore

from pickle import _Unpickler, Unpickler


Unpickler = _Unpickler  # type: ignore  # noqa: 811

modNameMap = {
    "model": "mitransformer.models.model",
    "tokeniser": "mitransformer.data.tokeniser",
    "trainer": "mitransformer.train.trainer"
}


old_find_class = Unpickler.find_class


def find_class(self, moduleName: str, name):
    if moduleName in modNameMap:
        moduleName = modNameMap[moduleName]
    return old_find_class(self, moduleName, name)


setattr(Unpickler, "find_class", find_class)


def renamed_load(file_obj):
    return Unpickler(file_obj).load()


globals()["load"] = renamed_load
