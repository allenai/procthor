import random
from typing import List

from procthor.databases import asset_id_database

from procthor.utils.types import Object

TYPES_TO_TOGGLE = {
    "DeskLamp",
    "FloorLamp",
}

TYPES_TO_DIRTY = {"Bed"}


def randomize_object_states(objects: List[Object]) -> None:
    """Randomize the states of objects."""

    def _randomize_states(objects: List[Object]) -> None:
        for obj in objects:
            obj_type = asset_id_database[obj["assetId"]]["objectType"]
            if obj_type in TYPES_TO_TOGGLE:
                obj["isOn"] = random.choice([True, False])
            if obj_type in TYPES_TO_DIRTY:
                obj["isDirty"] = random.choice([True, False])

            if "children" in obj:
                _randomize_states(obj["children"])

    _randomize_states(objects)
