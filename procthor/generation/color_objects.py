import random
from typing import List, TYPE_CHECKING

from procthor.databases import asset_id_database

if TYPE_CHECKING:
    pass
from procthor.utils.types import RGB, Object

OBJECT_TYPES_TO_COLOR_RANDOMIZE = {"Vase", "Statue", "Bottle"}

P_RANDOMIZE_COLORS = 0.7
"""Only randomize the colors of the object with p probability."""


def randomize_object_colors(objects: List[Object]) -> None:
    def _randomize_object_color(objects: List[Object]):
        """Recursively randomize the color of every object and child object."""
        for obj in objects:
            if (
                asset_id_database[obj["assetId"]]["objectType"]
                in OBJECT_TYPES_TO_COLOR_RANDOMIZE
                and random.random() < P_RANDOMIZE_COLORS
            ):
                obj["color"] = RGB(
                    r=random.random(), g=random.random(), b=random.random()
                )
            if "children" in obj:
                _randomize_object_color(obj["children"])

    _randomize_object_color(objects)
