import random

from ai2thor.controller import Controller

from procthor.databases import ProcTHORDatabase
from procthor.generation import PartialHouse
from procthor.utils.types import RGB, Skybox, Vector3, Split


def default_add_skybox(
    partial_house: PartialHouse,
    controller: Controller,
    pt_db: ProcTHORDatabase,
    split: Split,
) -> None:
    """Add a skybox to the scene."""
    skybox: Skybox = random.choice(list(pt_db.SKYBOXES.values()))
    time_of_day = skybox["timeOfDay"]

    partial_house.procedural_parameters["skyboxId"] = skybox["name"]

    lights = partial_house.procedural_parameters["lights"]
    directional_light = lights[0]
    point_lights = lights[1:]

    # NOTE: Set directional lights
    if time_of_day == "Midday":
        directional_light["intensity"] = 1
        directional_light["rgb"] = RGB(r=1.0, g=1.0, b=1.0)
        directional_light["rotation"] = Vector3(x=66, y=75, z=0)
        for point_light in point_lights:
            point_light["intensity"] = 0.45
    elif time_of_day == "GoldenHour":
        directional_light["intensity"] = 1
        directional_light["rgb"] = RGB(r=1.0, g=0.694, b=0.78)
        directional_light["rotation"] = Vector3(x=6, y=-166, z=0)
    elif time_of_day == "BlueHour":
        directional_light["intensity"] = 0.5
        directional_light["rgb"] = RGB(r=0.638, g=0.843, b=1.0)
        directional_light["rotation"] = Vector3(x=82, y=-30, z=0)
    elif time_of_day == "Midnight":
        raise Exception(
            "Currently do not support night skyboxes."
            " They appear too dark to see all of the objects."
        )
        directional_light["intensity"] = 0.3
        directional_light["rgb"] = RGB(r=0.93, g=0.965, b=1.0)
        directional_light["rotation"] = Vector3(x=41, y=-50, z=0)
