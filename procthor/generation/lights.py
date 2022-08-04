from typing import Dict

from ai2thor.controller import Controller
from shapely.geometry import Polygon

from procthor.databases import ProcTHORDatabase
from procthor.generation import PartialHouse
from procthor.utils.types import RGB, Light, LightShadow, Vector3, Split


def default_add_lights(
    partial_house: PartialHouse,
    controller: Controller,
    pt_db: ProcTHORDatabase,
    split: Split,
    floor_polygons: Dict[str, Polygon],
    ceiling_height: float,
) -> None:
    """Adds lights to the house.

    Lights include:
    - A point light to the centroid of each room.
    - A directional light.

    Args:
        house: HouseDict, the house to add lights to.
        floor_polygons: Dict[str, Polygon] maps each room's id to the shapely polygon
            of each room's floor.
    """
    # add directional light
    lights = [
        Light(
            id="DirectionalLight",
            position=Vector3(x=0.84, y=0.1855, z=-1.09),
            rotation=Vector3(x=43.375, y=-3.902, z=-63.618),
            shadow=LightShadow(
                type="Soft",
                strength=1,
                normalBias=0,
                bias=0,
                nearPlane=0.2,
                resolution="FromQualitySettings",
            ),
            type="directional",
            intensity=0.35,
            indirectMultiplier=1.0,
            rgb=RGB(r=1.0, g=1.0, b=1.0),
        )
    ]

    # add point lights
    for room in partial_house.room_types:
        room_id = room["id"]
        x = floor_polygons[room_id].centroid.x

        # NOTE: with the 2d top-down polygon, y maps to the z direction.
        z = floor_polygons[room_id].centroid.y

        room_id_num = int(room_id[room_id.rfind("|") + 1 :])

        # NOTE: The point lights may be overwritten by the skybox.
        lights.append(
            Light(
                id=f"light_{room_id_num}",
                type="point",
                position=Vector3(x=x, y=ceiling_height - 0.2, z=z),
                intensity=0.75,
                range=15,
                rgb=RGB(r=1.0, g=0.855, b=0.722),
                shadow=LightShadow(
                    type="Soft",
                    strength=1,
                    normalBias=0,
                    bias=0.05,
                    nearPlane=0.2,
                    resolution="FromQualitySettings",
                ),
            )
        )

    partial_house.procedural_parameters["lights"] = lights
