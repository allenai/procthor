import random
from typing import TYPE_CHECKING

from procthor.databases import ProcTHORDatabase

if TYPE_CHECKING:
    from . import PartialHouse


P_ALL_WALLS_SAME = 0.35
"""Probability that all wall materials are the same."""

P_ALL_FLOOR_SAME = 0.15
"""Probability that all floor materials are the same."""

P_SAMPLE_SOLID_WALL_COLOR = 0.5
"""Probability of sampling a solid wall color instead of a material."""


def randomize_wall_and_floor_materials(
    partial_house: "PartialHouse",
    pt_db: ProcTHORDatabase,
) -> None:
    """Randomize the materials on each wall and floor."""
    randomize_wall_materials(partial_house, pt_db=pt_db)
    randomize_floor_materials(partial_house, pt_db=pt_db)


def sample_wall_params(
    pt_db: ProcTHORDatabase,
):
    if random.random() < P_SAMPLE_SOLID_WALL_COLOR:
        return {
            "color": random.choice(pt_db.SOLID_WALL_COLORS),
            "material": "PureWhite",
        }
    return {
        "material": random.choice(pt_db.MATERIAL_DATABASE["Wall"]),
    }


def randomize_wall_materials(
    partial_house: "PartialHouse",
    pt_db: ProcTHORDatabase,
) -> None:
    """Randomize the materials on each wall."""
    # NOTE: randomize all the walls to the same material.
    if random.random() < P_ALL_WALLS_SAME:
        wall_params = sample_wall_params(pt_db=pt_db)
        for wall in partial_house.walls:
            for k, v in wall_params.items():
                wall[k] = v

        # NOTE: set the ceiling
        partial_house.procedural_parameters["ceilingMaterial"] = wall_params["material"]
        if "color" in wall_params:
            partial_house.procedural_parameters["ceilingColor"] = wall_params["color"]

        return

    # NOTE: independently randomize each room's materials.
    room_ids = set()
    for wall in partial_house.walls:
        room_ids.add(wall["roomId"])
    room_ids.add("ceiling")

    wall_params_per_room = dict()
    for room_id in room_ids:
        wall_params_per_room[room_id] = sample_wall_params(pt_db=pt_db)

    for wall in partial_house.walls:
        for k, v in wall_params_per_room[wall["roomId"]].items():
            wall[k] = v

    # NOTE: randomize ceiling material
    partial_house.procedural_parameters["ceilingMaterial"] = wall_params_per_room[
        "ceiling"
    ]["material"]
    if "color" in wall_params_per_room["ceiling"]:
        partial_house.procedural_parameters["ceilingColor"] = wall_params_per_room[
            "ceiling"
        ]["color"]


def randomize_floor_materials(
    partial_house: "PartialHouse",
    pt_db: ProcTHORDatabase,
) -> None:
    """Randomize the materials on each floor."""
    if random.random() < P_ALL_FLOOR_SAME:
        floor_material = random.choice(pt_db.MATERIAL_DATABASE["Wood"])
        for room in partial_house.room_types:
            room["floorMaterial"] = floor_material
        return

    for room in partial_house.room_types:
        room["floorMaterial"] = random.choice(pt_db.MATERIAL_DATABASE["Wood"])
