import random
from typing import TYPE_CHECKING

from ai2thor.controller import Controller

from procthor.constants import OUTDOOR_ROOM_ID
from procthor.utils.types import BoundaryGroups, Wall, Split
from ..databases import ProcTHORDatabase

if TYPE_CHECKING:
    from . import PartialHouse


def default_add_exterior_walls(
    partial_house: "PartialHouse",
    controller: Controller,
    pt_db: ProcTHORDatabase,
    split: Split,
    boundary_groups: BoundaryGroups,
) -> None:
    """Add walls to the outside of the house.

    Walls are one sided. So, adding exterior walls makes sure that when one looks
    from the window to another room, there is actually a wall there.
    """
    outdoor_boundary_groups = {
        bg: walls for bg, walls in boundary_groups.items() if OUTDOOR_ROOM_ID in bg
    }

    # NOTE: Intentionally not using solid colors as they're often too bright.
    material = random.choice(pt_db.MATERIAL_DATABASE["Wall"])

    house_walls = {wall["id"]: wall for wall in partial_house.walls}
    for bg, walls in outdoor_boundary_groups.items():
        room_id_n = bg[0] if bg[1] == OUTDOOR_ROOM_ID else bg[1]
        room_id = f"room|{room_id_n}"

        for (x0, z0), (x1, z1) in walls:
            wall_order = f"{min(x0, x1):.2f}|{min(z0, z1):.2f}|{max(x0, x1):.2f}|{max(z0, z1):.2f}"
            wall_id = f"wall|{room_id_n}|{wall_order}"
            house_wall = house_walls[wall_id]

            partial_house.walls.append(
                Wall(
                    id=f"wall|exterior|{wall_order}",
                    polygon=list(reversed(house_wall["polygon"])),
                    roomId=room_id,
                    material=material,
                )
            )
