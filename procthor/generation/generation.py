from collections import defaultdict
from typing import Dict, Literal, Optional, Sequence, Set

import numpy as np
from shapely.geometry import Polygon

from procthor.constants import FLOOR_Y, OUTDOOR_ROOM_ID
from procthor.utils.types import (
    BoundaryGroups,
    ProceduralParameters,
    RoomType,
    Vector3,
    Wall,
    XZPoly,
)
from .ceiling_height import sample_ceiling_height
from .floorplan_generation import generate_floorplan
from .house import HouseStructure, PartialHouse
from .interior_boundaries import sample_interior_boundary, DEFAULT_AVERAGE_ROOM_SIZE
from .room_specs import RoomSpec


def consolidate_walls(walls):
    """Join neighboring walls together.

    Example: if one of the walls is ::

        [
            ((0, 0), (0, 1)),
            ((0, 1), (0, 2)),
            ((0, 2), (0, 3)),
            ((0, 3), (0, 4)),
            ((0, 4), (0, 5)),
            ((0, 5), (0, 6)),
            ((0, 6), (0, 7)),
            ((0, 7), (0, 8)),
            ((0, 8), (0, 9)),
            ((0, 0), (1, 0)),
            ((1, 0), (2, 0)),
            ((2, 0), (3, 0)),
            ((3, 0), (4, 0)),
            ((4, 0), (5, 0)),
            ((5, 0), (6, 0)),
            ((6, 0), (7, 0)),
            ((7, 0), (8, 0)),
            ((8, 0), (9, 0))
        ]

    it becomes ::

        {
            ((0, 0), (0, 9)),
            ((0, 0), (9, 0))
        }
    """
    out = dict()
    for wall_group_id, wall_pairs in walls.items():
        wall_map = defaultdict(lambda: set())
        wall_map = dict()
        for wall in wall_pairs:
            if wall[0] not in wall_map:
                wall_map[wall[0]] = set()
            wall_map[wall[0]].add(wall[1])

        did_update = True
        while did_update:
            did_update = False
            for w1_1 in wall_map.copy():
                if w1_1 not in wall_map:
                    continue
                break_outer = False
                for w1_2 in wall_map[w1_1]:
                    if w1_2 in wall_map:
                        w2_1 = w1_2
                        for w2_2 in wall_map[w2_1]:
                            if (
                                w1_1[0] == w1_2[0] == w2_1[0] == w2_2[0]
                                or w1_1[1] == w1_2[1] == w2_1[1] == w2_2[1]
                            ):
                                wall_map[w2_1].remove(w2_2)
                                if not wall_map[w2_1]:
                                    del wall_map[w2_1]

                                wall_map[w1_1].remove(w2_1)
                                wall_map[w1_1].add(w2_2)

                                did_update = True
                                break_outer = True
                                break
                        if break_outer:
                            break
                    if break_outer:
                        break
        out[wall_group_id] = set([(w1, w2) for w1 in wall_map for w2 in wall_map[w1]])
    return out


def scale_boundary_groups(
    boundary_groups: BoundaryGroups, scale: float, precision: int = 3
) -> BoundaryGroups:
    out = dict()
    for key, lines in boundary_groups.items():
        scaled_lines = set()
        for ((x0, z0), (x1, z1)) in lines:
            scaled_lines.add(
                (
                    (round(x0 * scale, precision), round(z0 * scale, precision)),
                    (round(x1 * scale, precision), round(z1 * scale, precision)),
                )
            )
        out[key] = scaled_lines
    return out


def find_walls(floorplan: np.array):
    walls = defaultdict(list)
    for row in range(len(floorplan) - 1):
        for col in range(len(floorplan[0]) - 1):
            a = floorplan[row, col]
            b = floorplan[row, col + 1]
            if a != b:
                walls[(int(min(a, b)), int(max(a, b)))].append(
                    ((row - 1, col), (row, col))
                )
            b = floorplan[row + 1, col]
            if a != b:
                walls[(int(min(a, b)), int(max(a, b)))].append(
                    ((row, col - 1), (row, col))
                )
    return walls


def get_floor_polygons(xz_poly_map: dict) -> Dict[str, Polygon]:
    """Return a shapely Polygon for each floor in the room."""
    floor_polygons = dict()
    for room_id, xz_poly in xz_poly_map.items():
        floor_polygon = []
        for ((x0, z0), (x1, z1)) in xz_poly:
            floor_polygon.append((x0, z0))
        floor_polygon.append((x1, z1))
        floor_polygons[f"room|{room_id}"] = Polygon(floor_polygon)
    return floor_polygons


def get_wall_loop(walls: Sequence[XZPoly]):
    walls_left = set(walls)
    out = [walls[0]]
    walls_left.remove(walls[0])
    while walls_left:
        for wall in walls_left:
            if out[-1][1] == wall[0]:
                out.append(wall)
                walls_left.remove(wall)
                break
            elif out[-1][1] == wall[1]:
                out.append((wall[1], wall[0]))
                walls_left.remove(wall)
                break
        else:
            raise Exception(f"No connecting wall for {out[-1]}!")
    return out


def get_xz_poly_map(boundary_groups, room_ids: Set[int]) -> Dict[int, XZPoly]:
    out = dict()
    for room_id in room_ids:
        room_walls = []
        for k in [k for k in boundary_groups.keys() if room_id in k]:
            room_walls.extend(boundary_groups[k])

        room_wall_loop = get_wall_loop(room_walls)

        # determines if the loop is counter-clockwise, flips if it is
        edge_sum = 0
        for (x0, z0), (x1, z1) in room_wall_loop:
            dist = x0 * z1 - x1 * z0
            edge_sum += dist
        if edge_sum > 0:
            room_wall_loop = [(p1, p0) for p0, p1 in reversed(room_wall_loop)]

        out[room_id] = room_wall_loop
    return out


def default_sample_house_structure(
    interior_boundary: Optional[np.ndarray],
    room_ids: Set,
    room_spec: "RoomSpec",
    interior_boundary_scale: float,
    average_room_size: int = DEFAULT_AVERAGE_ROOM_SIZE,
) -> HouseStructure:
    if interior_boundary is None:
        interior_boundary = sample_interior_boundary(
            num_rooms=len(room_ids),
            average_room_size=average_room_size,
            dims=None if room_spec.dims is None else room_spec.dims(),
        )
    floorplan = generate_floorplan(
        room_spec=room_spec, interior_boundary=interior_boundary
    )

    # NOTE: Pad the floorplan with the outdoor room id to make
    # it easier to find walls.
    floorplan = np.pad(
        floorplan, pad_width=1, mode="constant", constant_values=OUTDOOR_ROOM_ID
    )
    rowcol_walls = find_walls(floorplan=floorplan)
    boundary_groups = consolidate_walls(walls=rowcol_walls)
    boundary_groups = scale_boundary_groups(
        boundary_groups=boundary_groups,
        scale=interior_boundary_scale,
    )
    xz_poly_map = get_xz_poly_map(boundary_groups=boundary_groups, room_ids=room_ids)
    ceiling_height = sample_ceiling_height()
    return HouseStructure(
        interior_boundary=interior_boundary,
        floorplan=floorplan,
        rowcol_walls=rowcol_walls,
        boundary_groups=boundary_groups,
        xz_poly_map=xz_poly_map,
        ceiling_height=ceiling_height,
    )


def create_empty_partial_house(
    xz_poly_map: Dict[int, XZPoly],
    room_type_map: Dict[int, Literal["Bedroom", "Bathroom", "Kitchen", "LivingRoom"]],
    ceiling_height: float,
) -> PartialHouse:
    walls = []
    for room_id, xz_poly in xz_poly_map.items():
        for ((x0, z0), (x1, z1)) in xz_poly:
            wall_id = f"wall|{room_id}|{min(x0, x1):.2f}|{min(z0, z1):.2f}|{max(x0, x1):.2f}|{max(z0, z1):.2f}"
            wall = Wall(
                id=wall_id,
                roomId=f"room|{room_id}",
                polygon=[
                    Vector3(x=x0, y=FLOOR_Y, z=z0),
                    Vector3(x=x1, y=FLOOR_Y, z=z1),
                    Vector3(x=x0, y=FLOOR_Y + ceiling_height, z=z0),
                    Vector3(x=x1, y=FLOOR_Y + ceiling_height, z=z1),
                ],
            )
            walls.append(wall)

    return PartialHouse(
        room_types=[
            RoomType(
                id=f"room|{room_id}",
                roomType=room_type_map[room_id],
                children=[],
                ceilings=[],
                floorPolygon=[
                    Vector3(x=x0, y=0, z=z0) for ((x0, z0), (x1, z1)) in xz_poly
                ],
            )
            for room_id, xz_poly in xz_poly_map.items()
        ],
        walls=walls,
        objects=[],
        procedural_parameters=ProceduralParameters(
            floorColliderThickness=1.0,
            receptacleHeight=0.7,
            lights=[],
            reflections=[],
        ),
    )
