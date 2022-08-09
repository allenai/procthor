"""Procedurally add windows and paintings to the house."""

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from ai2thor.controller import Controller
from shapely.geometry import LineString, MultiLineString, Polygon

from procthor.constants import OUTDOOR_ROOM_ID
from procthor.utils.types import (
    BoundaryGroups,
    BoundingBox,
    Object,
    Split,
    Vector3,
    Wall,
    Window,
)
from . import PartialHouse
from .objects import AssetGroup, ProceduralRoom
from ..databases import ProcTHORDatabase

MAX_CENTER_Y_HEIGHT = 3
"""Clips the height of the wall.

If the ceiling_height is > 3, then the position wall objects will be placed on the
wall will be centered vertically between (0, 3) meters.
"""

BETA_A = 12
"""Beta a parameter for sampling the y position of the asset."""

BETA_B = 12
"""Beta b parameter for sampling the y position of the asset."""


def add_padding_to_poly(
    poly: Sequence[Tuple[float, float]]
) -> Sequence[Tuple[float, float]]:
    """Add padding to a polygon.

    Makes it easier to subtract a polygon from a MultiLineString.
    """
    min_x = min(p[0] for p in poly)
    max_x = max(p[0] for p in poly)
    min_z = min(p[1] for p in poly)
    max_z = max(p[1] for p in poly)

    min_x -= POLYGON_PADDING
    max_x += POLYGON_PADDING
    min_z -= POLYGON_PADDING
    max_z += POLYGON_PADDING

    return [(min_x, min_z), (min_x, max_z), (max_x, max_z), (max_x, min_z)]


def get_assets_df(
    split: Split,
    asset_type: str,
    pt_db: ProcTHORDatabase,
) -> pd.DataFrame:
    """Return the available paintings to spawn into the scene."""
    return pd.DataFrame(
        [
            {
                "assetId": asset["assetId"],
                "xSize": asset["boundingBox"]["x"],
                "ySize": asset["boundingBox"]["y"],
                "zSize": asset["boundingBox"]["z"],
            }
            for asset in pt_db.ASSET_DATABASE[asset_type]
            if asset["split"] == split
        ]
    )


def get_boundary_strings(
    boundary_groups: Dict[Tuple[float, float], BoundaryGroups],
    from_room_types: Set[str],
    only_connected_to_outside: bool,
    room_type_map: Dict[int, str],
) -> Dict[int, MultiLineString]:
    """Get which rooms are connected to the outside and can have windows.

    Args:
        from_room_types: Which room types should and shouldn't be considered.
            For instance, pass in {"Bedroom", "Kitchen"} if you only want boundary
            strings from those rooms within the house.
        only_connected_to_outside: Should the boundaries only be boundaries between
            a room and the outside of the house? This may be particularly useful for
            windows.

    """
    room_boundary_groups = defaultdict(list)
    for (room0_id, room1_id), boundary_group in boundary_groups.items():
        if (
            only_connected_to_outside
            and room0_id != OUTDOOR_ROOM_ID
            and room1_id != OUTDOOR_ROOM_ID
        ):
            continue

        for room_id in [room0_id, room1_id]:
            if room_id != OUTDOOR_ROOM_ID and room_type_map[room_id] in from_room_types:
                room_boundary_groups[room_id].extend(list(boundary_group))

    for room_id in room_boundary_groups:
        room_boundary_groups[room_id] = MultiLineString(room_boundary_groups[room_id])
    return room_boundary_groups


def center_asset_y_position(
    asset_height: float, ceiling_height: float
) -> Tuple[float, float]:
    """Vertically center the asset in the middle of the wall.

    Returns the (min, max) bounds of the asset on the wall.
    """
    top_wall_position = min(ceiling_height, MAX_CENTER_Y_HEIGHT)
    diff = (top_wall_position - asset_height) / 2
    return diff, top_wall_position - diff


def sample_asset_y_position(
    asset_height: float,
    wall_object_heights: List[Dict[str, Any]],
    asset_top_down_poly: Polygon,
    ceiling_height: float,
) -> Tuple[float, float]:
    """Sample the asset position on middle of the wall.

    Parameters:
    - asset_height: the height of the wall asset.
    - wall_object_heights: the heights of all the wall objects in the room.
    - asset_line_string: the top down line string of the wall asset.

    Returns the (min, max) bounds of the asset on the wall.
    """
    bottom_open_position = 0.0
    if wall_object_heights:
        for wall_object in wall_object_heights:
            if wall_object["poly"].intersects(asset_top_down_poly):
                bottom_open_position = max(bottom_open_position, wall_object["height"])

    top_open_position = min(MAX_CENTER_Y_HEIGHT, ceiling_height)

    rand_range = top_open_position - bottom_open_position - asset_height
    center_y_height = (
        bottom_open_position
        + asset_height / 2
        + rand_range * np.random.beta(a=BETA_A, b=BETA_B)
    )

    return center_y_height + asset_height / 2, center_y_height - asset_height / 2


#%% Windows

ROOM_TYPES_WITH_WINDOWS = {"Bedroom", "Kitchen", "LivingRoom"}
"""The room types that windows can exist in."""

WINDOWS_PER_ROOM = {"population": [0, 1, 2], "weights": [0.125, 0.375, 0.5]}
"""Set the distribution over the number of windows attempted to be placed per room."""


def extract_wall_ids(
    boundary_strings: Dict[int, Union[LineString, MultiLineString]]
) -> Dict[int, Dict[str, Union[LineString, MultiLineString]]]:
    """Extract the wall id for each line string."""
    room_to_wall_to_string = dict()
    for room_id, boundary_multi_string in boundary_strings.items():
        wall_id_to_line_string = dict()
        for line_string in boundary_multi_string.geoms:
            start, end = line_string.boundary.geoms
            x1, z1 = start.x, start.y
            x2, z2 = end.x, end.y

            wall_id = (
                f"wall|{room_id}|"
                f"{min(x1, x2):.2f}|{min(z1, z2):.2f}|{max(x1, x2):.2f}|{max(z1, z2):.2f}"
            )
            wall_id_to_line_string[wall_id] = line_string
        room_to_wall_to_string[room_id] = wall_id_to_line_string
    return room_to_wall_to_string


def subtract_wall_assets(
    room_to_wall_to_string: Dict[int, Dict[str, Union[LineString, MultiLineString]]],
    rooms: Dict[int, ProceduralRoom],
    pt_db: ProcTHORDatabase,
    min_height: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Subtract assets on the edge of the room from the spawnable places for assets.

    This prevents spawning a window behind a Fridge, for instance.

    Parameters:
    - min_height: only subtract objects that are larger than min_height.

    Note that room_to_wall_to_string is mutated.

    Returns: a map from each room_id to a list of wall objects that can have a
        wall asset above it, and the height of that object.
    """
    wall_object_heights_per_room = dict()
    for room_id in room_to_wall_to_string.keys():
        wall_object_heights_per_room[room_id] = []
        for asset in rooms[room_id].assets:
            # NOTE: position filter
            if asset.anchor_type == "inMiddle":
                continue

            # NOTE: height filter
            if min_height is not None:
                # NOTE: assumes assets or asset groups are placed on the floor.
                asset_max_y = (
                    max(
                        pt_db.ASSET_ID_DATABASE[obj["assetId"]]["boundingBox"]["y"] / 2
                        + obj["position"]["y"]
                        for obj in asset.objects
                    )
                    if isinstance(asset, AssetGroup)
                    else (
                        pt_db.ASSET_ID_DATABASE[asset.asset_id]["boundingBox"]["y"] / 2
                        + asset.position["y"]
                    )
                )
                if asset_max_y < min_height:
                    x_bounds = sorted(
                        [asset.top_down_poly[0][0], asset.top_down_poly[2][0]]
                    )
                    z_bounds = sorted(
                        [asset.top_down_poly[0][1], asset.top_down_poly[2][1]]
                    )

                    x_bounds[0] -= POLYGON_PADDING
                    z_bounds[0] -= POLYGON_PADDING
                    x_bounds[1] += POLYGON_PADDING
                    z_bounds[1] += POLYGON_PADDING
                    poly = Polygon(
                        [
                            (x_bounds[0], z_bounds[0]),
                            (x_bounds[1], z_bounds[0]),
                            (x_bounds[1], z_bounds[1]),
                            (x_bounds[0], z_bounds[1]),
                        ]
                    )
                    wall_object_heights_per_room[room_id].append(
                        {"poly": poly, "height": asset_max_y}
                    )
                    continue

            padded_asset_poly = Polygon(
                add_padding_to_poly(poly=asset.top_down_poly_with_margin)
            )
            for wall_id in room_to_wall_to_string[room_id].keys():
                room_to_wall_to_string[room_id][wall_id] -= padded_asset_poly
    return wall_object_heights_per_room


def convert_multi_lines_to_single_lines(
    room_to_wall_to_string: Dict[int, Dict[str, Union[LineString, MultiLineString]]],
) -> None:
    """Converts all MultiLineStrings to individual LineStrings."""
    for room_id in list(room_to_wall_to_string.keys()).copy():
        for wall_id in list(room_to_wall_to_string[room_id].keys()).copy():
            line_string = room_to_wall_to_string[room_id][wall_id]
            if isinstance(line_string, MultiLineString):
                # NOTE: At least 1 asset was placed on this wall, causing it
                # to be cut up.
                room_to_wall_to_string[room_id][wall_id] = [
                    ls for ls in line_string.geoms
                ]
            else:
                # NOTE: No assets were on this wall.
                room_to_wall_to_string[room_id][wall_id] = [line_string]


def get_line_string_df_map(
    room_to_wall_to_string: Dict[int, Dict[str, Union[LineString, MultiLineString]]]
) -> Dict[int, pd.DataFrame]:
    """Return a map of each roomId to the line strings df."""
    rooms_lines_df_map = dict()
    for room_id, wall_id_to_line_string in room_to_wall_to_string.items():
        lines = []
        for wall_id, line_strings in wall_id_to_line_string.items():
            for line_string in line_strings:
                bounds = line_string.boundary.geoms
                if not bounds:
                    continue
                start, end = bounds
                x1, z1 = start.x, start.y
                x2, z2 = end.x, end.y
                lines.append(
                    {
                        "length": line_string.length,
                        "x1": min(x1, x2),
                        "x2": max(x1, x2),
                        "z1": min(z1, z2),
                        "z2": max(z1, z2),
                        "lineString": line_string,
                        "wallId": wall_id,
                    }
                )
        if lines:
            rooms_lines_df_map[room_id] = pd.DataFrame(lines)
    return rooms_lines_df_map


def filter_room_lines_df(
    room_lines_df: pd.DataFrame, min_asset_size: float
) -> pd.DataFrame:
    """Filter each line string to have a greater length than min_asset_size.

    Parameters:
    - min_asset_size: the minimum width an asset will take up on the wall.
    """
    return room_lines_df[room_lines_df["length"] > min_asset_size]


def add_windows(
    partial_house: PartialHouse,
    rooms: Dict[int, ProceduralRoom],
    boundary_groups: Dict[Tuple[float, float], BoundaryGroups],
    split: Split,
    room_type_map: Dict[int, str],
    wall_map: Dict[str, Wall],
    ceiling_height: float,
    pt_db: ProcTHORDatabase,
) -> None:
    """Add windows to the house."""
    window_boundary_strings = get_boundary_strings(
        boundary_groups=boundary_groups,
        from_room_types=ROOM_TYPES_WITH_WINDOWS,
        only_connected_to_outside=True,
        room_type_map=room_type_map,
    )
    room_to_wall_to_string = extract_wall_ids(boundary_strings=window_boundary_strings)
    subtract_wall_assets(
        room_to_wall_to_string=room_to_wall_to_string, rooms=rooms, pt_db=pt_db
    )
    convert_multi_lines_to_single_lines(room_to_wall_to_string=room_to_wall_to_string)
    rooms_lines_df_map = get_line_string_df_map(
        room_to_wall_to_string=room_to_wall_to_string
    )

    # NOTE: Doors and windows cannot share the same wall, per AI2-THOR limitation.
    remove_wall_ids = set()
    for door in partial_house.doors:
        if "exterior" in door["wall0"] is None:
            remove_wall_ids.add(door["wall1"])
        elif "exterior" in door["wall1"]:
            remove_wall_ids.add(door["wall0"])
    for room_id, room_lines_df in rooms_lines_df_map.items():
        rooms_lines_df_map[room_id] = room_lines_df[
            ~room_lines_df["wallId"].isin(remove_wall_ids)
        ]

    # NOTE: Get the window assets
    windows_df = get_assets_df(split=split, asset_type="Window", pt_db=pt_db)
    min_window_size = windows_df["xSize"].min()

    # NOTE: Sample windows to place
    partial_house.windows = []
    max_windows_in_rooms = random.choices(k=len(rooms_lines_df_map), **WINDOWS_PER_ROOM)
    for max_windows_in_room, (room_id, room_lines_df) in zip(
        max_windows_in_rooms, rooms_lines_df_map.items()
    ):
        room_lines_df = filter_room_lines_df(
            room_lines_df=room_lines_df, min_asset_size=min_window_size
        )
        for window_i in range(max_windows_in_room):
            # NOTE: No more space on the walls
            if not len(room_lines_df):
                break

            # NOTE: sample the line string
            room_line_i = random.choices(
                population=list(room_lines_df.index),
                weights=list(room_lines_df["length"]),
                k=1,
            )[0]
            room_line = room_lines_df.loc[room_line_i]

            # NOTE: sample the window
            window_candidates = windows_df[windows_df["xSize"] < room_line["length"]]
            window = window_candidates.sample()

            # NOTE: Choose the position of the window
            start_position = random.random() * (
                room_line["length"] - window["xSize"].iloc[0]
            )

            # NOTE: flips the position of the wall depending on how the polygon
            # is specified.
            wall_poly = wall_map[room_line["wallId"]]["polygon"]
            if abs(wall_poly[0]["x"] - wall_poly[1]["x"]) < 1e-3:
                # NOTE: changes along z
                p1 = wall_poly[0]["z"]
                p2 = wall_poly[1]["z"]
                start_position += room_line["z1"] - min(p1, p2)
            else:
                # NOTE: changes along x
                p1 = wall_poly[0]["x"]
                p2 = wall_poly[1]["x"]
                start_position += room_line["x1"] - min(p1, p2)
            if p1 < p2:
                min_x = start_position
            else:
                min_x = abs(p2 - p1) - start_position - window["xSize"].iloc[0]

            min_y, _ = center_asset_y_position(
                asset_height=window["ySize"].iloc[0], ceiling_height=ceiling_height
            )

            wall_order = room_line["wallId"][
                room_line["wallId"].find("|", len("wall|")) :
            ]

            asset_id = window["assetId"].iloc[0]
            wall_hole = pt_db.WALL_HOLES[asset_id]

            offset = wall_hole["offset"]

            # NOTE: similar to doors, x is the direction along the wall,
            # where z is currently being ignored.
            partial_house.windows.append(
                Window(
                    id=f"window|{room_id}|{window_i}",
                    assetId=window["assetId"].iloc[0],
                    room0=f"room|{room_id}",
                    room1=f"room|{room_id}",
                    wall0=room_line["wallId"],
                    wall1=f"wall|exterior{wall_order}",
                    boundingBox=BoundingBox(
                        min=Vector3(x=min_x - offset["x"], y=min_y, z=0),
                        max=Vector3(
                            x=min_x + wall_hole["max"]["x"] - offset["x"],
                            y=min_y + wall_hole["max"]["y"] - offset["y"],
                            z=0,
                        ),
                    ),
                    assetOffset=wall_hole["offset"],
                )
            )

            # NOTE: Currently, there is only support in AI2-THOR for one
            # window per wall.
            room_lines_df = room_lines_df[
                room_lines_df["wallId"] != room_line["wallId"]
            ]


#%% Paintings
PAINTINGS_PER_ROOM = {
    "population": [0, 1, 2, 3, 4],
    "weights": [0.05, 0.1, 0.5, 0.25, 0.1],
}
"""The distribution over the maximum paintings attempted to be placed per room."""

ROOM_TYPES_WITH_PAINTINGS = {"Bedroom", "Kitchen", "LivingRoom", "Bathroom"}
"""The room types that windows can exist in."""

PAINTING_WALL_PADDING = 5e-3
"""Avoids the clipping jitter on the other side of the wall."""

NO_PAINTINGS_ABOVE_ASSETS_OF_HEIGHT = 1.15
"""Place no paintings above assets of this height, in meters."""

POLYGON_PADDING: float = 1e-3
"""Adds around a line or polygon when subtracting it from another line.

Alleviates any comparison issues that arise to floating point arithmetic.
"""

ALLOW_DUPLICATE_PAINTINGS_IN_HOUSE = False
"""Allow two of the same painting in the same house."""


def subtract_doors_and_windows(
    room_to_wall_to_string: Dict[int, Union[LineString, MultiLineString]],
    partial_house: PartialHouse,
    wall_map: Dict[str, Wall],
    pt_db: ProcTHORDatabase,
) -> None:
    """Subtracts doors and walls from the wall line strings in room_to_wall_to_string."""
    # NOTE: Subtract empty walls that openly connect two rooms
    for room_id, wall_to_string in room_to_wall_to_string.items():
        for wall_id in wall_to_string.copy():
            if "empty" in wall_map[wall_id] and wall_map[wall_id]["empty"]:
                del wall_to_string[wall_id]

    # NOTE: subtract actual doors
    for obj in partial_house.windows + partial_house.doors:
        room_ids = []
        if obj["room0"] is not None:
            room_ids.append(int(obj["room0"][len("room|") :]))
        if obj["room1"] is not None:
            room_ids.append(int(obj["room1"][len("room|") :]))

        obj_width = pt_db.ASSET_ID_DATABASE[obj["assetId"]]["boundingBox"]["x"]
        center_pos_along_wall = (
            obj["boundingBox"]["min"]["x"] + obj["boundingBox"]["max"]["x"]
        ) / 2

        wall_0 = obj["wall0"] if obj["wall0"] is not None else obj["wall1"]
        p0 = wall_map[wall_0]["polygon"][0]
        p1 = wall_map[wall_0]["polygon"][1]
        if abs(p0["x"] - p1["x"]) < 1e-3:
            # NOTE: changes along z
            x = p0["x"]
            if p1["z"] > p0["z"]:
                poly = [
                    (
                        x - POLYGON_PADDING,
                        p0["z"] + center_pos_along_wall - obj_width / 2,
                    ),
                    (
                        x + POLYGON_PADDING,
                        p0["z"] + center_pos_along_wall - obj_width / 2,
                    ),
                    (
                        x + POLYGON_PADDING,
                        p0["z"] + center_pos_along_wall + obj_width / 2,
                    ),
                    (
                        x - POLYGON_PADDING,
                        p0["z"] + center_pos_along_wall + obj_width / 2,
                    ),
                ]
            else:
                poly = [
                    (
                        x - POLYGON_PADDING,
                        p0["z"] - center_pos_along_wall - obj_width / 2,
                    ),
                    (
                        x + POLYGON_PADDING,
                        p0["z"] - center_pos_along_wall - obj_width / 2,
                    ),
                    (
                        x + POLYGON_PADDING,
                        p0["z"] - center_pos_along_wall + obj_width / 2,
                    ),
                    (
                        x - POLYGON_PADDING,
                        p0["z"] - center_pos_along_wall + obj_width / 2,
                    ),
                ]
        else:
            # NOTE: changes along x
            z = p0["z"]
            if p1["x"] > p0["x"]:
                poly = [
                    (
                        p0["x"] + center_pos_along_wall - obj_width / 2,
                        z - POLYGON_PADDING,
                    ),
                    (
                        p0["x"] + center_pos_along_wall - obj_width / 2,
                        z + POLYGON_PADDING,
                    ),
                    (
                        p0["x"] + center_pos_along_wall + obj_width / 2,
                        z + POLYGON_PADDING,
                    ),
                    (
                        p0["x"] + center_pos_along_wall + obj_width / 2,
                        z - POLYGON_PADDING,
                    ),
                ]
            else:
                poly = [
                    (
                        p0["x"] - center_pos_along_wall - obj_width / 2,
                        z - POLYGON_PADDING,
                    ),
                    (
                        p0["x"] - center_pos_along_wall - obj_width / 2,
                        z + POLYGON_PADDING,
                    ),
                    (
                        p0["x"] - center_pos_along_wall + obj_width / 2,
                        z + POLYGON_PADDING,
                    ),
                    (
                        p0["x"] - center_pos_along_wall + obj_width / 2,
                        z - POLYGON_PADDING,
                    ),
                ]

        poly = Polygon(poly)
        for room_id in room_ids:
            for wall_id in room_to_wall_to_string[room_id]:
                room_to_wall_to_string[room_id][wall_id] -= poly


def get_wall_placement_info(
    wall_poly: List[Vector3],
    room_line: pd.DataFrame,
    asset: pd.DataFrame,
    start_position: float,
) -> Dict[str, Any]:
    """Gets the placement info for placing an asset on a wall."""
    px0 = wall_poly[0]["x"]
    px1 = wall_poly[1]["x"]
    pz0 = wall_poly[0]["z"]
    pz1 = wall_poly[1]["z"]

    if abs(px0 - px1) < 1e-3:
        # NOTE: placed along z
        x = room_line["x1"]
        center_z_position = (
            room_line["z1"] + start_position + asset["xSize"].iloc[0] / 2
        )

        asset_poly = [
            (
                x - POLYGON_PADDING,
                room_line["z1"] + start_position - POLYGON_PADDING,
            ),
            (
                x + POLYGON_PADDING,
                room_line["z1"] + start_position - POLYGON_PADDING,
            ),
            (
                x + POLYGON_PADDING,
                room_line["z1"]
                + start_position
                + asset["xSize"].iloc[0]
                + POLYGON_PADDING,
            ),
            (
                x - POLYGON_PADDING,
                room_line["z1"]
                + start_position
                + asset["xSize"].iloc[0]
                + POLYGON_PADDING,
            ),
        ]

        if pz1 > pz0:
            center_x_position = x + asset["zSize"].iloc[0] / 2 + PAINTING_WALL_PADDING
            rotation = 90
        else:
            center_x_position = x - asset["zSize"].iloc[0] / 2 - PAINTING_WALL_PADDING
            rotation = 270
    else:
        # NOTE: placed along x
        z = room_line["z1"]
        center_x_position = (
            room_line["x1"] + start_position + asset["xSize"].iloc[0] / 2
        )

        asset_poly = [
            (
                room_line["x1"] + start_position - POLYGON_PADDING,
                z - POLYGON_PADDING,
            ),
            (
                room_line["x1"] + start_position - POLYGON_PADDING,
                z + POLYGON_PADDING,
            ),
            (
                room_line["x1"]
                + start_position
                + asset["xSize"].iloc[0]
                + POLYGON_PADDING,
                z + POLYGON_PADDING,
            ),
            (
                room_line["x1"]
                + start_position
                + asset["xSize"].iloc[0]
                + POLYGON_PADDING,
                z - POLYGON_PADDING,
            ),
        ]

        if px1 > px0:
            center_z_position = z - asset["zSize"].iloc[0] / 2 - PAINTING_WALL_PADDING
            rotation = 180
        else:
            center_z_position = z + asset["zSize"].iloc[0] / 2 + PAINTING_WALL_PADDING
            rotation = 0

    poly = Polygon(asset_poly)
    return {
        "poly": poly,
        "rotation": rotation,
        "centerX": center_x_position,
        "centerZ": center_z_position,
    }


def setup_wall_placement(
    boundary_groups: Dict[Tuple[float, float], BoundaryGroups],
    room_type_map: Dict[int, str],
    partial_house: PartialHouse,
    wall_map: Dict[str, Wall],
    rooms: Dict[int, ProceduralRoom],
    pt_db: ProcTHORDatabase,
) -> Tuple[Any, Any]:
    painting_boundary_strings = get_boundary_strings(
        boundary_groups=boundary_groups,
        from_room_types=ROOM_TYPES_WITH_PAINTINGS,
        only_connected_to_outside=False,
        room_type_map=room_type_map,
    )
    room_to_wall_to_string = extract_wall_ids(
        boundary_strings=painting_boundary_strings
    )

    subtract_doors_and_windows(
        room_to_wall_to_string=room_to_wall_to_string,
        partial_house=partial_house,
        wall_map=wall_map,
        pt_db=pt_db,
    )
    wall_object_heights_per_room = subtract_wall_assets(
        room_to_wall_to_string=room_to_wall_to_string,
        rooms=rooms,
        min_height=NO_PAINTINGS_ABOVE_ASSETS_OF_HEIGHT,
        pt_db=pt_db,
    )

    convert_multi_lines_to_single_lines(room_to_wall_to_string=room_to_wall_to_string)
    rooms_lines_df_map = get_line_string_df_map(
        room_to_wall_to_string=room_to_wall_to_string
    )

    return rooms_lines_df_map, wall_object_heights_per_room


def add_paintings(
    partial_house: PartialHouse,
    rooms: Dict[int, ProceduralRoom],
    split: Split,
    wall_map: Dict[str, Wall],
    ceiling_height: float,
    tvs_per_room,
    rooms_lines_df_map,
    wall_object_heights_per_room,
    pt_db: ProcTHORDatabase,
) -> None:
    """Add paintings to the house."""
    paintings_df = get_assets_df(split=split, asset_type="Painting", pt_db=pt_db)

    max_paintings_in_rooms = random.choices(
        k=len(rooms_lines_df_map), **PAINTINGS_PER_ROOM
    )

    min_painting_size = paintings_df["xSize"].min()
    for max_paintings_in_room, (room_id, room_lines_df) in zip(
        max_paintings_in_rooms, rooms_lines_df_map.items()
    ):
        for painting_i in range(max_paintings_in_room):
            room_lines_df = filter_room_lines_df(
                room_lines_df=room_lines_df, min_asset_size=min_painting_size
            )

            # NOTE: No more space on the walls
            if not len(room_lines_df) or not len(paintings_df):
                break

            # NOTE: sample the line string
            room_line_i = random.choices(
                population=list(room_lines_df.index),
                weights=list(room_lines_df["length"]),
                k=1,
            )[0]
            room_line = room_lines_df.loc[room_line_i]

            # NOTE: sample the painting
            painting_candidates = paintings_df[
                paintings_df["xSize"] < room_line["length"]
            ]
            painting = painting_candidates.sample()

            # NOTE: Choose the position of the painting
            start_position = random.random() * (
                room_line["length"] - painting["xSize"].iloc[0]
            )

            wall_poly = wall_map[room_line["wallId"]]["polygon"]

            placement = get_wall_placement_info(
                wall_poly=wall_poly,
                room_line=room_line,
                asset=painting,
                start_position=start_position,
            )

            min_y, max_y = sample_asset_y_position(
                asset_height=painting["ySize"].iloc[0],
                wall_object_heights=wall_object_heights_per_room[room_id],
                asset_top_down_poly=placement["poly"],
                ceiling_height=ceiling_height,
            )
            center_y_position = (min_y + max_y) / 2

            partial_house.objects.append(
                Object(
                    id=f"{room_id}|{len(rooms[room_id].assets) + painting_i + tvs_per_room[room_id]}",
                    assetId=painting["assetId"].iloc[0],
                    position=Vector3(
                        x=placement["centerX"],
                        y=center_y_position,
                        z=placement["centerZ"],
                    ),
                    rotation=Vector3(x=0, y=placement["rotation"], z=0),
                    kinematic=True,
                )
            )

            # NOTE: subtract painting from valid locations in room
            room_lines_df = room_lines_df.drop(room_line_i)

            line_string = room_line["lineString"]
            line_string -= placement["poly"]

            if isinstance(line_string, MultiLineString):
                line_strings_to_add = [
                    ls for ls in line_string.geoms if ls.length > min_painting_size
                ]
            elif line_string.length > min_painting_size:
                line_strings_to_add = [line_string]
            else:
                line_strings_to_add = []

            if line_strings_to_add:
                lines_to_append = []
                for line_string in line_strings_to_add:
                    start, end = line_string.boundary.geoms
                    x1, z1 = start.x, start.y
                    x2, z2 = end.x, end.y
                    lines_to_append.append(
                        {
                            "length": line_string.length,
                            "x1": min(x1, x2),
                            "x2": max(x1, x2),
                            "z1": min(z1, z2),
                            "z2": max(z1, z2),
                            "lineString": line_string,
                            "wallId": room_line["wallId"],
                        }
                    )
                lines_to_append = pd.DataFrame(lines_to_append)
                room_lines_df = pd.concat(
                    [room_lines_df, lines_to_append], ignore_index=True
                )

            # NOTE: Don't allow the same painting to be spawned in.
            if not ALLOW_DUPLICATE_PAINTINGS_IN_HOUSE:
                paintings_df = paintings_df.drop(painting.index)
                min_painting_size = paintings_df["xSize"].min()


#%% add Televisions to the wall
VALID_WALL_TELEVISIONS = {
    "Television_14",
    "Television_16",
    "Television_25",
    "Television_27",
    "Television_3",
    "Television_9",
}

ROOMS_WITH_WALL_TELEVISIONS = {
    "LivingRoom": {"p": 0.8},
    "Kitchen": {"p": 0.25},
    "Bedroom": {"p": 0.4},
}
"""Probability of having a wall television in a room without a television."""


def add_televisions(
    partial_house: PartialHouse,
    rooms: Dict[int, ProceduralRoom],
    boundary_groups: Dict[Tuple[float, float], BoundaryGroups],
    split: Split,
    room_type_map: Dict[int, str],
    wall_map: Dict[str, Wall],
    ceiling_height: float,
    pt_db: ProcTHORDatabase,
):
    """Add paintings to the house."""
    rooms_lines_df_map, wall_object_heights_per_room = setup_wall_placement(
        boundary_groups=boundary_groups,
        room_type_map=room_type_map,
        partial_house=partial_house,
        wall_map=wall_map,
        rooms=rooms,
        pt_db=pt_db,
    )
    assets_df = get_assets_df(split=split, asset_type="Television", pt_db=pt_db)
    assets_df = assets_df[assets_df["assetId"].isin(VALID_WALL_TELEVISIONS)]
    if len(assets_df) == 0:
        return rooms_lines_df_map, wall_object_heights_per_room

    tvs_per_room = defaultdict(int)

    min_asset_size = assets_df["xSize"].min()
    for room_id, room_lines_df in rooms_lines_df_map.items():
        if (
            room_type_map[room_id] not in ROOMS_WITH_WALL_TELEVISIONS
            or random.random()
            > ROOMS_WITH_WALL_TELEVISIONS[room_type_map[room_id]]["p"]
        ):
            continue

        # NOTE: skip any rooms that already have a television
        room_has_television = False
        for asset in rooms[room_id].assets:
            if isinstance(asset, AssetGroup):
                for obj in asset.objects:
                    if (
                        pt_db.ASSET_ID_DATABASE[obj["assetId"]]["objectType"]
                        == "Television"
                    ):
                        room_has_television = True
            elif pt_db.ASSET_ID_DATABASE[asset.asset_id]["objectType"] == "Television":
                room_has_television = True
            if room_has_television:
                break
        if room_has_television:
            continue

        room_lines_df = filter_room_lines_df(
            room_lines_df=room_lines_df, min_asset_size=min_asset_size
        )

        # NOTE: No more space on the walls
        if not len(room_lines_df) or not len(assets_df):
            break

        # NOTE: sample the line string
        room_line_i = random.choices(
            population=list(room_lines_df.index),
            weights=list(room_lines_df["length"]),
            k=1,
        )[0]
        room_line = room_lines_df.loc[room_line_i]

        # NOTE: sample the asset
        asset_candidates = assets_df[assets_df["xSize"] < room_line["length"]]
        asset = asset_candidates.sample()

        # NOTE: Choose the position of the asset
        start_position = random.random() * (
            room_line["length"] - asset["xSize"].iloc[0]
        )

        wall_poly = wall_map[room_line["wallId"]]["polygon"]

        placement = get_wall_placement_info(
            wall_poly=wall_poly,
            room_line=room_line,
            asset=asset,
            start_position=start_position,
        )

        min_y, max_y = sample_asset_y_position(
            asset_height=asset["ySize"].iloc[0],
            wall_object_heights=wall_object_heights_per_room[room_id],
            asset_top_down_poly=placement["poly"],
            ceiling_height=ceiling_height,
        )
        center_y_position = (min_y + max_y) / 2

        partial_house.objects.append(
            Object(
                id=f"{room_id}|{len(rooms[room_id].assets)}",
                assetId=asset["assetId"].iloc[0],
                position=Vector3(
                    x=placement["centerX"],
                    y=center_y_position,
                    z=placement["centerZ"],
                ),
                rotation=Vector3(x=0, y=placement["rotation"], z=0),
                kinematic=True,
            )
        )

        # NOTE: subtract asset from valid locations in room
        rooms_lines_df_map[room_id] = rooms_lines_df_map[room_id].drop(room_line_i)

        line_string = room_line["lineString"]
        line_string -= placement["poly"]

        if isinstance(line_string, MultiLineString):
            line_strings_to_add = [ls for ls in line_string.geoms]
        elif line_string.length:
            line_strings_to_add = [line_string]
        else:
            line_strings_to_add = []

        if line_strings_to_add:
            lines_to_append = []
            for line_string in line_strings_to_add:
                start, end = line_string.boundary.geoms
                x1, z1 = start.x, start.y
                x2, z2 = end.x, end.y
                lines_to_append.append(
                    {
                        "length": line_string.length,
                        "x1": min(x1, x2),
                        "x2": max(x1, x2),
                        "z1": min(z1, z2),
                        "z2": max(z1, z2),
                        "lineString": line_string,
                        "wallId": room_line["wallId"],
                    }
                )
            lines_to_append = pd.DataFrame(lines_to_append)
            rooms_lines_df_map[room_id] = pd.concat(
                [rooms_lines_df_map[room_id], lines_to_append], ignore_index=True
            )
        tvs_per_room[room_id] += 1
    return tvs_per_room, rooms_lines_df_map, wall_object_heights_per_room


#%% Add wall objects
def default_add_wall_objects(
    partial_house: PartialHouse,
    controller: Controller,
    pt_db: ProcTHORDatabase,
    split: Split,
    rooms: Dict[int, ProceduralRoom],
    boundary_groups: BoundaryGroups,
    room_type_map: Dict[int, str],
    ceiling_height: float,
) -> None:
    """Add wall objects to the house."""
    wall_map = {w["id"]: w for w in partial_house.walls}
    add_windows(
        partial_house=partial_house,
        rooms=rooms,
        boundary_groups=boundary_groups,
        split=split,
        room_type_map=room_type_map,
        wall_map=wall_map,
        ceiling_height=ceiling_height,
        pt_db=pt_db,
    )
    tvs_per_room, rooms_lines_df_map, wall_object_heights_per_room = add_televisions(
        partial_house=partial_house,
        rooms=rooms,
        boundary_groups=boundary_groups,
        split=split,
        room_type_map=room_type_map,
        wall_map=wall_map,
        ceiling_height=ceiling_height,
        pt_db=pt_db,
    )

    add_paintings(
        partial_house=partial_house,
        rooms=rooms,
        split=split,
        wall_map=wall_map,
        ceiling_height=ceiling_height,
        tvs_per_room=tvs_per_room,
        rooms_lines_df_map=rooms_lines_df_map,
        wall_object_heights_per_room=wall_object_heights_per_room,
        pt_db=pt_db,
    )
