import copy
import itertools
import logging
import random
from collections import defaultdict
from collections.abc import Iterable
from typing import Dict, List, Set, Tuple, Union, TYPE_CHECKING, Any

import pandas as pd
from ai2thor.controller import Controller
from attr import field
from attrs import define
from shapely.geometry import Polygon

from procthor.constants import OUTDOOR_ROOM_ID
from procthor.utils.types import (
    BoundaryGroups,
    BoundingBox,
    Door,
    LeafRoom,
    MetaRoom,
    Split,
    Vector3,
    Wall,
)
from ..databases import ProcTHORDatabase

if TYPE_CHECKING:
    from . import PartialHouse

from .room_specs import RoomSpec

OPEN_ROOM_CONNECTIONS = [
    {"between": {"Kitchen", "LivingRoom"}, "p": 0.75, "pFrame": 0.5}
]
"""Which rooms may have door frames or no walls between them?

Parameters:
- between: The neighboring room types that may have an open room connection.
- p: probability of having a door frame or no walls between rooms.
- pFrame: probability of having a door frame instead of no walls.
"""

OPEN_ROOM_PADDING = 0.35
"""Padding per each room's side of an open room connection."""

PREFERRED_ENTRANCE_PADDING = 0.5
"""The amount of space to leave, in meters, behind of the door.

.. note::
    The entrance is on the opposite side of how the direction that the door
    opens.
"""

PADDING_IN_FRONT_OF_DOOR = 0.35
"""The amount of padding, in meters, in front of a door.

.. note::
    The front of the door is the direction that it opens.

.. note::
    This is in addition to the padding already provided to ensure that the door
    can fully open.
"""

PREFERRED_ROOMS_TO_OUTSIDE = {"Kitchen", "LivingRoom"}
"""Preferred room types to have doors to the outside."""

MIN_DOORS_TO_OUTSIDE = 1
"""Minimum number of rooms with doors to the outside."""

MAX_DOORS_TO_OUTSIDE = 1
"""Maximum number of rooms with doors to the outside."""

EPSILON = 1e-3
"""Small value to compare floats within a bound."""


def default_add_doors(
    partial_house: "PartialHouse",
    controller: Controller,
    pt_db: ProcTHORDatabase,
    split: Split,
) -> Dict[int, List[Polygon]]:
    """Add doors to the house."""
    assert split in {"train", "val", "test"}

    room_spec = partial_house.room_spec
    boundary_groups = partial_house.house_structure.boundary_groups

    room_spec_neighbors = get_room_spec_neighbors(room_spec=room_spec.spec)
    openings = select_openings(
        neighboring_rooms=set(boundary_groups.keys()),
        room_spec_neighbors=room_spec_neighbors,
        room_spec=room_spec,
    )
    door_walls = select_door_walls(
        openings=openings,
        boundary_groups=boundary_groups,
    )

    outdoor_openings = select_outdoor_openings(
        boundary_groups=boundary_groups, room_type_map=room_spec.room_type_map
    )
    outdoor_walls = select_door_walls(
        openings=outdoor_openings,
        boundary_groups=boundary_groups,
    )

    door_walls.update(outdoor_walls)
    polygons_to_subtract = add_door_meta(
        partial_house=partial_house,
        split=split,
        door_walls=door_walls,
        closed_doors=set(outdoor_walls.keys()),
        room_spec=room_spec,
        boundary_groups=boundary_groups,
        pt_db=pt_db,
    )
    return polygons_to_subtract


def select_outdoor_openings(
    boundary_groups: BoundaryGroups, room_type_map: Dict[int, str]
) -> List[Tuple[int, int]]:
    """Select which rooms have doors to the outside."""
    outdoor_candidates = [
        group for group in boundary_groups if OUTDOOR_ROOM_ID in group
    ]
    random.shuffle(outdoor_candidates)

    n_doors_target = random.randint(MIN_DOORS_TO_OUTSIDE, MAX_DOORS_TO_OUTSIDE)
    doors_to_outside = []

    # NOTE: Check preferred room types
    for room_id_1, room_id_2 in outdoor_candidates:
        room_id = room_id_1 if room_id_2 == OUTDOOR_ROOM_ID else room_id_2
        room_type = room_type_map[room_id]
        if room_type in PREFERRED_ROOMS_TO_OUTSIDE:
            doors_to_outside.append((room_id_1, room_id_2))
        if n_doors_target == len(doors_to_outside):
            return doors_to_outside
    if len(doors_to_outside) >= MIN_DOORS_TO_OUTSIDE:
        return doors_to_outside

    # NOTE: Check non preferred room types
    for room_id_1, room_id_2 in outdoor_candidates:
        room_id = room_id_1 if room_id_2 == OUTDOOR_ROOM_ID else room_id_2
        room_type = room_type_map[room_id]
        if room_type not in PREFERRED_ROOMS_TO_OUTSIDE:
            doors_to_outside.append((room_id_1, room_id_2))
        if n_doors_target == len(doors_to_outside):
            return doors_to_outside

    return doors_to_outside


def flatten(x):
    """Return the set of elements in an iterable collection.

    Example:
        input: [[{5}, {6}], [{555}, {667}], [{35}, {53}, {5, 6}, {555, 667}]]
        output: {5, 6, 35, 53, 555, 667}
    """
    return set([a for i in x for a in flatten(i)]) if isinstance(x, Iterable) else {x}


def get_room_spec_neighbors(
    room_spec: List[Union[MetaRoom, LeafRoom]]
) -> List[List[Set[int]]]:
    """Identify possible neighboring rooms from a room_spec.

    Here is an example, where the room_spec: ::

        [
            MetaRoom(
                children=[
                    LeafRoom(room_id=35),
                    LeafRoom(room_id=53),
                    MetaRoom(children=[LeafRoom(room_id=5), LeafRoom(room_id=6)]),
                    MetaRoom(children=[LeafRoom(room_id=555), LeafRoom(room_id=667)]),
                ],
            ),
            MetaRoom(children=[LeafRoom(room_id=48), LeafRoom(room_id=57)]),
        ]

    would return: ::

        [
            [{5}, {6}],
            [{555}, {667}],
            [{35}, {53}, {5, 6}, {555, 667}],
            [{48}, {57}],
            [{35, 5, 6, 555, 53, 667}, {48, 57}]
        ]

    Here, [{5}, {6}] indicates 5 and 6 must have a door between them,
    [{35, 5, 6, 555, 53, 667}, {48, 57}] indicates a door must connect from
    one of {35, 5, 6, 555, 53, 667} to one of {48, 57}, and [{35}, {53}, {5, 6},
    {555, 667}] indicates there must be a path between {35} to {53} to {5 or 6}
    to {555 to 667}.

    """
    out = []
    room_ids = [{room.room_id} for room in room_spec if isinstance(room, LeafRoom)]
    for room in room_spec:
        if isinstance(room, MetaRoom):
            child_ids = get_room_spec_neighbors(room.children)
            out.extend(child_ids)

            flattened_ids = flatten(child_ids)
            room_ids.append(flattened_ids)
    out.append(room_ids)
    return out


def randomly_prioritize_room_ids(
    room_id_pairs: List[Tuple[int, int]], room_spec: RoomSpec
) -> List[Tuple[int, int]]:
    """Random shuffling while moving rooms with avoid_doors_from_metarooms to back."""
    avoid_room_id_pairs = []
    prioritize_room_id_pairs = []
    for room_id_1, room_id_2 in room_id_pairs:
        avoid_on_1 = (
            isinstance(room_spec.room_map[room_id_1], LeafRoom)
            and room_spec.room_map[room_id_1].avoid_doors_from_metarooms
        )
        avoid_on_2 = (
            isinstance(room_spec.room_map[room_id_2], LeafRoom)
            and room_spec.room_map[room_id_2].avoid_doors_from_metarooms
        )
        if avoid_on_1 or avoid_on_2:
            avoid_room_id_pairs.append((room_id_1, room_id_2))
        else:
            prioritize_room_id_pairs.append((room_id_1, room_id_2))

    random.shuffle(prioritize_room_id_pairs)
    random.shuffle(avoid_room_id_pairs)
    return prioritize_room_id_pairs + avoid_room_id_pairs


def select_openings(
    neighboring_rooms: Set[Tuple[int, int]],
    room_spec_neighbors: List[Dict[int, Any]],
    room_spec: RoomSpec,
) -> List[Tuple[int, int]]:
    """Select which neighboring rooms should have doors between them.

    Args:
        neighboring_rooms: (roomId-1, roomId-2) of neighboring rooms, where
            roomId-2 > roomId-1.
        room_spec_neighbors: specifies which rooms can have connections next to each
            other, based on the room spec.

    Returns:
        The neighboring_rooms that can have doors between them.

    """
    selected_doors = []
    for group_neighbors in room_spec_neighbors:
        # NOTE: does not need a door if its the only leaf room.
        if len(group_neighbors) == 1:
            continue

        # NOTE: indexes that still need connecting rooms
        need_connections_between = list(range(len(group_neighbors)))
        while need_connections_between:
            next_room_i = random.choice(need_connections_between)
            other_room_is = [i for i in range(len(group_neighbors)) if i != next_room_i]
            random.shuffle(other_room_is)
            n1_subgroup = group_neighbors[next_room_i]
            for other_room_i in other_room_is:
                n2_subgroup = group_neighbors[other_room_i]
                combos = [
                    (a, b) if a < b else (b, a)
                    for a in n1_subgroup
                    for b in n2_subgroup
                ]
                combos = randomly_prioritize_room_ids(
                    room_id_pairs=combos, room_spec=room_spec
                )
                for door_combo in combos:
                    if door_combo in neighboring_rooms:
                        selected_doors.append(door_combo)
                        if next_room_i in need_connections_between:
                            need_connections_between.remove(next_room_i)
                        if other_room_i in need_connections_between:
                            need_connections_between.remove(other_room_i)
                        break

    return selected_doors


def select_door_walls(openings: List[Tuple[int, int]], boundary_groups: BoundaryGroups):
    chosen_openings = dict()
    for opening in openings:
        candidates = list(boundary_groups[opening])
        population = range(len(candidates))
        weights = [abs(c[1][0] - c[0][0]) + abs(c[1][1] - c[0][1]) for c in candidates]

        # Weights are the size of each wall. Since each wall has a size along a
        # single axis, Manhattan distance is equivalent to Euclidean distance
        chosen_opening = random.choices(population=population, weights=weights, k=1)[0]
        chosen_openings[opening] = candidates[chosen_opening]
    return chosen_openings


@define
class ProceduralFrame:
    """The frame part of a door."""

    room_0_id: int
    """The first room id."""

    room_1_id: int
    """The second room id."""

    wall_0: Wall
    """The first wall."""

    wall_1: Wall
    """The second wall."""

    wall_position_id: str
    """The unique identifier of the wall."""

    bounding_box: BoundingBox
    """The bounding box of the frame."""

    door_width: float
    """The width of the door."""

    door_open_size: float
    """The z-size of the door when it is open."""

    start_door_position: float
    """The start position of the door along the wall."""

    min_x_wall: float
    """The minimum x-coordinate of the wall."""

    max_x_wall: float
    """The maximum x-coordinate of the wall."""

    min_z_wall: float
    """The minimum z-coordinate of the wall."""

    max_z_wall: float
    """The maximum z-coordinate of the wall."""

    asset_id: str
    """The asset id of the door."""

    door_id: str = field(init=False)
    """The generated objectId of the door."""

    pt_db: ProcTHORDatabase

    def __attrs_post_init__(self) -> None:
        """Initialize the door id."""
        self.door_id = (
            f"door|{min(self.room_0_id, self.room_1_id)}|"
            f"{max(self.room_0_id, self.room_1_id)}"
        )

    def flip(self) -> None:
        # NOTE: Doors connected to the outside should not flip due to limitations
        # in the AI2-THOR json.
        if self.wall_1 is not None:
            self.wall_0, self.wall_1 = self.wall_1, self.wall_0
            self.room_0_id, self.room_1_id = self.room_1_id, self.room_0_id

    def polygon(self, entrance_padding: float, in_front_padding: float = 0) -> Polygon:
        """Add padding to the side of the door.

        Args:
            entrance_padding: The padding (in meters) before walking into the door.
            in_front_padding: The padding (in meters) to avoid objects in-front of
                the door's opening.

        """
        first_wall = self.wall_0 if self.wall_0 is not None else self.wall_1
        if self.max_x_wall - self.min_x_wall < EPSILON:
            # NOTE: min_x_wall \approx\equal max_x_wall
            x_wall = self.min_x_wall

            # placed along z
            flipped = first_wall["polygon"][0]["z"] > first_wall["polygon"][1]["z"]

            return (
                Polygon(
                    [
                        (
                            x_wall - entrance_padding,
                            self.max_z_wall - self.start_door_position,
                        ),
                        (
                            x_wall - entrance_padding,
                            self.max_z_wall
                            - self.start_door_position
                            - self.door_width,
                        ),
                        (
                            x_wall + self.door_open_size + in_front_padding,
                            self.max_z_wall
                            - self.start_door_position
                            - self.door_width,
                        ),
                        (
                            x_wall + self.door_open_size + in_front_padding,
                            self.max_z_wall - self.start_door_position,
                        ),
                    ]
                )
                if flipped
                else Polygon(
                    [
                        (
                            x_wall - self.door_open_size - in_front_padding,
                            self.min_z_wall + self.start_door_position,
                        ),
                        (
                            x_wall - self.door_open_size - in_front_padding,
                            self.min_z_wall
                            + self.start_door_position
                            + self.door_width,
                        ),
                        (
                            x_wall + entrance_padding,
                            self.min_z_wall
                            + self.start_door_position
                            + self.door_width,
                        ),
                        (
                            x_wall + entrance_padding,
                            self.min_z_wall + self.start_door_position,
                        ),
                    ]
                )
            )
        else:
            # NOTE: min_z_wall \approx\equal max_z_wall
            z_wall = self.min_z_wall

            # placed along x
            flipped = first_wall["polygon"][0]["x"] > first_wall["polygon"][1]["x"]
            return (
                Polygon(
                    [
                        (
                            self.max_x_wall - self.start_door_position,
                            z_wall - self.door_open_size - in_front_padding,
                        ),
                        (
                            self.max_x_wall
                            - self.start_door_position
                            - self.door_width,
                            z_wall - self.door_open_size - in_front_padding,
                        ),
                        (
                            self.max_x_wall
                            - self.start_door_position
                            - self.door_width,
                            z_wall + entrance_padding,
                        ),
                        (
                            self.max_x_wall - self.start_door_position,
                            z_wall + entrance_padding,
                        ),
                    ]
                )
                if flipped
                else Polygon(
                    [
                        (
                            self.min_x_wall + self.start_door_position,
                            z_wall - entrance_padding,
                        ),
                        (
                            self.min_x_wall
                            + self.start_door_position
                            + self.door_width,
                            z_wall - entrance_padding,
                        ),
                        (
                            self.min_x_wall
                            + self.start_door_position
                            + self.door_width,
                            z_wall + self.door_open_size + in_front_padding,
                        ),
                        (
                            self.min_x_wall + self.start_door_position,
                            z_wall + self.door_open_size + in_front_padding,
                        ),
                    ]
                )
            )

    def asdict(self) -> Door:
        room_0 = f"room|{self.room_0_id}"
        room_1 = f"room|{self.room_1_id}"
        if self.room_0_id == OUTDOOR_ROOM_ID:
            room_0 = room_1
        elif self.room_1_id == OUTDOOR_ROOM_ID:
            room_1 = room_0

        wall_0_id = (
            f"wall|{self.room_0_id}|{self.wall_position_id}"
            if self.room_0_id != OUTDOOR_ROOM_ID
            else f"wall|exterior|{self.wall_position_id}"
        )
        wall_1_id = (
            f"wall|{self.room_1_id}|{self.wall_position_id}"
            if self.room_1_id != OUTDOOR_ROOM_ID
            else f"wall|exterior|{self.wall_position_id}"
        )

        # TODO: This is a quick hack. Wall holes only contain info for doorways
        # right now, but they # share the same asset offset.
        asset_offset = self.pt_db.WALL_HOLES[
            self.asset_id.replace("Doorframe", "Doorway")
        ]["offset"]

        bbox_with_offset = copy.deepcopy(self.bounding_box)
        bbox_with_offset["min"]["x"] -= asset_offset["x"]
        bbox_with_offset["max"]["x"] -= asset_offset["x"]
        bbox_with_offset["max"]["y"] -= asset_offset["y"]

        return Door(
            id=self.door_id,
            room0=room_0,
            room1=room_1,
            wall0=wall_0_id,
            wall1=wall_1_id,
            assetId=self.asset_id,
            boundingBox=bbox_with_offset,
            assetOffset=asset_offset,
        )


@define
class ProceduralDoor(ProceduralFrame):
    """A door that can be opened."""

    is_open: bool
    """Whether the door is open."""

    def asdict(self) -> Door:
        door = super().asdict()
        door["openable"] = OUTDOOR_ROOM_ID not in [self.room_0_id, self.room_1_id]
        door["openness"] = 1 if self.is_open else 0
        return door


def fix_door_intersections(doors: List[ProceduralDoor]):
    """Try flipping doors until none of them intersect."""
    # Try with some padding first, so there's space to walk between rooms!
    for entrance_padding, in_front_padding in [
        (PREFERRED_ENTRANCE_PADDING, PADDING_IN_FRONT_OF_DOOR),
        (PREFERRED_ENTRANCE_PADDING, 0),
        (0, PADDING_IN_FRONT_OF_DOOR),
        (0, 0),
    ]:
        if entrance_padding == 0 or in_front_padding == 0:
            logging.warning("Might be unable to walk between rooms!")

        tried_combinations = set()
        next_door_flip_combinations = {tuple(False for _ in doors)}
        door_id_to_idx = {door.door_id: i for i, door in enumerate(doors)}

        while next_door_flip_combinations:
            flip_combination = random.choice(list(next_door_flip_combinations))
            next_door_flip_combinations.remove(flip_combination)
            tried_combinations.add(flip_combination)

            # flip the doors
            for door, flip in zip(doors, flip_combination):
                if flip:
                    door.flip()

            collisions = []
            for d0, d1 in itertools.combinations(doors, 2):

                if d0.polygon(
                    entrance_padding=(
                        entrance_padding
                        if isinstance(d0, ProceduralDoor)
                        else OPEN_ROOM_PADDING
                    ),
                    in_front_padding=(
                        in_front_padding
                        if isinstance(d0, ProceduralDoor)
                        else OPEN_ROOM_PADDING
                    ),
                ).intersects(
                    d1.polygon(
                        entrance_padding=(
                            entrance_padding
                            if isinstance(d1, ProceduralDoor)
                            else OPEN_ROOM_PADDING
                        ),
                        in_front_padding=(
                            in_front_padding
                            if isinstance(d1, ProceduralDoor)
                            else OPEN_ROOM_PADDING
                        ),
                    )
                ):
                    collisions.append((d0.door_id, d1.door_id))

            # terminate on success!
            if not collisions:
                return

            # undo the flips
            for door, flip in zip(doors, flip_combination):
                if flip:
                    door.flip()

            # add new combinations to try, based on the collisions
            new_combos = set()
            for d0_idx, d1_idx in collisions:
                d0_idx = door_id_to_idx[d0_idx]
                d1_idx = door_id_to_idx[d1_idx]

                # first flipped
                flip_combo = list(flip_combination)
                flip_combo[d0_idx] = not flip_combo[d0_idx]
                new_combos.add(tuple(flip_combo))

                # 2nd flipped
                flip_combo = list(flip_combination)
                flip_combo[d1_idx] = not flip_combo[d1_idx]
                new_combos.add(tuple(flip_combo))

                # both flipped
                flip_combo = list(flip_combination)
                flip_combo[d0_idx] = not flip_combo[d0_idx]
                flip_combo[d1_idx] = not flip_combo[d1_idx]
                new_combos.add(tuple(flip_combo))

            # add combinations that haven't been tried yet
            for combo in new_combos:
                if combo not in tried_combinations:
                    next_door_flip_combinations.add(combo)

    logging.warning(
        "There are colliding doors! Unable to find door flipping combination "
        "that doesn't result in a collision between doors."
    )


def add_door_meta(
    partial_house: "PartialHouse",
    split: Split,
    door_walls: Dict[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]],
    closed_doors: Set[Tuple[int, int]],
    room_spec: RoomSpec,
    boundary_groups: BoundaryGroups,
    pt_db: ProcTHORDatabase,
) -> Dict[int, List[Polygon]]:
    """Get the formatted metadata of the doors.

    Args:
        door_walls: which walls should have a connecting door. Example: ::

            {
                (35, 53): ((0, 3), (5, 3)),
                (48, 57): ((5, 4), (10, 4)),
                (35, 48): ((5, 4), (5, 7))
            },

            where the keys (e.g., (35, 53)) specify the roomIds and the values
            specify the wall positions.

    Returns:
        xz polygons to subtract from the open area of each room.

    """
    assert split in {"train", "val", "test"}

    # NOTE: open bounding box is always larger than normal bounding box.
    # Thus, we're effectively taking the max bounding box from a door.
    doors_df = pd.DataFrame(
        [
            {
                "assetId": d["assetId"],
                "openXSize": d["states"]["open"]["boundingBox"]["x"],
                "openYSize": d["states"]["open"]["boundingBox"]["y"],
                "openZSize": d["states"]["open"]["boundingBox"]["z"],
                "xSize": d["boundingBox"]["x"],
                "ySize": d["boundingBox"]["y"],
                "zSize": d["boundingBox"]["z"],
            }
            for d in pt_db.ASSET_DATABASE["Doorway"]
            if d["split"] == split
        ]
    )
    frames_df = pd.DataFrame(
        [
            {
                "assetId": d["assetId"],
                "xSize": d["boundingBox"]["x"],
                "ySize": d["boundingBox"]["y"],
                "zSize": d["boundingBox"]["z"],
            }
            for d in pt_db.ASSET_DATABASE["Doorframe"]
            if d["split"] == split
        ]
    )

    doors: List[ProceduralDoor] = []
    polygons_to_subtract = defaultdict(list)
    for orig_boundary, wall in door_walls.items():
        # NOTE: randomize which side of the door is open
        if OUTDOOR_ROOM_ID in orig_boundary:
            if orig_boundary[0] == OUTDOOR_ROOM_ID:
                # NOTE: put outdoor boundary 2nd since it cannot be first in the
                # json.
                boundary = (orig_boundary[1], orig_boundary[0])
            else:
                boundary = orig_boundary
        elif random.random() < 0.5:
            boundary = (orig_boundary[1], orig_boundary[0])
        else:
            boundary = orig_boundary

        # NOTE: set the default ASSETS_DF. It may be updated with frames_df.
        assets_df = doors_df
        use_frame = False

        # NOTE: Check if the connection between rooms may be open
        if all(b != OUTDOOR_ROOM_ID for b in boundary):
            open_connection_valid = False
            for orc in OPEN_ROOM_CONNECTIONS:
                if all(room_spec.room_type_map[b] in orc["between"] for b in boundary):
                    logging.debug("Has open room connection in house.")
                    open_connection_valid = True
                    connection = orc
                    break
            if open_connection_valid and random.random() < connection["p"]:
                # NOTE: Have an open connection
                if random.random() < connection["pFrame"]:
                    # NOTE: Use just a door frame.
                    logging.debug("Adding door frame.")
                    assets_df = frames_df
                    use_frame = True
                else:
                    # NOTE: Open up the entire wall.
                    logging.debug("Adding empty wall.")
                    walls = boundary_groups[orig_boundary]
                    wall_ids = set()
                    for other_wall in walls:
                        min_x = min(other_wall[0][0], other_wall[1][0])
                        max_x = max(other_wall[0][0], other_wall[1][0])
                        min_z = min(other_wall[0][1], other_wall[1][1])
                        max_z = max(other_wall[0][1], other_wall[1][1])
                        for room_id in boundary:
                            wall_ids.add(
                                f"wall|{room_id}|{min_x:.2f}|{min_z:.2f}|{max_x:.2f}|{max_z:.2f}"
                            )
                    for house_wall in partial_house.walls:
                        if house_wall["id"] in wall_ids:
                            house_wall["empty"] = True
                            wall_ids.remove(house_wall["id"])
                    assert not wall_ids

                    for other_wall in walls:
                        if other_wall[0][0] == other_wall[1][0]:
                            # NOTE: vertical wall
                            x = other_wall[0][0]
                            polygon = Polygon(
                                [
                                    [x - OPEN_ROOM_PADDING, other_wall[0][1]],
                                    [x - OPEN_ROOM_PADDING, other_wall[1][1]],
                                    [x + OPEN_ROOM_PADDING, other_wall[1][1]],
                                    [x + OPEN_ROOM_PADDING, other_wall[0][1]],
                                ]
                            )
                        elif other_wall[0][1] == other_wall[1][1]:
                            # NOTE: horizontal wall
                            z = other_wall[0][1]
                            polygon = Polygon(
                                [
                                    [other_wall[0][0], z - OPEN_ROOM_PADDING],
                                    [other_wall[1][0], z - OPEN_ROOM_PADDING],
                                    [other_wall[1][0], z + OPEN_ROOM_PADDING],
                                    [other_wall[0][0], z + OPEN_ROOM_PADDING],
                                ]
                            )
                        else:
                            raise ValueError(f"Invalid wall: {other_wall}")
                        for room_id in boundary:
                            polygons_to_subtract[room_id].append(polygon)

                    continue

        wall_size = abs(wall[1][0] - wall[0][0]) + abs(wall[1][1] - wall[0][1])
        valid_doors = assets_df[
            assets_df["xSize" if use_frame else "openXSize"] < wall_size
        ]
        door = valid_doors.sample()
        start_door_position = random.uniform(
            0, wall_size - door["xSize" if use_frame else "openXSize"].iloc[0]
        )

        # set the position of the door randomly between the wall
        # NOTE: z is currently ignored on the Unity side
        bounding_box = BoundingBox(
            min=Vector3(x=start_door_position, y=0, z=0),
            max=Vector3(
                x=start_door_position + door["xSize"].iloc[0],
                y=door["ySize"].iloc[0],
                z=0,
            ),
        )

        min_x_wall = min(wall[0][0], wall[1][0])
        min_z_wall = min(wall[0][1], wall[1][1])
        max_x_wall = max(wall[0][0], wall[1][0])
        max_z_wall = max(wall[0][1], wall[1][1])

        wall_position_id = (
            f"{min_x_wall:.2f}|{min_z_wall:.2f}|{max_x_wall:.2f}|{max_z_wall:.2f}"
        )

        wall_0_id = f"wall|{boundary[0]}|{wall_position_id}"
        wall_1_id = (
            None
            if boundary[1] == OUTDOOR_ROOM_ID
            else f"wall|{boundary[1]}|{wall_position_id}"
        )

        wall_map = {wall["id"]: wall for wall in partial_house.walls}

        wall_0 = next(w for w in partial_house.walls if w["id"] == wall_0_id)
        wall_1 = next((w for w in partial_house.walls if w["id"] == wall_1_id), None)

        # NOTE: a hole appears in the wall if openXSize is used from the Unity side.
        door_width = door["xSize"].iloc[0]

        door_open_size = door["zSize" if use_frame else "openZSize"].iloc[0]

        if use_frame:
            door = ProceduralFrame(
                room_0_id=boundary[0],
                room_1_id=boundary[1],
                wall_0=wall_0,
                wall_1=wall_1,
                wall_position_id=wall_position_id,
                bounding_box=bounding_box,
                door_width=door_width,
                door_open_size=door_open_size,
                start_door_position=start_door_position,
                min_x_wall=min_x_wall,
                max_x_wall=max_x_wall,
                min_z_wall=min_z_wall,
                max_z_wall=max_z_wall,
                asset_id=door["assetId"].iloc[0],
                pt_db=pt_db,
            )
        else:
            door = ProceduralDoor(
                room_0_id=boundary[0],
                room_1_id=boundary[1],
                wall_0=wall_0,
                wall_1=wall_1,
                wall_position_id=wall_position_id,
                bounding_box=bounding_box,
                door_width=door_width,
                door_open_size=door_open_size,
                start_door_position=start_door_position,
                min_x_wall=min_x_wall,
                max_x_wall=max_x_wall,
                min_z_wall=min_z_wall,
                max_z_wall=max_z_wall,
                asset_id=door["assetId"].iloc[0],
                is_open=orig_boundary not in closed_doors,
                pt_db=pt_db,
            )
        doors.append(door)

    fix_door_intersections(doors)

    partial_house.doors = []
    for door in doors:
        partial_house.doors.append(door.asdict())

        if isinstance(door, ProceduralDoor):
            entr_padding = PREFERRED_ENTRANCE_PADDING
            in_front_padding = PADDING_IN_FRONT_OF_DOOR
        else:
            entr_padding = OPEN_ROOM_PADDING
            in_front_padding = OPEN_ROOM_PADDING

        for room_id in (door.room_0_id, door.room_1_id):
            polygons_to_subtract[room_id].append(
                door.polygon(
                    entrance_padding=entr_padding,
                    in_front_padding=in_front_padding,
                )
            )

    return polygons_to_subtract
