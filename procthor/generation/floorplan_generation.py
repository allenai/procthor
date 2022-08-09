#%%
"""
Things to consider:
- Hallways:
    - Our algorithm starts by placing doors between rooms
      for which connectivity was explicitly declared. Next,
      it connects any hallway to all of its adjacent public
      rooms. Unconnected private rooms are then connected,
      if possible, to an adjacent public room. Publics rooms
      with no connections are connected to an adjacent public
      room as well. Finally, our last step is a reachability test.
      We examine all rooms and if any is not reachable from
      the hallway, we use the adjacency relationships between
      rooms to find a path to the unreachable room, and create
      the necessary door(s).
"""
import random
from typing import Dict, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

from procthor.constants import EMPTY_ROOM_ID, OUTDOOR_ROOM_ID
from procthor.generation.room_specs import RoomSpec
from procthor.utils.types import InvalidFloorplan, LeafRoom, MetaRoom


def visualize_floorplan(floorplan: np.array):
    colors = ["green", "blue", "red", "yellow", "black"]

    fig, ax = plt.subplots()
    for i, room_id in enumerate(np.unique(floorplan.flatten())):
        print(room_id)
        coords = np.argwhere(floorplan == room_id)
        ax.scatter(coords[:, 1], coords[:, 0], label=room_id, c=colors[i])
        ax.set_aspect("equal")
        ax.legend(loc=(1.04, 0))
    return fig


def select_room(
    rooms: Sequence[Union[LeafRoom, MetaRoom]]
) -> Union[LeafRoom, MetaRoom]:
    """
    From the paper:
        In SelectRoom the next room to be expanded is chosen
        on the basis of the defined size ratios r for each room.
        The chance for a room to be selected is its ratio relative to
        the total sum of ratios, defined in r. With this approach,
        variation is ensured, but the selection still respects the
        desired ratios of room areas.
    """
    total_ratio = sum(r.ratio for r in rooms)
    r = random.random() * total_ratio
    for room in rooms:
        r -= room.ratio
        if r <= 0:
            return room


def sample_initial_room_positions(
    rooms: Sequence[Union[LeafRoom, MetaRoom]], floorplan: np.ndarray
) -> None:
    """
    From the paper:
        However, placing initial positions adjacent to a
        wall does not always result in plausible results, as they
        tend to cause less regular room shapes. Therefore, we
        use a different approach. Based on the size ratio of the
        room and the total area of the building, we can estimate
        a desired area of the room. Cells positioned at least a
        specific distance (based on this estimated area) away
        from the walls are assigned a weight of 1. This results
        in much more plausible room shapes.

        This phase also deals with the defined adjacency con-
        straints. The adjacency constraints are always defined
        between two rooms, e.g. the bathroom should be next
        to the bedroom, the kitchen should be next to the living
        room, etc. When selecting the initial position of a room,
        we use the adjacency constraints to determine a list of
        rooms it should be adjacent to. We check whether these
        rooms already have an initial position. If there is, we
        alter the weights to high values in the surroundings of
        the initial positions of the rooms it should be adjacent to.
        This typically results in valid layouts; however there is a
        small chance that the algorithm grows another room in
        between the rooms that should be adjacent. To handle
        this case, we reset the generation process if some of the
        adjacency constraints were not met.

        Based on these grid weights, one cell is selected to place
        a room, and the weights around the selected cell are
        set to zero, to avoid several initial positions of different
        rooms to be too close to each other.
    """
    grid_weights = np.where(floorplan == EMPTY_ROOM_ID, 1, 0)
    for room in rooms:
        # make sure there is at least one open cell in the floorplan area.
        if (grid_weights == 0).all():
            raise InvalidFloorplan(
                "No empty cells in the floorplan to place room! This means the"
                " sampled interior boundary is too small for the room spec.\n"
                f"grid_weights:\n{grid_weights}"
                f"\nfloorplan\n{floorplan}"
            )

        # TODO: these weights could be updated by the adjacency constraints
        # and the hallways.
        # sample a grid cell by weight
        cell_idx = np.random.choice(
            grid_weights.size, p=grid_weights.ravel() / float(grid_weights.sum())
        )
        cell_y, cell_x = np.unravel_index(cell_idx, grid_weights.shape)

        # add the grid cell to the room
        room.min_x = cell_x
        room.min_y = cell_y
        room.max_x = cell_x + 1
        room.max_y = cell_y + 1

        # add the grid cell to the floorplan
        floorplan[cell_y, cell_x] = room.room_id

        # update the weights
        grid_weights[
            max(0, cell_y - 1) : min(grid_weights.shape[0], cell_y + 2),
            max(0, cell_x - 1) : min(grid_weights.shape[1], cell_x + 2),
        ] = 0


def grow_rect(room: Union[MetaRoom, LeafRoom], floorplan: np.ndarray) -> bool:
    """
    From the paper:
        The first phase of this algorithm is expanding rooms
        to rectangular shapes (GrowRect). In Fig. 3 we see an
        example of the start situation (a) and end (b) of the
        rectangular expansion phase for a building where rooms
        black, green and red have size ratios of, respectively, 8, 4
        and 2. Starting with rectangular expansion ensures two
        characteristics of real life floor plans: (i) a higher priority
        is given to obtain rectangular areas and (ii) the growth is
        done using the maximum space available, in a linear way.
        For this, all empty line intervals in the grid m to which
        the selected room can expand to are considered. The
        maximum growth, i.e. the longest line interval, which
        leads to a rectangular area is picked (randomly, if there
        are more than one candidates). A room remains available
        for selection until it can not grow more. This happens if
        there are no more directions available to grow or, in the
        rectangular expansion case, if the room has reached its
        maximum size. This condition also prevents starvation
        for lower ratio rooms, since size ratios have no relation
        with the total building area. In Fig.3 (b), all rooms have
        reached their maximum size."""
    # NOTE: check if room is already grown beyond the maximum size.
    maximum_size = room.ratio * 4
    if (room.max_x - room.min_x) * (room.max_y - room.min_y) > maximum_size:
        return False

    # NOTE: Find out how much the rectangle can grow in each direction.
    growth_sizes = {
        "right": (
            room.max_y - room.min_y
            if (
                room.max_x < floorplan.shape[1]
                and (
                    floorplan[room.min_y : room.max_y, room.max_x] == EMPTY_ROOM_ID
                ).all()
            )
            else 0
        ),
        "left": (
            room.max_y - room.min_y
            if (
                room.min_x > 0
                and (
                    floorplan[room.min_y : room.max_y, room.min_x - 1] == EMPTY_ROOM_ID
                ).all()
            )
            else 0
        ),
        "down": (
            room.max_x - room.min_x
            if (
                room.max_y < floorplan.shape[0]
                and (
                    floorplan[room.max_y, room.min_x : room.max_x] == EMPTY_ROOM_ID
                ).all()
            )
            else 0
        ),
        "up": (
            room.max_x - room.min_x
            if (
                room.min_y > 0
                and (
                    floorplan[room.min_y - 1, room.min_x : room.max_x] == EMPTY_ROOM_ID
                ).all()
            )
            else 0
        ),
    }

    max_growth_size = max(growth_sizes.values())

    # If there is no room to grow, return False
    if max_growth_size == 0:
        return False

    # NOTE: Pick a random max growth direction to grow.
    # From the paper: The maximum growth, i.e. the longest line interval, which
    # leads to a rectangular area is picked (randomly, if there are more than
    # one candidates).
    growth_direction = random.choice(
        [
            growth_direction
            for growth_direction, growth_size in growth_sizes.items()
            if growth_size == max_growth_size
        ]
    )

    # NOTE: Grow the room in the chosen direction.
    if growth_direction == "right":
        room.max_x += 1
        floorplan[room.min_y : room.max_y, room.max_x - 1] = room.room_id
    elif growth_direction == "left":
        room.min_x -= 1
        floorplan[room.min_y : room.max_y, room.min_x] = room.room_id
    elif growth_direction == "down":
        room.max_y += 1
        floorplan[room.max_y - 1, room.min_x : room.max_x] = room.room_id
    elif growth_direction == "up":
        room.min_y -= 1
        floorplan[room.min_y, room.min_x : room.max_x] = room.room_id

    return True


def grow_l_shape(room, floorplan):
    """
    From the paper:
        Of course, this first phase does not ensure that all available space
        gets assigned to a room. In the second phase, all rooms are again
        considered for further expansion, now allowing for non-rectangular
        shapes. The maximum growth line is again selected, in order to maximize
        efficient space use, i.e. to avoid narrow L-shaped edges. In this phase,
        the maximum size for each room is no longer considered, since the
        algorithm attempts to fill all the remaining empty space. Furthermore we
        included mechanisms for preventing U-shaped rooms. Fig. 3 (c)
        illustrates the result of the L-shaped growth step on the previous
        example. The final phase scans the grid for remaining empty space; this
        space is directly assigned to the room which fills most of the adjacent
        area.
    """
    # NOTE: Find out how much the rectangle can grow in each direction.
    growth_sizes = {
        "right": (
            [
                y
                for y in range(room.min_y, room.max_y)
                if (
                    floorplan[y, room.max_x] == EMPTY_ROOM_ID
                    and floorplan[y, room.max_x - 1] == room.room_id
                )
            ]
            if (room.max_x < floorplan.shape[1])
            else []
        ),
        "left": (
            [
                y
                for y in range(room.min_y, room.max_y)
                if (
                    floorplan[y, room.min_x - 1] == EMPTY_ROOM_ID
                    and floorplan[y, room.min_x] == room.room_id
                )
            ]
            if (room.min_x > 0)
            else []
        ),
        "down": (
            [
                x
                for x in range(room.min_x, room.max_x)
                if (
                    floorplan[room.max_y, x] == EMPTY_ROOM_ID
                    and floorplan[room.max_y - 1, x] == room.room_id
                )
            ]
            if (room.max_y < floorplan.shape[0])
            else []
        ),
        "up": (
            [
                x
                for x in range(room.min_x, room.max_x)
                if (
                    floorplan[room.min_y - 1, x] == EMPTY_ROOM_ID
                    and floorplan[room.min_y, x] == room.room_id
                )
            ]
            if (room.min_y > 0)
            else []
        ),
    }

    max_growth_size = max(growth_sizes.values(), key=len)
    if len(max_growth_size) == 0:
        return False

    # NOTE: Pick a random max growth direction to grow.
    growth_direction = random.choice(
        [
            growth_direction
            for growth_direction, growth_size in growth_sizes.items()
            if growth_size == max_growth_size
        ]
    )
    if growth_direction == "right":
        for y in growth_sizes["right"]:
            floorplan[y, room.max_x] = room.room_id
        room.max_x += 1
    elif growth_direction == "left":
        for y in growth_sizes["left"]:
            floorplan[y, room.min_x - 1] = room.room_id
        room.min_x -= 1
    elif growth_direction == "down":
        for x in growth_sizes["down"]:
            floorplan[room.max_y, x] = room.room_id
        room.max_y += 1
    elif growth_direction == "up":
        for x in growth_sizes["up"]:
            floorplan[room.min_y - 1, x] = room.room_id
        room.min_y -= 1

    return True


def expand_rooms(
    rooms: Sequence[Union[LeafRoom, MetaRoom]], floorplan: np.ndarray
) -> None:
    """Assign rooms from a given hierarchy to the floorplan.

    From the paper:
        Algorithm 1 outlines the expansion of rooms in our
        method. It starts with a grid m containing the initial
        positions of each room. It then picks one room at a time,
        selected from a set of available rooms (SelectRoom), and
        expands the room shape to the maximum rectangular
        space available (GrowRect). This is done until no more
        rectangular expansions are possible. At this point, the
        process resets rooms to the initial set, but now considers
        expansions that lead to L-shaped rooms (GrowLShape).
    """

    # NOTE: Initial center placement of each room
    sample_initial_room_positions(rooms, floorplan)

    # NOTE: grow rectangles
    rooms_to_grow = set(rooms)
    while rooms_to_grow:
        room = select_room(rooms_to_grow)
        can_grow = grow_rect(room, floorplan)
        if not can_grow:
            rooms_to_grow.remove(room)

    # NOTE: grow L-Shape
    rooms_to_grow = set(rooms)
    while rooms_to_grow:
        room = select_room(rooms_to_grow)
        can_grow = grow_l_shape(room, floorplan)
        if not can_grow:
            rooms_to_grow.remove(room)


def _set_ideal_ratios(
    ideal_ratios: Dict[int, float],
    rooms: Sequence[Union[MetaRoom, LeafRoom]],
    parent_sum: float = 1,
) -> None:
    """Set the ideal ratio size of each room in the floorplan.

    After calling this method, ideal ratios becomes something like:
        {
            2: 0.3,
            3: 0.1,
            4: 0.2,
            5: 0.4
        }
    where the sum of the values is 1. The value indicates the ideal absolute size
    of the room relative to the entire floorplan.
    """
    room_ratio_sum = sum([room.ratio for room in rooms])
    for room in rooms:
        ideal_ratios[room.room_id] = room.ratio / room_ratio_sum * parent_sum
        if isinstance(room, MetaRoom):
            _set_ideal_ratios(
                ideal_ratios=ideal_ratios,
                rooms=room.children,
                parent_sum=ideal_ratios[room.room_id],
            )
            del ideal_ratios[room.room_id]


def get_ratio_overlap_score(room_spec: RoomSpec, floorplan: np.ndarray) -> float:
    """Calculate the difference between the ratios in floorplan and room_spec."""
    # NOTE: Get the average ratio overlap of all rooms
    ideal_ratios = {}
    _set_ideal_ratios(ideal_ratios, rooms=room_spec.spec)

    actual_ratios = {}
    occupied_cells = (floorplan != OUTDOOR_ROOM_ID).sum()
    for room_id in room_spec.room_type_map:
        actual_ratios[room_id] = (floorplan == room_id).sum() / occupied_cells

    # NOTE: Get the average ratio overlap of all rooms
    ratio_overlap = sum(
        [min(actual_ratios[room_id], ideal_ratios[room_id]) for room_id in ideal_ratios]
    )

    return ratio_overlap


def score_floorplan(room_spec: RoomSpec, floorplan: np.ndarray) -> float:
    """Calculate the quality of the floorplan based on the room specifications."""
    # TODO: Consider ranking by adjacency constraints, maybe hallway connections.
    ratio_overlap_score = get_ratio_overlap_score(room_spec, floorplan)
    return ratio_overlap_score


def recursively_expand_rooms(
    rooms: Sequence[Union[LeafRoom, MetaRoom]], floorplan: np.ndarray
) -> None:
    """Assign rooms to the floorplan and expand it if it is a MetaRoom."""
    expand_rooms(rooms, floorplan)
    for room in rooms:
        if isinstance(room, MetaRoom):
            floorplan_mask = floorplan == room.room_id
            floorplan[floorplan_mask] = EMPTY_ROOM_ID
            recursively_expand_rooms(
                room.children,
                floorplan[room.min_y : room.max_y, room.min_x : room.max_x],
            )


def generate_floorplan(
    room_spec: np.ndarray,
    interior_boundary: np.ndarray,
    candidate_generations: int = 100,
) -> np.ndarray:
    """Generate a floorplan for the given room spec and interior boundary.

    Args:
        room_spec: Room spec for the floorplan.
        interior_boundary: Interior boundary of the floorplan.
        candidate_generations: Number of candidate generations to generate. The
            best candidate floorplan is returned.
    """
    # NOTE: If there is only one room, the floorplan will always be the same.
    if len(room_spec.room_type_map) == 1:
        candidate_generations = 1

    best_floorplan = None
    best_score = float("-inf")
    for _ in range(candidate_generations):
        floorplan = interior_boundary.copy()
        try:
            recursively_expand_rooms(rooms=room_spec.spec, floorplan=floorplan)
        except InvalidFloorplan:
            continue
        else:
            score = score_floorplan(room_spec=room_spec, floorplan=floorplan)
            if best_floorplan is None or score > best_score:
                best_floorplan = floorplan
                best_score = score

    if best_floorplan is None:
        raise InvalidFloorplan(
            "Failed to generate a valid floorplan all candidate_generations="
            f"{candidate_generations} times from the interior boundary!"
            " This means the sampled interior boundary is too small for the room"
            " spec. Try again with a another interior boundary.\n"
            f"interior_boundary:\n{interior_boundary}\n, room_spec:\n{room_spec}"
        )

    return best_floorplan
