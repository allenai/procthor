#%%
import logging
import queue
import random
from typing import Optional, Tuple

import numpy as np
from procthor.constants import OUTDOOR_ROOM_ID
from scipy.ndimage.measurements import label

DEFAULT_AVERAGE_ROOM_SIZE = 3
"""Average room size in meters"""

DEFAULT_MIN_HOUSE_SIDE_LENGTH = 2
"""Min length of a side of the house (in meters)."""

DEFAULT_MAX_BOUNDARY_CUT_AREA = 6
"""Max area of a single chop along the boundary."""


def count_components(boundary):
    boundary = boundary != 1
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    _, num_components = label(boundary, structure)
    return num_components


def is_valid_cut(boundary, chop_side, z_cut, x_cut):
    # Apply cut and validate
    if chop_side == 0:
        # NOTE: top-right corner
        boundary[:z_cut, -x_cut:] = OUTDOOR_ROOM_ID
    elif chop_side == 1:
        # NOTE: top-left corner
        boundary[:z_cut, :x_cut] = OUTDOOR_ROOM_ID
    elif chop_side == 2:
        # NOTE: bottom-left corner
        boundary[-z_cut:, :x_cut] = OUTDOOR_ROOM_ID
    elif chop_side == 3:
        # NOTE: bottom-right corner
        boundary[-z_cut:, -x_cut:] = OUTDOOR_ROOM_ID

    num_components = count_components(boundary=boundary)
    return num_components == 1


def get_n_cuts(num_rooms: int) -> int:
    return round(np.random.beta(a=0.5 * num_rooms, b=6) * 10)


def sample_interior_boundary(
    num_rooms: int,
    average_room_size: int = DEFAULT_AVERAGE_ROOM_SIZE,
    min_house_side_length: int = DEFAULT_MIN_HOUSE_SIDE_LENGTH,
    max_boundary_cut_area: int = DEFAULT_MAX_BOUNDARY_CUT_AREA,
    dims: Optional[Tuple[int, int]] = None,
) -> np.array:
    """Sample a boundary for the interior of a house.

    Parameters:
        num_rooms: The number of rooms in the house.
        dims: The (x_size, z_size) dimensions of the house.
    """
    assert num_rooms > 0

    # NOTE: -1 * average_room_size and +1 * average_room_size adds in some
    # variance. The +1 makes high is inclusive.
    if dims is None:
        x_size, z_size = np.random.randint(
            low=max(
                min_house_side_length,
                np.sqrt(num_rooms) * average_room_size - 1 * average_room_size // 2,
            ),
            high=(
                np.sqrt(num_rooms) * average_room_size + 1 * average_room_size // 2 + 1
            ),
            size=2,
        )
    else:
        x_size, z_size = dims

    boundary = np.zeros((z_size, x_size), dtype=int)

    n_cuts = get_n_cuts(num_rooms=num_rooms)
    logging.debug(f"Number of cuts: {n_cuts}")

    chop_sides = np.random.randint(0, 4, size=n_cuts)

    for chop_side in chop_sides:
        x_cut = np.random.randint(
            low=1, high=max(2, min(x_size - 1, max_boundary_cut_area // 2))
        )
        z_cut_candidates = []

        i = 1
        while x_cut * i <= max_boundary_cut_area and i + 1 <= z_size:
            z_cut_candidates.append(i)
            i += 1

        z_cut = random.choice(z_cut_candidates)

        if is_valid_cut(
            np.copy(boundary), chop_side, z_cut, x_cut
        ):
            if chop_side == 0:
                # NOTE: top-right corner
                boundary[:z_cut, -x_cut:] = OUTDOOR_ROOM_ID
            elif chop_side == 1:
                # NOTE: top-left corner
                boundary[:z_cut, :x_cut] = OUTDOOR_ROOM_ID
            elif chop_side == 2:
                # NOTE: bottom-left corner
                boundary[-z_cut:, :x_cut] = OUTDOOR_ROOM_ID
            elif chop_side == 3:
                # NOTE: bottom-right corner
                boundary[-z_cut:, -x_cut:] = OUTDOOR_ROOM_ID

    return boundary
