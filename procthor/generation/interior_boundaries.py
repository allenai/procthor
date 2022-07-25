#%%
import logging
import random
from typing import Optional, Tuple

import numpy as np
from procthor.constants import OUTDOOR_ROOM_ID

AVERAGE_ROOM_SIZE = 3
"""Average room size in meters"""

MIN_HOUSE_SIDE_LENGTH = 2
"""Min length of a side of the house (in meters)."""

MAX_BOUNDARY_CUT_AREA = 6
"""Max area of a single chop along the boundary."""


def get_n_cuts(num_rooms: int) -> int:
    return round(np.random.beta(a=0.5 * num_rooms, b=6) * 10)


def sample_interior_boundary(
    num_rooms: int, dims: Optional[Tuple[int, int]] = None
) -> np.array:
    """Sample a boundary for the interior of a house.

    Parameters:
        num_rooms: The number of rooms in the house.
        dims: The (x_size, z_size) dimensions of the house.
    """
    assert num_rooms > 0

    # NOTE: -1 * AVERAGE_ROOM_SIZE and +1 * AVERAGE_ROOM_SIZE adds in some
    # variance. The +1 makes high is inclusive.
    if dims is None:
        x_size, z_size = np.random.randint(
            low=max(
                MIN_HOUSE_SIDE_LENGTH,
                np.sqrt(num_rooms) * AVERAGE_ROOM_SIZE - 1 * AVERAGE_ROOM_SIZE // 2,
            ),
            high=(
                np.sqrt(num_rooms) * AVERAGE_ROOM_SIZE + 1 * AVERAGE_ROOM_SIZE // 2 + 1
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
            low=1, high=max(2, min(x_size - 1, MAX_BOUNDARY_CUT_AREA // 2))
        )
        z_cut_candidates = []

        i = 1
        while x_cut * i <= MAX_BOUNDARY_CUT_AREA and i + 1 <= z_size:
            z_cut_candidates.append(i)
            i += 1

        z_cut = random.choice(z_cut_candidates)

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
