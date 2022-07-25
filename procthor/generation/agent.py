import random
from typing import Dict, Literal, TypedDict

from procthor.utils.types import Vector3
from .objects import ProceduralRoom

ROTATIONS = [0, 90, 180, 270]
"""Valid starting rotations for the agent in the scene."""

AGENT_HORIZON: Literal[-30, 0, 30, 60] = 30
"""The starting camera horizon of the agent."""

AGENT_IS_STANDING = True
"""Set if the agent starts in a standing position."""

GRID_SIZE = 0.25
"""The navigable gridSize of the room in AI2-THOR."""

AGENT_Y_HEIGHT = 0.95
"""This is the position that allows the agent to teleport on the ground."""


class AgentPose(TypedDict):
    """The full pose of the agent, used for `TeleportFull`."""

    position: Vector3
    """The position of the agent."""

    rotation: Vector3
    """The rotation of the agent."""

    standing: bool
    """Whether the agent is standing or not."""

    horizon: Literal[-30, 0, 30, 60]
    """The starting camera horizon of the agent."""


def generate_starting_pose(rooms: Dict[int, ProceduralRoom]) -> AgentPose:
    """Choose the starting agent pose in the house."""

    rooms = list(rooms.values())
    random.shuffle(rooms)
    for room in rooms:
        rect = room.sample_next_rectangle(choose_largest_rectangle=True)
        if rect:
            x0, z0, x1, z1 = rect
            mid_x = (x0 + x1) / 2
            mid_z = (z0 + z1) / 2

            x = round(mid_x / GRID_SIZE) * GRID_SIZE
            z = round(mid_z / GRID_SIZE) * GRID_SIZE
            break
    else:
        raise Exception("Unable to place the agent in any room in the house!")

    return AgentPose(
        position=Vector3(x=x, y=AGENT_Y_HEIGHT, z=z),
        rotation=Vector3(x=0, y=random.choice(ROTATIONS), z=0),
        standing=AGENT_IS_STANDING,
        horizon=AGENT_HORIZON,
    )
