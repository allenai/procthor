USE_ITHOR_SPLITS = False
"""Determines if the iTHOR object splits should be used spawning objects."""

OPENNESS_RANDOMIZATIONS = {
    "Box": {"population": [0, 1], "weights": [0.5, 0.5]},
    "Laptop": {"population": [0, 1, "any"], "weights": [0.4, 0.4, 0.2]},
}
"""Parameters that specify the openness randomization of an object.

Currently assumes that opening the object roughly doesn't influcence any
other objects in the scene (e.g., there are no objects on top of the object,
opening doesn't cause the size of the object to expand in the x/z direction.)
"""

SCHEMA = "0.0.1"
"""The schema version of the json file to create the house."""

MARGIN = {
    "middle": 0.35,
    "edge": {"front": 0.5, "back": 0, "sides": 0},
    "corner": {"front": 0.5, "back": 0, "sides": 0},
}
"""The margin between different objects."""

PADDING_AGAINST_WALL = 0.05
"""Padding, or extra space, added to each object.

This helps keep objects from colliding into the wall.
"""

P_CHOOSE_ASSET_GROUP = 0.6
"""The probability of choosing a semantic asset group over a standalone asset."""

MAX_INTERSECTING_OBJECT_RETRIES = 5
"""Number of retries to sample from asset group if any objects within it collide."""

P_W1_ASSET_SKIPPED = 0.8
"""Probability of skipping a weight 1 asset, when there are only weight 1 assets available.

Avoids the problem of weight 1 assets always appearing in rooms when max_floor_objects
is large.

Note that this number is often compounded, relative to max_floor_objects and
the number of w2 and asset groups available.
"""

P_CHOOSE_EDGE = 0.7
"""Probability of placing an object at the edge of a room.

When sampling a rectangle that is at the edge of the room, this denotes the
probability that the sampled object should be placed at the edge of the rectangle
instead of in the middle.
"""

P_LARGEST_RECTANGLE = 0.8
"""Probability that the largest possible rectangle gets chosen.

Among all possible rectangles with which to place an object, this denotes the
probability of choosing the largest remaining one.
"""

MIN_RECTANGLE_SIDE_SIZE = 0.5
"""The minimum rectangle size per side, in meters, that can be chosen."""

PROCTHOR_INITIALIZATION = dict(commit_id="391b3fae4d4cc026f1522e5acf60953560235971", scene="Procedural")
"Base AI2-THOR initialization parameters for ProcTHOR."

FLOOR_Y = 0
"""Position of the floor in meters."""

OUTDOOR_ROOM_ID = 1
"""The roomId of the entries in the matrix outside of the generated house."""

EMPTY_ROOM_ID = 0
"""The roomId of the entries in the matrix that are empty."""
