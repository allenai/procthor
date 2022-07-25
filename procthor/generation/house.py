import enum
import gzip
import json
import logging
import random
from collections import Counter
from functools import total_ordering
from typing import Dict, Optional, List, Tuple

import numpy as np
from ai2thor.controller import Controller
from attrs import define
from moviepy.editor import ImageSequenceClip
from shapely.geometry import Point

from procthor.constants import PROCTHOR_INITIALIZATION, SCHEMA, FLOOR_Y
from procthor.generation.agent import AgentPose, generate_starting_pose
from procthor.utils.types import (
    BoundingBox,
    HouseDict,
    Vector3,
    Wall,
    Door,
    Window,
    Object,
    ProceduralParameters,
    RoomType,
)
from .objects import ProceduralRoom
from .room_specs import RoomSpec


def snake_to_camel_case(s: str):
    split = s.split("_")
    return split[0] + "".join(part.capitalize() for part in split[1:])


@define
class HouseStructure:
    interior_boundary: np.ndarray
    floorplan: np.ndarray
    rowcol_walls: Dict[Tuple[int, int], List[Tuple[int, int]]]
    boundary_groups: Dict[Tuple[int, int], List[Tuple[float, float]]]
    xz_poly_map: Dict[int, List[Tuple[float, float]]]
    ceiling_height: float


@define
class House:
    data: HouseDict
    rooms: Dict[int, ProceduralRoom]  # TODO: Should be `rooms_map`
    interior_boundary: np.array
    room_spec: RoomSpec
    add_metadata: bool = True

    def __attrs_post_init__(self) -> None:
        if self.add_metadata:
            self._add_metadata()

    def _add_metadata(self) -> None:
        self.data["metadata"] = {
            "agent": self.choose_agent_pose(),
            "roomSpecId": self.room_spec.room_spec_id,
            "schema": SCHEMA,
        }

    def to_json(self, filename: Optional[str] = None, compressed: bool = False) -> str:
        json_rep = json.dumps(self.data, sort_keys=True, indent=4)
        if filename:
            if compressed:
                assert filename.endswith(".json.gz")
                with gzip.open(filename, "wt", encoding="UTF-8") as f:
                    json.dump(self.data, f)
            else:
                assert filename.endswith(".json")
                with open(filename, "w") as f:
                    f.write(json_rep)
        return json_rep

    @property
    def bounds(self) -> BoundingBox:
        out = BoundingBox(
            min=Vector3(x=float("inf"), y=float("inf"), z=float("inf")),
            max=Vector3(x=float("-inf"), y=float("-inf"), z=float("-inf")),
        )

        for wall in self.data["walls"]:
            for poly in wall["polygon"]:
                for k in ["x", "y", "z"]:
                    if poly[k] < out["min"][k]:
                        out["min"][k] = poly[k]
                    if poly[k] > out["max"][k]:
                        out["max"][k] = poly[k]

        return out

    def choose_agent_pose(self) -> AgentPose:
        """Generate a starting position for the default agent in the house."""
        return generate_starting_pose(self.rooms)

    def validate(self, controller: Controller) -> Dict[str, str]:
        """Validate that the house is useable.

        Returns a dictionary of warnings or errors.
        """
        warnings = {}
        controller.reset(renderImage=False)
        event = controller.step(
            action="CreateHouse", house=self.data, renderImage=False
        )
        if not event:
            warnings["CreateHouse"] = "Failed to create house."
            logging.warning(warnings)
            self.data["metadata"]["warnings"] = warnings
            return warnings
        event = controller.step(
            action="TeleportFull",
            **self.data["metadata"]["agent"],
            renderImage=False,
        )
        if not event:
            warnings["TeleportFull"] = "Unable to teleport to starting position."
            logging.warning(warnings)
            self.data["metadata"]["warnings"] = warnings
            return warnings
        event = controller.step(action="GetReachablePositions", renderImage=False)
        if not event:
            warnings["GetReachablePositions"] = "Failed to get reachable positions"
            logging.warning(warnings)
            self.data["metadata"]["warnings"] = warnings
            return warnings
        rps = event.metadata["actionReturn"]

        random.shuffle(rps)

        # Check that every room is reachable
        points_per_room = Counter({r: 0 for r in self.rooms})
        for p in rps:
            point = Point(p["x"], p["z"])
            for room_id in self.rooms.keys():
                room_poly = self.rooms[room_id].room_polygon.polygon
                if room_poly.contains(point):
                    points_per_room[room_id] += 1
                    break
            else:
                warnings[f"Unreachable point: {p}"] = "Unreachable point."

            if all(num_points >= 5 for num_points in points_per_room.values()):
                break
        else:
            unnavigable_rooms = list(
                filter(lambda r: points_per_room[r] < 5, points_per_room)
            )
            warnings[
                "RoomsNotNavigable"
            ] = f"Rooms {unnavigable_rooms} / {len(points_per_room)} are not navigable."

        if warnings:
            logging.warning(warnings)

        self.data["metadata"]["warnings"] = warnings

        return warnings

    @staticmethod
    def top_down_video(
        data: HouseDict,
        controller: Controller,
        filename: str,
        width: int = 1024,
        height: int = 1024,
    ) -> None:
        """Saves a top-down video of the house."""
        if controller is None:
            controller = Controller(
                width=width, height=height, **PROCTHOR_INITIALIZATION
            )

        controller.step(action="CreateHouse", house=data)
        max_meters = max(
            max(p["x"], p["z"]) for wall in data["walls"] for p in wall["polygon"]
        )
        raise NotImplementedError("Update max_meters from data.")

        if not controller.last_event.third_party_camera_frames:
            controller.step(
                action="AddThirdPartyCamera",
                position=dict(x=0, y=0, z=0),
                rotation=dict(x=0, y=0, z=0),
                skyboxColor="white",
            )

        dist = 2.75
        images = []
        for angle in range(0, 360, 45):
            controller.step(
                action="UpdateThirdPartyCamera",
                position=dict(
                    x=max_meters / 2 - np.sin(angle * np.pi / 180) * dist,
                    y=max_meters + 1.5,
                    z=max_meters / 2 - np.cos(angle * np.pi / 180) * dist,
                ),
                rotation=dict(x=65, y=angle, z=0),
            )
            images.append(controller.last_event.third_party_camera_frames[0])
        imsn = ImageSequenceClip(images, fps=2)
        imsn.write_videofile(filename)


@total_ordering
class NextSamplingStage(enum.Enum):
    STRUCTURE = 0
    DOORS = 1
    LIGHTS = 2
    SKYBOX = 3
    EXTERIOR_WALLS = 4
    ROOMS = 5
    FLOOR_OBJS = 6
    WALL_OBJS = 7
    SMALL_OBJS = 8
    COMPLETE = 9

    def __lt__(self, other: "NextSamplingStage"):
        if self.__class__ is other.__class__:
            return self.value < other.value
        raise NotImplementedError


@define
class PartialHouse:
    house_structure: HouseStructure
    room_spec: RoomSpec
    procedural_parameters: ProceduralParameters

    room_types: Optional[List[RoomType]] = None
    doors: Optional[List[Door]] = None
    windows: Optional[List[Window]] = None
    objects: Optional[List[Object]] = None
    walls: Optional[List[Wall]] = None

    rooms: Optional[Dict[int, ProceduralRoom]] = None  # TODO: Should be `rooms_map`
    next_sampling_stage: Optional[NextSamplingStage] = NextSamplingStage.STRUCTURE

    @classmethod
    def from_structure_and_room_spec(
        cls,
        house_structure: HouseStructure,
        room_spec: RoomSpec,
    ) -> "PartialHouse":
        walls = []
        for room_id, xz_poly in house_structure.xz_poly_map.items():
            for ((x0, z0), (x1, z1)) in xz_poly:
                wall_id = f"wall|{room_id}|{min(x0, x1):.2f}|{min(z0, z1):.2f}|{max(x0, x1):.2f}|{max(z0, z1):.2f}"
                wall = Wall(
                    id=wall_id,
                    roomId=f"room|{room_id}",
                    polygon=[
                        Vector3(x=x0, y=FLOOR_Y, z=z0),
                        Vector3(x=x1, y=FLOOR_Y, z=z1),
                        Vector3(x=x0, y=FLOOR_Y + house_structure.ceiling_height, z=z0),
                        Vector3(x=x1, y=FLOOR_Y + house_structure.ceiling_height, z=z1),
                    ],
                )
                walls.append(wall)

        return PartialHouse(
            house_structure=house_structure,
            room_spec=room_spec,
            room_types=[
                RoomType(
                    id=f"room|{room_id}",
                    roomType=room_spec.room_type_map[room_id],
                    children=[],
                    ceilings=[],
                    floorPolygon=[
                        Vector3(x=x0, y=0, z=z0) for ((x0, z0), (x1, z1)) in xz_poly
                    ],
                )
                for room_id, xz_poly in house_structure.xz_poly_map.items()
            ],
            walls=walls,
            objects=[],
            procedural_parameters=ProceduralParameters(
                floorColliderThickness=1.0,
                receptacleHeight=0.7,
                lights=[],
                reflections=[],
            ),
            next_sampling_stage=NextSamplingStage.DOORS,
        )

    def to_house_dict(self) -> HouseDict:
        house_dict = {}
        from_to_map = {
            "room_types": "rooms",
            "doors": "doors",
            "windows": "windows",
            "objects": "objects",
            "walls": "walls",
            "procedural_parameters": "proceduralParameters",
        }
        for k in from_to_map:
            val = getattr(self, k)
            if val is not None:
                house_dict[from_to_map[k]] = val

        return HouseDict(**house_dict)

    def to_house(self) -> House:
        assert all(
            getattr(self, k) is not None
            for k in ["rooms", "doors", "windows", "objects", "walls"]
        )

        return House(
            data=self.to_house_dict(),
            rooms=self.rooms,
            interior_boundary=self.house_structure.interior_boundary,
            room_spec=self.room_spec,
        )

    def advance_sampling_stage(self):
        self.next_sampling_stage = next(
            (
                nss
                for nss in NextSamplingStage
                if nss.value == self.next_sampling_stage.value + 1
            ),
            self.next_sampling_stage.COMPLETE,
        )

    def reset(self):
        raise NotImplementedError
