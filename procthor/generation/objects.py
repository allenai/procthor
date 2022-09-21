import copy
import random
from collections import defaultdict
from statistics import mean
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import pandas as pd
from ai2thor.controller import Controller
from attr import field
from attrs import define
from procthor.constants import (
    MARGIN,
    MAX_INTERSECTING_OBJECT_RETRIES,
    MIN_RECTANGLE_SIDE_SIZE,
    OPENNESS_RANDOMIZATIONS,
    P_CHOOSE_ASSET_GROUP,
    P_CHOOSE_EDGE,
    P_LARGEST_RECTANGLE,
    P_W1_ASSET_SKIPPED,
    PADDING_AGAINST_WALL,
)
from procthor.databases import ProcTHORDatabase, get_spawnable_asset_group_info
from procthor.utils.types import Object, Split, Vector3
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

if TYPE_CHECKING:
    from . import PartialHouse

from .asset_groups import AssetGroupGenerator

P_ALLOW_HOUSE_PLANT_GROUP = 0.5
"""Quick hack to prevent oversampling the house plants on the floor of the room."""

P_ALLOW_TV_GROUP = 0.5
"""Quick hack to prevent oversampling the standalone televisions on the floor.

(Standalone tvs are just tvs on tv stands, not those with chairs/couches).

This makes wall tvs more likely.
"""


class ChosenAsset(TypedDict):
    xSize: float
    ySize: float
    zSize: float
    assetId: str
    rotated: bool


class ChosenAssetGroup(TypedDict):
    assetGroupName: str
    xSize: float
    ySize: float
    zSize: float
    rotated: bool
    objects: Any
    bounds: Any
    allowDuplicates: bool


def is_chosen_asset_group(data: dict) -> bool:
    """Determine if a dict is from a ChosenAsset or ChosenAssetGroup."""
    return "objects" in data


def sample_openness(obj_type: str) -> float:
    """Sample the openness of an object."""
    openness = random.choices(**OPENNESS_RANDOMIZATIONS[obj_type])[0]
    if openness == "any":
        openness = random.random()
    return openness


@define
class Asset:
    asset_id: str
    """The unique AI2-THOR identifier of the asset."""

    top_down_poly: Sequence[Tuple[float, float]]
    """Full bounding box area that the polygon takes up. Including padding, excludes margin."""

    top_down_poly_with_margin: Sequence[Tuple[float, float]]
    """Full bounding box area that the polygon takes up. Includes padding and margin."""

    rotation: int
    """The yaw rotation of the object."""

    position: Vector3
    """Center position of the asset. Includes padding, excludes margin."""

    anchor_type: Literal["inCorner", "onEdge", "inMiddle"]
    """Specifies the location of the asset in the room."""

    room_id: int
    """The id of the procedural room."""

    object_n: int
    """The number of assets/asset groups in the scene before placing this asset."""

    states: Dict[str, Any]

    pt_db: ProcTHORDatabase

    poly_xs: List[float] = field(init=False)
    poly_zs: List[float] = field(init=False)

    margined_poly_xs: List[float] = field(init=False)
    margined_poly_zs: List[float] = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.poly_xs = [p[0] for p in self.top_down_poly]
        self.poly_zs = [p[1] for p in self.top_down_poly]

        self.margined_poly_xs = [p[0] for p in self.top_down_poly_with_margin]
        self.margined_poly_zs = [p[1] for p in self.top_down_poly_with_margin]

    @property
    def asset_dict(self) -> Object:
        return Object(
            id=f"{self.room_id}|{self.object_n}",
            position=self.position,
            rotation=Vector3(x=0, y=self.rotation, z=0),
            assetId=self.asset_id,
            kinematic=bool(
                self.pt_db.PLACEMENT_ANNOTATIONS.loc[
                    self.pt_db.ASSET_ID_DATABASE[self.asset_id]["objectType"]
                ]["isKinematic"]
            ),
            **self.states,
        )


@define
class AssetGroup:
    asset_group_name: str
    """The name of the asset group."""

    top_down_poly: Sequence[Tuple[float, float]]
    """Full bounding box area that the polygon takes up. Including padding, excludes margin."""

    top_down_poly_with_margin: Sequence[Tuple[float, float]]
    """Full bounding box area that the polygon takes up. Includes padding and margin."""

    objects: List[Dict[str, Any]]
    """
    Must have keys for:
    - position: Vector3 the center position of the asset in R^3 (includes padding).
    - rotation: float the top down rotation of the object in degrees
    - assetId: str
    - instanceId: str the instanceId of the asset within the group.
    """

    anchor_type: Literal["inCorner", "onEdge", "inMiddle"]
    """Specifies the location of the asset group in the room."""

    room_id: int
    """The id of the procedural room."""

    object_n: int
    """The number of assets/asset groups in the scene before placing this asset."""

    pt_db: ProcTHORDatabase

    poly_xs: List[float] = field(init=False)
    poly_zs: List[float] = field(init=False)

    margined_poly_xs: List[float] = field(init=False)
    margined_poly_zs: List[float] = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.poly_xs = [p[0] for p in self.top_down_poly]
        self.poly_zs = [p[1] for p in self.top_down_poly]

        self.margined_poly_xs = [p[0] for p in self.top_down_poly_with_margin]
        self.margined_poly_zs = [p[1] for p in self.top_down_poly_with_margin]

    @property
    def assets_dict(self) -> List[Object]:
        # NOTE: Assign "above" objects to be children to parent receptacle
        asset_group_metadata = self.pt_db.ASSET_GROUPS[self.asset_group_name][
            "assetMetadata"
        ]
        parent_children_pairs = []
        for child_id, metadata in asset_group_metadata.items():
            if (
                "parentInstanceId" in metadata
                and metadata["position"]["verticalAlignment"] == "above"
            ):
                parent_id = str(metadata["parentInstanceId"])
                parent_children_pairs.append((parent_id, child_id))

        objects = {
            obj["instanceId"]: Object(
                id=f"{self.room_id}|{self.object_n}|{i}",
                position=obj["position"],
                rotation=Vector3(x=0, y=obj["rotation"], z=0),
                assetId=obj["assetId"],
                kinematic=bool(
                    self.pt_db.PLACEMENT_ANNOTATIONS.loc[
                        self.pt_db.ASSET_ID_DATABASE[obj["assetId"]]["objectType"]
                    ]["isKinematic"]
                ),
            )
            for i, obj in enumerate(self.objects)
        }

        # NOTE: assign children to "children" object and then remove them as keys
        # in the objects dict.
        child_instance_ids = set()
        for parent_id, child_id in parent_children_pairs:
            if "children" not in objects[parent_id]:
                objects[parent_id]["children"] = []
            objects[parent_id]["children"].append(objects[child_id])
            child_instance_ids.add(child_id)
        for child_id in child_instance_ids:
            del objects[child_id]

        return list(objects.values())


class OrthogonalPolygon:
    def __init__(self, polygon) -> None:
        self.polygon = polygon
        self._set_attributes()

    def _set_attributes(self) -> None:
        if isinstance(self.polygon, MultiPolygon):
            points = set()
            for poly in self.polygon.geoms:
                points.update(set(poly.exterior.coords))
                for interior in poly.interiors:
                    points.update(set(interior.coords[:]))
        else:
            points = set(self.polygon.exterior.coords)
            for interior in self.polygon.interiors:
                points.update(set(interior.coords[:]))

        self.xs = [p[0] for p in points]
        self.zs = [p[1] for p in points]

        self.unique_xs = sorted(list(set(self.xs)))
        self.unique_zs = sorted(list(set(self.zs)))

        self.x_edges_map = self._set_x_edges_map(points)
        self.z_edges_map = self._set_z_edges_map(points)

        # NOTE: get_neighboring_rectangles() sets the area
        self.get_neighboring_rectangles()

    def _set_x_edges_map(self, points: Set[Tuple[float, float]]):
        out = defaultdict(list)
        points = list(points)
        for p0, p1 in zip(points, points[1:]):
            if p0[0] == p1[0]:
                out[p0[0]].append(sorted([p0[1], p1[1]]))
        return out

    def _set_z_edges_map(self, points: Set[Tuple[float, float]]):
        out = defaultdict(list)
        points = list(points)
        for p0, p1 in zip(points, points[1:]):
            if p0[1] == p1[1]:
                out[p0[1]].append(sorted([p0[0], p1[0]]))
        return out

    def get_neighboring_rectangles(self) -> Set[Tuple[float, float, float, float]]:
        out = set()
        area = 0
        for x0, x1 in zip(self.unique_xs, self.unique_xs[1:]):
            for z0, z1 in zip(self.unique_zs, self.unique_zs[1:]):
                mid_x = (x0 + x1) / 2
                mid_z = (z0 + z1) / 2
                if self.is_point_inside((mid_x, mid_z)):
                    out.add((x0, z0, x1, z1))
                    area += (x1 - x0) * (z1 - z0)
        return out

    def _join_neighboring_rectangles(
        self, rects: Set[Tuple[float, float, float, float]]
    ) -> Tuple[float, float, float, float]:
        orig_rects = rects.copy()
        out = set()
        for rect1 in rects.copy():
            x0_0, z0_0, x1_0, z1_0 = rect1
            points1 = {(x0_0, z0_0), (x0_0, z1_0), (x1_0, z1_0), (x1_0, z0_0)}
            for rect2 in rects - {rect1}:
                x0_1, z0_1, x1_1, z1_1 = rect2
                points2 = {(x0_1, z0_1), (x0_1, z1_1), (x1_1, z1_1), (x1_1, z0_1)}
                if len(points1 & points2) == 2:
                    out.add(
                        (
                            min(x0_0, x1_0, x0_1, x1_1),
                            min(z0_0, z1_0, z0_1, z1_1),
                            max(x0_0, x1_0, x0_1, x1_1),
                            max(z0_0, z1_0, z0_1, z1_1),
                        )
                    )
        return out - orig_rects

    def get_all_rectangles(self) -> Set[Tuple[float, float, float, float]]:
        neighboring_rectangles = self.get_neighboring_rectangles().copy()
        curr_rects = neighboring_rectangles
        while True:
            rect_candidates = self._join_neighboring_rectangles(curr_rects)
            rects = curr_rects | rect_candidates
            if len(rects) == len(curr_rects):
                return curr_rects
            curr_rects = rects

    def _get_edge_cross_count(self, point: Tuple[float, float]) -> int:
        edge_cross_count = 0
        for x in self.unique_xs:
            if x > point[0]:
                break
            for y0, y1 in self.x_edges_map[x]:
                if y0 < point[1] < y1:
                    edge_cross_count += 1
                    break
        return edge_cross_count

    def is_point_inside(self, point: Tuple[float, float]) -> bool:
        return self.polygon.contains(Point(*point))

    def subtract(self, polygon: Polygon) -> None:
        self.polygon -= polygon
        self._set_attributes()

    def __repr__(self) -> str:
        return self.polygon.__repr__()

    @staticmethod
    def add_margin_to_top_down_poly(
        poly: Sequence[Tuple[float, float]],
        rotation: Literal[0, 90, 180, 270],
        anchor_type: Literal["inCorner", "onEdge", "inMiddle"],
    ) -> Sequence[Tuple[float, float]]:
        """Adds margin to a top-down polygon."""
        if rotation not in {0, 90, 180, 270}:
            raise ValueError(
                f"rotation must be in {{0, 90, 180, 270}}. Got {rotation}."
            )
        if anchor_type not in {"inCorner", "onEdge", "inMiddle"}:
            raise ValueError(
                'anchor_type must be in {{"inCorner", "onEdge", "inMiddle"}}.'
                f" You gave {anchor_type}."
            )

        min_x = min(p[0] for p in poly)
        max_x = max(p[0] for p in poly)

        min_z = min(p[1] for p in poly)
        max_z = max(p[1] for p in poly)

        if anchor_type == "inMiddle":
            # NOTE: add margin to each side of a middle object.
            margin = MARGIN["middle"]

            min_x -= margin
            max_x += margin

            min_z -= margin
            max_z += margin
        else:
            if anchor_type == "onEdge":
                front_space = MARGIN["edge"]["front"]
                back_space = MARGIN["edge"]["back"]
                side_space = MARGIN["edge"]["sides"]
            elif anchor_type == "inCorner":
                front_space = MARGIN["corner"]["front"]
                back_space = MARGIN["corner"]["back"]
                side_space = MARGIN["corner"]["sides"]

            if rotation == 0:
                max_z += front_space
                min_z -= back_space
                max_x += side_space
                min_x -= side_space
            elif rotation == 90:
                max_x += front_space
                min_x -= back_space
                max_z += side_space
                min_z -= side_space
            elif rotation == 180:
                min_z -= front_space
                max_z += back_space
                max_x += side_space
                min_x -= side_space
            elif rotation == 270:
                min_x -= front_space
                max_x += back_space
                max_z += side_space
                min_z -= side_space

        return [(min_x, min_z), (min_x, max_z), (max_x, max_z), (max_x, min_z)]

    @staticmethod
    def get_top_down_poly(
        anchor_location: Tuple[float, float],
        anchor_delta: int,
        asset_bb: Dict[str, float],
        rotated: bool,
    ) -> List[Tuple[float, float]]:
        """Return the top-down polygon from an asset."""
        x, z = anchor_location
        rot1, rot2 = ("z", "x") if rotated else ("x", "z")

        if anchor_delta == 0:
            top_down_poly = [
                (x, z),
                (x, z + asset_bb[rot2] + PADDING_AGAINST_WALL),
                (
                    x - asset_bb[rot1] - PADDING_AGAINST_WALL,
                    z + asset_bb[rot2] + PADDING_AGAINST_WALL,
                ),
                (x - asset_bb[rot1] - PADDING_AGAINST_WALL, z),
            ]
        elif anchor_delta == 1:
            top_down_poly = [
                (x - asset_bb[rot1] / 2, z),
                (x + asset_bb[rot1] / 2, z),
                (x + asset_bb[rot1] / 2, z + asset_bb[rot2] + PADDING_AGAINST_WALL),
                (x - asset_bb[rot1] / 2, z + asset_bb[rot2] + PADDING_AGAINST_WALL),
            ]
        elif anchor_delta == 2:
            top_down_poly = [
                (x, z),
                (x, z + asset_bb[rot2] + PADDING_AGAINST_WALL),
                (
                    x + asset_bb[rot1] + PADDING_AGAINST_WALL,
                    z + asset_bb[rot2] + PADDING_AGAINST_WALL,
                ),
                (x + asset_bb[rot1] + PADDING_AGAINST_WALL, z),
            ]
        elif anchor_delta == 3:
            top_down_poly = [
                (x, z + asset_bb[rot2] / 2),
                (x, z - asset_bb[rot2] / 2),
                (x - asset_bb[rot1] - PADDING_AGAINST_WALL, z - asset_bb[rot2] / 2),
                (x - asset_bb[rot1] - PADDING_AGAINST_WALL, z + asset_bb[rot2] / 2),
            ]
        elif anchor_delta == 4:
            top_down_poly = [
                (x - asset_bb[rot1] / 2, z - asset_bb[rot2] / 2),
                (x + asset_bb[rot1] / 2, z - asset_bb[rot2] / 2),
                (x + asset_bb[rot1] / 2, z + asset_bb[rot2] / 2),
                (x - asset_bb[rot1] / 2, z + asset_bb[rot2] / 2),
            ]
        elif anchor_delta == 5:
            top_down_poly = [
                (x, z + asset_bb[rot2] / 2),
                (x, z - asset_bb[rot2] / 2),
                (x + asset_bb[rot1] + PADDING_AGAINST_WALL, z - asset_bb[rot2] / 2),
                (x + asset_bb[rot1] + PADDING_AGAINST_WALL, z + asset_bb[rot2] / 2),
            ]
        elif anchor_delta == 6:
            top_down_poly = [
                (x, z),
                (x, z - asset_bb[rot2] - PADDING_AGAINST_WALL),
                (
                    x - asset_bb[rot1] - PADDING_AGAINST_WALL,
                    z - asset_bb[rot2] - PADDING_AGAINST_WALL,
                ),
                (x - asset_bb[rot1] - PADDING_AGAINST_WALL, z),
            ]
        elif anchor_delta == 7:
            top_down_poly = [
                (x - asset_bb[rot1] / 2, z),
                (x + asset_bb[rot1] / 2, z),
                (x + asset_bb[rot1] / 2, z - asset_bb[rot2] - PADDING_AGAINST_WALL),
                (x - asset_bb[rot1] / 2, z - asset_bb[rot2] - PADDING_AGAINST_WALL),
            ]
        elif anchor_delta == 8:
            top_down_poly = [
                (x, z),
                (x, z - asset_bb[rot2] - PADDING_AGAINST_WALL),
                (
                    x + asset_bb[rot1] + PADDING_AGAINST_WALL,
                    z - asset_bb[rot2] - PADDING_AGAINST_WALL,
                ),
                (x + asset_bb[rot1] + PADDING_AGAINST_WALL, z),
            ]
        else:
            raise Exception(
                f"Unknown anchor anchor_delta: {anchor_delta}. Must be an int in [0:8]."
            )

        return top_down_poly


class ProceduralRoom:
    def __init__(
        self,
        polygon: Sequence[Tuple[int, int]],
        room_type: Literal["Kitchen", "LivingRoom", "Bedroom", "Bathroom"],
        room_id: int,
        split: Split,
        door_polygons: List[Polygon],
        pt_db: ProcTHORDatabase,
    ) -> None:
        """

        Parameters:
        - room_type: str must be in {"Kitchen", "LivingRoom", "Bedroom", "Bathroom"}.
        - split: str must be in {"train", "val", "test"}.
        """
        assert room_type in {"Kitchen", "LivingRoom", "Bedroom", "Bathroom"}
        assert split in {"train", "val", "test"}

        self.room_polygon = OrthogonalPolygon(polygon=copy.deepcopy(polygon))
        self.open_polygon = OrthogonalPolygon(polygon=copy.deepcopy(polygon))

        self.door_polygons = door_polygons
        self._cut_out_doors()

        self.room_type = room_type
        self.room_id = room_id

        self.split = split

        self.pt_db = pt_db
        self.assets: List[Union[Asset, AssetGroup]] = []
        self.last_rectangles: Optional[Set[Tuple[float, float, float, float]]] = None

    def _cut_out_doors(self) -> None:
        for door_polygon in self.door_polygons:
            self.open_polygon.subtract(door_polygon)

    def add_asset(self, asset: Union[Asset, AssetGroup]) -> None:
        """Add an asset to the room.

        Assumes that the asset can be placed
        """
        self.assets.append(asset)
        self.open_polygon.subtract(Polygon(asset.top_down_poly_with_margin))

    def sample_next_rectangle(
        self, choose_largest_rectangle: bool = False, cache_rectangles: bool = False
    ) -> Optional[Tuple[float, float, float, float]]:
        """

        If None is returned, all the rectangles are greater in size than min_size.
        """
        if cache_rectangles:
            assert self.last_rectangles is not None, "Unable to cache rectangles!"
            rectangles = self.last_rectangles
        else:
            rectangles = self.open_polygon.get_all_rectangles()
            self.last_rectangles = rectangles

        if len(rectangles) == 0:
            return None

        if choose_largest_rectangle or random.random() < P_LARGEST_RECTANGLE:
            # NOTE: p(epsilon) = choose largest area
            max_area = 0
            out: Optional[Tuple[float, float, float, float]] = None
            for rect in rectangles:
                x0, z0, x1, z1 = rect
                area = (x1 - x0) * (z1 - z0)
                if area > max_area:
                    max_area = area
                    out = rect
            return out

        # NOTE: p(1 - epsilon) = randomly choose rect, weighted by area
        rectangles = list(rectangles)
        population = []
        weights = []
        for rect in rectangles:
            x0, z0, x1, z1 = rect
            xdist = x1 - x0
            zdist = z1 - z0
            if xdist < MIN_RECTANGLE_SIDE_SIZE or zdist < MIN_RECTANGLE_SIDE_SIZE:
                continue
            area = xdist * zdist
            weights.append(area)
            population.append(rect)
        if not weights:
            return None
        return random.choices(population=population, weights=weights, k=1)[0]

    def sample_place_asset_in_rectangle(
        self,
        asset: Union[ChosenAsset, ChosenAssetGroup],
        rectangle: Tuple[float, float, float, float],
        anchor_type: Literal["inCorner", "onEdge", "inMiddle"],
        x_info: Any,
        z_info: Any,
        anchor_delta: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8],
    ) -> None:
        """Places a chosen asset somewhere inside of the rectangle."""
        bb = dict(
            x=asset["xSize"] if not asset["rotated"] else asset["zSize"],
            y=asset["ySize"],
            z=asset["zSize"] if not asset["rotated"] else asset["xSize"],
        )
        if anchor_type == "inCorner":
            x, z = x_info, z_info
            if anchor_delta == 0:
                rotation = 270 if asset["rotated"] else 0
            elif anchor_delta == 2:
                rotation = 90 if asset["rotated"] else 0
            elif anchor_delta == 6:
                rotation = 270 if asset["rotated"] else 180
            elif anchor_delta == 8:
                rotation = 90 if asset["rotated"] else 180
        elif anchor_type == "onEdge":
            xs, zs = x_info, z_info
            if anchor_delta in {1, 7}:
                # randomize x
                x0, x1 = xs
                x_length = x1 - x0
                full_rand_dist = x_length - bb["x"]
                rand_dist = random.random() * full_rand_dist
                x = x0 + rand_dist + bb["x"] / 2
                z = sum(zs) / 2
                rotation = 180 if anchor_delta == 7 else 0
            elif anchor_delta in {3, 5}:
                # randomize z
                z0, z1 = zs
                z_length = z1 - z0
                full_rand_dist = z_length - bb["z"]
                rand_dist = random.random() * full_rand_dist
                x = sum(xs) / 2
                z = z0 + rand_dist + bb["z"] / 2
                rotation = 90 if anchor_delta == 5 else 270
        elif anchor_type == "inMiddle":
            x, z = 0, 0
            x0, z0, x1, z1 = rectangle
            x_length = x1 - x0
            z_length = z1 - z0
            full_x_rand_dist = x_length - bb["x"]
            full_z_rand_dist = z_length - bb["z"]
            rand_x_dist = random.random() * full_x_rand_dist
            rand_z_dist = random.random() * full_z_rand_dist
            x = x0 + rand_x_dist + bb["x"] / 2
            z = z0 + rand_z_dist + bb["z"] / 2
            rotation = random.choice([90, 270] if asset["rotated"] else [0, 180])
        top_down_poly = OrthogonalPolygon.get_top_down_poly(
            anchor_location=(x, z),
            anchor_delta=anchor_delta,
            asset_bb=bb,
            rotated=False,
        )
        center_x = mean(p[0] for p in top_down_poly)
        center_z = mean(p[1] for p in top_down_poly)

        top_down_poly_with_margin = OrthogonalPolygon.add_margin_to_top_down_poly(
            poly=top_down_poly,
            rotation=rotation,
            anchor_type=anchor_type,
        )

        if is_chosen_asset_group(asset):
            objects = []
            for obj in asset["objects"]:
                x = obj["position"]["x"]
                z = obj["position"]["z"]

                # NOTE: center the origin of the asset group at (0, 0)
                x -= asset["bounds"]["x"]["center"]
                z -= asset["bounds"]["z"]["center"]

                # NOTE: rotate the asset about the origin
                rotation_rad = -rotation * np.pi / 180.0
                x, z = (
                    x * np.cos(rotation_rad) - z * np.sin(rotation_rad),
                    x * np.sin(rotation_rad) + z * np.cos(rotation_rad),
                )

                x += center_x
                z += center_z

                objects.append(
                    {
                        "assetId": obj["assetId"],
                        # NOTE: adds the base rotation to the object's rotation
                        "rotation": (obj["rotation"] + rotation) % 360,
                        "position": Vector3(x=x, y=obj["position"]["y"], z=z),
                        "instanceId": obj["instanceId"],
                    }
                )
            self.add_asset(
                AssetGroup(
                    asset_group_name=asset["assetGroupName"],
                    top_down_poly=top_down_poly,
                    objects=objects,
                    top_down_poly_with_margin=top_down_poly_with_margin,
                    anchor_type=anchor_type,
                    room_id=self.room_id,
                    object_n=len(self.assets),
                    pt_db=self.pt_db,
                )
            )
        else:
            states = {}

            # NOTE: Don't randomize the openness in any of the asset groups.
            # NOTE: assumes opening the object doesn't change its top-down bbox.
            obj_type = self.pt_db.ASSET_ID_DATABASE[asset["assetId"]]["objectType"]
            if (
                obj_type in OPENNESS_RANDOMIZATIONS
                and "CanOpen"
                in self.pt_db.ASSET_ID_DATABASE[asset["assetId"]]["secondaryProperties"]
            ):
                states["openness"] = sample_openness(obj_type)

            self.add_asset(
                Asset(
                    asset_id=asset["assetId"],
                    top_down_poly=top_down_poly,
                    top_down_poly_with_margin=top_down_poly_with_margin,
                    rotation=rotation,
                    position=Vector3(
                        x=center_x,
                        y=bb["y"] / 2,
                        z=center_z,
                    ),
                    anchor_type=anchor_type,
                    room_id=self.room_id,
                    object_n=len(self.assets),
                    states=states,
                    pt_db=self.pt_db,
                )
            )

    @staticmethod
    def sample_rotation(
        asset: Dict[str, Any], rect_x_length: float, rect_z_length: float
    ) -> bool:
        valid_rotated = []
        if asset["xSize"] < rect_x_length and asset["zSize"] < rect_z_length:
            valid_rotated.append(False)
        if asset["xSize"] < rect_z_length and asset["zSize"] < rect_x_length:
            valid_rotated.append(True)
        return random.choice(valid_rotated)

    def place_asset_group(
        self,
        asset_group: pd.DataFrame,
        set_rotated: Optional[bool],
        rect_x_length: float,
        rect_z_length: float,
    ) -> Optional[ChosenAssetGroup]:
        """

        Returns None if the asset group collides on each attempt (very unlikely).
        """
        asset_group_generator: AssetGroupGenerator = asset_group[
            "assetGroupGenerator"
        ].iloc[0]

        # NOTE: sample object placement from within the asset group generator.
        for _ in range(MAX_INTERSECTING_OBJECT_RETRIES):
            object_placement = asset_group_generator.sample_object_placement()

            # NOTE: check for collisions
            (
                any_collisions,
                intersecting_objects,
            ) = asset_group_generator.get_intersecting_objects(
                object_placement=object_placement["objects"]
            )
            if not any_collisions:
                if set_rotated is None:
                    set_rotated = ProceduralRoom.sample_rotation(
                        asset=asset_group.to_dict(orient="records")[0],
                        rect_x_length=rect_x_length,
                        rect_z_length=rect_z_length,
                    )

                return ChosenAssetGroup(
                    assetGroupName=asset_group["assetGroupName"].iloc[0],
                    xSize=object_placement["bounds"]["x"]["length"],
                    ySize=object_placement["bounds"]["y"]["length"],
                    zSize=object_placement["bounds"]["z"]["length"],
                    rotated=set_rotated,
                    objects=object_placement["objects"],
                    bounds=object_placement["bounds"],
                    allowDuplicates=asset_group["allowDuplicates"].iloc[0],
                )

    def place_asset(
        self,
        asset: pd.DataFrame,
        set_rotated: Optional[bool],
        rect_x_length: float,
        rect_z_length: float,
    ) -> ChosenAsset:
        # NOTE: convert the pd dataframe to a dict
        asset.reset_index(drop=False, inplace=True)
        asset = asset.to_dict(orient="records")[0]

        # NOTE: Choose the rotation if both were valid.
        if set_rotated is None:
            set_rotated = ProceduralRoom.sample_rotation(
                asset=asset, rect_x_length=rect_x_length, rect_z_length=rect_z_length
            )
        asset["rotated"] = set_rotated

        return ChosenAsset(**asset)

    def sample_anchor_location(
        self,
        rectangle: Tuple[float, float, float, float],
    ) -> Tuple[Optional[float], Optional[float], int, str]:
        """Chooses which object to place in the rectangle.

        Returns:
            The (x, z, anchor_delta, anchor_type), where anchor_delta specifies
            the direction of where to place the object. Specifically, it can be though
            of with axes: ::

                0   1   2
                    |
                3 - 4 - 5
                    |
                6   7   8

            where

            * 4 specifies to center the object at (x, z)

            * 8 specifies that the object should go to the bottom right of (x, z)

            * 0 specifies that the object should go to the upper left of (x, z)

            * 1 specifies that the object should go to the upper middle of (x, z)

            and so on.

            The :code:`anchor_type` is:

            * "inCorner" of open area in the scene.

            * "onEdge" of open area in the scene.

            * "inMiddle" of the scene, not next to any other objects.

        """
        x0, z0, x1, z1 = rectangle

        # Place the object in a corner of the room
        rect_corners = [(x0, z0, 2), (x0, z1, 8), (x1, z1, 6), (x1, z0, 0)]
        random.shuffle(rect_corners)
        epsilon = 1e-3
        corners = []
        for x, z, anchor_delta in rect_corners:
            q1 = self.room_polygon.is_point_inside((x + epsilon, z + epsilon))
            q2 = self.room_polygon.is_point_inside((x - epsilon, z + epsilon))
            q3 = self.room_polygon.is_point_inside((x - epsilon, z - epsilon))
            q4 = self.room_polygon.is_point_inside((x + epsilon, z - epsilon))
            if (q1 and q3 and not q2 and not q4) or (q2 and q4 and not q1 and not q3):
                # DiagCorner
                corners.append((x, z, anchor_delta, "inCorner"))
            elif (
                (q1 and not q2 and not q3 and not q4)
                or (q2 and not q1 and not q3 and not q4)
                or (q3 and not q1 and not q2 and not q4)
                or (q4 and not q1 and not q2 and not q3)
            ):
                corners.append((x, z, anchor_delta, "inCorner"))
        if corners:
            return random.choice(corners)

        # Place the object on an edge of the room
        edges = []
        rect_edge_lines = [
            (LineString([(x0, z0), (x1, z0)]), 1),
            (LineString([(x0, z0), (x0, z1)]), 5),
            (LineString([(x1, z0), (x1, z1)]), 3),
            (LineString([(x0, z1), (x1, z1)]), 7),
        ]
        random.shuffle(rect_edge_lines)
        room_outer_lines = LineString(self.room_polygon.polygon.exterior.coords)
        for rect_edge_line, anchor_delta in rect_edge_lines:
            if room_outer_lines.contains(rect_edge_line):
                xs = [p[0] for p in rect_edge_line.coords]
                zs = [p[1] for p in rect_edge_line.coords]
                edges.append((xs, zs, anchor_delta, "onEdge"))
        if edges and random.random() < P_CHOOSE_EDGE:
            return random.choice(edges)

        # Place an object in the middle of the room
        return (None, None, 4, "inMiddle")

    def save_viz(self, path) -> None:
        import matplotlib.pyplot as plt

        self.visualize()
        plt.savefig(path, bbox_inches="tight")

    def visualize(self) -> "plt.Axes":
        import matplotlib.pyplot as plt

        poly = self.room_polygon.polygon.exterior.coords
        xs = [p[0] for p in poly]
        zs = [p[1] for p in poly]
        room_type_to_color = {
            "Kitchen": "#ffd6e7",
            "LivingRoom": "#d9f7be",
            "Bedroom": "#fff1b8",
            "Bathroom": "#bae7ff",
        }
        plt.fill(
            xs,
            zs,
            room_type_to_color[self.room_type],
            label=f"{self.room_type} ({self.room_id})",
        )
        plt.plot(xs, zs, "#000000")

        for door in self.door_polygons:
            xs = [p[0] for p in door.exterior.coords]
            zs = [p[1] for p in door.exterior.coords]
            plt.fill(xs, zs, "#000000", alpha=0.35)

        plt.xlabel("$x$")
        plt.ylabel("$z$")
        plt.title("Room Layout")

        open_poly = self.open_polygon
        min_x = open_poly.unique_xs[0]
        max_x = open_poly.unique_xs[-1]

        min_z = open_poly.unique_zs[0]
        max_z = open_poly.unique_zs[-1]

        grid_params = dict(c="#8c8c8c", alpha=0.35)
        margin = 0  # 0.35
        for z in reversed(open_poly.unique_zs):
            plt.plot([min_x - margin, max_x + margin], [z, z], "--", **grid_params)

        for x in open_poly.unique_xs:
            plt.plot([x, x], [min_z - margin, max_z + margin], "--", **grid_params)

        for x in open_poly.unique_xs:
            for z in open_poly.unique_zs:
                plt.scatter(x, z, zorder=2, c="#8c8c8c", s=3)

        for asset in self.assets:
            label = (
                (" " * 4 + asset.asset_id)
                if isinstance(asset, Asset)
                else (" " * 4 + asset.asset_group_name)
            )
            plt.fill(asset.margined_poly_xs, asset.margined_poly_zs, "k", alpha=0.15)
            plt.fill(asset.poly_xs, asset.poly_zs, label=label, alpha=0.5)

        plt.gca().set_aspect("equal")

        plt.gca().legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        # plt.xlim([min_x - 0.5, max_x + 0.5])
        # plt.ylim([min_z - 0.5, max_z + 0.5])
        plt.box(False)
        plt.tick_params(axis="both", which="both", length=0)
        plt.gca().set_facecolor("white")
        plt.tight_layout()
        return plt.gca()

    def __repr__(self) -> str:
        # self.visualize()
        return ""


def sample_and_add_floor_asset(
    room: ProceduralRoom,
    rectangle: Tuple[float, float, float, float],
    anchor_type: str,
    anchor_delta: int,
    allow_house_plant_group: bool,
    allow_tv_group: bool,
    spawnable_assets: pd.DataFrame,
    spawnable_asset_groups: pd.DataFrame,
    priority_asset_types: List[str],
    pt_db: ProcTHORDatabase,
) -> Optional[Union[ChosenAsset, ChosenAssetGroup]]:
    """Chooses an asset to place in the room.

    Args:
        rectangle: The size of the outer rectangle that the object must
            fit inside of.
        anchor_type: The anchor type, in :code:`{"inCorner", "inMiddle", "onEdge"}`.

    Returns:
        The chosen asset or asset group. The rotated flag is set to
        :code:`True` if the x and z sides should be flipped.

    """
    set_rotated = None

    # NOTE: Choose the valid rotations
    x0, z0, x1, z1 = rectangle
    rect_x_length = x1 - x0
    rect_z_length = z1 - z0

    # NOTE: add margin to each object.
    # NOTE: z is the forward direction on each object.
    # Therefore, we only add space in front of the object.
    if anchor_type == "onEdge":
        x_margin = 2 * MARGIN["edge"]["sides"]
        z_margin = (
            MARGIN["edge"]["front"] + MARGIN["edge"]["back"] + PADDING_AGAINST_WALL
        )
    elif anchor_type == "inCorner":
        x_margin = 2 * MARGIN["corner"]["sides"] + PADDING_AGAINST_WALL
        z_margin = (
            MARGIN["corner"]["front"] + MARGIN["corner"]["back"] + PADDING_AGAINST_WALL
        )
    elif anchor_type == "inMiddle":
        # NOTE: add space to both sides
        x_margin = 2 * MARGIN["middle"]
        z_margin = 2 * MARGIN["middle"]

    # NOTE: define the size filters
    if anchor_delta in {1, 7}:
        # NOTE: should not be rotated
        size_filter = lambda assets_df: (
            (assets_df["xSize"] + x_margin < rect_x_length)
            & (assets_df["zSize"] + z_margin < rect_z_length)
        )
        set_rotated = False
    elif anchor_delta in {3, 5}:
        # NOTE: must be rotated
        size_filter = lambda assets_df: (
            (assets_df["zSize"] + z_margin < rect_x_length)
            & (assets_df["xSize"] + x_margin < rect_z_length)
        )
        set_rotated = True
    else:
        # NOTE: either rotated or not rotated works
        size_filter = lambda assets_df: (
            (
                (assets_df["xSize"] + x_margin < rect_x_length)
                & (assets_df["zSize"] + z_margin < rect_z_length)
            )
            | (
                (assets_df["zSize"] + z_margin < rect_x_length)
                & (assets_df["xSize"] + x_margin < rect_z_length)
            )
        )

    # NOTE: make sure anchor types and sizes fit
    asset_group_candidates = spawnable_asset_groups[
        spawnable_asset_groups[anchor_type] & size_filter(spawnable_asset_groups)
    ]
    if not allow_house_plant_group:
        # NOTE: quick hack to avoid oversampling floor house plants.
        asset_group_candidates = asset_group_candidates[
            asset_group_candidates["assetGroupName"] != "floor-house-plant"
        ]
    if not allow_tv_group:
        # NOTE: quick hack to avoid oversampling tv.
        asset_group_candidates = asset_group_candidates[
            asset_group_candidates["assetGroupName"] != "television"
        ]

    asset_candidates = spawnable_assets[
        spawnable_assets[anchor_type] & size_filter(spawnable_assets)
    ]

    # NOTE: try using a priority asset type if one needs to be placed
    if priority_asset_types:
        for asset_type in priority_asset_types:
            # NOTE: see if there are any semantic asset groups with the asset
            asset_groups_with_type = asset_group_candidates[
                asset_group_candidates[f"has{asset_type}"]
            ]

            # NOTE: see if assets can spawn by themselves
            can_spawn_standalone = (
                pt_db.PLACEMENT_ANNOTATIONS[
                    pt_db.PLACEMENT_ANNOTATIONS.index == asset_type
                ][f"in{room.room_type}s"].iloc[0]
                > 0
            )
            assets_with_type = None
            if can_spawn_standalone:
                assets_with_type = asset_candidates[
                    asset_candidates["assetType"] == asset_type
                ]

            # NOTE: try using an asset group first
            if len(asset_groups_with_type) and (
                assets_with_type is None or random.random() <= P_CHOOSE_ASSET_GROUP
            ):
                # NOTE: Try using an asset group
                asset_group = asset_groups_with_type.sample()
                chosen_asset_group = room.place_asset_group(
                    asset_group=asset_group,
                    set_rotated=set_rotated,
                    rect_x_length=rect_x_length,
                    rect_z_length=rect_z_length,
                )
                if chosen_asset_group is not None:
                    return chosen_asset_group

            # NOTE: try using a standalone asset
            if assets_with_type is not None and len(assets_with_type):
                # NOTE: try spawning in standalone
                asset = assets_with_type.sample()
                return room.place_asset(
                    asset=asset,
                    set_rotated=set_rotated,
                    rect_x_length=rect_x_length,
                    rect_z_length=rect_z_length,
                )

    # NOTE: try using an asset group
    if len(asset_group_candidates) and random.random() <= P_CHOOSE_ASSET_GROUP:
        # NOTE: use an asset group if you can
        asset_group = asset_group_candidates.sample()
        chosen_asset_group = room.place_asset_group(
            asset_group=asset_group,
            set_rotated=set_rotated,
            rect_x_length=rect_x_length,
            rect_z_length=rect_z_length,
        )
        if chosen_asset_group is not None:
            return chosen_asset_group

    # NOTE: Skip weight 1 assets with a probability of P_W1_ASSET_SKIPPED
    if random.random() <= P_W1_ASSET_SKIPPED:
        asset_candidates = asset_candidates[
            asset_candidates[f"in{room.room_type}s"] != 1
        ]

    # NOTE: no assets fit the anchor_type and size criteria
    if not len(asset_candidates):
        return None

    # NOTE: this is a sampling by asset type
    asset_type = random.choice(asset_candidates["assetType"].unique())
    asset = asset_candidates[asset_candidates["assetType"] == asset_type].sample()
    return room.place_asset(
        asset=asset,
        set_rotated=set_rotated,
        rect_x_length=rect_x_length,
        rect_z_length=rect_z_length,
    )


def default_add_rooms(
    partial_house: "PartialHouse",
    controller: Controller,
    pt_db: ProcTHORDatabase,
    split: str,
    floor_polygons: Dict[str, Polygon],
    room_type_map: Dict[int, str],
    door_polygons: Dict[int, List[Polygon]],
) -> None:
    """Add rooms

    Args:
        floor_polygons: Maps each room's id to the shapely polygon of each room's
            floor.

    """
    assert partial_house.rooms is None or len(partial_house.rooms) == 0

    rooms = dict()
    for room_id, room_type in room_type_map.items():
        polygon = floor_polygons[f"room|{room_id}"]
        room = ProceduralRoom(
            polygon=polygon,
            room_type=room_type,
            room_id=room_id,
            split=split,
            door_polygons=door_polygons[room_id],
            pt_db=pt_db,
        )
        rooms[room_id] = room

    partial_house.rooms = rooms


def default_add_floor_objects(
    partial_house: "PartialHouse",
    controller: Controller,
    pt_db: ProcTHORDatabase,
    split: Split,
    max_floor_objects: int,
    p_allow_house_plant_group: float = P_ALLOW_HOUSE_PLANT_GROUP,
    p_allow_tv_group: float = P_ALLOW_TV_GROUP,
) -> None:
    """Add objects to each room.

    Args:
        floor_polygons: Maps each room's id to the shapely polygon of each room's
            floor.

    """
    assert partial_house.objects is None or len(partial_house.objects) == 0

    partial_house.objects = []
    for room in partial_house.rooms.values():
        allow_house_plant_group = random.random() < p_allow_house_plant_group
        allow_tv_group = random.random() < p_allow_tv_group

        floor_types, spawnable_assets = pt_db.FLOOR_ASSET_DICT[
            (room.room_type, room.split)
        ]
        priority_asset_types = copy.deepcopy(pt_db.PRIORITY_ASSET_TYPES[room.room_type])
        random.shuffle(priority_asset_types)

        spawnable_asset_group_info = get_spawnable_asset_group_info(
            split=room.split, controller=controller, pt_db=pt_db
        )
        spawnable_asset_groups = spawnable_asset_group_info[
            spawnable_asset_group_info[f"in{room.room_type}s"] > 0
        ]

        asset = None
        for i in range(max_floor_objects):
            cache_rectangles = i != 0 and asset is None
            if cache_rectangles:
                # NOTE: Don't resample failed rectangles
                room.last_rectangles.remove(rectangle)
                rectangle = room.sample_next_rectangle(cache_rectangles=True)
            else:
                rectangle = room.sample_next_rectangle()

            if rectangle is None:
                break

            x_info, z_info, anchor_delta, anchor_type = room.sample_anchor_location(
                rectangle
            )
            asset = sample_and_add_floor_asset(
                room=room,
                rectangle=rectangle,
                anchor_type=anchor_type,
                anchor_delta=anchor_delta,
                allow_house_plant_group=allow_house_plant_group,
                allow_tv_group=allow_tv_group,
                spawnable_assets=spawnable_assets,
                spawnable_asset_groups=spawnable_asset_groups,
                priority_asset_types=priority_asset_types,
                pt_db=pt_db,
            )
            # NOTE: no asset within the asset group could be placed inside of the
            # rectangle.
            if asset is None:
                continue

            room.sample_place_asset_in_rectangle(
                asset=asset,
                rectangle=rectangle,
                anchor_type=anchor_type,
                x_info=x_info,
                z_info=z_info,
                anchor_delta=anchor_delta,
            )
            added_asset_types = []
            if "assetType" in asset:
                added_asset_types.append(asset["assetType"])
            else:
                added_asset_types.extend([o["assetType"] for o in asset["objects"]])

                if not asset["allowDuplicates"]:
                    spawnable_asset_groups = spawnable_asset_groups.query(
                        f"assetGroupName!='{asset['assetGroupName']}'"
                    )

            for asset_type in added_asset_types:
                # Remove spawned object types from `priority_asset_types` when appropriate
                if asset_type in priority_asset_types:
                    priority_asset_types.remove(asset_type)

                allow_duplicates_of_asset_type = pt_db.PLACEMENT_ANNOTATIONS.loc[
                    asset_type
                ]["multiplePerRoom"]

                if not allow_duplicates_of_asset_type:
                    # NOTE: Remove all asset groups that have the type
                    spawnable_asset_groups = spawnable_asset_groups[
                        ~spawnable_asset_groups[f"has{asset_type}"]
                    ]

                    # NOTE: Remove all standalone assets that have the type
                    spawnable_assets = spawnable_assets[
                        spawnable_assets["assetType"] != asset_type
                    ]

        # NOTE: add the formatted assets
        for asset in room.assets:
            if isinstance(asset, AssetGroup):
                partial_house.objects.extend(asset.assets_dict)
            else:
                partial_house.objects.append(asset.asset_dict)
