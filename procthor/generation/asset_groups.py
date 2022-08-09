import copy
import random
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from ai2thor.controller import Controller
from attr import field
from attrs import define
from trimesh import Trimesh
from trimesh.collision import CollisionManager

from procthor.databases import ProcTHORDatabase
from procthor.utils.types import Split, Vector3


@define
class AssetGroupGenerator:
    name: str
    """The name of the asset group."""

    split: Split
    """The split for the asset group."""

    data: Dict[str, Any]
    """The parsed json data of the asset group."""

    controller: Controller
    """Procedural AI2-THOR Controller."""

    cache: Dict[str, Any] = field(init=False)

    pt_db: ProcTHORDatabase

    def __attrs_post_init__(self):
        """Preprocesses the asset group data to be in a better format.

        - Tansforms asset_metadata[assetIds] from
          [asset_type]: [...assetIds] to [...(asset_type, asset_id)].
        - Transforms the asset database from [asset_type]: [...asset]
          to [asset_type]: [...[asset_id]: [asset]].
        """
        self.data = copy.deepcopy(self.data)
        self.cache = {}

        def flatten_asset_ids() -> None:
            """Tansforms asset_metadata[assetIds] from [asset_type]: [...assetIds]
            to [...(asset_type, asset_id)] and filter assets by split.
            """
            for asset_metadata in self.data["assetMetadata"].values():
                out = []
                for asset_type, asset_ids in asset_metadata["assetIds"].items():
                    for asset_id in asset_ids:
                        if self.pt_db.ASSET_ID_DATABASE[asset_id]["split"] in {
                            None,
                            self.split,
                        }:
                            out.append((asset_type, asset_id))
                if not out:
                    raise Exception(
                        f"No valid asset groups for {self.name} with {self.split} split!"
                    )
                asset_metadata["assetIds"] = out

        def flip_x_axis() -> None:
            """
            .. note::
                AI2-THOR requires flipping the x-axis for the group editor to
                appear correct.
            """
            for instance_id in self.data["assetMetadata"].keys():
                self.data["assetMetadata"][instance_id]["position"]["x"] = -self.data[
                    "assetMetadata"
                ][instance_id]["position"]["x"]

        flatten_asset_ids()
        flip_x_axis()

    @property
    def dimensions(self) -> Vector3:
        """Get the dimensions of the asset group.

        The dimensions are set to the maximum possible extent of the asset
        group, independently in each direction.

        TODO: Consider accounting for randomness in the dtheta dimensions.
        """
        if "dimensions" not in self.cache:
            self._set_dimensions()
        return self.cache["dimensions"]

    def _set_dimensions(self) -> None:
        asset_group_assets = {
            asset["name"]: set([asset_id for asset_type, asset_id in asset["assetIds"]])
            for asset in self.data["assetMetadata"].values()
        }

        assets = {
            asset_name: pd.DataFrame(
                [
                    {
                        "assetId": asset_id,
                        "assetType": self.pt_db.ASSET_ID_DATABASE[asset_id][
                            "objectType"
                        ],
                        "split": self.pt_db.ASSET_ID_DATABASE[asset_id]["split"],
                        "xSize": self.pt_db.ASSET_ID_DATABASE[asset_id]["boundingBox"][
                            "x"
                        ],
                        "ySize": self.pt_db.ASSET_ID_DATABASE[asset_id]["boundingBox"][
                            "y"
                        ],
                        "zSize": self.pt_db.ASSET_ID_DATABASE[asset_id]["boundingBox"][
                            "z"
                        ],
                    }
                    for asset_id in asset_ids
                ]
            )
            for asset_name, asset_ids in asset_group_assets.items()
        }

        max_y = -np.inf
        chosen_asset_ids = {"largestXAssets": dict(), "largestZAssets": dict()}
        for asset_name, asset_df in assets.items():
            x_max_asset_id = asset_df.iloc[asset_df["xSize"].idxmax()]["assetId"]
            z_max_asset_id = asset_df.iloc[asset_df["zSize"].idxmax()]["assetId"]

            chosen_asset_ids["largestXAssets"][asset_name] = (
                self.pt_db.ASSET_ID_DATABASE[x_max_asset_id]["objectType"],
                x_max_asset_id,
            )
            chosen_asset_ids["largestZAssets"][asset_name] = (
                self.pt_db.ASSET_ID_DATABASE[z_max_asset_id]["objectType"],
                z_max_asset_id,
            )

            if asset_df["ySize"].max() > max_y:
                max_y = asset_df["ySize"].max()

        # TODO: eventually turn off randomness.
        x_dim_assets = self.sample_object_placement(
            chosen_asset_ids=chosen_asset_ids["largestXAssets"]
        )
        z_dim_assets = self.sample_object_placement(
            chosen_asset_ids=chosen_asset_ids["largestZAssets"]
        )

        self.cache["dimensions"] = Vector3(
            x=x_dim_assets["bounds"]["x"]["length"],
            y=max_y,
            z=z_dim_assets["bounds"]["z"]["length"],
        )

    @staticmethod
    def bounding_boxes_intersect(
        bbox1: Tuple[Vector3, Vector3],
        bbox2: Tuple[Vector3, Vector3],
        epsilon: float = 1e-3,
    ) -> bool:
        """
        Bounding boxes should be in the form of (min xyz points, max xyz points).
        """
        return all(
            (
                (
                    bbox1[k]["min"] - epsilon
                    <= bbox2[k]["min"]
                    <= bbox1[k]["max"] + epsilon
                )
                or (
                    bbox1[k]["min"] - epsilon
                    <= bbox2[k]["max"]
                    <= bbox1[k]["max"] + epsilon
                )
                or (
                    bbox2[k]["min"] - epsilon
                    <= bbox1[k]["min"]
                    <= bbox2[k]["max"] + epsilon
                )
                or (
                    bbox2[k]["min"] - epsilon
                    <= bbox1[k]["max"]
                    <= bbox2[k]["max"] + epsilon
                )
            )
            for k in ["x", "y", "z"]
        )

    def on_top_of_parent(self, instance_id_1: str, instance_id_2: str) -> bool:
        """Check if either instance id is on top of the other."""
        return (
            self.data["assetMetadata"][instance_id_1].get("parentInstanceId", None)
            == int(instance_id_2)
            and self.data["assetMetadata"][instance_id_1]["position"][
                "verticalAlignment"
            ]
            == "above"
        ) or (
            int(instance_id_1)
            == self.data["assetMetadata"][instance_id_2].get("parentInstanceId", None)
            and self.data["assetMetadata"][instance_id_2]["position"][
                "verticalAlignment"
            ]
            == "above"
        )

    def get_intersecting_objects(self, object_placement) -> bool:
        """Return True if any of the objects in object_placement are intersecting.

        Args:
            object_placement: must be in the form of the returned output from
                sample_object_placement().

        Remove clipping from the asset group.

        Takes longer but ends up making the group look more realistic.
        """
        # NOTE: run quick checks first to see if any bounding boxes are even
        # colliding. Only continue to expensive checks if there are collisions.
        for obj1, obj2 in combinations(object_placement, 2):
            if not self.on_top_of_parent(
                obj1["instanceId"], obj2["instanceId"]
            ) and AssetGroupGenerator.bounding_boxes_intersect(
                obj1["bbox"], obj2["bbox"]
            ):
                break
        else:
            return False, set()

        self.controller.reset()
        collision_manager = CollisionManager()
        for i, obj in enumerate(object_placement):
            instance_id = f"{obj['instanceId']}-{i}"
            self.controller.step(
                action="SpawnAsset",
                assetId=obj["assetId"],
                generatedId=instance_id,
                position=obj["position"],
                rotation=dict(x=0, y=obj["rotation"], z=0),
                renderImage=False,
            )

            # NOTE: extract the meshes
            # NOTE: some assets have multiple mesh components that make up
            # a single mesh (e.g., Fridges).
            asset_geometry = self.controller.step(
                action="GetInSceneAssetGeometry",
                objectId=instance_id,
                triangles=True,
                renderImage=False,
            ).metadata["actionReturn"]
            for j, mesh_info in enumerate(asset_geometry):
                # NOTE: Swaps y and z dimensions
                vertices = np.array(
                    [[p["x"], p["z"], p["y"]] for p in mesh_info["vertices"]]
                )
                triangles = np.array(mesh_info["triangles"]).reshape(-1, 3)[
                    :, [0, 2, 1]
                ]
                collision_manager.add_object(
                    name=f"{instance_id}-{j}",
                    mesh=Trimesh(vertices=vertices, faces=triangles),
                )
        is_colliding, colliding_meshes = collision_manager.in_collision_internal(
            return_names=True
        )

        # NOTE: Ignores self collisions and ignores collisions from objects
        # where the parent is
        if is_colliding:
            true_colliding_meshes = set()
            for mesh1, mesh2 in colliding_meshes:
                mesh1_id = mesh1[: mesh1.find("-")]
                mesh2_id = mesh2[: mesh2.find("-")]
                if (mesh1_id != mesh2_id) and not self.on_top_of_parent(
                    mesh1_id, mesh2_id
                ):
                    true_colliding_meshes.add((mesh1, mesh2))
            return bool(true_colliding_meshes), true_colliding_meshes

        return is_colliding, colliding_meshes

    @staticmethod
    def rotate_bounding_box(
        theta: float,
        bbox_size: Dict[str, float],
        x_center: float = 0,
        z_center: float = 0,
    ) -> Dict[str, Dict[str, float]]:
        """Rotate a top-down 2D bounding box.

        Args:
            theta: The rotation of the bounding box in degrees.
            bbox_size: The size of the bounding box. Must have keys for {"x", "z"}.
            x_center: The center x position of the bounding box.
            z_center: The center z position of the bounding box.
        """
        bb_corners = [
            (x_center + bbox_size["x"] / 2, z_center + bbox_size["z"] / 2),
            (x_center - bbox_size["x"] / 2, z_center + bbox_size["z"] / 2),
            (x_center - bbox_size["x"] / 2, z_center - bbox_size["z"] / 2),
            (x_center + bbox_size["x"] / 2, z_center - bbox_size["z"] / 2),
        ]
        theta_rad = theta * np.pi / 180.0
        for i, (x, z) in enumerate(bb_corners):
            x_ = (
                x_center
                + (x - x_center) * np.cos(theta_rad)
                + (z - z_center) * np.sin(theta_rad)
            )
            z_ = (
                z_center
                - (x - x_center) * np.sin(theta_rad)
                + (z - z_center) * np.cos(theta_rad)
            )
            bb_corners[i] = (x_, z_)
        return {
            "x": {
                "min": min(bb_corners, key=lambda bb_corner: bb_corner[0])[0],
                "max": max(bb_corners, key=lambda bb_corner: bb_corner[0])[0],
            },
            "z": {
                "min": min(bb_corners, key=lambda bb_corner: bb_corner[1])[1],
                "max": max(bb_corners, key=lambda bb_corner: bb_corner[1])[1],
            },
        }

    def sample_object_placement(
        self,
        allow_clipping: bool = True,
        floor_position: float = 0,
        use_thumbnail_assets: bool = False,
        chosen_asset_ids: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Sample object placement.

        Args:
            chosen_asset_ids: Maps from the "name" in assetMetadata to the chosen
                (assetType, assetId) of that asset. Note that assets with the same
                name have the same chosen assetId.
            floor_position: The position of the floor.
            use_thumbnail_assets If the randomly chosen asset should be the one
                shown in the thumbnail specified in the JSON.

        Returns:
            A dict mapping each assetId to an (x, y, z) position.
        """
        if not allow_clipping:
            raise NotImplementedError(
                "Currently, only allow_clipping == True is supported."
            )

        out = {
            "objects": [],
            "bounds": {
                "x": {
                    "min": float("inf"),
                    "max": float("-inf"),
                },
                "z": {
                    "min": float("inf"),
                    "max": float("-inf"),
                },
            },
        }

        asset_stack = [
            {"parentId": None, "tree": parent_asset}
            for parent_asset in reversed(self.data["treeData"])
        ]

        # assets with the same name have the same assetIds chosen
        if chosen_asset_ids is None:
            chosen_asset_ids = dict()

        # quick lookup for the placement of parent positions
        parent_asset_lookup = dict()

        while asset_stack:
            asset = asset_stack.pop()
            instance_id = str(asset["tree"]["instanceId"])

            if "children" in asset["tree"]:
                for child_asset in asset["tree"]["children"]:
                    asset_stack.append({"parentId": instance_id, "tree": child_asset})

            asset_metadata = self.data["assetMetadata"][instance_id]

            # NOTE: choose the asset id
            name = asset_metadata["name"]
            if name in chosen_asset_ids:
                asset_type, asset_id = chosen_asset_ids[name]
            elif use_thumbnail_assets:
                asset_id = asset_metadata["shownAssetId"]
                asset_type = self.pt_db.ASSET_ID_DATABASE[asset_id]["objectType"]
            else:
                asset_type, asset_id = random.choice(asset_metadata["assetIds"])
            chosen_asset_ids[name] = (asset_type, asset_id)

            # set the y position of the asset
            bbox_size = self.pt_db.ASSET_ID_DATABASE[asset_id]["boundingBox"]

            # NOTE: add in randomness
            dtheta = asset_metadata["randomness"]["dtheta"]
            theta_offset = random.random() * dtheta * 2 - dtheta
            theta = asset_metadata["rotation"] + theta_offset

            # calculate the bounding box after rotating the object.
            bbox_bounds = AssetGroupGenerator.rotate_bounding_box(
                theta=theta, bbox_size=bbox_size
            )

            # NOTE: determine where to place the asset
            if asset["parentId"] is None:
                # NOTE: position represents an absolute position
                center_position = asset_metadata["position"]
                x_center, z_center = center_position["x"], center_position["z"]
                y_center = floor_position + bbox_size["y"] / 2
                bbox_bounds["y"] = {
                    "min": floor_position,
                    "max": floor_position + bbox_size["y"],
                }
            else:
                # NOTE: position is relative to the parent
                parent = parent_asset_lookup[asset["parentId"]]

                x_center = parent["position"]["x"] + asset_metadata["position"]["x"]
                z_center = parent["position"]["z"] + asset_metadata["position"]["z"]

                parent_x_length = -(
                    parent["bbox"]["x"]["max"] - parent["bbox"]["x"]["min"]
                )
                parent_z_length = (
                    parent["bbox"]["z"]["max"] - parent["bbox"]["z"]["min"]
                )

                anchor = asset_metadata["position"]["relativeAnchorToParent"]
                if anchor in {0, 1, 2}:
                    z_center -= parent_z_length / 2
                elif anchor in {6, 7, 8}:
                    z_center += parent_z_length / 2

                if anchor in {0, 3, 6}:
                    x_center -= parent_x_length / 2
                elif anchor in {2, 5, 8}:
                    x_center += parent_x_length / 2

                x_alignment = asset_metadata["position"]["xAlignment"]
                z_alignment = asset_metadata["position"]["zAlignment"]

                bbox_x_length = bbox_bounds["x"]["max"] - bbox_bounds["x"]["min"]
                bbox_z_length = bbox_bounds["z"]["max"] - bbox_bounds["z"]["min"]

                if x_alignment == 0:
                    x_center -= bbox_x_length / 2
                elif x_alignment == 2:
                    x_center += bbox_x_length / 2

                if z_alignment == 0:
                    z_center -= bbox_z_length / 2
                elif z_alignment == 2:
                    z_center += bbox_z_length / 2

                if asset_metadata["position"]["verticalAlignment"] == "nextTo":
                    y_center = parent["floorPosition"] + bbox_size["y"] / 2
                    bbox_bounds["y"] = {
                        "min": parent["floorPosition"],
                        "max": parent["floorPosition"] + bbox_size["y"],
                    }
                elif asset_metadata["position"]["verticalAlignment"] == "above":
                    # NOTE: This is naive. It places objects at of the parent
                    # object's bounding box height. Consider a more advanced height
                    # calculation that looks at the contours of an object instead
                    # of just using the bounding box.
                    y_center = (
                        parent["floorPosition"] + parent["height"] + bbox_size["y"] / 2
                    )
                    bbox_bounds["y"] = {
                        "min": parent["floorPosition"] + parent["height"],
                        "max": parent["floorPosition"]
                        + parent["height"]
                        + bbox_size["y"],
                    }

            bbox_bounds["x"]["min"] += x_center
            bbox_bounds["x"]["max"] += x_center
            bbox_bounds["z"]["min"] += z_center
            bbox_bounds["z"]["max"] += z_center

            for k in ["x", "z"]:
                if bbox_bounds[k]["min"] < out["bounds"][k]["min"]:
                    out["bounds"][k]["min"] = bbox_bounds[k]["min"]
            for k in ["x", "z"]:
                if bbox_bounds[k]["max"] > out["bounds"][k]["max"]:
                    out["bounds"][k]["max"] = bbox_bounds[k]["max"]
            out["bounds"]["y"] = bbox_bounds["y"]

            parent_asset_lookup[instance_id] = {
                "position": {"x": x_center, "y": y_center, "z": z_center},
                "floorPosition": y_center - bbox_size["y"] / 2,
                "height": bbox_size["y"],
                "bbox": bbox_bounds,
                "assetId": asset_id,
            }

            out["objects"].append(
                {
                    "instanceId": instance_id,
                    "assetId": asset_id,
                    "assetType": asset_type,
                    "position": {"x": x_center, "y": y_center, "z": z_center},
                    "rotation": theta,
                    "bbox": bbox_bounds,
                }
            )

        # NOTE: cache the center and length of the group
        for k in ["x", "y", "z"]:
            out["bounds"][k]["length"] = (
                out["bounds"][k]["max"] - out["bounds"][k]["min"]
            )
            out["bounds"][k]["center"] = (
                out["bounds"][k]["min"] + out["bounds"][k]["max"]
            ) / 2

        return out

    @staticmethod
    def generate_thumbnail(
        controller: Controller,
        placement: Any,
        save_as: Optional[str] = None,
    ) -> Image:
        controller.reset(scene="Procedural")
        for i, obj in enumerate(placement["objects"]):
            controller.step(
                action="SpawnAsset",
                assetId=obj["assetId"],
                generatedId=f"asset_{i}",
                position=dict(
                    x=obj["position"]["x"] - placement["bounds"]["x"]["center"],
                    y=obj["position"]["y"],
                    z=obj["position"]["z"] - placement["bounds"]["z"]["center"],
                ),
                rotation=dict(x=0, y=obj["rotation"], z=0),
            )

        controller.step(
            action="AddThirdPartyCamera",
            rotation=dict(x=90, y=180, z=0),
            position=dict(x=0, y=20, z=0),
            orthographic=True,
            skyboxColor="white",
            orthographicSize=max([placement["bounds"][k]["length"] for k in ["x", "z"]])
            * 0.75,
        )

        im = Image.fromarray(controller.last_event.third_party_camera_frames[0])
        if save_as:
            im.save(save_as)

        return im
