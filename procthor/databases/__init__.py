import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from ai2thor.controller import Controller
from procthor.constants import USE_ITHOR_SPLITS
from procthor.utils.types import Split


def _load_json_from_database(json_file: str) -> Union[list, dict]:
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, json_file)
    with open(filepath, "r") as f:
        return json.load(f)


def _get_solid_wall_colors() -> dict:
    return _load_json_from_database("solid-wall-colors.json")


def _get_material_database() -> dict:
    return _load_json_from_database("material-database.json")


def _get_asset_database() -> dict:
    return (
        _load_json_from_database("procthor-ithor-split-asset-database.json")
        if USE_ITHOR_SPLITS
        else _load_json_from_database("asset-database.json")
    )


def _get_skyboxes() -> list:
    return _load_json_from_database("skyboxes.json")


def _get_asset_id_database() -> dict:
    asset_type_database = _get_asset_database()
    asset_id_database = dict()
    for assets in asset_type_database.values():
        for asset in assets:
            asset_id_database[asset["assetId"]] = asset
    return asset_id_database


def _get_ai2thor_object_metadata() -> dict:
    return _load_json_from_database("ai2thor-object-metadata.json")


def _get_placement_annotations(fillna: bool = True) -> pd.DataFrame:
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, "placement-annotations.json")

    df = pd.read_json(filepath)
    if fillna:
        for key in ["inKitchens", "inLivingRooms", "inBedrooms", "inBathrooms"]:
            df[key] = df[key].fillna(0).astype(np.uint8)
        for key in [
            "inCorner",
            "inMiddle",
            "onEdge",
            "onFloor",
            "onWall",
            "isPickupable",
            "isKinematic",
            "isStructure",
            "multiplePerRoom",
        ]:
            df[key] = df[key].fillna(0).astype(bool)
    df.index.names = ["assetType"]
    return df


def _get_asset_groups() -> Dict[str, Any]:
    """Maps each asset group to the web metadata of the asset group."""
    out = dict()
    dirname = os.path.join(os.path.dirname(__file__), "asset_groups")
    for fname in os.listdir(dirname):
        if not fname.endswith(".json"):
            continue
        with open(f"{dirname}/{fname}", "r") as f:
            out[fname[: -len(".json")]] = json.load(f)
    return out


def _get_object_in_receptacles() -> Dict[str, List[str]]:
    return _load_json_from_database("receptacles.json")


def _get_assets_df() -> pd.DataFrame:
    asset_id_db = _get_asset_id_database()
    return pd.DataFrame(
        [
            {
                "split": asset["split"],
                "assetId": asset["assetId"],
                "objectType": asset["objectType"],
                "xSize": asset["boundingBox"]["x"],
                "ySize": asset["boundingBox"]["y"],
                "zSize": asset["boundingBox"]["z"],
            }
            for asset in asset_id_db.values()
        ]
    )


def _get_wall_holes() -> Dict[str, Any]:
    return _load_json_from_database("wall-holes.json")


@lru_cache(maxsize=None)
def get_spawnable_asset_group_info(
    split: Split, controller: Controller
) -> pd.DataFrame:
    from procthor.generation.asset_groups import AssetGroupGenerator

    asset_groups = _get_asset_groups()
    asset_database = _get_asset_database()

    data = []
    for asset_group_name, asset_group_data in asset_groups.items():
        asset_group_generator = AssetGroupGenerator(
            name=asset_group_name,
            split=split,
            data=asset_group_data,
            controller=controller,
        )

        dims = asset_group_generator.dimensions
        group_properties = asset_group_data["groupProperties"]

        # NOTE: This is kinda naive, since a single asset in the asset group
        # could map to multiple different types of asset types (e.g., Both Chair
        # and ArmChair could be in the same asset).
        # NOTE: use the asset_group_generator.data instead of asset_group_data
        # since it only includes assets from a given split.
        asset_types_in_group = set(
            asset_type
            for asset in asset_group_generator.data["assetMetadata"].values()
            for asset_type, asset_id in asset["assetIds"]
        )
        group_data = {
            "assetGroupName": asset_group_name,
            "assetGroupGenerator": asset_group_generator,
            "xSize": dims["x"],
            "ySize": dims["y"],
            "zSize": dims["z"],
            "inBathrooms": group_properties["roomWeights"]["bathrooms"],
            "inBedrooms": group_properties["roomWeights"]["bedrooms"],
            "inKitchens": group_properties["roomWeights"]["kitchens"],
            "inLivingRooms": group_properties["roomWeights"]["livingRooms"],
            "allowDuplicates": group_properties["properties"]["allowDuplicates"],
            "inCorner": group_properties["location"]["corner"],
            "onEdge": group_properties["location"]["edge"],
            "inMiddle": group_properties["location"]["middle"],
        }

        # NOTE: Add which types are in this asset group
        for asset_type in asset_database.keys():
            group_data[f"has{asset_type}"] = asset_type in asset_types_in_group

        data.append(group_data)

    return pd.DataFrame(data)


solid_wall_colors = _get_solid_wall_colors()
material_database = _get_material_database()
skyboxes = _get_skyboxes()
objects_in_receptacles = _get_object_in_receptacles()
asset_database = _get_asset_database()
asset_id_database = _get_asset_id_database()
placement_annotations = _get_placement_annotations()
ai2thor_object_metadata = _get_ai2thor_object_metadata()
asset_groups = _get_asset_groups()
assets_df = _get_assets_df()
wall_holes = _get_wall_holes()
