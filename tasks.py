from invoke import task

ASSET_DATABASE_PATH = "procthor/databases/asset-database.json"
MATERIAL_DATABASE_PATH = "procthor/databases/material-database.json"
ASSET_IMAGE_SIZE = 450

#%% Utils
@task
def set_version(c, version):
    """Writes the version upon a release."""
    for filename in ["setup.py", "procthor/__init__.py"]:
        with open(filename, "r") as f:
            file = f.read()
        file = file.replace("<REPLACE_WITH_VERSION>", version)
        with open(filename, "w") as f:
            f.write(file)


'''
from ai2thor.controller import Controller
from PIL import Image
import numpy as np

from houses.constants import PROCTHOR_INITIALIZATION
from houses.databases import ASSET_DATABASE, MATERIAL_DATABASE, PLACEMENT_ANNOTATIONS


#%% Material Database
def reset_material_database():
    controller = Controller(**PROCTHOR_INITIALIZATION)
    update_material_database(controller)
    update_material_visualizations()


def save_material_database(MATERIAL_DATABASE):
    for materials in MATERIAL_DATABASE.values():
        materials.sort()

    with open(MATERIAL_DATABASE_PATH, "w") as f:
        f.write(json.dumps(MATERIAL_DATABASE, indent=4, sort_keys=True))


def update_material_database(controller: Controller) -> None:
    controller.reset(scene="Procedural")
    event = controller.step(action="GetMaterials")
    save_material_database(event.metadata["actionReturn"])


def update_material_visualizations():
    controller = Controller(branch="nanna-cmb", scene="Procedural")
    for material_group, materials in MATERIAL_DATABASE.items():
        print(material_group)
        for material in materials:
            print("|", end="")
            controller.reset()
            controller.step(
                action="CreateMaterialBall",
                materialGroup=material_group,
                material=material,
            )
            controller.step(
                action="AddThirdPartyCamera",
                position=dict(x=0, y=0, z=-0.85),
                rotation=dict(x=0, y=0, z=0),
                skyboxColor="white",
            )
            os.makedirs(f"materials/{material_group}", exist_ok=True)
            image = Image.fromarray(controller.last_event.third_party_camera_frames[0])
            image.save(f"materials/{material_group}/{material}.png")
    shutil.make_archive("materials", "zip", "materials")


#%% Asset Database
def reset_asset_database():
    controller = Controller(**PROCTHOR_INITIALIZATION)

    update_asset_database_by_type(controller)
    set_scenes_per_asset(controller)
    set_object_states(controller)
    set_materials_per_asset_id()
    create_splits()
    update_asset_images()


def save_asset_database(asset_database_by_type: dict):
    for assets in asset_database_by_type.values():
        assets.sort(key=lambda asset: asset["assetId"])

    with open(ASSET_DATABASE_PATH, "w") as f:
        f.write(json.dumps(asset_database_by_type, indent=4, sort_keys=True))


def create_splits() -> None:
    overrides = 0
    for asset_type, assets in ASSET_DATABASE.items():
        assets.sort(key=lambda asset: asset["assetId"])
        random.seed(42)
        if len(assets) < 6:
            for asset in assets:
                asset["split"] = None
        else:
            random.shuffle(assets)
            train, val, test = np.split(
                assets, [int(len(assets) * 2 / 3), int(len(assets) * 5 / 6)]
            )
            for asset in train:
                asset["split"] = "train"
            for asset in val:
                asset["split"] = "val"
            for asset in test:
                asset["split"] = "test"

        # manual overrides
        for asset in assets:
            if asset_type == "Fridge":
                if asset["assetId"] == "Fridge_11":
                    assert asset["split"] == "train"
                    asset["split"] = "test"
                    overrides += 1
                elif asset["assetId"] == "Fridge_1":
                    assert asset["split"] == "test"
                    asset["split"] = "train"
                    overrides += 1
            elif asset_type == "Doorway":
                if asset["assetId"] == "Doorway_8":
                    assert asset["split"] == "test"
                    asset["split"] = "train"
                    overrides += 1
                elif asset["assetId"] == "Doorway_Double_3":
                    assert asset["split"] == "train"
                    asset["split"] = "test"
                    overrides += 1

    assert overrides == 4
    save_asset_database(ASSET_DATABASE)


def update_asset_database_by_type(controller: Controller) -> None:
    """Set the initial metadata in the asset database and fix each asset's bounding boxes."""
    controller.reset(scene="Procedural")

    # _asset_database contains un-filtered objects
    _asset_database = controller.step(action="GetAssetDatabase").metadata[
        "actionReturn"
    ]

    asset_database_by_type = defaultdict(list)
    for asset_id, asset_meta in _asset_database.items():
        if asset_id.endswith("_Master"):
            # these are duplicate prefabs
            continue
        asset_meta["assetId"] = asset_id
        asset_database_by_type[asset_meta["objectType"]].append(asset_meta)

    # remove unnecessary objects
    del asset_database_by_type["EggCracked"]
    del asset_database_by_type["StoveKnob"]

    # fix bounding boxes
    for asset_type in sorted(asset_database_by_type.keys()):
        print(asset_type, end=": ")

        for i, asset in enumerate(asset_database_by_type[asset_type]):
            print("|", end="")
            asset_id = asset["assetId"]

            controller.reset()
            controller.step(
                action="SpawnAsset", assetId=asset_id, generatedId="asset_0"
            )
            asset = next(
                obj
                for obj in controller.last_event.metadata["objects"]
                if obj["objectId"] == "asset_0"
            )
            bb = asset["axisAlignedBoundingBox"]["size"]
            asset_database_by_type[asset_type][i]["boundingBox"] = bb
        print()

    save_asset_database(asset_database_by_type)


def update_bounding_box():
    controller = Controller(**PROCTHOR_INITIALIZATION)
    for obj_type in ["Bed", "TennisRacket"]:
        for i in range(len(ASSET_DATABASE[obj_type])):
            controller.step(action="DestroyHouse")
            asset = ASSET_DATABASE[obj_type][i]
            controller.step(
                action="SpawnAsset",
                assetId=asset["assetId"],
                generatedId=asset["assetId"],
            )
            asset = next(
                obj
                for obj in controller.last_event.metadata["objects"]
                if obj["objectId"] == asset["assetId"]
            )
            bb = asset["axisAlignedBoundingBox"]["size"]
            ASSET_DATABASE[obj_type][i]["boundingBox"] = bb
    save_asset_database(ASSET_DATABASE)


def set_materials_per_asset_id():
    """Update each asset with the set of materials it has from the material database."""
    controller = Controller(branch="nanna-cmb", scene="Procedural")

    material_id_to_group = {
        material: material_group
        for material_group, materials in MATERIAL_DATABASE.items()
        for material in materials
    }

    for asset_group, assets in ASSET_DATABASE.items():
        print(asset_group, end=": ")
        for asset in assets:
            print("|", end="")
            controller.step(
                action="SpawnAsset", assetId=asset["assetId"], generatedId="asset_0"
            )
            event = controller.step(action="GetMaterialsOnObject", objectId="asset_0")

            materials = [
                [material_id_to_group[material], material]
                for material in event.metadata["actionReturn"]
                if material in material_id_to_group
            ]
            asset["materials"] = materials

            controller.step(action="DisableObject", objectId="asset_0")
        print()

    save_asset_database(ASSET_DATABASE)


def set_scenes_per_asset(controller: Controller):
    """Update the asset database with each scene that each assetId appears in."""
    asset_id_to_scenes = defaultdict(list)
    scene_names = [
        scene[: -len("_physics")] if scene.endswith("_physics") else scene
        for scene in controller.scene_names()
    ]

    for i, scene in enumerate(scene_names):
        print(f"{i}/{len(scene_names)}")
        event = controller.reset(scene, procedural=False)
        asset_ids = set(
            [obj["assetId"] for obj in event.metadata["objects"] if obj["assetId"]]
        )
        for asset_id in asset_ids:
            asset_id_to_scenes[asset_id].append(scene)

    for assets in ASSET_DATABASE.values():
        for asset in assets:
            asset["scenes"] = asset_id_to_scenes[asset["assetId"]]

    save_asset_database(ASSET_DATABASE)


def update_asset_images(
    asset_types: Optional[Set[str]] = None,
    asset_ids: Optional[Set[str]] = None,
    sides: Optional[Set[str]] = None,
) -> None:
    controller = Controller(
        cameraNearPlane=1e-10,
        cameraFarPlane=1e5,
        makeAgentsVisible=False,
        renderInstanceSegmentation=True,
        width=ASSET_IMAGE_SIZE,
        height=ASSET_IMAGE_SIZE,
        renderDepthImage=True,
        **PROCTHOR_INITIALIZATION,
    )
    i_obj = 0
    total_objects = sum([len(objs) for objs in ASSET_DATABASE.values()])

    LARGE_DISTANCE = 20
    base_dir = "images/orthographic"
    for asset_type in sorted(ASSET_DATABASE.keys()):
        if asset_types is not None and asset_type not in asset_types:
            continue
        print(asset_type, end=": ")
        folder = f"{base_dir}/{asset_type}"
        os.makedirs(folder, exist_ok=True)

        for asset in ASSET_DATABASE[asset_type]:
            asset_id = asset["assetId"]
            if asset_ids is not None and asset_id not in asset_ids:
                continue

            i_obj += 1
            print("|", end="")

            controller.reset()
            controller.step(
                action="RandomizeLighting",
                randomizeColor=False,
                brightness=(1.25, 1.25),
            )
            controller.step(
                action="SpawnAsset", assetId=asset_id, generatedId="asset_0"
            )
            asset = next(
                obj
                for obj in controller.last_event.metadata["objects"]
                if obj["objectId"] == "asset_0"
            )
            bb = asset["axisAlignedBoundingBox"]["size"]

            controller.step(
                action="AddThirdPartyCamera",
                position=dict(x=0, y=0, z=0),
                rotation=dict(x=0, y=0, z=0),
                orthographic=True,
                skyboxColor="white",
                orthographicSize=max(
                    bb["x"],
                    bb["z"],
                    bb["y"],
                )
                * 0.75,
            )

            camera_sides = {
                "bottom": dict(x=270, y=180, z=0),
                "top": dict(x=90, y=180, z=0),
                "front": dict(x=0, y=180, z=0),
                "back": dict(x=0, y=0, z=0),
                "right": dict(x=0, y=270, z=0),
                "left": dict(x=0, y=90, z=0),
            }
            for side, rotation in camera_sides.items():
                if sides is not None and side not in sides:
                    continue
                controller.step(
                    action="TeleportObject",
                    objectId="asset_0",
                    position=asset["position"],
                    rotation=rotation,
                    forceAction=True,
                )
                asset = next(
                    obj
                    for obj in controller.last_event.metadata["objects"]
                    if obj["objectId"] == "asset_0"
                )
                bb = asset["axisAlignedBoundingBox"]
                controller.step(
                    action="UpdateThirdPartyCamera",
                    position=dict(
                        x=bb["center"]["x"], y=bb["center"]["y"], z=-LARGE_DISTANCE
                    ),
                )

                # save the transparent frame
                event = controller.last_event
                rgb_frame = event.third_party_camera_frames[0]
                r, g, b = np.rollaxis(rgb_frame, axis=-1)
                seg_frame = event.third_party_instance_segmentation_frames[0]

                a = np.zeros_like(r)
                for key in event.object_id_to_color:
                    if key.startswith("asset_0"):
                        a += (
                            np.all(
                                seg_frame == event.object_id_to_color[key], axis=-1
                            ).astype(np.uint8)
                            * 255
                        )
                depth_frame = event.third_party_depth_frames[0]
                a += (depth_frame == 0).astype(np.uint8) * 255
                Image.fromarray(np.stack([r, g, b, a], axis=2), "RGBA").save(
                    f"{folder}/{asset_id}-{side}.png"
                )
        print(f" {i_obj}/{total_objects}")

    shutil.make_archive(base_dir, "zip", base_dir)


def add_max_image_pixel_length():
    for asset_type, assets in ASSET_DATABASE.items():
        print(asset_type)
        sides = ["back", "bottom", "front", "left", "right", "top"]
        max_asset_type_lengths = dict()
        for asset in assets:
            asset_id = asset["assetId"]
            max_length = 0
            for side in sides:
                im_path = f"orthographic/{asset_type}/{asset_id}-{side}.png"
                im = Image.open(im_path)

                bbox = im.getbbox()
                if bbox is None:
                    # ignores fully transparent images
                    continue

                left, upper, right, lower = bbox
                width = right - left
                height = lower - upper
                if width > max_length:
                    max_length = width
                if height > max_length:
                    max_length = height
            if max_length == 0:
                print(f"Zero length with assetId {asset_id}!")
            asset["maxImagePixelLength"] = max_length / ASSET_IMAGE_SIZE
            max_asset_type_lengths[asset_id] = max_length / ASSET_IMAGE_SIZE
    save_asset_database(ASSET_DATABASE)


def set_object_states(controller):
    """Set properties in the metadata for different object states."""

    def encompass_points(bbox, points) -> None:
        for point in points:
            px, py, pz = point
            for k, p in zip(["x", "y", "z"], [px, py, pz]):
                if p < bbox["min"][k]:
                    bbox["min"][k] = p
                if p > bbox["max"][k]:
                    bbox["max"][k] = p

    controller.reset(scene="Procedural")

    for asset_group, assets in ASSET_DATABASE.items():
        for i in range(len((assets))):
            ASSET_DATABASE[asset_group][i]["states"] = dict()

    for object_type in ASSET_DATABASE:
        print(object_type)
        for i, asset in enumerate(ASSET_DATABASE[object_type]):
            if "CanOpen" not in asset["secondaryProperties"]:
                continue

            print("|", end="")
            controller.reset()
            assert controller.step(
                action="SpawnAsset",
                assetId=asset["assetId"],
                generatedId="asset_0",
                raise_for_failure=True,
            )
            for obj in controller.last_event.metadata["objects"]:
                if obj["openable"]:
                    assert controller.step(
                        action="OpenObject",
                        objectId=obj["objectId"],
                        forceAction=True,
                    )

            bbox = {
                "min": dict(x=float("inf"), y=float("inf"), z=float("inf")),
                "max": dict(x=float("-inf"), y=float("-inf"), z=float("-inf")),
            }
            for obj in controller.last_event.metadata["objects"]:
                encompass_points(bbox, obj["axisAlignedBoundingBox"]["cornerPoints"])

            bbox_size = {k: bbox["max"][k] - bbox["min"][k] for k in ["x", "y", "z"]}
            assert any(bbox_size[k] > 1e-3 for k in ["x", "y", "z"])
            ASSET_DATABASE[object_type][i]["states"]["open"] = dict(
                boundingBox=bbox_size
            )

        print()

    save_asset_database(ASSET_DATABASE)


if __name__ == "__main__":
    controller = Controller(branch="nanna")
    set_object_states(controller)


#%%
def save_ai2thor_object_metadata():
    """Save the starting object metadata in each AI2-THOR scene."""
    controller = Controller(**PROCTHOR_INITIALIZATION)
    scenes = {
        "kitchens": [f"FloorPlan{i}" for i in range(1, 31)],
        "living_rooms": [f"FloorPlan{200 + i}" for i in range(1, 31)],
        "bedrooms": [f"FloorPlan{300 + i}" for i in range(1, 31)],
        "bathrooms": [f"FloorPlan{400 + i}" for i in range(1, 31)],
        "robothor": (
            [f"FloorPlan_Train{i}_{j}" for i in range(1, 13) for j in range(1, 6)]
            + [f"FloorPlan_Val{i}_{j}" for i in range(1, 4) for j in range(1, 6)]
        ),
    }

    out = dict()
    for scene_group, scenes in scenes.items():
        object_meta = []
        for scene in scenes:
            event = controller.reset(scene=scene, procedural=False)
            object_meta.append(event.metadata["objects"])
        out[scene_group] = object_meta

    with open("procthor/databases/ai2thor-object-metadata.json", "w") as f:
        f.write(json.dumps(out, indent=4, sort_keys=True))


#%%
def assign_object_groups():
    placeable_object_types = PLACEMENT_ANNOTATIONS[
        (
            (PLACEMENT_ANNOTATIONS["isPickupable"] == False)
            & (PLACEMENT_ANNOTATIONS["onWall"] == False)
        )
        | (
            (
                (PLACEMENT_ANNOTATIONS["isPickupable"] == True)
                | (PLACEMENT_ANNOTATIONS["onWall"] == True)
            )
            & (PLACEMENT_ANNOTATIONS["onFloor"] == True)
        )
    ]

    object_groups = {
        obj_type: [asset["assetId"] for asset in ASSET_DATABASE[obj_type]]
        for obj_type in placeable_object_types.index
    }
    with open("procthor/databases/object-groups.json", "w") as f:
        f.write(json.dumps(object_groups, indent=4, sort_keys=True))


import json
from collections import Counter, defaultdict

#%%
from houses.databases import AI2THOR_OBJECT_METADATA, PLACEMENT_ANNOTATIONS


def save_receptacles():
    used_asset_types = set(PLACEMENT_ANNOTATIONS.index)

    receptacle_types = set()
    for room_type, scenes in AI2THOR_OBJECT_METADATA.items():
        for objects in scenes:
            for obj in objects:
                if obj["parentReceptacles"]:
                    parents = obj["parentReceptacles"]
                    for parent in parents:
                        parent_type = parent[: parent.find("|")]
                        if (
                            parent_type in used_asset_types
                            and obj["objectType"] in used_asset_types
                        ):
                            receptacle_types.add(parent_type)

    receptacle_counts = defaultdict(Counter)
    for room_type, scenes in AI2THOR_OBJECT_METADATA.items():
        for objects in scenes:
            for obj in objects:
                if obj["objectType"] in receptacle_types:
                    receptacle_counts[obj["objectType"]]["_count"] += 1
                if obj["parentReceptacles"]:
                    parents = obj["parentReceptacles"]
                    for parent in parents:
                        parent_type = parent[: parent.find("|")]
                        if (
                            parent_type in receptacle_types
                            and obj["objectType"] in used_asset_types
                            and not obj["objectType"]
                            in {
                                "Sofa",
                                "Television",
                                "CounterTop",
                                "Chair",
                                "DiningTable",
                                "ShelvingUnit",
                                "SideTable",
                                "Toilet",
                                "Desk",
                                "Sink",
                                "Fridge",
                            }
                        ):
                            receptacle_counts[parent_type][obj["objectType"]] += 1
    out = defaultdict(dict)
    for receptacle, child_objects in receptacle_counts.items():
        parent_count = child_objects["_count"]
        for key, child_count in child_objects.items():
            if key == "_count":
                continue
            out[receptacle][key] = {
                "p": child_count / parent_count,
                "count": child_count,
            }
    out = dict(out)

    with open("procthor/databases/receptacles.json", "w") as f:
        f.write(json.dumps(out, indent=2, sort_keys=True))


#%% wall holes


def set_wall_holes():
    import json

    from ai2thor.controller import Controller

    from houses.constants import PROCTHOR_INITIALIZATION
    from houses.databases import ASSET_DATABASE

    controller = Controller(**PROCTHOR_INITIALIZATION)

    window_ids = [obj["assetId"] for obj in ASSET_DATABASE["Window"]]
    door_ids = [obj["assetId"] for obj in ASSET_DATABASE["Doorway"]]

    hole_ids = window_ids + door_ids
    out = dict()
    for hole_id in hole_ids:
        event = controller.step(action="GetAssetHoleMetadata", assetId=hole_id)
        out[hole_id] = event.metadata["actionReturn"]

    with open("procthor/databases/wall-holes.json", "w") as f:
        f.write(json.dumps(out, indent=4, sort_keys=True))
'''
