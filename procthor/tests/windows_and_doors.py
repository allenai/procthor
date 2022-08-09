#%% generates visualizations of windows and doors on the wall.

#%%

# %%
import json
import os

from procthor.databases import DEFAULT_PROCTHOR_DATABASE

if __name__ == "__main__":
    os.chdir("../..")
    house = {
        "proceduralParameters": {
            "ceilingMaterial": "CeramicTiles3",
            "lights": [
                {
                    "id": "DirectionalLight",
                    "indirectMultiplier": 1.0,
                    "intensity": 0.75,
                    "position": {"x": 0.84, "y": 0.1855, "z": -1.09},
                    "rgb": {"a": 1.0, "b": 1.0, "g": 1.0, "r": 1.0},
                    "rotation": {"x": 43.375, "y": -3.902, "z": -63.618},
                    "shadow": {
                        "bias": 0.0,
                        "nearPlane": 0.1,
                        "normalBias": 0.91,
                        "resolution": "VeryHigh",
                        "strength": 0.116,
                        "type": "Soft",
                    },
                    "type": "directional",
                }
            ],
            "skyboxId": "SkyMountain",
        },
        "walls": [],
        "doors": [],
        "windows": [],
    }

    windows = [
        (window, "windows")
        for window in DEFAULT_PROCTHOR_DATABASE.ASSET_DATABASE["Window"]
    ]
    doors = [
        (door, "doors") for door in DEFAULT_PROCTHOR_DATABASE.ASSET_DATABASE["Doorway"]
    ]
    for i, (asset, key) in enumerate(windows + doors):
        asset_id = asset["assetId"]

        z = 0
        house["walls"].append(
            {
                "id": f"wall|{i}",
                "material": "Walldrywall3",
                "polygon": [
                    {"x": i * 4, "y": 0, "z": z},
                    {"x": i * 4 + 3, "y": 0, "z": z},
                    {"x": i * 4 + 3, "y": 3, "z": z},
                    {"x": i * 4, "y": 3, "z": z},
                ],
            }
        )

        # placement in left corner
        house[key].append(
            {
                "assetId": asset_id,
                "assetOffset": DEFAULT_PROCTHOR_DATABASE.WALL_HOLES[asset_id]["offset"],
                "boundingBox": {
                    "min": {
                        "x": 0,
                        "y": 0,
                        "z": 0,
                    },
                    "max": {
                        "x": (
                            DEFAULT_PROCTHOR_DATABASE.WALL_HOLES[asset_id]["max"]["x"]
                        ),
                        "y": 0 + asset["boundingBox"]["y"],
                        "z": 0,
                    },
                },
                "id": f"obj|{i}",
                "wall0": f"wall|{i}",
                "wall1": None,
            }
        )

        # NOTE: place windows in the center
        z = 5
        house["walls"].append(
            {
                "id": f"wall|{i}|1",
                "material": "Walldrywall3",
                "polygon": [
                    {"x": i * 4, "y": 0, "z": z},
                    {"x": i * 4 + 3, "y": 0, "z": z},
                    {"x": i * 4 + 3, "y": 3, "z": z},
                    {"x": i * 4, "y": 3, "z": z},
                ],
            }
        )

        start_at_x = (3 - asset["boundingBox"]["x"]) / 2
        house[key].append(
            {
                "assetId": asset_id,
                "assetOffset": DEFAULT_PROCTHOR_DATABASE.WALL_HOLES[asset_id]["offset"],
                "boundingBox": {
                    "min": {"x": start_at_x, "y": 0.5, "z": 0},
                    "max": {
                        "x": start_at_x
                        + DEFAULT_PROCTHOR_DATABASE.WALL_HOLES[asset_id]["max"]["x"],
                        "y": 0.5 + asset["boundingBox"]["y"],
                        "z": 0,
                    },
                },
                "id": f"obj|{i}|1",
                "wall0": f"wall|{i}|1",
                "wall1": None,
            }
        )

        # NOTE: place windows on the right edge
        z = 10
        house["walls"].append(
            {
                "id": f"wall|{i}|2",
                "material": "Walldrywall3",
                "polygon": [
                    {"x": i * 4, "y": 0, "z": z},
                    {"x": i * 4 + 3, "y": 0, "z": z},
                    {"x": i * 4 + 3, "y": 3, "z": z},
                    {"x": i * 4, "y": 3, "z": z},
                ],
            }
        )
        house[key].append(
            {
                "assetId": asset_id,
                "assetOffset": DEFAULT_PROCTHOR_DATABASE.WALL_HOLES[asset_id]["offset"],
                "boundingBox": {
                    "min": {
                        "x": (3 - asset["boundingBox"]["x"]),
                        "y": 0.5,
                        "z": 0,
                    },
                    "max": {
                        "x": (
                            3
                            - asset["boundingBox"]["x"]
                            + DEFAULT_PROCTHOR_DATABASE.WALL_HOLES[asset_id]["max"]["x"]
                        ),
                        "y": 0.5 + asset["boundingBox"]["y"],
                        "z": 0,
                    },
                },
                "id": f"obj|{i}|2",
                "wall0": f"wall|{i}|2",
                "wall1": None,
            }
        )

        # NOTE: flip polygon, left corner
        z = 15
        house["walls"].append(
            {
                "id": f"wall|{i}|3",
                "material": "Walldrywall3",
                "polygon": [
                    {"x": i * 4 + 3, "y": 0, "z": z},
                    {"x": i * 4, "y": 0, "z": z},
                    {"x": i * 4, "y": 3, "z": z},
                    {"x": i * 4 + 3, "y": 3, "z": z},
                ],
            }
        )
        house[key].append(
            {
                "assetId": asset_id,
                "assetOffset": DEFAULT_PROCTHOR_DATABASE.WALL_HOLES[asset_id]["offset"],
                "boundingBox": {
                    "min": {
                        "x": (3 - asset["boundingBox"]["x"]),
                        "y": 3 - asset["boundingBox"]["y"],
                        "z": 0,
                    },
                    "max": {
                        "x": (
                            3
                            - asset["boundingBox"]["x"]
                            + DEFAULT_PROCTHOR_DATABASE.WALL_HOLES[asset_id]["max"]["x"]
                        ),
                        "y": 3,
                        "z": 0,
                    },
                },
                "id": f"obj|{i}|3",
                "wall0": f"wall|{i}|3",
                "wall1": None,
            }
        )

        # NOTE: flip polygon, right corner
        z = 20
        house["walls"].append(
            {
                "id": f"wall|{i}|4",
                "material": "Walldrywall3",
                "polygon": [
                    {"x": i * 4 + 3, "y": 0, "z": z},
                    {"x": i * 4, "y": 0, "z": z},
                    {"x": i * 4, "y": 3, "z": z},
                    {"x": i * 4 + 3, "y": 3, "z": z},
                ],
            }
        )
        house[key].append(
            {
                "assetId": asset_id,
                "assetOffset": DEFAULT_PROCTHOR_DATABASE.WALL_HOLES[asset_id]["offset"],
                "boundingBox": {
                    "min": {
                        "x": 0,
                        "y": 0,
                        "z": 0,
                    },
                    "max": {
                        "x": (
                            0
                            + DEFAULT_PROCTHOR_DATABASE.WALL_HOLES[asset_id]["max"]["x"]
                        ),
                        "y": 0 + asset["boundingBox"]["y"],
                        "z": 0,
                    },
                },
                "id": f"obj|{i}|4",
                "wall0": f"wall|{i}|4",
                "wall1": None,
            }
        )

    with open("temp.json", "w") as f:
        f.write(json.dumps(house, indent=4, sort_keys=True))
