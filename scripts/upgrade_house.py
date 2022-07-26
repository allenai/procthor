import argparse
import json
import os

from procthor.constants import SCHEMA
from procthor.utils.upgrade_house_version import HouseUpgradeManager
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Convert house json to latest version.")
parser.add_argument(
    "source",
    metavar="source_filename",
    type=str,
    help="Source json, should be an array of houses or single house",
)
parser.add_argument(
    "output", metavar="output_filename", type=str, help="Output filename"
)

parser.add_argument("-o", "--overwrite", action="store_true")
parser.add_argument("-i", "--indent", action="store_true")
parser.add_argument("-s", "--sort_keys", action="store_true")


args = parser.parse_args()

with open(args.source, "r") as f:
    houses = json.load(f)
    houses = houses if isinstance(houses, list) else [houses]
    result = []
    for i in tqdm(range(len(houses))):
        house = houses[i]
        out_house = HouseUpgradeManager.upgrade_to(house, SCHEMA)
        result.append(out_house)
    if len(result) == 1:
        result = result[0]
    if os.path.exists(args.output) and not args.overwrite:
        raise ValueError(
            f"Output file '{args.output}' already exists. Pass -o option to overwrite it."
        )
    with open(args.output, "w") as fo:
        options = {"sort_keys": args.sort_keys}
        if args.indent:
            options["indent"] = 4
        json.dump(result, fo, **options)
