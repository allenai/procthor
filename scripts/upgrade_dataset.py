import copy
import gzip
import json

import prior
from procthor.constants import SCHEMA
from procthor.utils.upgrade_house_version import HouseUpgradeManager
from tqdm import tqdm

dataset = prior.load_dataset("procthor-10k")
dataset_copy = copy.deepcopy(dataset)
splits = ["train", "val", "test"]
out = {}
for s in splits:
    print(f'Converting "{s}" split into latest version "{SCHEMA}":')
    new_split = []
    for i in tqdm(range(len(dataset_copy[s]))):
        house = dataset_copy[s][i]
        out_house = HouseUpgradeManager.upgrade_to(house, SCHEMA)
        new_split.append(out_house)
    out[s] = new_split
with gzip.open(f"dataset_v{SCHEMA}.json.gz", "wt") as fo:
    fo.write(json.dumps(out))
