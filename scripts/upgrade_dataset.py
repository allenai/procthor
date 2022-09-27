from multiprocessing import Pool
import prior
import os
import sys
import json
from tqdm import tqdm
import copy

# Needed for running locally
sys.path.append(".")

from procthor.utils.upgrade_house_version import HouseUpgradeManager
from procthor.constants import SCHEMA

dataset = prior.load_dataset("procthor-10k")
dataset_copy = copy.deepcopy(dataset)
splits = ["train", "val", "test"]
for s in splits:
    print(f'Converting "{s}" split into latest version "{SCHEMA}":')
    new_split = []
    for i in tqdm(range(len(dataset_copy[s]))):
        house = dataset_copy[s][i]
        out_house = HouseUpgradeManager.upgrade_to(house, SCHEMA)
        new_split.append(out_house)
