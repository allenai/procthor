from multiprocessing import Pool
import prior
import os
import sys
import json
from tqdm import tqdm
import copy

# Needed for running locally
sys.path.append('.')

from procthor.utils.upgrade_house_version import HouseUpgradeManager
from procthor.constants import CURRENT_VERSION

dataset_copy = copy.deepcopy(dataset)
splits = ['train', 'val', 'test']
for s in splits:
    print(f'Converting "{s}" split into latest version "{CURRENT_VERSION}":')
    new_split = []
    for i in tqdm(range(len(dataset_copy[s]))):
        house = dataset_copy[s][i]
        out_house = HouseUpgradeManager.upgrade_to(house, CURRENT_VERSION)
        new_split.append(out_house)