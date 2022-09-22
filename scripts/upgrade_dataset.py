from multiprocessing import Pool
import prior
import os
import sys
import json

sys.path.append('.')

# os.chdir("..")

from procthor.utils.upgrade_house_version import HouseUpgradeManager
from procthor.constants import LATEST_VERSION


# with Pool(processes=processes) as p:
dataset = prior.load_dataset("procthor-10k")
# print(dataset['test'][0])
with open('test_0.json', 'w') as f:
    json.dump(dataset['test'][0], f, indent=4)
latest_version = HouseUpgradeManager.upgrade_to(dataset['test'][0], LATEST_VERSION)

with open('../ai2thor/unity/Assets/Resources/rooms/test_out_0.json', 'w') as f:
    json.dump(latest_version, f, indent=4)
# print(latest_version)
# r = p.map(generate_house, range(10_000_000_000))