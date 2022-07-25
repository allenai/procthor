import os
from datetime import datetime
from multiprocessing import Pool, Value
from time import sleep

import torch
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from procthor.constants import PROCTHOR_INITIALIZATION
from procthor.generation import HouseGenerator, PROCTHOR10K_ROOM_SPEC_SAMPLER

print("Starting at", datetime.now())


split = "train"
processes = 60
counter = Value("i", 5400 - processes)
n_gpus = torch.cuda.device_count()

# n_gpus = 1
# house_generators = [
#     HouseGenerator(
#         split=split,
#         controller=Controller(
#             x_display=f":0.{i}", quality="Low", **PROCTHOR_INITIALIZATION
#         ),
#     )
#     for i in range(n_gpus)
# ]

# house_generator = HouseGenerator(split=split)
house_generators = {}


def generate_house(i: int) -> None:
    global counter
    global house_generator
    global n_gpus

    pid = os.getpid()
    print(f"Using {pid}")
    if pid not in house_generators:
        gpu_i = pid % n_gpus
        controller = Controller(
            gpu_device=gpu_i,
            platform=CloudRendering,
            quality="Low",
            **PROCTHOR_INITIALIZATION,
        )
        house_generators[pid] = HouseGenerator(
            controller=controller,
            split=split,
            room_spec_sampler=PROCTHOR10K_ROOM_SPEC_SAMPLER,
        )

    house_generator = house_generators[pid]

    # NOTE: sometimes house_generator.sample() hangs
    room_spec = None
    while True:
        house_generator.room_spec = room_spec
        house, _ = house_generator.sample()
        house.validate(house_generator.controller)
        if house.data["metadata"]["warnings"]:
            # NOTE: Keep the room spec the same to avoid sampling bias.
            house_generator.room_spec = house.room_spec
            continue
        break

    with counter.get_lock():
        counter.value += 1
    sleep(0.1)

    print(i, counter.value)

    house.to_json(f"big-dataset/{split}/{pid}-{counter.value}.json.gz", compressed=True)


with Pool(processes=processes) as p:
    r = p.map(generate_house, range(10_000_000_000))
