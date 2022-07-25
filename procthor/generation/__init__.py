import copy
import logging
import random
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Union

import numpy as np
from ai2thor.controller import Controller
from attr import Attribute, field
from attrs import define
from procthor.constants import PROCTHOR_INITIALIZATION
from procthor.utils.types import InvalidFloorplan, SamplingVars, Split

from .ceiling_height import sample_ceiling_height
from .color_objects import randomize_object_colors
from .doors import add_doors
from .exterior_walls import add_exterior_walls
from .generation import (
    create_empty_partial_house,
    find_walls,
    get_floor_polygons,
    get_xz_poly_map,
    sample_house_structure,
    scale_boundary_groups,
)
from .house import House, HouseStructure, NextSamplingStage, PartialHouse
from .interior_boundaries import sample_interior_boundary
from .layer import assign_layer_to_rooms
from .lights import add_lights
from .materials import randomize_wall_and_floor_materials
from .object_states import randomize_object_states
from .objects import add_floor_objects, add_rooms
from .room_specs import PROCTHOR10K_ROOM_SPEC_SAMPLER, RoomSpec, RoomSpecSampler
from .skyboxes import add_skybox
from .small_objects import add_small_objects
from .wall_objects import add_wall_objects


@define
class HouseGenerator:
    split: Split = field()
    seed: Optional[int] = None
    room_spec: Optional[Union[RoomSpec, str]] = None
    room_spec_sampler: Optional[RoomSpecSampler] = None
    interior_boundary: Optional[np.array] = None
    controller: Optional[Controller] = None
    partial_house: Optional[PartialHouse] = None

    @split.validator
    def _valid_split(self, attribute: Attribute, value: Split) -> None:
        if value not in {"train", "val", "test"}:
            raise ValueError(f'split={value} must be in {{"train", "val", "test"}}')

    def __attrs_post_init__(self) -> None:
        if self.seed is None:
            self.seed = random.randint(0, 2**15)
            logging.debug(f"Using seed {self.seed}")
        if self.seed is not None:
            self.set_seed(self.seed)
        if isinstance(self.room_spec, str):
            self.room_spec = self.room_spec_sampler[self.room_spec]

    def set_seed(self, seed: int) -> None:
        # TODO: Probably should not be done on a global level
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        if self.controller is not None:
            self.controller.step(action="SetRandomSeed", seed=seed, renderImage=False)

    def sample(
        self,
        partial_house: Optional[PartialHouse] = None,
        return_partial_houses: bool = False,
    ) -> Tuple[House, Dict[NextSamplingStage, PartialHouse]]:
        """Sample a house specification compatible with AI2-THOR."""
        if self.controller is None:
            # NOTE: assumes images are never used by this Controller.
            self.controller = Controller(quality="Low", **PROCTHOR_INITIALIZATION)
            if self.seed is not None:
                self.controller.step(
                    action="SetRandomSeed", seed=self.seed, renderImage=False
                )

        sampling_stage_to_ph = {}
        sampling_vars = SamplingVars.sample()

        if partial_house is None:
            # NOTE: grabbing existing values
            if self.partial_house is not None:
                room_spec = self.partial_house.room_spec
                interior_boundary = self.partial_house.house_structure.interior_boundary
            else:
                room_spec = self.room_spec_sampler.sample()
                interior_boundary = None

            # NOTE: sample house structure via rejection sampling.
            room_ids = set(room_spec.room_type_map.keys())
            for _ in range(10):
                try:
                    house_structure = sample_house_structure(
                        interior_boundary=interior_boundary,
                        room_ids=room_ids,
                        room_spec=room_spec,
                        interior_boundary_scale=sampling_vars.interior_boundary_scale,
                    )
                    break
                except InvalidFloorplan:
                    pass
            else:
                raise Exception(
                    f"Failed to generate interior boundary and floorplan"
                    "for room_spec={room_spec}!"
                )

            partial_house = PartialHouse.from_structure_and_room_spec(
                house_structure=house_structure,
                room_spec=room_spec,
            )
        else:
            assert partial_house.next_sampling_stage.value > NextSamplingStage.STRUCTURE

        if return_partial_houses:
            sampling_stage_to_ph[partial_house.next_sampling_stage] = copy.deepcopy(
                partial_house
            )

        @contextmanager
        def advance_and_record_partial(partial_house: PartialHouse):
            try:
                yield None
            finally:
                partial_house.advance_sampling_stage()
                if return_partial_houses:
                    sampling_stage_to_ph[
                        partial_house.next_sampling_stage
                    ] = copy.deepcopy(partial_house)

        # NOTE: DOORS
        if partial_house.next_sampling_stage <= NextSamplingStage.DOORS:
            with advance_and_record_partial(partial_house):
                door_polygons = add_doors(
                    partial_house=partial_house,
                    split=self.split,
                )
                randomize_wall_and_floor_materials(partial_house)

        floor_polygons = get_floor_polygons(
            xz_poly_map=partial_house.house_structure.xz_poly_map
        )

        if partial_house.next_sampling_stage <= NextSamplingStage.LIGHTS:
            with advance_and_record_partial(partial_house):
                add_lights(
                    partial_house=partial_house,
                    floor_polygons=floor_polygons,
                    ceiling_height=partial_house.house_structure.ceiling_height,
                )

        if partial_house.next_sampling_stage <= NextSamplingStage.SKYBOX:
            with advance_and_record_partial(partial_house):
                add_skybox(partial_house=partial_house)

        # NOTE: added after `randomize_wall_and_floor_materials` on purpose
        if partial_house.next_sampling_stage <= NextSamplingStage.EXTERIOR_WALLS:
            with advance_and_record_partial(partial_house):
                add_exterior_walls(
                    partial_house=partial_house,
                    boundary_groups=partial_house.house_structure.boundary_groups,
                )

        if partial_house.next_sampling_stage <= NextSamplingStage.ROOMS:
            with advance_and_record_partial(partial_house):
                add_rooms(
                    partial_house=partial_house,
                    floor_polygons=floor_polygons,
                    room_type_map=partial_house.room_spec.room_type_map,
                    split=self.split,
                    door_polygons=door_polygons,
                    controller=self.controller,
                )

        if partial_house.next_sampling_stage <= NextSamplingStage.FLOOR_OBJS:
            with advance_and_record_partial(partial_house):
                add_floor_objects(
                    partial_house=partial_house,
                    max_floor_objects=sampling_vars.max_floor_objects,
                )
                floor_objects = [*partial_house.objects]
                randomize_object_colors(objects=floor_objects)
                randomize_object_states(objects=floor_objects)

        if partial_house.next_sampling_stage <= NextSamplingStage.WALL_OBJS:
            with advance_and_record_partial(partial_house):
                add_wall_objects(
                    partial_house=partial_house,
                    rooms=partial_house.rooms,
                    boundary_groups=partial_house.house_structure.boundary_groups,
                    split=self.split,
                    room_type_map=partial_house.room_spec.room_type_map,
                    ceiling_height=partial_house.house_structure.ceiling_height,
                )
                wall_objects = [*partial_house.objects[len(floor_objects) :]]
                randomize_object_colors(objects=wall_objects)
                randomize_object_states(objects=wall_objects)

        if partial_house.next_sampling_stage <= NextSamplingStage.SMALL_OBJS:
            with advance_and_record_partial(partial_house):
                add_small_objects(
                    partial_house=partial_house,
                    rooms=partial_house.rooms,
                    split=self.split,
                    controller=self.controller,
                )
                small_objects = [
                    *partial_house.objects[len(floor_objects) + len(wall_objects) :]
                ]
                randomize_object_colors(objects=small_objects)
                randomize_object_states(objects=small_objects)

        assign_layer_to_rooms(partial_house=partial_house)

        sampling_stage_to_ph[partial_house.next_sampling_stage] = partial_house

        return partial_house.to_house(), sampling_stage_to_ph
