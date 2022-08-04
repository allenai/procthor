import random
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from attr import Attribute, field
from attrs import define
from procthor.utils.types import LeafRoom, MetaRoom


@define
class RoomSpec:
    room_spec_id: str
    sampling_weight: float = field()
    spec: List[Union[LeafRoom, MetaRoom]]

    dims: Optional[Callable[[], Tuple[int, int]]] = None
    """The (x_size, z_size) dimensions of the house.

    Note that this size will later be scaled up by interior_boundary_scale.
    """

    room_type_map: Dict[
        int, Literal["Bedroom", "Bathroom", "Kitchen", "LivingRoom"]
    ] = field(init=False)
    room_map: Dict[int, Union[LeafRoom, MetaRoom]] = field(init=False)

    @sampling_weight.validator
    def ge_0(self, attribute: Attribute, value: float):
        if value <= 0:
            raise ValueError(f"sampling_weight must be > 0! You gave {value}.")

    def _set_meta_room_ids(
        self,
        spec: List[Union[LeafRoom, MetaRoom]],
        used_ids: Set[int],
        start_at: int = 2,
    ) -> Set[int]:
        """Assign a room_id to each MetaRoom in the RoomSpec."""
        used_ids = used_ids.copy()
        for room in spec:
            if isinstance(room, MetaRoom):
                i = 0
                while (room_id := start_at + i) in used_ids:
                    i += 1

                used_ids.add(room_id)
                room.room_id = room_id
                self.room_map[room_id] = room

                used_ids = self._set_meta_room_ids(
                    spec=room.children, used_ids=used_ids, start_at=room_id + 1
                )
            else:
                self.room_map[room.room_id] = room
        return used_ids

    def _get_room_type_map(
        self, spec: List[Union[MetaRoom, LeafRoom]]
    ) -> Dict[int, str]:
        """Set room_type_map to the room_id -> room type for each room in the room spec."""
        room_ids = dict()
        for room in spec:
            if isinstance(room, MetaRoom):
                room_ids.update(self._get_room_type_map(room.children))
            else:
                room_ids[room.room_id] = room.room_type
        return room_ids

    def __attrs_post_init__(self) -> None:
        self.room_type_map = self._get_room_type_map(spec=self.spec)
        self.room_map = dict()
        self._set_meta_room_ids(spec=self.spec, used_ids=set(self.room_type_map.keys()))


@define
class RoomSpecSampler:
    room_specs: List[RoomSpec] = field()

    room_spec_map: Dict[str, RoomSpec] = field(init=False)
    weights: List[float] = field(init=False)

    @room_specs.validator
    def unique_room_spec_ids(self, attribute: Attribute, value: List[RoomSpec]) -> None:
        room_spec_ids = set()
        for room_spec in value:
            if room_spec.room_spec_id in room_spec_ids:
                raise ValueError(
                    "Each RoomSpec must have a unique room_spec_id."
                    f" You gave duplicate room_spec_id: {room_spec.room_spec_id}."
                )
            room_spec_ids.add(room_spec.room_spec_id)

    def __attrs_post_init__(self) -> None:
        self.room_spec_map = dict()
        self.weights = []
        for room_spec in self.room_specs:
            self.room_spec_map[room_spec.room_spec_id] = room_spec
            self.weights.append(room_spec.sampling_weight)

    def __getitem__(self, room_spec_id: str) -> RoomSpec:
        return self.room_spec_map[room_spec_id]

    def sample(self, k: int = 1) -> Union[RoomSpec, List[RoomSpec]]:
        """Return a RoomSpec with weighted sampling."""
        sample = random.choices(self.room_specs, weights=self.weights, k=k)
        return sample[0] if k == 1 else sample


PROCTHOR10K_ROOM_SPEC_SAMPLER = RoomSpecSampler(
    [
        RoomSpec(
            dims=lambda: (random.randint(13, 16), random.randint(5, 8)),
            room_spec_id="8-room-3-bed",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=4,
                    children=[
                        MetaRoom(
                            ratio=2,
                            children=[
                                LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
                                LeafRoom(room_id=3, ratio=3, room_type="LivingRoom"),
                            ],
                        ),
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=4, ratio=2, room_type="LivingRoom"),
                                LeafRoom(
                                    room_id=5,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                    ],
                ),
                MetaRoom(
                    ratio=1,
                    children=[
                        LeafRoom(room_id=6, ratio=1, room_type="Bedroom"),
                    ],
                ),
                MetaRoom(
                    ratio=1,
                    children=[
                        LeafRoom(room_id=7, ratio=1, room_type="Bedroom"),
                    ],
                ),
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=8, ratio=1, room_type="Bedroom"),
                        LeafRoom(
                            room_id=9,
                            ratio=1,
                            room_type="Bathroom",
                            avoid_doors_from_metarooms=True,
                        ),
                    ],
                ),
            ],
        ),
        RoomSpec(
            room_spec_id="7-room-3-bed",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=3,
                    children=[
                        MetaRoom(
                            ratio=2,
                            children=[
                                LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
                                LeafRoom(room_id=3, ratio=3, room_type="LivingRoom"),
                            ],
                        ),
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=4, ratio=2, room_type="LivingRoom"),
                                LeafRoom(
                                    room_id=5,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                    ],
                ),
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=6, ratio=2, room_type="Bedroom"),
                        LeafRoom(room_id=7, ratio=2, room_type="Bedroom"),
                        LeafRoom(room_id=8, ratio=2, room_type="Bedroom"),
                    ],
                ),
            ],
        ),
        RoomSpec(
            room_spec_id="12-room-3-bed",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=1,
                    children=[
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
                                LeafRoom(room_id=3, ratio=3, room_type="LivingRoom"),
                            ],
                        ),
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=4, ratio=1, room_type="LivingRoom"),
                                LeafRoom(room_id=5, ratio=1, room_type="LivingRoom"),
                            ],
                        ),
                    ],
                ),
                MetaRoom(
                    ratio=1,
                    children=[
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=6, ratio=2, room_type="Bedroom"),
                                LeafRoom(
                                    room_id=7,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=8, ratio=2, room_type="Bedroom"),
                                LeafRoom(
                                    room_id=9,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=10, ratio=2, room_type="Bedroom"),
                                LeafRoom(
                                    room_id=11,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        RoomSpec(
            room_spec_id="12-room",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=2,
                    children=[
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
                                LeafRoom(room_id=3, ratio=3, room_type="LivingRoom"),
                            ],
                        ),
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=4, ratio=2, room_type="Bedroom"),
                                LeafRoom(
                                    room_id=5,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                    ],
                ),
                MetaRoom(
                    ratio=3,
                    children=[
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=6, ratio=2, room_type="Bedroom"),
                                LeafRoom(
                                    room_id=7,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=8, ratio=2, room_type="Bedroom"),
                                LeafRoom(
                                    room_id=9,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                        MetaRoom(
                            ratio=1,
                            children=[
                                LeafRoom(room_id=10, ratio=2, room_type="Bedroom"),
                                LeafRoom(
                                    room_id=11,
                                    ratio=1,
                                    room_type="Bathroom",
                                    avoid_doors_from_metarooms=True,
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        RoomSpec(
            room_spec_id="4-room",
            sampling_weight=5,
            spec=[
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=4, ratio=2, room_type="Bedroom"),
                        LeafRoom(
                            room_id=5,
                            ratio=1,
                            room_type="Bathroom",
                            avoid_doors_from_metarooms=True,
                        ),
                    ],
                ),
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=6, ratio=3, room_type="Kitchen"),
                        LeafRoom(room_id=7, ratio=2, room_type="LivingRoom"),
                    ],
                ),
            ],
        ),
        RoomSpec(
            room_spec_id="2-bed-1-bath",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
                        LeafRoom(
                            room_id=3,
                            ratio=2,
                            room_type="Bathroom",
                            avoid_doors_from_metarooms=True,
                        ),
                        LeafRoom(room_id=4, ratio=3, room_type="LivingRoom"),
                    ],
                ),
                LeafRoom(room_id=5, ratio=1, room_type="Bedroom"),
                LeafRoom(room_id=6, ratio=1, room_type="Bedroom"),
            ],
        ),
        RoomSpec(
            room_spec_id="5-room",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=4, ratio=2, room_type="Bedroom"),
                        LeafRoom(
                            room_id=5,
                            ratio=1,
                            room_type="Bathroom",
                            avoid_doors_from_metarooms=True,
                        ),
                    ],
                ),
                LeafRoom(room_id=6, ratio=2, room_type="Bedroom"),
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=8, ratio=3, room_type="Kitchen"),
                        LeafRoom(room_id=9, ratio=2, room_type="LivingRoom"),
                    ],
                ),
            ],
        ),
        RoomSpec(
            room_spec_id="2-bed-2-bath",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=4, ratio=2, room_type="Bedroom"),
                        LeafRoom(
                            room_id=5,
                            ratio=1,
                            room_type="Bathroom",
                            avoid_doors_from_metarooms=True,
                        ),
                    ],
                ),
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=6, ratio=2, room_type="Bedroom"),
                        LeafRoom(
                            room_id=7,
                            ratio=1,
                            room_type="Bathroom",
                            avoid_doors_from_metarooms=True,
                        ),
                    ],
                ),
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=8, ratio=3, room_type="Kitchen"),
                        LeafRoom(room_id=9, ratio=2, room_type="LivingRoom"),
                    ],
                ),
            ],
        ),
        RoomSpec(
            room_spec_id="bedroom-bathroom",
            sampling_weight=2,
            spec=[
                LeafRoom(room_id=2, ratio=2, room_type="Bedroom"),
                LeafRoom(room_id=3, ratio=1, room_type="Bathroom"),
            ],
        ),
        RoomSpec(
            room_spec_id="kitchen-living-bedroom-room",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=6, ratio=3, room_type="Kitchen"),
                        LeafRoom(room_id=7, ratio=2, room_type="LivingRoom"),
                    ],
                ),
                LeafRoom(room_id=2, ratio=1, room_type="Bedroom"),
            ],
        ),
        RoomSpec(
            room_spec_id="kitchen-living-bedroom-room2",
            sampling_weight=1,
            spec=[
                MetaRoom(
                    ratio=2,
                    children=[
                        LeafRoom(room_id=6, ratio=1, room_type="Kitchen"),
                        LeafRoom(room_id=7, ratio=1, room_type="LivingRoom"),
                    ],
                ),
                LeafRoom(room_id=2, ratio=1, room_type="Bedroom"),
            ],
        ),
        RoomSpec(
            room_spec_id="kitchen-living-room",
            sampling_weight=2,
            spec=[
                LeafRoom(room_id=2, ratio=1, room_type="Kitchen"),
                LeafRoom(room_id=3, ratio=1, room_type="LivingRoom"),
            ],
        ),
        RoomSpec(
            room_spec_id="kitchen",
            sampling_weight=1,
            spec=[LeafRoom(room_id=2, ratio=1, room_type="Kitchen")],
        ),
        RoomSpec(
            room_spec_id="living-room",
            sampling_weight=1,
            spec=[LeafRoom(room_id=2, ratio=1, room_type="LivingRoom")],
        ),
        RoomSpec(
            room_spec_id="bedroom",
            sampling_weight=1,
            spec=[LeafRoom(room_id=2, ratio=1, room_type="Bedroom")],
        ),
        RoomSpec(
            # scale=1.25?
            room_spec_id="bathroom",
            sampling_weight=1,
            spec=[LeafRoom(room_id=2, ratio=1, room_type="Bathroom")],
        ),
    ]
)
