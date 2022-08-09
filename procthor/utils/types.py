import random
from typing import Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union

from attrs import define

from procthor.constants import OUTDOOR_ROOM_ID

Split = Literal["train", "val", "test"]
XZPoly = List[Tuple[Tuple[float, float], Tuple[float, float]]]
BoundaryGroups = Dict[Tuple[int, int], List[Tuple[float, float]]]


class InvalidFloorplan(Exception):
    pass


@define
class SamplingVars:
    interior_boundary_scale: float
    """Amount in meters with which each interior boundary is scaled.

    This is often useful to set more than 1 because most single doors are between
    :math:`(1.0 : 1.1)` meters. Thus, if a scale of :math:`1.0` was used, there wouldn't
    be any eligable doors with a single panel wall separator.
    """

    max_floor_objects: int
    """The maximum number of objects that can be placed on the floor of each room."""

    @classmethod
    def sample(cls) -> "SamplingVars":
        return SamplingVars(
            interior_boundary_scale=random.uniform(1.6, 2.2),
            max_floor_objects=random.choices(
                population=[1, 4, 5, 6, 7],
                weights=[1 / 200, 1 / 100, 1 / 50, 1 / 10, 173 / 200],
                k=1,
            )[0],
        )


class Skybox(TypedDict):
    name: str
    timeOfDay: Literal["Midday", "GoldenHour", "BlueHour", "Midnight"]


class Vector3(TypedDict):
    x: float
    y: float
    z: float


class BoundingBox(TypedDict):
    max: Vector3
    min: Vector3


class Door(TypedDict):
    id: str
    assetId: str
    boundingBox: BoundingBox
    openness: float
    openable: bool
    room0: str
    room1: str
    wall0: str
    wall1: str


class Window(TypedDict):
    id: str
    assetId: str
    boundingBox: BoundingBox
    room0: str
    room1: Optional[str]
    wall0: str
    wall1: Optional[str]
    assetOffset: Vector3
    """A piece of metadata used for cutting walls. Allows the cutout to not be
    the entire size of the window."""


class Object(TypedDict):
    id: str
    assetId: str
    """The id of the asset in the asset database."""

    rotation: Vector3
    """The global rotation of the object."""

    position: Vector3
    """The global (x, y, z) position of the object."""

    children: List["Object"]
    """Objects that are parented to the receptacle."""

    kinematic: bool
    """True if the object can be moved, False otherwise.

    Large objects, such as Fridges and Toilets often shouldn't be moveable, and 
    can result in a variety of bugs.
    """


class RGB(TypedDict):
    r: float
    g: float
    b: float


class RGBA(TypedDict):
    r: float
    g: float
    b: float
    a: float


class LightShadow(TypedDict):
    bias: float
    nearPlane: float
    normalBias: float
    resolution: str
    strength: float
    type: str


# TODO: account for either directional or point light!
class Light(TypedDict):
    id: str
    """The name of the light."""

    position: Vector3
    """The global position of the light in the scene."""

    rotation: Vector3
    """The global rotation of the light in the scene."""

    type: Literal["point", "directional", "spot"]

    rgb: RGBA
    """The color of the light."""

    # optional
    linkedObjectId: str
    """Links the light to a toggleabe object.
    
    When the object is toggled, the light is toggled.
    """

    indirectMultiplier: float
    """Use this value to vary the intensity of indirect light.

    If you set Indirect Multiplier to a value lower than 1, the bounced light becomes
    dimmer with every bounce. A value higher than 1 makes light brighter with each
    bounce. This is useful, for example, when a dark surface in shadow (such as the
    interior of a cave) needs to be brighter in order to make detail visible. If you
    want to use Realtime Global Illumination, but want to limit a single real-time
    Light so that it only emits direct light, set its Indirect Multiplier to 0.

    From Unity: https://docs.unity3d.com/Manual/class-Light.html
    """

    intensity: float
    """The Intensity of a light is multiplied with the Light color.

    The value can be between 0 and 8. This allows you to create over bright lights.

    From Unity: https://docs.unity3d.com/ScriptReference/Light-intensity.html
    """

    range: float
    """Define how far the light emitted from the center of the object travels
    (Point and Spot lights only).

    From Unity: https://docs.unity3d.com/Manual/class-Light.html
    """

    shadow: LightShadow
    """Only used for directional lights."""


class ProceduralParameters(TypedDict):
    ceilingMaterial: str
    ceilingColor: RGB
    lights: List[Light]

    reflections: list  # TODO: Annotate this!
    skyboxId: str

    # receptacle collider height for the floor
    receptacleHeight: float

    # TODO: what is this?
    floorColliderThickness: float


class RoomType(TypedDict):
    id: str
    roomType: Literal["Kitchen", "LivingRoom", "Bedroom", "Bathroom"]
    floorMaterial: str
    floorPolygon: List[Vector3]

    ceilings: list  # TODO: what is this?
    children: list  # TODO: what is this?


class Wall(TypedDict):
    id: str
    polygon: List[Vector3]
    material: str
    roomId: str

    # NOTE: optional
    empty: bool


class HouseDict(TypedDict):
    doors: Optional[List[Door]]
    windows: Optional[List[Window]]
    objects: Optional[List[Object]]
    proceduralParameters: Optional[ProceduralParameters]
    rooms: Optional[List[RoomType]]
    walls: Optional[List[Wall]]


class LeafRoom:
    def __init__(
        self,
        room_id: int,
        ratio: int,
        room_type: Optional[Literal["Kitchen", "LivingRoom", "Bedroom", "Bathroom"]],
        avoid_doors_from_metarooms: bool = False,
    ):
        """
        Parameters:
        - avoid_doors_from_metarooms: prioritize having only 1 door, if possible.
          For example, bathrooms often only have 1 door.
        """
        assert room_type in {"Kitchen", "LivingRoom", "Bedroom", "Bathroom", None}
        if room_id in {0, OUTDOOR_ROOM_ID}:
            raise Exception(f"room_id of 0 and {OUTDOOR_ROOM_ID} are reserved!")

        self.avoid_doors_from_metarooms = avoid_doors_from_metarooms
        self.room_id = room_id
        self.room_type = room_type
        self.ratio = ratio

    def __repr__(self):
        return (
            "LeafRoom(\n"
            f"    room_id={self.room_id},\n"
            f"    ratio={self.ratio},\n"
            f"    room_type={self.room_type}\n"
            ")"
        )

    def __str__(self):
        return self.__repr__()


class MetaRoom:
    def __init__(
        self,
        ratio: int,
        children: Sequence[Union[LeafRoom, "MetaRoom"]],
        room_type: Optional[str] = None,
    ):
        self.ratio = ratio
        self.children = children
        self.room_type = room_type
        self._room_id = None

    @property
    def room_id(self):
        if self._room_id is None:
            raise RuntimeError("room_id not set in MetaRoom!")
        return self._room_id

    @room_id.setter
    def room_id(self, room_id: int):
        if room_id in {0, OUTDOOR_ROOM_ID}:
            raise Exception(f"room_id of 0 and {OUTDOOR_ROOM_ID} are reserved!")
        self._room_id = room_id

    def __repr__(self):
        return f"MetaRoom(ratio={self.ratio}, children={self.children})"

    def __str__(self):
        return self.__repr__()
