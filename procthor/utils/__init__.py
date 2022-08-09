from typing import Tuple, Union, Dict

from ai2thor.controller import Controller


class DebugController(Controller):
    def __init__(self, *args, **kwargs) -> None:
        self.steps = []
        super().__init__(*args, **kwargs)

    def step(self, **kwargs):
        out = super().step(**kwargs)
        if "raise_for_failure" in kwargs:
            del kwargs["raise_for_failure"]

        if kwargs["action"] not in {"GetScenesInBuild"}:
            self.steps.append(kwargs)

        return out


def is_equal(
    point0: Union[int, float, Tuple[int, int], Dict[str, float]],
    point1: Union[int, float, Tuple[int, int], Dict[str, float]],
    threshold: float = 1e-2,
) -> bool:
    """Return True if all entries in point0 are within threshold of pairwise entries
    in point1.

    Assumes point0 and point1 have comparable types.
    """
    if isinstance(point0, tuple):
        for p0, p1 in zip(point0, point1):
            if abs(p0 - p1) > threshold:
                return False
        return True
    elif isinstance(point0, (int, float)):
        return abs(point0 - point1) <= threshold
    elif isinstance(point0, dict):
        for p0, p1 in zip(point0.values(), point1.values()):
            if abs(p0 - p1) > threshold:
                return False
        return True
    else:
        raise Exception(f"Unknown types for point0={point0}, point1={point1}")
