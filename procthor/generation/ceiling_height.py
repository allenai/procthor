from typing import Optional

import numpy as np

MIN_CEILING_HEIGHT = 2.5
"""Minimum ceiling height in meters."""

MAX_CEILING_HEIGHT = 7
"""Maximum ceiling height in meters."""

BETA_A = 1.25
"""Beta a distribution parameter."""

BETA_B = 5.5
"""Beta b distribution parameter."""


def sample_ceiling_height(size: Optional[float] = None) -> float:
    """Sample a ceiling height from the distribution."""
    height = MIN_CEILING_HEIGHT + np.random.beta(a=BETA_A, b=BETA_B, size=size) * (
        MAX_CEILING_HEIGHT - MIN_CEILING_HEIGHT
    )
    return height
