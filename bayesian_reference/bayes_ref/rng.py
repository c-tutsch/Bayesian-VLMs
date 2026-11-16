"""Random number generator helpers."""

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Return a NumPy Generator seeded with ``seed``.

    Thin wrapper around ``np.random.default_rng`` so all randomness in the
    module goes through one entry point.
    """
    return np.random.default_rng(seed)


def spawn_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """Return ``n`` statistically independent Generators derived from ``seed``.

    Uses ``np.random.SeedSequence(seed).spawn(n)``, so the result is
    reproducible for a given ``(seed, n)`` and the child streams do not overlap.
    """
    children = np.random.SeedSequence(seed).spawn(n)
    return [np.random.default_rng(child) for child in children]
