"""Exact integer reward arithmetic, option-set sampling and the simulated user's choice."""

import numpy as np

from bayes_ref import config
from bayes_ref.world import all_flights

_GRID_ATOL = 1e-9


def _to_int_grid(x: np.ndarray, scale: int, name: str) -> np.ndarray:
    scaled = np.asarray(x, dtype=np.float64) * scale
    rounded = np.rint(scaled)
    if scaled.size and np.max(np.abs(scaled - rounded)) >= _GRID_ATOL:
        raise ValueError(f"{name}: values are not on the grid of step 1/{scale}: {x!r}")
    return rounded.astype(np.int64)


def to_int_features(phi: np.ndarray) -> np.ndarray:
    """Features in units of 1/FEATURE_SCALE. Any shape in, int64 of the same shape out.

    Raises ValueError if any value is off the 0.1 grid by 1e-9 or more.
    """
    return _to_int_grid(phi, config.FEATURE_SCALE, "features")


def to_int_theta(theta: np.ndarray) -> np.ndarray:
    """Reward weights in units of 1/THETA_SCALE. Any shape in, int64 of the same shape out.

    Raises ValueError if any value is off the 0.5 grid by 1e-9 or more.
    """
    return _to_int_grid(theta, config.THETA_SCALE, "theta")


def rewards_int(options: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Integer rewards (FEATURE_SCALE * THETA_SCALE * true reward) of options (K, 4) under theta (4,). Shape (K,)."""
    return to_int_features(options) @ to_int_theta(theta)


def rewards_int_all(options: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    """Integer rewards of options (K, 4) under every theta in Theta (U, 4). Shape (K, U)."""
    return to_int_features(options) @ to_int_theta(Theta).T


def tied_best(rewards: np.ndarray) -> np.ndarray:
    """Sorted indices of all entries of rewards (K,) attaining the exact maximum. Shape (n_tied,)."""
    rewards = np.asarray(rewards)
    return np.flatnonzero(rewards == rewards.max())


def has_max_tie(theta: np.ndarray, options: np.ndarray) -> bool:
    """True if more than one of options (K, 4) attains the maximum reward under theta (4,)."""
    return len(tied_best(rewards_int(options, theta))) > 1


def user_choice(theta: np.ndarray, options: np.ndarray, rng: np.random.Generator) -> int:
    """Index of the option (K, 4) the user with theta (4,) picks: argmax reward, ties uniformly at random."""
    best = tied_best(rewards_int(options, theta))
    return int(best[rng.integers(len(best))])


def sample_option_set(rng: np.random.Generator) -> np.ndarray:
    """N_OPTIONS_PER_SET distinct flights drawn uniformly from all_flights(). Shape (3, 4) float64.

    Sampled WITHOUT replacement within a set: none of the 3120 interaction option sets
    (data/eval/interaction/flight.jsonl) or the 100 heldout sets (data/eval/heldout/flight.json)
    contains the same flight twice. With replacement about 2.4 duplicate sets would be expected
    across those 3220 sets, so this is weak evidence (P(0 duplicates) is about 0.09).
    """
    flights = all_flights()
    idx = rng.choice(len(flights), size=config.N_OPTIONS_PER_SET, replace=False)
    return flights[idx]


def sample_option_set_no_tie(rng: np.random.Generator, theta: np.ndarray, max_tries: int = 1000) -> np.ndarray:
    """Like sample_option_set, resampled until theta (4,) has a unique best option. Shape (3, 4).

    Raises RuntimeError if no tie-free set was found within max_tries draws.
    """
    for _ in range(max_tries):
        options = sample_option_set(rng)
        if not has_max_tie(theta, options):
            return options
    raise RuntimeError(f"no option set without a max tie for theta={theta!r} after {max_tries} tries")
