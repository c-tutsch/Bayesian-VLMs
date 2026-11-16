"""The oracle assistant: knows the user's true theta and recommends among the true tied-best options."""

import numpy as np

from bayes_ref.choice import rewards_int, tied_best


class OracleAssistant:
    """Scores options with the user's true theta (4,)."""

    def __init__(self, theta: np.ndarray) -> None:
        self.theta = np.asarray(theta, dtype=np.float64)

    def update(self, options: np.ndarray, chosen: int) -> None:
        """No-op: the oracle already knows theta. Present so it can be driven like BayesianAssistant."""

    def scores(self, options: np.ndarray) -> np.ndarray:
        """Exact integer rewards of options (K, 4) under theta. Shape (K,) int64; a stack (..., K, 4) gives (..., K)."""
        return rewards_int(options, self.theta)

    def candidates(self, options: np.ndarray) -> np.ndarray:
        """Sorted indices of options (K, 4) attaining the max true reward. Shape (n_tied,)."""
        return tied_best(self.scores(options))

    def recommend(self, options: np.ndarray, rng: np.random.Generator) -> int:
        """Index of the recommended option among options (K, 4): uniformly random among candidates()."""
        best = self.candidates(options)
        return int(best[rng.integers(len(best))])
