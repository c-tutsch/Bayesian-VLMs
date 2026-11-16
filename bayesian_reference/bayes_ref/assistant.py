"""The Bayesian Assistant: indicator likelihood over the 624 thetas, exact integer posterior weights."""

import numpy as np

from bayes_ref import config
from bayes_ref.choice import rewards_int_all, tied_best, to_int_features, to_int_theta
from bayes_ref.world import build_theta


class EmptyPosteriorError(RuntimeError):
    """Raised when no theta is consistent with the observations under the current tie rule."""


def likelihood_weights(options: np.ndarray, chosen: int, Theta: np.ndarray, rule: str) -> np.ndarray:
    """Integer likelihood of observing `chosen` among options (K, 4) under each theta in Theta (U, 4). Shape (U,) int64.

    For each theta, m = max reward over the options and k = number of options attaining m.
    'simple': 1 if the chosen option attains m, else 0.
    'strict': 1 if the chosen option attains m and k == 1, else 0.
    'exact':  WEIGHT_SCALE // k if the chosen option attains m, else 0 (proportional to 1/k).
    """
    if rule not in config.TIE_RULE_CHOICES:
        raise ValueError(f"unknown tie rule {rule!r}, expected one of {config.TIE_RULE_CHOICES}")
    rewards = rewards_int_all(options, Theta)  # (K, U)
    m = rewards.max(axis=0)
    k = (rewards == m).sum(axis=0)
    attains = rewards[chosen] == m
    if rule == "simple":
        w = attains
    elif rule == "strict":
        w = attains & (k == 1)
    else:
        return np.where(attains, config.WEIGHT_SCALE // k, 0).astype(np.int64)
    return w.astype(np.int64)


class BayesianAssistant:
    """Uniform prior over Theta (U, 4); posterior kept as exact integer weights w (U,) int64."""

    def __init__(self, tie_rule: str | None = None, Theta: np.ndarray | None = None) -> None:
        self.tie_rule = config.TIE_RULE_LIKELIHOOD if tie_rule is None else tie_rule
        if self.tie_rule not in config.TIE_RULE_CHOICES:
            raise ValueError(f"unknown tie rule {self.tie_rule!r}, expected one of {config.TIE_RULE_CHOICES}")
        self.Theta = build_theta() if Theta is None else np.asarray(Theta, dtype=np.float64)
        self._theta_int = to_int_theta(self.Theta)  # (U, 4)
        self.reset()

    def reset(self) -> None:
        """Back to the uniform prior: w = ones (U,)."""
        self._w = np.ones(len(self.Theta), dtype=np.int64)

    def update(self, options: np.ndarray, chosen: int) -> None:
        """Multiply w by likelihood_weights(options (K, 4), chosen). Raises EmptyPosteriorError if w becomes all 0."""
        w = self._w * likelihood_weights(options, chosen, self.Theta, self.tie_rule)
        if not w.any():
            raise EmptyPosteriorError(
                f"posterior is empty under tie rule {self.tie_rule!r} after observing choice {chosen} "
                f"among options {np.asarray(options).tolist()}"
            )
        self._w = w

    @property
    def n_consistent(self) -> int:
        """Number of thetas with positive weight."""
        return int(np.count_nonzero(self._w > 0))

    @property
    def weights(self) -> np.ndarray:
        """Copy of the integer weights. Shape (U,) int64."""
        return self._w.copy()

    def posterior(self) -> np.ndarray:
        """Normalised posterior w / w.sum(). Shape (U,) float64."""
        return self._w / self._w.sum()

    def theta_hat(self) -> np.ndarray:
        """Posterior mean of theta in true units. Shape (4,) float64."""
        return (self._w @ self._theta_int) / (self._w.sum() * config.THETA_SCALE)

    def scores(self, options: np.ndarray) -> np.ndarray:
        """Exact integer scores of options (K, 4). Shape (K,) int64; a stack (..., K, 4) gives (..., K).

        Equals options @ theta_hat() times the positive factor FEATURE_SCALE * THETA_SCALE * w.sum(),
        so the argmax is identical to that of the posterior-mean reward, without float rounding.
        """
        return to_int_features(options) @ (self._w @ self._theta_int)

    def candidates(self, options: np.ndarray) -> np.ndarray:
        """Sorted indices of options (K, 4) attaining the max score; all K indices at round 0. Shape (n_tied,)."""
        return tied_best(self.scores(options))

    def recommend(self, options: np.ndarray, rng: np.random.Generator) -> int:
        """Index of the recommended option among options (K, 4): uniformly random among candidates()."""
        best = self.candidates(options)
        return int(best[rng.integers(len(best))])
