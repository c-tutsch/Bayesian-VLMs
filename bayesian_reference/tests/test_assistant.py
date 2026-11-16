"""Tests for bayes_ref.assistant."""

import numpy as np
import pytest

from bayes_ref import config
from bayes_ref.assistant import BayesianAssistant, EmptyPosteriorError, likelihood_weights
from bayes_ref.choice import sample_option_set, sample_option_set_no_tie, user_choice
from bayes_ref.world import build_theta

RULES = config.TIE_RULE_CHOICES


def _departure_set(departures: tuple[float, ...]) -> np.ndarray:
    return np.array([[d, 0.5, 0.5, 0.5] for d in departures])


def _true_index(Theta: np.ndarray, theta: np.ndarray) -> int:
    return int(np.flatnonzero(np.all(Theta == theta, axis=1))[0])


@pytest.mark.parametrize(
    "rule, n_consistent, weight_sum",
    [("simple", 374, 374), ("strict", 250, 250), ("exact", 374, 1748)],
)
def test_hand_example(rule, n_consistent, weight_sum):
    a = BayesianAssistant(rule)
    a.update(_departure_set((0.2, 0.5, 0.8)), 2)
    assert a.n_consistent == n_consistent
    assert a.weights.sum() == weight_sum

    Theta = build_theta()
    dep_pos = Theta[:, 0] > 0
    dep_zero = Theta[:, 0] == 0
    w = a.weights
    assert np.all(w[dep_pos] == (6 if rule == "exact" else 1))
    assert np.all(w[dep_zero] == {"simple": 1, "strict": 0, "exact": 2}[rule])
    assert np.all(w[Theta[:, 0] < 0] == 0)

    fresh = _departure_set((0.9, 0.1, 0.4))
    s = a.scores(fresh)
    assert s.dtype == np.int64
    assert s[0] > s[2] > s[1] > 0
    np.testing.assert_array_equal(a.candidates(fresh), [0])
    assert a.recommend(fresh, np.random.default_rng(0)) == 0


def test_round0_uniform():
    a = BayesianAssistant()
    options = sample_option_set(np.random.default_rng(5))
    np.testing.assert_array_equal(a.scores(options), [0, 0, 0])
    np.testing.assert_array_equal(a.candidates(options), [0, 1, 2])
    rng = np.random.default_rng(config.DEFAULT_SEED)
    n = 3000
    counts = np.bincount([a.recommend(options, rng) for _ in range(n)], minlength=3)
    assert np.all((0.30 * n <= counts) & (counts <= 0.37 * n)), counts


@pytest.mark.parametrize("rule", RULES)
def test_posterior_and_theta_hat(rule):
    rng = np.random.default_rng(11)
    Theta = build_theta()
    theta = Theta[rng.integers(len(Theta))]
    a = BayesianAssistant(rule, Theta)
    for _ in range(3):
        options = sample_option_set_no_tie(rng, theta)
        a.update(options, user_choice(theta, options, rng))
        p = a.posterior()
        assert p.dtype == np.float64 and p.shape == (len(Theta),)
        assert abs(p.sum() - 1.0) < 1e-12
        if rule == "exact":
            np.testing.assert_allclose(p, a.weights / a.weights.sum(), rtol=0, atol=1e-15)
        np.testing.assert_allclose(a.theta_hat(), p @ Theta, rtol=0, atol=1e-12)


@pytest.mark.parametrize("rule", RULES)
def test_true_theta_stays_consistent(rule):
    rng = np.random.default_rng(21)
    Theta = build_theta()
    for u in rng.choice(len(Theta), size=200, replace=False):
        theta = Theta[u]
        a = BayesianAssistant(rule, Theta)
        prev = a.n_consistent
        for _ in range(config.N_ROUNDS):
            options = sample_option_set_no_tie(rng, theta) if rule == "strict" else sample_option_set(rng)
            a.update(options, user_choice(theta, options, rng))
            assert a.weights[u] > 0, f"user {theta} lost under {rule}"
            if rule in ("simple", "strict"):
                assert a.n_consistent <= prev
            prev = a.n_consistent


def _recommendation_sequence(seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    Theta = build_theta()
    theta = Theta[rng.integers(len(Theta))]
    a = BayesianAssistant(None, Theta)
    recs = []
    for _ in range(config.N_ROUNDS):
        options = sample_option_set(rng)
        recs.append(a.recommend(options, rng))
        a.update(options, user_choice(theta, options, rng))
    return recs


def test_same_seed_same_recommendations():
    for seed in range(20):
        assert _recommendation_sequence(seed) == _recommendation_sequence(seed)


def test_unknown_rule():
    with pytest.raises(ValueError):
        likelihood_weights(_departure_set((0.2, 0.8)), 0, build_theta(), "bogus")
    with pytest.raises(ValueError):
        BayesianAssistant("bogus")


def test_empty_posterior_strict():
    options = _departure_set((0.2, 0.8))
    a = BayesianAssistant("strict")
    a.update(options, 0)  # only departure weight < 0 survives
    with pytest.raises(EmptyPosteriorError, match="strict"):
        a.update(options, 1)  # needs departure weight > 0
    assert a.n_consistent == 250  # state unchanged after the failed update


@pytest.mark.parametrize("rule", ["simple", "exact"])
def test_contradictory_departure_only_keeps_zero_departure(rule):
    """With departure-only options the 124 thetas with zero departure weight tie and survive both observations."""
    options = _departure_set((0.2, 0.8))
    a = BayesianAssistant(rule)
    a.update(options, 0)
    a.update(options, 1)
    assert a.n_consistent == 124
    assert np.all(build_theta()[a.weights > 0, 0] == 0)
