"""Smoke tests for the Step 0 project skeleton."""

import numpy as np

from bayes_ref import config
from bayes_ref.rng import make_rng, spawn_rngs


def test_config_constants_consistent():
    assert len(config.THETA_VALUES) ** config.N_FEATURES - 1 == config.N_USERS_EXPECTED
    assert len(config.STOPS_VALUES) == config.N_LEVELS_STOPS
    assert len(config.FEATURE_NAMES) == config.N_FEATURES
    assert config.TIE_RULE_LIKELIHOOD in ("simple", "exact")
    assert all(0.0 <= v <= 1.0 for v in config.STOPS_VALUES)
    assert all(0.0 < v < 1.0 for v in config.TARGET_ACCURACY.values())
    assert max(config.TARGET_ACCURACY) <= config.N_ROUNDS


def test_make_rng_reproducible():
    a = make_rng(42).random(5)
    b = make_rng(42).random(5)
    np.testing.assert_array_equal(a, b)


def test_make_rng_different_seeds_differ():
    assert not np.array_equal(make_rng(1).random(5), make_rng(2).random(5))


def test_spawn_rngs_independent_and_reproducible():
    rngs = spawn_rngs(0, 3)
    assert len(rngs) == 3
    assert all(isinstance(r, np.random.Generator) for r in rngs)
    first = [r.random() for r in rngs]
    assert len(set(first)) == 3

    again = [r.random() for r in spawn_rngs(0, 3)]
    assert first == again


def test_package_imports():
    import bayes_ref

    assert bayes_ref.__doc__
