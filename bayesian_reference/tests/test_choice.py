"""Tests for bayes_ref.choice (data checks are skipped when ./data is missing)."""

import math
from pathlib import Path

import numpy as np
import pytest

from bayes_ref import config
from bayes_ref.choice import (
    has_max_tie,
    rewards_int,
    rewards_int_all,
    sample_option_set,
    sample_option_set_no_tie,
    tied_best,
    to_int_features,
    to_int_theta,
    user_choice,
)
from bayes_ref.data_io import load_eval_interactions, load_heldout
from bayes_ref.world import all_flights, build_theta

ROOT = Path(__file__).resolve().parent.parent
INTERACTION = ROOT / config.EVAL_INTERACTION_PATH
HELDOUT = ROOT / config.EVAL_HELDOUT_PATH

needs_interaction = pytest.mark.skipif(not INTERACTION.exists(), reason=f"missing {INTERACTION}")
needs_heldout = pytest.mark.skipif(not HELDOUT.exists(), reason=f"missing {HELDOUT}")


@pytest.fixture(scope="module")
def interaction():
    return load_eval_interactions(INTERACTION)


@pytest.fixture(scope="module")
def heldout():
    return load_heldout(HELDOUT)


def _n_sets_with_duplicate(option_sets: np.ndarray) -> int:
    ints = to_int_features(option_sets)
    return sum(len({tuple(row) for row in s}) < len(s) for s in ints)


def _chi2_sf_even_df(x: float, df: int) -> float:
    """Chi-square survival function, closed form valid for even df only."""
    assert df % 2 == 0 and df > 0
    half = x / 2.0
    return math.exp(-half) * sum(half**i / math.factorial(i) for i in range(df // 2))


def test_tie_rule_config_consistent():
    assert config.TIE_RULE_CHOICES == ("simple", "strict", "exact")
    assert config.TIE_RULE_LIKELIHOOD in config.TIE_RULE_CHOICES


def test_int_helpers():
    np.testing.assert_array_equal(to_int_features(np.linspace(0, 1, 11)), np.arange(11))
    np.testing.assert_array_equal(to_int_features(np.array(config.STOPS_VALUES)), [0, 5, 10])
    np.testing.assert_array_equal(to_int_theta(np.array(config.THETA_VALUES)), [-2, -1, 0, 1, 2])
    assert to_int_features(np.zeros((2, 3, 4))).shape == (2, 3, 4)
    assert to_int_features(np.zeros(4)).dtype == np.int64
    assert to_int_theta(np.zeros(4)).dtype == np.int64


@pytest.mark.parametrize("bad", [0.123, 0.05, 1.0 + 1e-6])
def test_to_int_features_rejects_off_grid(bad):
    with pytest.raises(ValueError):
        to_int_features(np.array([0.2, bad, 0.0, 0.5]))


def test_to_int_theta_rejects_off_grid():
    with pytest.raises(ValueError):
        to_int_theta(np.array([0.25, 0.0, 0.0, 0.0]))


def test_hand_example():
    options = np.array([
        [0.2, 0.9, 0.0, 0.5],
        [0.8, 0.3, 0.5, 0.1],
        [0.5, 0.5, 1.0, 0.7],
    ])
    theta = np.array([-1.0, 0.0, 1.0, 0.5])
    np.testing.assert_allclose(options @ theta, [0.05, -0.25, 0.85], atol=1e-12)
    r = rewards_int(options, theta)
    assert r.dtype == np.int64
    np.testing.assert_array_equal(r, [1, -5, 17])
    np.testing.assert_array_equal(tied_best(r), [2])
    assert not has_max_tie(theta, options)
    assert user_choice(theta, options, np.random.default_rng(0)) == 2


def test_hand_tie_example():
    options = np.array([
        [0.5, 0.1, 0.0, 0.9],
        [0.5, 0.7, 1.0, 0.2],
        [0.2, 0.4, 0.5, 0.6],
    ])
    theta = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(tied_best(rewards_int(options, theta)), [0, 1])
    assert has_max_tie(theta, options)

    rng = np.random.default_rng(config.DEFAULT_SEED)
    n = 2000
    counts = np.bincount([user_choice(theta, options, rng) for _ in range(n)], minlength=3)
    assert counts[2] == 0
    assert 0.45 * n <= counts[0] <= 0.55 * n
    assert 0.45 * n <= counts[1] <= 0.55 * n


def test_rewards_int_all_matches_rewards_int():
    rng = np.random.default_rng(1)
    theta_all = build_theta()
    for _ in range(50):
        options = sample_option_set(rng)
        users = rng.choice(len(theta_all), size=50, replace=False)
        Theta = theta_all[users]
        R = rewards_int_all(options, Theta)
        assert R.shape == (3, 50) and R.dtype == np.int64
        for u in range(len(Theta)):
            np.testing.assert_array_equal(R[:, u], rewards_int(options, Theta[u]))


def test_sample_option_set():
    flights = {tuple(r) for r in all_flights()}
    rng = np.random.default_rng(2)
    for _ in range(200):
        options = sample_option_set(rng)
        assert options.shape == (3, 4) and options.dtype == np.float64
        assert all(tuple(r) in flights for r in options)
        assert len({tuple(r) for r in options}) == 3
    a = [sample_option_set(np.random.default_rng(7)) for _ in range(2)]
    np.testing.assert_array_equal(a[0], a[1])
    rng1, rng2 = np.random.default_rng(8), np.random.default_rng(8)
    for _ in range(20):
        np.testing.assert_array_equal(sample_option_set(rng1), sample_option_set(rng2))


def test_sample_option_set_no_tie():
    rng = np.random.default_rng(3)
    theta_all = build_theta()
    for u in rng.choice(len(theta_all), size=200, replace=False):
        options = sample_option_set_no_tie(rng, theta_all[u])
        assert options.shape == (3, 4)
        assert not has_max_tie(theta_all[u], options)


def test_sample_option_set_no_tie_gives_up():
    with pytest.raises(RuntimeError):
        sample_option_set_no_tie(np.random.default_rng(0), np.array([1.0, 0.0, 0.0, 0.0]), max_tries=0)


@needs_interaction
def test_a_interaction_unique_int_argmax_equals_user_idx(interaction):
    n_users, n_rounds = interaction["user_idx"].shape
    for u in range(n_users):
        for t in range(n_rounds):
            best = tied_best(rewards_int(interaction["options"][u, t], interaction["reward_fn"][u]))
            assert len(best) == 1, f"user {u} round {t}: max tie {best}"
            assert best[0] == interaction["user_idx"][u, t], f"user {u} round {t}"


@needs_heldout
def test_b_heldout_choice_in_tied_best(heldout):
    n_users, n_sets = heldout["user_idxs"].shape
    n_ties = n_first = n_last = n_middle = 0
    tie_sizes = {}
    for s in range(n_sets):
        R = rewards_int_all(heldout["options"][s], heldout["reward_fn"])
        for u in range(n_users):
            best = tied_best(R[:, u])
            chosen = heldout["user_idxs"][u, s]
            assert chosen in best, f"user {u} set {s}: chose {chosen}, tied_best {best}"
            if len(best) > 1:
                n_ties += 1
                tie_sizes[len(best)] = tie_sizes.get(len(best), 0) + 1
                if chosen == best[0]:
                    n_first += 1
                elif chosen == best[-1]:
                    n_last += 1
                else:
                    n_middle += 1
    print(
        f"\n[heldout] {n_users * n_sets} cases, {n_ties} with a max tie (expect 3315); tie sizes {tie_sizes}; "
        f"stored choice is first of tied_best {n_first}, last {n_last}, middle {n_middle}"
    )


@needs_interaction
@needs_heldout
def test_c_duplicate_flights_in_option_sets(interaction, heldout):
    inter_sets = interaction["options"].reshape(-1, config.N_OPTIONS_PER_SET, config.N_FEATURES)
    print(
        f"\n[duplicates] interaction: {_n_sets_with_duplicate(inter_sets)}/{len(inter_sets)} sets with a repeated flight; "
        f"heldout: {_n_sets_with_duplicate(heldout['options'])}/{len(heldout['options'])}"
    )


@needs_interaction
def test_c_interaction_feature_level_frequencies(interaction):
    flat = to_int_features(interaction["options"].reshape(-1, config.N_FEATURES))
    print(f"\n[levels] {len(flat)} options in the interaction file")
    for j, name in enumerate(config.FEATURE_NAMES):
        levels = np.array([0, 5, 10]) if name == "stops" else np.arange(11)
        counts = np.array([np.sum(flat[:, j] == v) for v in levels])
        assert counts.sum() == len(flat)
        expected = len(flat) / len(levels)
        chi2 = float(np.sum((counts - expected) ** 2 / expected))
        p = _chi2_sf_even_df(chi2, len(levels) - 1)
        print(f"  {name:15s} counts {counts.tolist()}  chi2={chi2:.2f} df={len(levels) - 1} p={p:.3f}")
