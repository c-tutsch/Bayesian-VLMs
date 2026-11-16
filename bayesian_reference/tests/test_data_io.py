"""Tests for bayes_ref.data_io against the real Zenodo files (skipped when ./data is missing)."""

import json
from pathlib import Path

import numpy as np
import pytest

from bayes_ref import config
from bayes_ref.data_io import load_eval_interactions, load_heldout
from bayes_ref.world import all_flights, build_theta, text_to_features

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


def _strip_flight_prefix(text: str) -> str:
    return text.split(": ", 1)[1]


def _assert_on_flight_grid(name: str, options: np.ndarray) -> None:
    flat = options.reshape(-1, config.N_FEATURES)
    flights = all_flights()
    close = np.all(np.isclose(flat[:, None, :], flights[None, :, :], rtol=0.0, atol=1e-9), axis=2)
    n_matches = close.sum(axis=1)
    assert np.all(n_matches == 1), f"{name}: {np.sum(n_matches != 1)} option vectors not matching exactly one flight"
    nearest = flights[close.argmax(axis=1)]
    max_dev = np.abs(flat - nearest).max()
    print(f"\n[{name}] {len(flat)} option vectors on the all_flights() grid, max |deviation| = {max_dev:.3e}")
    assert max_dev < 1e-9


@needs_interaction
def test_interaction_shapes(interaction):
    assert interaction["reward_fn"].shape == (624, 4)
    assert interaction["options"].shape == (624, 5, 3, 4)
    assert interaction["user_idx"].shape == (624, 5)
    assert interaction["idx"].shape == (624,)
    assert np.all((interaction["user_idx"] >= 0) & (interaction["user_idx"] < 3))


@needs_heldout
def test_heldout_shapes(heldout):
    assert heldout["options"].shape == (100, 3, 4)
    assert heldout["reward_fn"].shape == (624, 4)
    assert heldout["user_idxs"].shape == (624, 100)
    assert np.all((heldout["user_idxs"] >= 0) & (heldout["user_idxs"] < 3))


@needs_interaction
def test_a_reward_fns_equal_build_theta(interaction):
    rows = {tuple(r) for r in interaction["reward_fn"]}
    assert len(rows) == 624
    assert rows == {tuple(r) for r in build_theta()}


@needs_interaction
def test_b_interaction_options_on_grid(interaction):
    _assert_on_flight_grid("interaction", interaction["options"])


@needs_heldout
def test_b_heldout_options_on_grid(heldout):
    _assert_on_flight_grid("heldout", heldout["options"])


@needs_interaction
def test_c_interaction_text_matches_numeric():
    n = 0
    with open(INTERACTION, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for rnd, rnd_num in zip(rec["rounds"], rec["rounds_numpy"], strict=True):
                for text, num in zip(rnd["options"], rnd_num, strict=True):
                    np.testing.assert_array_equal(text_to_features(text), num, err_msg=text)
                    np.testing.assert_array_equal(text_to_features(_strip_flight_prefix(text)), num, err_msg=text)
                    n += 1
    assert n == 624 * 5 * 3


@needs_heldout
def test_c_heldout_text_matches_numeric():
    obj = json.loads(HELDOUT.read_text(encoding="utf-8"))
    n = 0
    for texts, nums in zip(obj["all_options"], obj["all_options_numpy"], strict=True):
        for text, num in zip(texts, nums, strict=True):
            np.testing.assert_array_equal(text_to_features(text), num, err_msg=text)
            n += 1
    assert n == 100 * 3


@needs_interaction
def test_d_interaction_choice_is_argmax(interaction):
    rewards = np.einsum("urkd,ud->urk", interaction["options"], interaction["reward_fn"])
    np.testing.assert_array_equal(rewards.argmax(axis=2), interaction["user_idx"])


@needs_heldout
def test_e_heldout_choice_is_an_argmax(heldout):
    rewards = np.einsum("skd,ud->usk", heldout["options"], heldout["reward_fn"])
    best = rewards.max(axis=2, keepdims=True)
    is_max = np.isclose(rewards, best, rtol=0.0, atol=1e-9)
    chosen = np.take_along_axis(is_max, heldout["user_idxs"][:, :, None], axis=2)[:, :, 0]
    assert chosen.all(), f"{np.sum(~chosen)} heldout choices are not an argmax"

    n_total = chosen.size
    n_first_argmax = int(np.sum(rewards.argmax(axis=2) == heldout["user_idxs"]))
    n_ties = int(np.sum(is_max.sum(axis=2) > 1))
    print(
        f"\n[heldout] {n_first_argmax}/{n_total} choices equal np.argmax of float64 options @ reward_fn; "
        f"{n_ties}/{n_total} cases have a max-tie (atol 1e-9)"
    )
