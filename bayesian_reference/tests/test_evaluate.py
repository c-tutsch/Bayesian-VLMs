"""Tests for bayes_ref.evaluate and bayes_ref.oracle (data tests skipped when ./data is missing)."""

import json
from pathlib import Path

import numpy as np
import pytest

from bayes_ref import config
from bayes_ref.data_io import load_eval_interactions, load_heldout
from bayes_ref.evaluate import (
    agreement,
    agreement_curve,
    bayesian_factory,
    candidate_sets,
    evaluate_predictions,
    expected_success,
    heldout_labels,
    in_candidates,
    oracle_factory,
    read_predictions,
    sampled_recs,
    success_rate_curve,
    write_predictions,
)
from bayes_ref.oracle import OracleAssistant

ROOT = Path(__file__).resolve().parent.parent
INTERACTION = ROOT / config.EVAL_INTERACTION_PATH
HELDOUT = ROOT / config.EVAL_HELDOUT_PATH
needs_data = pytest.mark.skipif(
    not (INTERACTION.exists() and HELDOUT.exists()), reason=f"missing {INTERACTION} or {HELDOUT}"
)
FULL_SHAPE = (config.N_USERS_EXPECTED, len(config.EVAL_OBS_ROUNDS), config.N_FRESH_SETS)

T, F = True, False


def _hand_cands() -> np.ndarray:
    """(N=1, R=2, S=2, K=3) candidate sets."""
    return np.array([[[[T, T, T], [F, T, F]], [[T, F, T], [F, T, T]]]])


# ---------------------------------------------------------------- synthetic hand tests


def test_expected_success_hand():
    labels = np.array([[1, 2]])
    np.testing.assert_array_equal(expected_success(_hand_cands(), labels), [[[1 / 3, 0.0], [0.0, 0.5]]])


def test_agreement_hand():
    recs = np.array([[[2, 1], [1, 2]]])
    np.testing.assert_array_equal(agreement(recs, _hand_cands()), [[[1 / 3, 1.0], [0.0, 0.5]]])
    np.testing.assert_array_equal(in_candidates(recs, _hand_cands()), [[[T, T], [F, T]]])


def test_empty_candidate_set_raises():
    cands = _hand_cands()
    cands[0, 1, 1] = False
    with pytest.raises(ValueError):
        expected_success(cands, np.array([[0, 0]]))
    with pytest.raises(ValueError):
        sampled_recs(cands, np.random.default_rng(0))


def test_sampled_recs_hand():
    n = 6000
    cands = np.zeros((n, 3, 3), dtype=bool)
    cands[:, 0] = [F, F, T]
    cands[:, 1] = [T, F, T]
    cands[:, 2] = [T, T, T]
    recs = sampled_recs(cands, np.random.default_rng(3))
    assert recs.shape == (n, 3) and recs.dtype == np.int64
    assert np.all(recs[:, 0] == 2)
    assert set(np.unique(recs[:, 1])) == {0, 2}
    frac0 = np.mean(recs[:, 1] == 0)
    counts = np.bincount(recs[:, 2], minlength=3) / n
    print(f"\n[sampled_recs] P(0 | {{0,2}}) = {frac0:.4f}, freqs over {{0,1,2}} = {counts.round(4).tolist()}")
    assert abs(frac0 - 0.5) < 0.03
    assert np.all(np.abs(counts - 1 / 3) < 0.03)
    np.testing.assert_array_equal(recs, sampled_recs(cands, np.random.default_rng(3)))
    assert not np.array_equal(recs, sampled_recs(cands, np.random.default_rng(4)))


def test_success_rate_curve_hand():
    values = np.array([[[1.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [1.0, 0.0]]])  # (N=2, R=2, S=2)
    c = success_rate_curve(values)
    np.testing.assert_allclose(c["mean"], [0.25, 0.75])
    per_user = np.array([[0.5, 1.0], [0.0, 0.5]])
    np.testing.assert_allclose(c["se"], per_user.std(axis=0, ddof=config.SE_DDOF) / np.sqrt(2))
    np.testing.assert_array_equal(agreement_curve(values)["mean"], c["mean"])


def test_oracle_hand():
    options = np.array([[0.2, 0.5, 0.5, 0.3], [0.2, 0.1, 0.5, 0.3], [0.9, 0.0, 0.0, 0.3]])
    o = OracleAssistant(np.array([0.0, -1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(o.scores(options), [-10, -2, 0])
    np.testing.assert_array_equal(o.candidates(options), [2])
    o_tie = OracleAssistant(np.array([0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(o_tie.candidates(options), [0, 1, 2])
    rng = np.random.default_rng(0)
    assert {o_tie.recommend(options, rng) for _ in range(100)} == {0, 1, 2}


# ---------------------------------------------------------------- prediction files


@pytest.fixture(scope="module")
def random_pred_file(tmp_path_factory):
    recs = np.random.default_rng(7).integers(0, config.N_OPTIONS_PER_SET, size=FULL_SHAPE)
    path = tmp_path_factory.mktemp("pred") / "random.jsonl"
    write_predictions(recs, path)
    return recs, path


def test_predictions_round_trip(random_pred_file):
    recs, path = random_pred_file
    back = read_predictions(path)
    assert back.shape == FULL_SHAPE and back.dtype == np.int64
    np.testing.assert_array_equal(back, recs)
    first = json.loads(path.read_text(encoding="utf-8").split("\n", 1)[0])
    assert first == {"user": 0, "n_obs": 0, "set": 0, "rec": int(recs[0, 0, 0])}


def test_read_predictions_deleted_line(random_pred_file, tmp_path):
    _, path = random_pred_file
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    bad = tmp_path / "deleted.jsonl"
    bad.write_text("".join(lines[:1234] + lines[1235:]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        read_predictions(bad)


def test_read_predictions_duplicated_line(random_pred_file, tmp_path):
    _, path = random_pred_file
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    bad = tmp_path / "duplicated.jsonl"
    bad.write_text("".join(lines[:1235] + [lines[1234]] + lines[1235:]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_predictions(bad)


@pytest.mark.parametrize(
    "line",
    [
        '{"user": 0, "n_obs": 0, "set": 0, "rec": 3}',
        '{"user": 624, "n_obs": 0, "set": 0, "rec": 0}',
        '{"user": 0, "n_obs": 6, "set": 0, "rec": 0}',
        '{"user": 0, "n_obs": 0, "set": 100, "rec": 0}',
        '{"user": 0, "n_obs": 0, "set": 0, "rec": 1.0}',
        '{"user": 0, "n_obs": 0, "set": 0}',
    ],
)
def test_read_predictions_bad_record(tmp_path, line):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_predictions(bad)


# ---------------------------------------------------------------- real eval files


@pytest.fixture(scope="module")
def data():
    interaction = load_eval_interactions(INTERACTION)
    heldout = load_heldout(HELDOUT)
    return interaction, heldout, heldout_labels(interaction, heldout)


@pytest.fixture(scope="module")
def bayes_cands(data):
    interaction, heldout, _ = data
    return candidate_sets(bayesian_factory, interaction, heldout)


@pytest.fixture(scope="module")
def bayes_pred_file(bayes_cands, tmp_path_factory):
    path = tmp_path_factory.mktemp("bayes") / "bayes_seed0.jsonl"
    write_predictions(sampled_recs(bayes_cands, np.random.default_rng(0)), path)
    return path


@needs_data
def test_oracle_labels_in_candidates(data):
    interaction, heldout, labels = data
    cands = candidate_sets(oracle_factory, interaction, heldout)
    assert cands.shape == (*FULL_SHAPE, config.N_OPTIONS_PER_SET)
    inside = in_candidates(labels, cands)
    n_ties = int((cands.sum(-1) > 1).sum())
    exp = success_rate_curve(expected_success(cands, labels))["mean"]
    print(f"\n[oracle] labels in candidate set: {int(inside.sum())}/{inside.size}; "
          f"tied cells: {n_ties}; expected success under random tie-breaking: {exp.round(4).tolist()}")
    assert inside.all()


@needs_data
def test_bayesian_success_curve(data, bayes_cands):
    _, _, labels = data
    mean = success_rate_curve(expected_success(bayes_cands, labels))["mean"]
    print(f"\n[bayes {config.TIE_RULE_LIKELIHOOD}] expected success per n_obs: {mean.round(4).tolist()}")
    assert abs(mean[0] - 1 / 3) < 0.01
    assert np.all(np.diff(mean) >= 0)
    assert 0.55 <= mean[1] <= 0.60
    assert 0.78 <= mean[-1] <= 0.83


@needs_data
def test_evaluate_example_bayes_predictions(data, bayes_cands, bayes_pred_file):
    interaction, heldout, _ = data
    ev = evaluate_predictions(bayes_pred_file, interaction, heldout)
    agree = ev["agreement"]["mean"]
    n_ties = (bayes_cands.sum(-1) > 1).sum(axis=(0, 2))
    print(f"\n[example bayes predictions] agreement per n_obs: {agree.round(4).tolist()}")
    print(f"[example bayes predictions] tied Bayesian cells per n_obs: {n_ties.tolist()}")
    print(f"[example bayes predictions] success per n_obs: {ev['success']['mean'].round(4).tolist()}")
    # n_obs = 0: uniform prior, every set is a 3-way tie, so expected agreement is exactly 1/3.
    assert np.all(bayes_cands[:, 0])
    assert agree[0] == pytest.approx(1 / 3, abs=1e-12)
    assert np.all((agree[1:] >= 0.9) & (agree[1:] <= 1.0))
    assert np.all(agree[n_ties == 0] == 1.0)
    recs = read_predictions(bayes_pred_file)
    assert in_candidates(recs, bayes_cands).all()


@needs_data
def test_same_seed_same_prediction_file(bayes_cands, bayes_pred_file, tmp_path):
    again = tmp_path / "again.jsonl"
    write_predictions(sampled_recs(bayes_cands, np.random.default_rng(0)), again)
    assert again.read_bytes() == bayes_pred_file.read_bytes()
    other = tmp_path / "other.jsonl"
    write_predictions(sampled_recs(bayes_cands, np.random.default_rng(1)), other)
    assert other.read_bytes() != bayes_pred_file.read_bytes()
