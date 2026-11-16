"""Tests for subset prediction files, "rec": null failures and shift_diagnostics (data tests skipped when ./data is missing)."""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from bayes_ref import config
from bayes_ref.data_io import load_eval_interactions, load_heldout
from bayes_ref.evaluate import (
    bayesian_factory,
    candidate_sets,
    evaluate_predictions,
    heldout_labels,
    in_candidates,
    oracle_factory,
    read_predictions,
    sampled_recs,
    shift_diagnostics,
    subset_candidate_sets,
    success_rate_curve,
    write_predictions,
)

ROOT = Path(__file__).resolve().parent.parent
INTERACTION = ROOT / config.EVAL_INTERACTION_PATH
HELDOUT = ROOT / config.EVAL_HELDOUT_PATH
needs_data = pytest.mark.skipif(
    not (INTERACTION.exists() and HELDOUT.exists()), reason=f"missing {INTERACTION} or {HELDOUT}"
)

USERS = [3, 10, 500]
N_OBS = (1, 5)
SETS = list(range(0, 100, 7))
SUB_SHAPE = (len(USERS), len(N_OBS), len(SETS))
FULL_LINES = config.N_USERS_EXPECTED * len(config.EVAL_OBS_ROUNDS) * config.N_FRESH_SETS


def _write_with_nulls(path: Path, recs: np.ndarray, failed: np.ndarray, users, n_obs, sets) -> None:
    """Like write_predictions, but writes "rec": null where failed is True."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for a, u in enumerate(users):
            for i, n in enumerate(n_obs):
                for b, k in enumerate(sets):
                    r = None if failed[a, i, b] else int(recs[a, i, b])
                    f.write(json.dumps({"user": u, "n_obs": n, "set": k, "rec": r}) + "\n")


@pytest.fixture
def sub_file(tmp_path):
    recs = np.random.default_rng(11).integers(0, config.N_OPTIONS_PER_SET, size=SUB_SHAPE)
    path = tmp_path / "sub.jsonl"
    write_predictions(recs, path, USERS, N_OBS, SETS)
    return recs, path


# ---------------------------------------------------------------- synthetic: subset files


def test_subset_round_trip(sub_file):
    recs, path = sub_file
    back = read_predictions(path, USERS, N_OBS, SETS)
    assert back.shape == SUB_SHAPE and back.dtype == np.int64
    np.testing.assert_array_equal(back, recs)
    lines = path.read_text(encoding="utf-8").split("\n")
    assert len(lines) == recs.size + 1 and lines[-1] == ""
    assert "\r" not in path.read_text(encoding="utf-8")


def test_subset_axes_hand(tmp_path):
    recs = np.zeros(SUB_SHAPE, dtype=np.int64)
    recs[2, 0, 1] = 2  # user 500, n_obs 1, set 7
    recs[0, 1, 14] = 1  # user 3, n_obs 5, set 98
    path = tmp_path / "hand.jsonl"
    write_predictions(recs, path, USERS, N_OBS, SETS)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    nonzero = [r for r in records if r["rec"] != 0]
    assert nonzero == [
        {"user": 3, "n_obs": 5, "set": 98, "rec": 1},
        {"user": 500, "n_obs": 1, "set": 7, "rec": 2},
    ]
    assert records[0] == {"user": 3, "n_obs": 1, "set": 0, "rec": 0}
    assert records[-1] == {"user": 500, "n_obs": 5, "set": 98, "rec": 0}
    keys = [(r["user"], r["n_obs"], r["set"]) for r in records]
    assert keys == sorted(keys)
    back = read_predictions(path, USERS, N_OBS, SETS)
    assert back[2, 0, 1] == 2 and back[0, 1, 14] == 1 and back.sum() == 3


def _mutate(lines: list[str], kind: str) -> list[str]:
    if kind == "missing":
        return lines[:5] + lines[6:]
    if kind == "duplicate":
        return lines[:6] + [lines[5]] + lines[6:]
    if kind == "outside":
        return lines + ['{"user": 4, "n_obs": 1, "set": 0, "rec": 0}\n']
    if kind == "outside_null":
        return lines + ['{"user": 3, "n_obs": 2, "set": 0, "rec": null}\n']
    if kind == "rec3":
        return lines[:5] + ['{"user": 3, "n_obs": 1, "set": 35, "rec": 3}\n'] + lines[6:]
    if kind == "rec-1":
        return lines[:5] + ['{"user": 3, "n_obs": 1, "set": 35, "rec": -1}\n'] + lines[6:]
    if kind == "missing_key":
        return lines[:5] + ['{"user": 3, "n_obs": 1, "set": 35}\n'] + lines[6:]
    if kind == "rec_str":
        return lines[:5] + ['{"user": 3, "n_obs": 1, "set": 35, "rec": "1"}\n'] + lines[6:]
    if kind == "rec_bool":
        return lines[:5] + ['{"user": 3, "n_obs": 1, "set": 35, "rec": true}\n'] + lines[6:]
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind, match",
    [
        ("missing", r"missing: 1 record\(s\), first user=3 n_obs=1 set=35"),
        ("duplicate", r"duplicate: 1 record\(s\), first line 7"),
        ("outside", r"outside declared subset: 1 record\(s\), first line 91"),
        ("outside_null", r"outside declared subset: 1 record\(s\)"),
        ("rec3", r"rec out of range: 1 record\(s\), first line 6"),
        ("rec-1", r"rec out of range: 1 record\(s\)"),
        ("missing_key", r"wrong keys: 1 record\(s\), first line 6"),
        ("rec_str", r"wrong types: 1 record\(s\)"),
        ("rec_bool", r"wrong types: 1 record\(s\)"),
    ],
)
def test_read_predictions_subset_raises(sub_file, tmp_path, kind, match):
    _, path = sub_file
    bad = tmp_path / f"{kind}.jsonl"
    bad.write_text("".join(_mutate(path.read_text(encoding="utf-8").splitlines(keepends=True), kind)), encoding="utf-8")
    with pytest.raises(ValueError, match=match) as e:
        read_predictions(bad, USERS, N_OBS, SETS)
    print(f"\n[{kind}] {e.value}")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"users": [10, 3]}, "sorted"),
        ({"users": [3, 3]}, "duplicate"),
        ({"users": [624]}, "out of range"),
        ({"users": [-1]}, "out of range"),
        ({"users": []}, "empty"),
        ({"n_obs": (5, 1)}, "sorted"),
        ({"n_obs": (6,)}, "out of range"),
        ({"sets": [0, 100]}, "out of range"),
        ({"sets": [1.0]}, "ints"),
        ({"users": "0-49"}, "sequence"),
    ],
)
def test_bad_subset_arguments(sub_file, tmp_path, kwargs, match):
    _, path = sub_file
    with pytest.raises(ValueError, match=match):
        read_predictions(path, **kwargs)
    with pytest.raises(ValueError, match=match):
        write_predictions(np.zeros((1, 1, 1), dtype=np.int64), tmp_path / "x.jsonl", **kwargs)
    with pytest.raises(ValueError, match=match):  # validated before any data is loaded
        evaluate_predictions(path, **kwargs)


def test_default_call_requires_full_file(sub_file):
    _, path = sub_file
    with pytest.raises(ValueError, match=rf"{FULL_LINES} records expected") as e:
        read_predictions(path)
    assert f"missing: {FULL_LINES - np.prod(SUB_SHAPE)} record(s)" in str(e.value)
    print(f"\n[default call on a subset file] {e.value}")


def test_write_predictions_wrong_shape(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        write_predictions(np.zeros((3, 2, 14), dtype=np.int64), tmp_path / "x.jsonl", USERS, N_OBS, SETS)


def test_read_nulls_synthetic(sub_file, tmp_path):
    recs, _ = sub_file
    failed = np.zeros(SUB_SHAPE, dtype=bool)
    failed[1, 0, 3] = failed[2, 1, 0] = True
    path = tmp_path / "nulls.jsonl"
    _write_with_nulls(path, recs, failed, USERS, N_OBS, SETS)
    back, back_failed = read_predictions(path, USERS, N_OBS, SETS, return_failed=True)
    np.testing.assert_array_equal(back_failed, failed)
    np.testing.assert_array_equal(back[~failed], recs[~failed])
    assert np.all(back[failed] == -1)


def test_success_rate_curve_one_user_nan_se():
    values = np.array([[[1.0, 0.0], [1.0, 1.0]]])  # (N=1, R=2, S=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = success_rate_curve(values)
    np.testing.assert_allclose(c["mean"], [0.5, 1.0])
    assert np.all(np.isnan(c["se"]))


def test_subset_same_recs_same_bytes(tmp_path):
    recs = np.random.default_rng(5).integers(0, 3, size=SUB_SHAPE)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_predictions(recs, a, USERS, N_OBS, SETS)
    write_predictions(np.random.default_rng(5).integers(0, 3, size=SUB_SHAPE), b, USERS, N_OBS, SETS)
    assert a.read_bytes() == b.read_bytes()


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
def bayes_recs(bayes_cands):
    return sampled_recs(bayes_cands, np.random.default_rng(0))


def _obs_pos(n_obs) -> list[int]:
    return [config.EVAL_OBS_ROUNDS.index(n) for n in n_obs]


@needs_data
def test_subset_candidate_sets_match_full(data, bayes_cands):
    interaction, heldout, _ = data
    sub = subset_candidate_sets(bayesian_factory, interaction, heldout, USERS, N_OBS, SETS)
    np.testing.assert_array_equal(sub, bayes_cands[np.ix_(USERS, _obs_pos(N_OBS), SETS)])


@needs_data
def test_bayes_subset_matches_full_arrays(data, bayes_cands, bayes_recs, tmp_path):
    interaction, heldout, labels = data
    users, n_obs = range(50), (1, 5)
    path = tmp_path / "bayes_sub.jsonl"
    write_predictions(bayes_recs[:50][:, _obs_pos(n_obs)], path, users, n_obs)
    ev = evaluate_predictions(path, interaction, heldout, users=users, n_obs=n_obs)

    hit = (bayes_recs == labels[:, None, :]).mean(axis=2)  # (N, R) per-user, full file
    agree = (in_candidates(bayes_recs, bayes_cands) / bayes_cands.sum(-1)).mean(axis=2)
    for key, per_user in (("success", hit), ("agreement", agree)):
        sub = per_user[:50][:, _obs_pos(n_obs)]
        np.testing.assert_allclose(ev[key]["mean"], sub.mean(axis=0), rtol=0, atol=1e-12)
        np.testing.assert_allclose(ev[key]["se"], sub.std(axis=0, ddof=config.SE_DDOF) / np.sqrt(50), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(ev["fail_rate"]["mean"], [0.0, 0.0])
    assert ev["n_obs"] == [1, 5] and ev["n_users"] == 50 and ev["n_sets"] == 100 and ev["n_lines"] == 50 * 2 * 100
    assert not ev["diagnostics"]["suspicious"]
    print(f"\n[bayes users 0-49, n_obs (1, 5)] success {ev['success']['mean'].round(4).tolist()} "
          f"agreement {ev['agreement']['mean'].round(4).tolist()}")


@needs_data
def test_one_user_nan_se(data, bayes_recs, tmp_path):
    interaction, heldout, _ = data
    path = tmp_path / "one.jsonl"
    write_predictions(bayes_recs[[7]][:, _obs_pos(N_OBS)], path, [7], N_OBS)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ev = evaluate_predictions(path, interaction, heldout, users=[7], n_obs=N_OBS)
    for key in ("success", "agreement", "fail_rate"):
        assert np.all(np.isnan(ev[key]["se"])) and np.all(np.isfinite(ev[key]["mean"]))
    assert np.isnan(ev["diagnostics"]["success_user_shift"])


@needs_data
def test_oracle_subset_file(data, tmp_path):
    interaction, heldout, _ = data
    users = range(20)
    cands = subset_candidate_sets(oracle_factory, interaction, heldout, users, N_OBS)
    path = tmp_path / "oracle.jsonl"
    write_predictions(sampled_recs(cands, np.random.default_rng(0)), path, users, N_OBS)
    ev = evaluate_predictions(path, interaction, heldout, users=users, n_obs=N_OBS)
    print(f"\n[oracle users 0-19] success {ev['success']['mean'].round(4).tolist()}")
    assert np.all((ev["success"]["mean"] >= 0.95) & (ev["success"]["mean"] <= 1.0))
    assert not ev["diagnostics"]["suspicious"]


@needs_data
def test_shift_diagnostics(data, bayes_recs, tmp_path):
    interaction, heldout, _ = data
    users, n_obs = range(100), (5,)
    good = bayes_recs[:100][:, _obs_pos(n_obs)]
    variants = {
        "as-is": good,
        "rec+1": (good + 1) % 3,
        "user shift": np.roll(good, -1, axis=0),  # row u holds the predictions for user u+1
    }
    diags = {}
    for name, recs in variants.items():
        path = tmp_path / f"{name}.jsonl"
        write_predictions(recs, path, users, n_obs)
        diags[name] = shift_diagnostics(path, users, n_obs, interaction=interaction, heldout=heldout)
        d = diags[name]
        print(f"\n[shift {name}] as-is {d['success']:.4f}, rec+1 {d['success_rec_plus_1']:.4f}, "
              f"rec-1 {d['success_rec_minus_1']:.4f}, user shift {d['success_user_shift']:.4f}, "
              f"suspicious {d['suspicious']} {d['reasons']}")
    assert not diags["as-is"]["suspicious"]
    assert diags["rec+1"]["suspicious"]
    assert diags["rec+1"]["success_rec_minus_1"] == pytest.approx(diags["as-is"]["success"], abs=1e-12)
    assert diags["user shift"]["suspicious"]
    assert diags["user shift"]["success_user_shift"] == pytest.approx(diags["as-is"]["success"], abs=1e-12)


@needs_data
def test_subset_same_seed_same_bytes(data, tmp_path):
    interaction, heldout, _ = data
    n_obs = (0, 1, 5)  # n_obs = 0 is all 3-way ties, so different seeds give different files
    paths = []
    for name, seed in (("a", 0), ("b", 0), ("c", 1)):
        cands = subset_candidate_sets(bayesian_factory, interaction, heldout, USERS, n_obs, SETS)
        paths.append(tmp_path / f"{name}.jsonl")
        write_predictions(sampled_recs(cands, np.random.default_rng(seed)), paths[-1], USERS, n_obs, SETS)
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert paths[0].read_bytes() != paths[2].read_bytes()


@needs_data
def test_ten_percent_nulls(data, bayes_cands, bayes_recs, tmp_path):
    interaction, heldout, labels = data
    users, n_obs = range(50), (1, 5)
    recs = bayes_recs[:50][:, _obs_pos(n_obs)]
    cands = bayes_cands[:50][:, _obs_pos(n_obs)]
    rng = np.random.default_rng(42)
    failed = np.zeros(recs.shape, dtype=bool)
    for i in range(len(n_obs)):  # exactly 10% of the (user, set) cells per n_obs
        flat = rng.choice(50 * 100, size=500, replace=False)
        failed[flat // 100, i, flat % 100] = True
    path = tmp_path / "nulls.jsonl"
    _write_with_nulls(path, recs, failed, users, n_obs, range(100))
    ev = evaluate_predictions(path, interaction, heldout, users=users, n_obs=n_obs)

    lab = labels[:50]
    exp_success = ((recs == lab[:, None, :]) & ~failed).mean(axis=2).mean(axis=0)
    exp_agree = np.where(failed, 0.0, in_candidates(recs, cands) / cands.sum(-1)).mean(axis=2).mean(axis=0)
    np.testing.assert_allclose(ev["success"]["mean"], exp_success, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ev["agreement"]["mean"], exp_agree, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ev["fail_rate"]["mean"], [0.10, 0.10], rtol=0, atol=1e-12)
    d = ev["diagnostics"]
    ok = ~failed
    exp_diag = (recs == lab[:, None, :])[ok].mean()
    assert d["success"] == pytest.approx(exp_diag, abs=1e-12)
    assert not d["suspicious"]
    print(f"\n[10% nulls] success {ev['success']['mean'].round(4).tolist()} (hand {exp_success.round(4).tolist()}), "
          f"fail_rate {ev['fail_rate']['mean'].tolist()}, diagnostics success (non-failed) {d['success']:.4f}")


@needs_data
def test_all_nulls(data, tmp_path):
    interaction, heldout, _ = data
    users, n_obs = range(5), (0, 1, 5)
    shape = (5, 3, 100)
    path = tmp_path / "all_null.jsonl"
    _write_with_nulls(path, np.zeros(shape, dtype=np.int64), np.ones(shape, dtype=bool), users, n_obs, range(100))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ev = evaluate_predictions(path, interaction, heldout, users=users, n_obs=n_obs)
    np.testing.assert_array_equal(ev["success"]["mean"], 0.0)
    np.testing.assert_array_equal(ev["agreement"]["mean"], 0.0)
    np.testing.assert_array_equal(ev["fail_rate"]["mean"], 1.0)
    assert np.isnan(ev["diagnostics"]["success"]) and not ev["diagnostics"]["suspicious"]
