"""Tests for scripts/compare_runs.py and scripts/paired_compare.py on synthetic eval data (real-data tests skip)."""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from bayes_ref import config
from bayes_ref.choice import sample_option_set, user_choice
from bayes_ref.data_io import load_eval_interactions, load_heldout
from bayes_ref.evaluate import (
    bayesian_factory,
    evaluate_predictions,
    expected_success,
    heldout_labels,
    oracle_factory,
    read_predictions,
    sampled_recs,
    subset_candidate_sets,
    success_rate_curve,
    write_predictions,
)
from bayes_ref.world import build_theta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import compare_runs  # noqa: E402
import paired_compare  # noqa: E402

INTERACTION = ROOT / config.EVAL_INTERACTION_PATH
HELDOUT = ROOT / config.EVAL_HELDOUT_PATH
needs_data = pytest.mark.skipif(
    not (INTERACTION.exists() and HELDOUT.exists()), reason=f"missing {INTERACTION} or {HELDOUT}"
)
EXAMPLE_PAPER = ROOT / "docs" / "paper_approx_example.json"

USERS = list(range(8))
N_OBS = (0, 1, 5)
SETS = list(range(12))
SHAPE = (len(USERS), len(N_OBS), len(SETS))
FEATURES = ["departure_time", "duration", "number_of_stops", "price"]


# ---------------------------------------------------------------- synthetic eval data


def _synthetic_world(seed: int = 123) -> tuple[dict, dict]:
    """Synthetic interaction / heldout dicts in the loaders' format: 624 users, consistent simulated choices.

    The heldout users are a permutation of the interaction users, so labels are matched by reward_fn.
    """
    rng = np.random.default_rng(seed)
    theta = build_theta()
    n, r, s = config.N_USERS_EXPECTED, config.N_ROUNDS, config.N_FRESH_SETS
    options = np.array([[sample_option_set(rng) for _ in range(r)] for _ in range(n)])
    user_idx = np.array([[user_choice(theta[u], options[u, j], rng) for j in range(r)] for u in range(n)])
    held_options = np.array([sample_option_set(rng) for _ in range(s)])
    perm = rng.permutation(n)
    held_idxs = np.array([[user_choice(theta[u], held_options[k], rng) for k in range(s)] for u in perm])
    interaction = {"reward_fn": theta, "options": options, "user_idx": user_idx, "idx": np.arange(n)}
    heldout = {"options": held_options, "reward_fn": theta[perm], "user_idxs": held_idxs}
    return interaction, heldout


def _write_world(interaction: dict, heldout: dict, directory: Path) -> tuple[Path, Path]:
    """Write the synthetic world as Zenodo-format files readable by load_eval_interactions / load_heldout."""
    inter_path, held_path = directory / "interaction.jsonl", directory / "heldout.json"
    with open(inter_path, "w", encoding="utf-8", newline="\n") as f:
        for u in range(len(interaction["reward_fn"])):
            rounds = [{"options": ["-", "-", "-"], "user_idx": int(c)} for c in interaction["user_idx"][u]]
            f.write(json.dumps({"idx": u, "reward_fn": interaction["reward_fn"][u].tolist(), "seed": 0,
                                "features": FEATURES, "rounds": rounds,
                                "rounds_numpy": interaction["options"][u].tolist()}) + "\n")
    held = {
        "features": FEATURES,
        "all_options": [["-", "-", "-"]] * len(heldout["options"]),
        "all_options_numpy": heldout["options"].tolist(),
        "user_idxs": [{"reward_fn": rf.tolist(), "idxs": idxs.tolist()}
                      for rf, idxs in zip(heldout["reward_fn"], heldout["user_idxs"])],
    }
    held_path.write_text(json.dumps(held), encoding="utf-8")
    return inter_path, held_path


@pytest.fixture(scope="module")
def world():
    return _synthetic_world()


@pytest.fixture(scope="module")
def world_files(world, tmp_path_factory):
    return _write_world(*world, tmp_path_factory.mktemp("world"))


@pytest.fixture(scope="module")
def labels(world):
    return heldout_labels(*world)[np.ix_(USERS, SETS)]  # (U', S')


def _write_with_nulls(path: Path, recs: np.ndarray, failed: np.ndarray, users, n_obs, sets, keep=None) -> Path:
    """Like write_predictions, with "rec": null where failed; keep (same shape bool) drops records where False."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for a, u in enumerate(users):
            for i, n in enumerate(n_obs):
                for b, k in enumerate(sets):
                    if keep is not None and not keep[a, i, b]:
                        continue
                    r = None if failed[a, i, b] else int(recs[a, i, b])
                    f.write(json.dumps({"user": u, "n_obs": n, "set": k, "rec": r}) + "\n")
    return path


def _lab3(labels: np.ndarray) -> np.ndarray:
    return np.broadcast_to(labels[:, None, :], SHAPE).copy()


# ---------------------------------------------------------------- compare_runs: argument parsing


def test_parse_run_spec():
    assert compare_runs.parse_run_spec("qwen2.5-32b=path/to/run.jsonl") == ("qwen2.5-32b", Path("path/to/run.jsonl"))
    assert compare_runs.parse_run_spec("a=b=c.jsonl") == ("a", Path("b=c.jsonl"))
    for bad in ("path/to/run.jsonl", "=x.jsonl", "name=", " =x"):
        with pytest.raises(ValueError, match="NAME=PATH"):
            compare_runs.parse_run_spec(bad)
    with pytest.raises(ValueError, match="duplicate"):
        compare_runs.parse_runs(["a=x.jsonl", "a=y.jsonl"])
    for reserved in ("bayesian", "Oracle"):
        with pytest.raises(ValueError, match="reserved"):
            compare_runs.parse_runs([f"{reserved}=x.jsonl"])


# ---------------------------------------------------------------- compare_runs: table values


@pytest.fixture(scope="module")
def three_runs(world, labels, tmp_path_factory):
    """perfect (rec = label), shifted ((label+1)%3, 1/4 null) and random recs on USERS x N_OBS x SETS."""
    d = tmp_path_factory.mktemp("runs")
    lab = _lab3(labels)
    rng = np.random.default_rng(7)
    failed = np.zeros(SHAPE, dtype=bool)
    failed[:, :, :3] = True  # 3 of 12 sets null for every user and n_obs
    paths = {
        "perfect": d / "perfect.jsonl",
        "shifted": _write_with_nulls(d / "shifted.jsonl", (lab + 1) % 3, failed, USERS, N_OBS, SETS),
        "random": d / "random.jsonl",
    }
    write_predictions(lab, paths["perfect"], USERS, N_OBS, SETS)
    write_predictions(rng.integers(0, 3, size=SHAPE), paths["random"], USERS, N_OBS, SETS)
    return paths


def test_compare_table_values(world, labels, three_runs):
    interaction, heldout = world
    res = compare_runs.run_comparison(three_runs, interaction, heldout, USERS, N_OBS, SETS)
    rows = {(r["n_obs"], r["name"]): r for r in res["rows"]}
    assert len(res["rows"]) == len(N_OBS) * (len(three_runs) + 2)
    assert [r["name"] for r in res["rows"][:5]] == ["perfect", "shifted", "random", "bayesian", "oracle"]
    assert set(res["rows"][0]) == set(compare_runs.TABLE_COLUMNS)

    for n in N_OBS:
        assert rows[n, "perfect"]["success"] == 1.0 and rows[n, "perfect"]["success_se"] == 0.0
        assert rows[n, "perfect"]["fail_rate"] == 0.0
        assert rows[n, "shifted"]["success"] == 0.0 and rows[n, "shifted"]["agreement"] >= 0.0
        assert rows[n, "shifted"]["fail_rate"] == pytest.approx(0.25, abs=1e-12)
        assert rows[n, "shifted"]["fail_rate_se"] == 0.0
        for ref in ("bayesian", "oracle"):
            assert rows[n, ref]["kind"] == "reference" and rows[n, ref]["agreement"] is None
        assert rows[n, "perfect"]["paper_approx"] is None and rows[n, "perfect"]["deviation"] is None

    # every run row equals evaluate_predictions on the same subset
    for name, path in three_runs.items():
        ev = evaluate_predictions(path, interaction, heldout, users=USERS, n_obs=N_OBS, sets=SETS)
        for i, n in enumerate(N_OBS):
            for key in ("success", "agreement", "fail_rate"):
                assert rows[n, name][key] == pytest.approx(ev[key]["mean"][i], abs=1e-12)
                assert rows[n, name][f"{key}_se"] == pytest.approx(ev[key]["se"][i], abs=1e-12, nan_ok=True)

    # references are expected success of the subset candidate sets
    for ref, factory in (("bayesian", bayesian_factory), ("oracle", oracle_factory)):
        cands = subset_candidate_sets(factory, interaction, heldout, USERS, N_OBS, SETS)
        curve = success_rate_curve(expected_success(cands, labels))
        for i, n in enumerate(N_OBS):
            assert rows[n, ref]["success"] == pytest.approx(curve["mean"][i], abs=1e-12)
            assert rows[n, ref]["success_se"] == pytest.approx(curve["se"][i], abs=1e-12)
    # at n_obs = 0 the Bayesian assistant is uniform: expected success exactly 1/3, agreement of any rec 1/3
    assert rows[0, "bayesian"]["success"] == pytest.approx(1 / 3, abs=1e-12)
    assert rows[0, "random"]["agreement"] == pytest.approx(1 / 3, abs=1e-12)
    assert rows[0, "shifted"]["agreement"] == pytest.approx(0.75 / 3, abs=1e-12)  # nulls score 0


def test_compare_subset_mismatch_raises(world, three_runs):
    interaction, heldout = world
    with pytest.raises(ValueError, match="missing"):  # file covers N_OBS (0, 1, 5); declaring more is an error
        compare_runs.run_comparison(three_runs, interaction, heldout, USERS, (0, 1, 2, 5), SETS)
    with pytest.raises(ValueError, match="outside declared subset"):
        compare_runs.run_comparison(three_runs, interaction, heldout, USERS[:4], N_OBS, SETS)
    with pytest.raises(ValueError, match="sorted"):
        compare_runs.run_comparison(three_runs, interaction, heldout, USERS[::-1], N_OBS, SETS)


# ---------------------------------------------------------------- paper-approx file and deviation maths


def test_paper_deviation_maths():
    assert compare_runs.paper_deviation(0.45, 0.01, 0.42) == pytest.approx((0.03, 3.0))
    assert compare_runs.paper_deviation(0.40, 0.02, 0.42) == pytest.approx((-0.02, -1.0))
    dev, z = compare_runs.paper_deviation(0.5, float("nan"), 0.42)
    assert dev == pytest.approx(0.08) and math.isnan(z)
    dev, z = compare_runs.paper_deviation(1.0, 0.0, 0.81)
    assert dev == pytest.approx(0.19) and math.isnan(z)
    _, z = compare_runs.paper_deviation(1 / 3, 4e-17, 0.33)  # float noise of an exactly constant curve
    assert math.isnan(z)


def test_example_paper_file():
    paper = compare_runs.load_paper_approx(EXAMPLE_PAPER)
    assert paper["values"] == {
        "qwen2.5-7b": {1: 0.37, 5: 0.37},
        "qwen2.5-32b": {1: 0.42, 5: 0.49},
        "bayesian": {1: 0.58, 5: 0.81},
    }
    assert "approximate" in paper["description"].lower()
    assert set(paper["sources"]) == set(paper["values"])


@pytest.mark.parametrize(
    "obj, match",
    [
        ({"value": {}}, "key 'values'"),
        ({"values": {}}, "non-empty"),
        ({"values": {"m": {}}}, "non-empty"),
        ({"values": {"m": {"x": 0.5}}}, "not an integer"),
        ({"values": {"m": {"6": 0.5}}}, "must be one of"),
        ({"values": {"m": {"01": 0.5}}}, "must be one of"),
        ({"values": {"m": {"1": 1.5}}}, r"in \[0, 1\]"),
        ({"values": {"m": {"1": "0.5"}}}, r"in \[0, 1\]"),
        ({"values": {"m": {"1": True}}}, r"in \[0, 1\]"),
        ({"values": {"m": {"1": 0.5}}, "sources": {"m": 1}}, "sources"),
        ({"values": {"m": {"1": 0.5}}, "extra": 1}, "optional"),
    ],
)
def test_paper_file_invalid(tmp_path, obj, match):
    path = tmp_path / "paper.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        compare_runs.load_paper_approx(path)


def test_compare_with_paper_values(world, three_runs, tmp_path):
    interaction, heldout = world
    path = tmp_path / "paper.json"
    path.write_text(json.dumps({"values": {
        "random": {"1": 0.40, "5": 0.30, "2": 0.5},
        "bayesian": {"0": 0.33, "5": 0.81},
        "not-a-run": {"1": 0.5},
    }}), encoding="utf-8")
    paper = compare_runs.load_paper_approx(path)
    res = compare_runs.run_comparison(three_runs, interaction, heldout, USERS, N_OBS, SETS, paper)
    rows = {(r["n_obs"], r["name"]): r for r in res["rows"]}
    for (n, name), value in {(1, "random"): 0.40, (5, "random"): 0.30, (0, "bayesian"): 0.33,
                             (5, "bayesian"): 0.81}.items():
        r = rows[n, name]
        assert r["paper_approx"] == value
        assert r["deviation"] == pytest.approx(r["success"] - value, abs=1e-12)
        if r["success_se"] > 0:
            assert r["deviation_over_se"] == pytest.approx((r["success"] - value) / r["success_se"], rel=1e-12)
        else:
            assert math.isnan(r["deviation_over_se"])
    assert rows[0, "bayesian"]["success_se"] == 0.0  # uniform posterior: exactly 1/3 for every user
    assert rows[0, "random"]["paper_approx"] is None and rows[1, "bayesian"]["paper_approx"] is None
    assert res["paper_unmatched"] == ["not-a-run"]
    assert res["paper_skipped"] == [["random", 2]]


# ---------------------------------------------------------------- paired_compare


def test_read_partial_matches_read_predictions(three_runs):
    recs, failed, present = paired_compare.read_partial_predictions(three_runs["shifted"], USERS, N_OBS, SETS)
    back, back_failed = read_predictions(three_runs["shifted"], USERS, N_OBS, SETS, return_failed=True)
    assert present.all()
    np.testing.assert_array_equal(recs, back)
    np.testing.assert_array_equal(failed, back_failed)


@pytest.mark.parametrize(
    "line, match",
    [
        ('{"user": 99, "n_obs": 1, "set": 0, "rec": 0}', "outside declared subset"),
        ('{"user": 0, "n_obs": 0, "set": 0, "rec": 1}', "duplicate"),
        ('{"user": 0, "n_obs": 1, "set": 0, "rec": 3}', "rec out of range"),
        ('{"user": 0, "n_obs": 1, "set": 0}', "wrong keys"),
        ('{"user": 0, "n_obs": 1, "set": 0, "rec": "1"}', "wrong types"),
        ("not json", "invalid JSON"),
    ],
)
def test_read_partial_raises(three_runs, tmp_path, line, match):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(three_runs["perfect"].read_text(encoding="utf-8") + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        paired_compare.read_partial_predictions(bad, USERS, N_OBS, SETS)


def test_change_counts_hand():
    labels = np.array([[0, 1, 2, 0, 1, 2, 0, 1]])
    recs_a = np.array([[0, 0, 2, 1, -1, 2, -1, 1]])
    fail_a = recs_a == -1
    recs_b = np.array([[0, 1, 0, 2, 1, -1, -1, 1]])
    fail_b = recs_b == -1
    used = np.ones_like(labels, dtype=bool)
    used[0, 7] = False
    c = paired_compare.change_counts(recs_a, fail_a, recs_b, fail_b, labels, used)
    # case: 0 same-correct, 1 wrong->correct, 2 correct->wrong, 3 wrong->wrong, 4 null->rec,
    #       5 rec->null, 6 both null (unchanged), 7 not used
    assert c == {"n_cases": 7, "changed": 5, "correct_to_wrong": 1, "wrong_to_correct": 1, "wrong_to_wrong": 1,
                 "null_involved": 2, "both_null": 1}


def test_change_counts_add_up_random():
    rng = np.random.default_rng(3)
    shape = (40, 3, 50)
    labels = rng.integers(0, 3, size=shape)
    recs_a, recs_b = rng.integers(0, 3, size=shape), rng.integers(0, 3, size=shape)
    fail_a, fail_b = rng.random(shape) < 0.1, rng.random(shape) < 0.1
    recs_a[fail_a] = -1
    recs_b[fail_b] = -1
    used = rng.random(shape) < 0.8
    c = paired_compare.change_counts(recs_a, fail_a, recs_b, fail_b, labels, used)
    assert sum(c[k] for k in paired_compare.CHANGE_KINDS) == c["changed"]
    changed_by_hand = 0
    for idx in zip(*np.nonzero(used)):
        a = None if fail_a[idx] else recs_a[idx]
        b = None if fail_b[idx] else recs_b[idx]
        changed_by_hand += a != b
    assert c["changed"] == changed_by_hand and c["n_cases"] == used.sum()


def test_paired_complete_files(world, labels, three_runs):
    interaction, heldout = world
    a, b = ("random", three_runs["random"]), ("shifted", three_runs["shifted"])
    res = paired_compare.run_paired(a, b, interaction, heldout, USERS, N_OBS, SETS)
    assert res["join"] == {"declared": int(np.prod(SHAPE)), "in_a": int(np.prod(SHAPE)), "in_b": int(np.prod(SHAPE)),
                           "shared": int(np.prod(SHAPE)), "only_a": 0, "only_b": 0, "in_neither": 0}
    ev_a = evaluate_predictions(a[1], interaction, heldout, users=USERS, n_obs=N_OBS, sets=SETS)
    ev_b = evaluate_predictions(b[1], interaction, heldout, users=USERS, n_obs=N_OBS, sets=SETS)
    recs_a = read_predictions(a[1], USERS, N_OBS, SETS)
    recs_b, fail_b = read_predictions(b[1], USERS, N_OBS, SETS, return_failed=True)
    lab = _lab3(labels)
    for i, r in enumerate(res["rows"]):
        for key, ev in (("a", ev_a), ("b", ev_b)):
            for metric in ("success", "agreement", "fail_rate"):
                assert r[f"{metric}_{key}"] == pytest.approx(ev[metric]["mean"][i], abs=1e-12)
            assert r[f"success_{key}_se"] == pytest.approx(ev["success"]["se"][i], abs=1e-12)
        d = ((recs_b == lab) & ~fail_b)[:, i].mean(axis=1) - (recs_a == lab)[:, i].mean(axis=1)
        assert r["diff_b_minus_a"] == pytest.approx(d.mean(), abs=1e-12)
        assert r["diff_se"] == pytest.approx(d.std(ddof=config.SE_DDOF) / math.sqrt(len(USERS)), abs=1e-12)
        assert r["n_users"] == len(USERS) and r["n_cases"] == len(USERS) * len(SETS)
        assert sum(r[k] for k in paired_compare.CHANGE_KINDS) == r["changed"]
        assert r["null_involved"] == len(USERS) * 3 and r["both_null"] == 0
        assert r["changed_share"] == pytest.approx(r["changed"] / r["n_cases"])
        # B = (label+1)%3 is never correct, so no wrong->correct
        assert r["wrong_to_correct"] == 0


def test_paired_join_mismatch_and_intersect(world, labels, three_runs, tmp_path):
    interaction, heldout = world
    lab = _lab3(labels)
    no_fail = np.zeros(SHAPE, dtype=bool)
    keep_a = np.ones(SHAPE, dtype=bool)
    keep_a[:, :, -2:] = False  # A lacks sets 10, 11
    keep_b = np.ones(SHAPE, dtype=bool)
    keep_b[-1] = False  # B lacks user 7
    pa = _write_with_nulls(tmp_path / "a.jsonl", lab, no_fail, USERS, N_OBS, SETS, keep_a)
    pb = _write_with_nulls(tmp_path / "b.jsonl", (lab + 1) % 3, no_fail, USERS, N_OBS, SETS, keep_b)
    a, b = ("A", pa), ("B", pb)
    with pytest.raises(ValueError, match="do not cover the same cases") as e:
        paired_compare.run_paired(a, b, interaction, heldout, USERS, N_OBS, SETS)
    shared = int((keep_a & keep_b).sum())
    assert f"shared {shared:,}" in str(e.value) and "--intersect" in str(e.value)

    res = paired_compare.run_paired(a, b, interaction, heldout, USERS, N_OBS, SETS, intersect=True)
    assert res["join"]["shared"] == shared == 7 * 3 * 10
    assert res["join"]["only_a"] == 1 * 3 * 10 and res["join"]["only_b"] == 7 * 3 * 2
    for r in res["rows"]:
        assert r["n_users"] == 7 and r["n_cases"] == 70
        assert r["success_a"] == 1.0 and r["success_b"] == 0.0 and r["diff_b_minus_a"] == -1.0
        assert r["changed"] == 70 == r["correct_to_wrong"]

    # both files cover the same cases, but not the declared subset
    pb2 = _write_with_nulls(tmp_path / "b2.jsonl", lab, no_fail, USERS, N_OBS, SETS, keep_a)
    with pytest.raises(ValueError, match="not the whole declared subset"):
        paired_compare.run_paired(a, ("B", pb2), interaction, heldout, USERS, N_OBS, SETS)

    # no shared case at all
    keep_c = ~keep_a
    pc = _write_with_nulls(tmp_path / "c.jsonl", lab, no_fail, USERS, N_OBS, SETS, keep_c)
    with pytest.raises(ValueError, match="share no"):
        paired_compare.run_paired(a, ("C", pc), interaction, heldout, USERS, N_OBS, SETS, intersect=True)


def test_paired_intersect_unequal_users(world, labels, tmp_path):
    """Per-user means over each user's own shared cases; users without shared cases at an n_obs are left out."""
    interaction, heldout = world
    lab = _lab3(labels)
    no_fail = np.zeros(SHAPE, dtype=bool)
    keep = np.ones(SHAPE, dtype=bool)
    keep[0, :, 6:] = False  # user 0: 6 of 12 sets
    keep[1, 2] = False  # user 1: nothing at n_obs 5
    recs_b = lab.copy()
    recs_b[0, :, :3] = (lab[0, :, :3] + 1) % 3  # user 0 wrong on 3 of its 6 shared sets
    pa = _write_with_nulls(tmp_path / "a.jsonl", lab, no_fail, USERS, N_OBS, SETS)
    pb = _write_with_nulls(tmp_path / "b.jsonl", recs_b, no_fail, USERS, N_OBS, SETS, keep)
    res = paired_compare.run_paired(("A", pa), ("B", pb), interaction, heldout, USERS, N_OBS, SETS, intersect=True)
    r0, r5 = res["rows"][0], res["rows"][2]
    assert r0["n_users"] == 8 and r5["n_users"] == 7
    # per-user B success: user 0 = 0.5, others 1.0
    assert r0["success_b"] == pytest.approx((0.5 + 7) / 8)
    assert r5["success_b"] == pytest.approx((0.5 + 6) / 7)
    d5 = np.array([-0.5] + [0.0] * 6)
    assert r5["diff_se"] == pytest.approx(d5.std(ddof=1) / math.sqrt(7))
    assert r0["changed"] == 3 == r0["correct_to_wrong"]


# ---------------------------------------------------------------- CLI smoke tests


def test_compare_runs_cli(world_files, three_runs, tmp_path, capsys):
    inter, held = world_files
    out = tmp_path / "out"
    argv = [*(f"{k}={v}" for k, v in three_runs.items()),
            "--users", "0-7", "--n-obs", "0,1,5", "--sets", "0-11",
            "--interaction", str(inter), "--heldout", str(held),
            "--table-md", str(out / "t.md"), "--table-csv", str(out / "t.csv"), "--table-json", str(out / "t.json"),
            "--plot", str(out / "curves.png"), "--plot-rounds", str(out / "rounds.png"),
            "--note", "synthetic data, single seed", "--paper-approx", str(EXAMPLE_PAPER)]
    assert compare_runs.main(argv) == 0
    text = capsys.readouterr().out
    assert "standard error of the mean over users" in text and "Deviation from approximate paper values" in text
    assert "unmatched paper names" in text and "qwen2.5-7b" in text
    assert "MISALIGNED" in text  # the shifted run is flagged by the alignment diagnostics
    for name in ("t.md", "t.csv", "t.json", "curves.png", "rounds.png"):
        assert (out / name).stat().st_size > 0
    with open(out / "t.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == compare_runs.TABLE_COLUMNS and len(rows) == 3 * 5
    j = json.loads((out / "t.json").read_text(encoding="utf-8"))
    assert j["n_obs"] == [0, 1, 5] and len(j["rows"]) == 15 and "over users" in j["se"]
    bayes5 = next(r for r in j["rows"] if r["name"] == "bayesian" and r["n_obs"] == 5)
    assert bayes5["paper_approx"] == 0.81
    assert bayes5["deviation"] == pytest.approx(bayes5["success"] - 0.81)
    md = (out / "t.md").read_text(encoding="utf-8")
    assert "| n_obs | name | success |" in md and "deviation/SE" in md


def test_compare_runs_cli_errors(world_files, three_runs, capsys):
    inter, held = world_files
    base = ["--interaction", str(inter), "--heldout", str(held)]
    assert compare_runs.main([f"perfect={three_runs['perfect']}", *base]) == 1  # default subset = full file
    assert "missing" in capsys.readouterr().err
    assert compare_runs.main([f"perfect={three_runs['perfect']}", "--users", "0-x", *base]) == 1
    assert "--users: cannot parse" in capsys.readouterr().err
    assert compare_runs.main([str(three_runs["perfect"]), *base]) == 1
    assert "NAME=PATH" in capsys.readouterr().err
    args = [f"perfect={three_runs['perfect']}", "--users", "0-7", "--n-obs", "0,1,5", "--sets", "0-11", *base]
    assert compare_runs.main([*args, "--plot-rounds", "x.png", "--rounds", "2"]) == 1
    assert "--rounds" in capsys.readouterr().err


def test_paired_compare_cli(world_files, three_runs, tmp_path, capsys):
    inter, held = world_files
    out = tmp_path / "out"
    argv = [f"random={three_runs['random']}", f"shifted={three_runs['shifted']}",
            "--users", "0-7", "--n-obs", "0,1,5", "--sets", "0-11",
            "--interaction", str(inter), "--heldout", str(held),
            "--table-md", str(out / "p.md"), "--table-csv", str(out / "p.csv"), "--table-json", str(out / "p.json")]
    assert paired_compare.main(argv) == 0
    text = capsys.readouterr().out
    assert "standard error of the mean over users" in text and "shared 288" in text
    assert "correct->wrong" in text and "null involved" in text
    with open(out / "p.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == paired_compare.TABLE_COLUMNS and len(rows) == 3
    j = json.loads((out / "p.json").read_text(encoding="utf-8"))
    assert j["a"]["name"] == "random" and j["join"]["shared"] == 288 and not j["intersect"]
    assert "| n_obs | n_users |" in (out / "p.md").read_text(encoding="utf-8")

    assert paired_compare.main([f"x={three_runs['random']}", f"x={three_runs['shifted']}", *argv[2:]]) == 1
    assert "same name" in capsys.readouterr().err


# ---------------------------------------------------------------- real eval files


@needs_data
def test_compare_runs_real_data_bayes_file(tmp_path):
    interaction, heldout = load_eval_interactions(INTERACTION), load_heldout(HELDOUT)
    users, n_obs = list(range(30)), (0, 1, 5)
    cands = subset_candidate_sets(bayesian_factory, interaction, heldout, users, n_obs)
    path = tmp_path / "bayes.jsonl"
    write_predictions(sampled_recs(cands, np.random.default_rng(0)), path, users, n_obs)
    res = compare_runs.run_comparison({"bayes-seed0": path}, interaction, heldout, users, n_obs)
    ev = evaluate_predictions(path, interaction, heldout, users=users, n_obs=n_obs)
    rows = {(r["n_obs"], r["name"]): r for r in res["rows"]}
    for i, n in enumerate(n_obs):
        assert rows[n, "bayes-seed0"]["success"] == pytest.approx(ev["success"]["mean"][i], abs=1e-12)
        # a sample from the Bayesian candidate sets has expected agreement equal to its own mean 1/|A|
        assert rows[n, "bayes-seed0"]["agreement"] > 0.0
    assert rows[0, "bayesian"]["success"] == pytest.approx(1 / 3, abs=1e-12)
