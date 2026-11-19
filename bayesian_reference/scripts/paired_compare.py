"""Paired comparison of two prediction files A and B on the same cases (e.g. text vs OCR input, or two models).

    python scripts/paired_compare.py NAME_A=PATH_A NAME_B=PATH_B [--users 0-49|all] [--n-obs 1,5|all] [--sets all]
        [--intersect] [--table-md F.md] [--table-csv F.csv] [--table-json F.json] [--interaction F] [--heldout F]

The records are joined on (user, n_obs, set) within the subset declared with --users / --n-obs / --sets (default:
all). Lines are validated as in evaluate_file.py; a line outside the declared subset is always an error. By default
both files must cover the declared subset completely; otherwise the script reports the coverage and exits with 1.
With --intersect only the cases present in both files are used, and their number is reported.

Per n_obs: success of A and B, difference B - A with a paired SE over users, agreement with the Bayesian assistant
per file, fail rate per file, the share of changed recommendations (rec_A != rec_B, where null counts as a value of
its own) and the direction of each change:
    correct->wrong   A correct, B wrong (both not null)
    wrong->correct   A wrong, B correct (both not null)
    wrong->wrong     both wrong and different (both not null)
    null involved    exactly one of A, B is null
These four add up to the number of changed cases; "both null" cases are unchanged and reported separately.

Success, agreement and fail rate are means over users of the per-user means over that user's used cases (identical
to evaluate_file.py for complete files). The paired SE is the SE of the per-user differences B - A. Users without
used cases at some n_obs are left out there. The "+-" values are the SE: standard error of the mean over users (not
comparable to the paper's error bars, see README). Change counts are pooled over all used cases.
"""

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bayes_ref import config  # noqa: E402
from bayes_ref.data_io import load_eval_interactions, load_heldout  # noqa: E402
from bayes_ref.evaluate import (  # noqa: E402
    _resolve_subset,
    agreement,
    bayesian_factory,
    heldout_labels,
    subset_candidate_sets,
)
from compare_runs import (  # noqa: E402
    SE_TEXT,
    _cell,
    md_table,
    parse_run_spec,
    text_table,
    write_csv,
    write_json,
    write_text,
)
from evaluate_file import fmt, parse_ids  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_PRED_KEYS = ("user", "n_obs", "set", "rec")
CHANGE_KINDS = ("correct_to_wrong", "wrong_to_correct", "wrong_to_wrong", "null_involved")

TABLE_COLUMNS = [
    "n_obs", "name_a", "name_b", "n_users", "n_cases",
    "success_a", "success_a_se", "success_b", "success_b_se", "diff_b_minus_a", "diff_se",
    "agreement_a", "agreement_a_se", "agreement_b", "agreement_b_se", "fail_rate_a", "fail_rate_b",
    "changed", "changed_share", *CHANGE_KINDS, "both_null",
]


def read_partial_predictions(
    path: str | Path,
    users: Sequence[int] | None = None,
    n_obs: Sequence[int] | None = None,
    sets: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a predictions file that may cover only part of the declared subset. Shapes (U', R', S').

    Returns (recs int64, -1 where null or absent; failed bool, "rec": null; present bool, record present).
    Same line validation as evaluate.read_predictions (invalid JSON, wrong keys, wrong types, rec out of range,
    outside declared subset, duplicate), but missing records are not an error; the caller decides via present.
    Raises ValueError listing per problem kind the count and the first offending record.
    """
    users_t, obs_t, sets_t = _resolve_subset(users, n_obs, sets)
    u_pos = {u: i for i, u in enumerate(users_t)}
    o_pos = {n: i for i, n in enumerate(obs_t)}
    s_pos = {k: i for i, k in enumerate(sets_t)}
    shape = (len(users_t), len(obs_t), len(sets_t))
    recs = np.full(shape, -1, dtype=np.int64)
    failed = np.zeros(shape, dtype=bool)
    present = np.zeros(shape, dtype=bool)
    problems: dict[str, list] = {}

    def problem(kind: str, detail: str) -> None:
        problems.setdefault(kind, [0, detail])[0] += 1

    path = Path(path)
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            where = f"line {line_no}: {line.strip()[:200]}"
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                problem("invalid JSON", f"{where} ({e})")
                continue
            if not isinstance(rec, dict) or set(rec) != set(_PRED_KEYS):
                problem("wrong keys", f"{where} (expected exactly {list(_PRED_KEYS)})")
                continue
            u, n, k, r = (rec[key] for key in _PRED_KEYS)
            if not all(type(v) is int for v in (u, n, k)) or not (r is None or type(r) is int):
                problem("wrong types", f"{where} (user, n_obs, set must be int; rec int or null)")
                continue
            if r is not None and not 0 <= r < config.N_OPTIONS_PER_SET:
                problem("rec out of range", f"{where} (rec must be null or 0..{config.N_OPTIONS_PER_SET - 1}, zero-based)")
                continue
            if u not in u_pos or n not in o_pos or k not in s_pos:
                problem("outside declared subset", where)
                continue
            idx = (u_pos[u], o_pos[n], s_pos[k])
            if present[idx]:
                problem("duplicate", f"{where} (user={u} n_obs={n} set={k} already present)")
                continue
            present[idx] = True
            if r is None:
                failed[idx] = True
            else:
                recs[idx] = r
    if problems:
        lines = [f"  {kind}: {count} record(s), first {first}" for kind, (count, first) in problems.items()]
        raise ValueError(f"{path}: invalid predictions file:\n" + "\n".join(lines))
    return recs, failed, present


def join_report(present_a: np.ndarray, present_b: np.ndarray) -> dict[str, int]:
    """Coverage counts of two present masks of the declared subset (same shape)."""
    return {
        "declared": int(present_a.size),
        "in_a": int(present_a.sum()),
        "in_b": int(present_b.sum()),
        "shared": int((present_a & present_b).sum()),
        "only_a": int((present_a & ~present_b).sum()),
        "only_b": int((~present_a & present_b).sum()),
        "in_neither": int((~present_a & ~present_b).sum()),
    }


def check_join(report: dict[str, int], intersect: bool) -> None:
    """Raise ValueError unless both files cover the declared subset completely (or, with intersect, share a case)."""
    if intersect:
        if report["shared"] == 0:
            raise ValueError("--intersect: the two files share no (user, n_obs, set) case in the declared subset")
        return
    if report["in_a"] == report["in_b"] == report["declared"]:
        return
    if report["only_a"] or report["only_b"]:
        reason = "the two files do not cover the same cases"
    else:
        reason = "both files cover the same cases, but not the whole declared subset (declare it explicitly)"
    raise ValueError(f"{reason}: {format_join(report)}; use --intersect to compare only the shared cases")


def format_join(report: dict[str, int]) -> str:
    return (f"declared {report['declared']:,}, in A {report['in_a']:,}, in B {report['in_b']:,}, "
            f"shared {report['shared']:,}, only in A {report['only_a']:,}, only in B {report['only_b']:,}, "
            f"in neither {report['in_neither']:,}")


def _mean_se(per_user: np.ndarray) -> tuple[float, float]:
    """Mean and SE (ddof=config.SE_DDOF) of per-user values (n,); SE NaN when n - ddof <= 0, mean NaN when n = 0."""
    n = len(per_user)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(per_user.mean())
    if n - config.SE_DDOF <= 0:
        return mean, float("nan")
    return mean, float(per_user.std(ddof=config.SE_DDOF) / math.sqrt(n))


def change_counts(
    recs_a: np.ndarray, failed_a: np.ndarray, recs_b: np.ndarray, failed_b: np.ndarray, labels: np.ndarray,
    used: np.ndarray,
) -> dict[str, int]:
    """Changed recommendations between A and B on the used cases. All arrays same shape; recs -1 where failed.

    changed: rec_A != rec_B, null counting as a value of its own. The CHANGE_KINDS partition the changed cases.
    both_null: unchanged cases where both are null.
    """
    both_ok = ~failed_a & ~failed_b
    changed = used & ((failed_a != failed_b) | (both_ok & (recs_a != recs_b)))
    ca, cb = recs_a == labels, recs_b == labels
    out = {
        "n_cases": int(used.sum()),
        "changed": int(changed.sum()),
        "correct_to_wrong": int((changed & both_ok & ca & ~cb).sum()),
        "wrong_to_correct": int((changed & both_ok & ~ca & cb).sum()),
        "wrong_to_wrong": int((changed & both_ok & ~ca & ~cb).sum()),
        "null_involved": int((changed & ~both_ok).sum()),
        "both_null": int((used & failed_a & failed_b).sum()),
    }
    return out


def paired_rows(
    a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], used: np.ndarray, labels: np.ndarray,
    bayes_cands: np.ndarray, obs: Sequence[int], names: tuple[str, str] = ("A", "B"),
) -> list[dict[str, Any]]:
    """One row per n_obs (TABLE_COLUMNS keys). a, b: (recs, failed) each (U', R', S'); used (U', R', S') bool;
    labels (U', S'); bayes_cands (U', R', S', K) bool; obs the declared n_obs (R')."""
    lab = np.broadcast_to(labels[:, None, :], used.shape)
    metrics = {}
    for key, (recs, failed) in (("a", a), ("b", b)):
        ok = ~failed
        metrics[key] = {
            "success": ((recs == lab) & ok).astype(np.float64),
            "agreement": np.where(ok, agreement(np.where(ok, recs, 0), bayes_cands), 0.0),
            "fail_rate": failed.astype(np.float64),
        }
    rows = []
    for i, n in enumerate(obs):
        m = used[:, i, :]  # (U', S')
        count = m.sum(axis=1)
        keep = count > 0

        def per_user(values: np.ndarray) -> np.ndarray:
            return (values[:, i, :] * m).sum(axis=1)[keep] / count[keep]

        row: dict[str, Any] = {"n_obs": n, "name_a": names[0], "name_b": names[1], "n_users": int(keep.sum())}
        for key in ("a", "b"):
            for metric in ("success", "agreement"):
                row[f"{metric}_{key}"], row[f"{metric}_{key}_se"] = _mean_se(per_user(metrics[key][metric]))
            row[f"fail_rate_{key}"] = _mean_se(per_user(metrics[key]["fail_rate"]))[0]
        row["diff_b_minus_a"], row["diff_se"] = _mean_se(
            per_user(metrics["b"]["success"]) - per_user(metrics["a"]["success"]))
        counts = change_counts(a[0][:, i], a[1][:, i], b[0][:, i], b[1][:, i], labels, m)
        row.update(counts)
        row["changed_share"] = counts["changed"] / counts["n_cases"] if counts["n_cases"] else float("nan")
        rows.append(row)
    return rows


def run_paired(
    a: tuple[str, Path], b: tuple[str, Path],
    interaction: dict[str, np.ndarray], heldout: dict[str, np.ndarray],
    users: Sequence[int] | None = None, n_obs: Sequence[int] | None = None, sets: Sequence[int] | None = None,
    intersect: bool = False,
) -> dict[str, Any]:
    """Read, join and compare two prediction files. Returns n_obs, users, sets, join (join_report), rows.

    Raises ValueError on invalid files or subsets and, without intersect, unless both files cover the declared
    subset completely (with intersect, if they share no case).
    """
    users_t, obs_t, sets_t = _resolve_subset(users, n_obs, sets)
    recs_a, failed_a, present_a = read_partial_predictions(a[1], users_t, obs_t, sets_t)
    recs_b, failed_b, present_b = read_partial_predictions(b[1], users_t, obs_t, sets_t)
    report = join_report(present_a, present_b)
    check_join(report, intersect)
    used = present_a & present_b
    labels = heldout_labels(interaction, heldout)[np.ix_(users_t, sets_t)]
    bayes_cands = subset_candidate_sets(bayesian_factory, interaction, heldout, users_t, obs_t, sets_t)
    rows = paired_rows((recs_a, failed_a), (recs_b, failed_b), used, labels, bayes_cands, obs_t, (a[0], b[0]))
    return {"n_obs": list(obs_t), "users": list(users_t), "sets": list(sets_t), "join": report, "rows": rows,
            "intersect": intersect}


def _pm(row: dict, key: str, sign: bool = False) -> str:
    if math.isnan(row[key]):
        return "-"
    text = fmt(row[key], row[f"{key}_se"]).replace(" ", "")
    return f"+{text}" if sign and row[key] >= 0 else text


def _nan_dash(value: float) -> str:
    return "-" if math.isnan(value) else _cell(value)


def tables(rows: Sequence[dict]) -> list[tuple[list[str], list[list[str]]]]:
    """(header, body) of the success/agreement table and of the change table."""
    a, b = rows[0]["name_a"], rows[0]["name_b"]
    t1 = (["n_obs", "n_users", f"success {a}", f"success {b}", f"diff {b}-{a}", f"agreement {a}",
           f"agreement {b}", f"fail_rate {a}", f"fail_rate {b}"],
          [[str(r["n_obs"]), str(r["n_users"]), _pm(r, "success_a"), _pm(r, "success_b"),
            _pm({"d": r["diff_b_minus_a"], "d_se": r["diff_se"]}, "d", sign=True),
            _pm(r, "agreement_a"), _pm(r, "agreement_b"), _nan_dash(r["fail_rate_a"]), _nan_dash(r["fail_rate_b"])]
           for r in rows])
    t2 = (["n_obs", "cases", "changed", "share", "correct->wrong", "wrong->correct", "wrong->wrong",
           "null involved", "both null"],
          [[str(r["n_obs"]), f"{r['n_cases']:,}", f"{r['changed']:,}", _nan_dash(r["changed_share"]),
            *(f"{r[k]:,}" for k in CHANGE_KINDS), f"{r['both_null']:,}"] for r in rows])
    return [t1, t2]


def describe_lines(res: dict, a: tuple[str, Path], b: tuple[str, Path], args_text: dict[str, str]) -> list[str]:
    used = res["join"]["shared"]
    mode = (f"--intersect: using the {used:,} shared cases of {res['join']['declared']:,} declared"
            if res["intersect"] else f"all {used:,} declared cases present in both files")
    return [
        f"A = {a[0]}: {a[1]}",
        f"B = {b[0]}: {b[1]}",
        f"subset: {len(res['users'])} users ({args_text['users']}), n_obs {res['n_obs']}, "
        f"{len(res['sets'])} sets ({args_text['sets']})",
        f"join on (user, n_obs, set): {format_join(res['join'])}",
        mode,
        f"mean +- SE over users; diff = B - A with paired SE (SE of the per-user differences); {SE_TEXT}",
        "changed = rec_A != rec_B (null counts as a value); correct->wrong + wrong->correct + wrong->wrong + "
        "null involved = changed",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="NAME_A=PATH_A, e.g. text=path/to/text_run.jsonl")
    ap.add_argument("b", help="NAME_B=PATH_B, e.g. ocr=path/to/ocr_run.jsonl")
    ap.add_argument("--users", default="all")
    ap.add_argument("--n-obs", default="all")
    ap.add_argument("--sets", default="all")
    ap.add_argument("--intersect", action="store_true", help="use only the cases present in both files")
    ap.add_argument("--table-md", type=Path, default=None)
    ap.add_argument("--table-csv", type=Path, default=None)
    ap.add_argument("--table-json", type=Path, default=None)
    ap.add_argument("--interaction", type=Path, default=ROOT / config.EVAL_INTERACTION_PATH)
    ap.add_argument("--heldout", type=Path, default=ROOT / config.EVAL_HELDOUT_PATH)
    args = ap.parse_args(argv)

    try:
        a, b = parse_run_spec(args.a), parse_run_spec(args.b)
        if a[0] == b[0]:
            raise ValueError(f"A and B have the same name {a[0]!r}")
        users = parse_ids(args.users, "users")
        n_obs = parse_ids(args.n_obs, "n-obs")
        sets = parse_ids(args.sets, "sets")
        interaction = load_eval_interactions(args.interaction)
        heldout = load_heldout(args.heldout)
        res = run_paired(a, b, interaction, heldout, users, n_obs, sets, args.intersect)
    except (ValueError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    args_text = {"users": args.users, "sets": args.sets}
    lines = describe_lines(res, a, b, args_text)
    print("\n".join(lines) + "\n")
    for header, body in tables(res["rows"]):
        print(text_table(header, body))

    if args.table_md:
        text = "\n".join(f"- {line}" for line in lines) + "\n"
        for header, body in tables(res["rows"]):
            text += "\n" + md_table(header, body)
        write_text(text, args.table_md)
        print(f"wrote {args.table_md}")
    if args.table_csv:
        write_csv(res["rows"], TABLE_COLUMNS, args.table_csv)
        print(f"wrote {args.table_csv}")
    if args.table_json:
        write_json({"a": {"name": a[0], "predictions": str(a[1])}, "b": {"name": b[0], "predictions": str(b[1])},
                    "se": SE_TEXT, **res}, args.table_json)
        print(f"wrote {args.table_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
