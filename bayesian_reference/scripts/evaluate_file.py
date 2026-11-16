"""Evaluate a predictions JSONL file against the stored heldout labels and the Bayesian reference.

    python scripts/evaluate_file.py PREDICTIONS.jsonl [--users 0-49|0,3,7|all] [--n-obs 1,5|1-5|all]
        [--sets 0-99|all] [--name MODEL] [--plot outputs/x.png] [--json outputs/x.json]

The subset the file covers must be declared with --users / --n-obs / --sets (default: all, i.e. the full
374,400-record file). Exit code 1 on any validation error, with the message printed.

The "+-" values are the SE: standard error of the mean over users (not comparable to the paper's error bars, see
README).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from bayes_ref import config
from bayes_ref.data_io import load_eval_interactions, load_heldout
from bayes_ref.evaluate import (
    bayesian_factory,
    evaluate_predictions,
    expected_success,
    heldout_labels,
    oracle_factory,
    subset_candidate_sets,
    success_rate_curve,
)

ROOT = Path(__file__).resolve().parent.parent


def parse_ids(text: str, name: str) -> list[int] | None:
    """Parse "all", "a-b" (inclusive), "a,b,c" or a mix like "0-4,7" into a list of ints; "all" -> None."""
    text = text.strip()
    if text.lower() == "all":
        return None
    out: list[int] = []
    for token in text.split(","):
        token = token.strip()
        try:
            if "-" in token:
                lo, hi = (int(t) for t in token.split("-"))
                if lo > hi:
                    raise ValueError
                out.extend(range(lo, hi + 1))
            else:
                out.append(int(token))
        except ValueError:
            raise ValueError(f"--{name}: cannot parse {token!r} in {text!r} (use 'all', 'a-b' or 'a,b,c')") from None
    return out


def fmt(mean: float, se: float) -> str:
    se_text = "   nan" if math.isnan(se) else f"{se:.4f}"
    return f"{mean:.4f}+-{se_text}"


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if math.isnan(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def banner(lines: list[str]) -> None:
    print("\n" + "!" * 78)
    for line in lines:
        print(f"!!! {line}")
    print("!" * 78)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--users", default="all")
    ap.add_argument("--n-obs", default="all")
    ap.add_argument("--sets", default="all")
    ap.add_argument("--name", default=None, help="model name for the table, plot and JSON (default: file stem)")
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    name = args.name or args.predictions.stem

    try:
        users = parse_ids(args.users, "users")
        n_obs = parse_ids(args.n_obs, "n-obs")
        sets = parse_ids(args.sets, "sets")
        interaction = load_eval_interactions(ROOT / config.EVAL_INTERACTION_PATH)
        heldout = load_heldout(ROOT / config.EVAL_HELDOUT_PATH)
        ev = evaluate_predictions(args.predictions, interaction, heldout, users=users, n_obs=n_obs, sets=sets)
    except (ValueError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    obs = ev["n_obs"]
    users_l = list(range(config.N_USERS_EXPECTED)) if users is None else users
    sets_l = list(range(config.N_FRESH_SETS)) if sets is None else sets
    labels = heldout_labels(interaction, heldout)[np.ix_(users_l, sets_l)]
    bayes = subset_candidate_sets(bayesian_factory, interaction, heldout, users_l, obs, sets_l)
    oracle = subset_candidate_sets(oracle_factory, interaction, heldout, users_l, obs, sets_l)
    ref = {
        "bayesian": success_rate_curve(expected_success(bayes, labels)),
        "oracle": success_rate_curve(expected_success(oracle, labels)),
    }
    n_cases = labels.size
    n_oracle_ties = int((oracle[:, 0].sum(-1) > 1).sum())

    print(f"file: {args.predictions}  (model: {name})")
    print(f"subset: {ev['n_users']} users ({args.users}), n_obs {obs}, {ev['n_sets']} sets ({args.sets}); "
          f"{ev['n_lines']} records")
    print("success = rec == stored label; agreement = 1/|A| if rec in the Bayesian candidate set A;")
    print("references are expected success under uniform tie-breaking on the same subset; mean +- se over users\n")
    head = f"{'n_obs':>5} | {'success':>15} | {'agreement':>15} | {'fail_rate':>9} | {'Bayesian ref':>15} | {'oracle ref':>15} | {'paper':>5}"
    print(head)
    print("-" * len(head))
    for i, n in enumerate(obs):
        target = f"{config.TARGET_ACCURACY[n]:.2f}" if n in config.TARGET_ACCURACY else "-"
        print(f"{n:>5} | {fmt(ev['success']['mean'][i], ev['success']['se'][i]):>15} | "
              f"{fmt(ev['agreement']['mean'][i], ev['agreement']['se'][i]):>15} | "
              f"{ev['fail_rate']['mean'][i]:>9.4f} | "
              f"{fmt(ref['bayesian']['mean'][i], ref['bayesian']['se'][i]):>15} | "
              f"{fmt(ref['oracle']['mean'][i], ref['oracle']['se'][i]):>15} | {target:>5}")
    print(f"\nOracle < 1 because {n_oracle_ties:,} of {n_cases:,} held-out cases in this subset contain true ties.")

    d = ev["diagnostics"]
    print("\nAlignment diagnostics (heuristic; cannot prove alignment). Success pooled over n_obs >= "
          f"{config.SHIFT_DIAG_MIN_N_OBS}, failed records ignored:")
    print(f"  file as-is          {d['success']:.4f}")
    print(f"  rec -> (rec+1)%3    {d['success_rec_plus_1']:.4f}")
    print(f"  rec -> (rec-1)%3    {d['success_rec_minus_1']:.4f}")
    print(f"  user axis shifted   {d['success_user_shift']:.4f}")
    print(f"  suspicious: {d['suspicious']}")
    warn = []
    if d["suspicious"]:
        warn += ["WARNING: predictions look MISALIGNED (0/1-based rec or shifted user order?)", *d["reasons"]]
    high_fail = [(n, f) for n, f in zip(obs, ev["fail_rate"]["mean"]) if f > config.FAIL_RATE_WARN]
    if high_fail:
        warn.append(f"WARNING: fail_rate (\"rec\": null) above {config.FAIL_RATE_WARN:.0%} at n_obs "
                    + ", ".join(f"{n} ({f:.1%})" for n, f in high_fail))
    if warn:
        banner(warn)

    if args.plot:
        from bayes_ref.plotting import plot_curves

        curves = {
            name: {"n_obs": obs, **ev["success"]},
            "Bayesian (expected)": {"n_obs": obs, **ref["bayesian"]},
            "Oracle (expected)": {"n_obs": obs, **ref["oracle"]},
        }
        plot_curves(curves, args.plot, ylabel="accuracy on held-out sets",
                    note=f"Oracle < 1 because {n_oracle_ties:,} of {n_cases:,} held-out cases contain true ties")
        print(f"\nwrote {args.plot}")
    if args.json:
        out = {
            "name": name,
            "predictions": str(args.predictions),
            "users": users_l,
            "sets": sets_l,
            **ev,
            "reference": ref,
            "oracle_tied_cases": n_oracle_ties,
            "n_cases_per_n_obs": n_cases,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(to_jsonable(out), f, indent=1)
            f.write("\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
