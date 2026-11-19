"""Compare several prediction files with each other and with the Bayesian and oracle references.

    python scripts/compare_runs.py NAME=PATH [NAME=PATH ...] [--users 0-49|all] [--n-obs 1,5|all] [--sets all]
        [--table-md F.md] [--table-csv F.csv] [--table-json F.json] [--plot F.png] [--plot-rounds F.png]
        [--rounds 1,5] [--note TEXT] [--paper-approx FILE.json] [--interaction F] [--heldout F]

Every file must cover exactly the subset declared with --users / --n-obs / --sets (default: all, i.e. the full
374,400-record file), validated as in evaluate_file.py. For each observed round n_obs and each run the table shows
success, agreement with the Bayesian assistant and fail rate ("rec": null), next to the Bayesian and oracle
references (expected success under uniform tie-breaking on the same subset).

--paper-approx FILE.json adds the deviation of each run (and of the references) from approximate paper values,
as deviation = ours - paper and deviation / SE. This is a report, not a pass/fail test. Format:

    {
      "description": "optional free text",
      "values":  {"NAME": {"N_OBS": accuracy, ...}, ...},
      "sources": {"NAME": "optional free text", ...}
    }

NAME is a run name given as NAME=PATH, or "bayesian" / "oracle" for the references; N_OBS is a string "0".."5";
accuracy is in [0, 1]. Names without a matching run are listed as unmatched; rounds outside the declared n_obs are
listed as skipped. See docs/paper_approx_example.json.

The "+-" values are the SE: standard error of the mean over users (not comparable to the paper's error bars, see
README). Exit code 1 on any validation error, with the message printed.
"""

import argparse
import csv
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
    bayesian_factory,
    evaluate_predictions,
    expected_success,
    heldout_labels,
    oracle_factory,
    subset_candidate_sets,
    success_rate_curve,
)
from evaluate_file import banner, fmt, parse_ids, to_jsonable  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_NAMES = ("bayesian", "oracle")
REFERENCE_LABELS = {"bayesian": "Bayesian (expected)", "oracle": "Oracle (expected)"}
SE_TEXT = "SE = standard error of the mean over users (not comparable to the paper's error bars, see README)"
_PAPER_KEYS = {"description", "values", "sources"}

TABLE_COLUMNS = [
    "n_obs", "name", "kind", "success", "success_se", "agreement", "agreement_se", "fail_rate", "fail_rate_se",
    "n_users", "n_sets", "paper_approx", "deviation", "deviation_over_se",
]


def parse_run_spec(text: str) -> tuple[str, Path]:
    """Split "NAME=PATH" at the first "=". Raises ValueError if there is no "=" or NAME or PATH is empty."""
    name, sep, path = text.partition("=")
    name = name.strip()
    if not sep or not name or not path.strip():
        raise ValueError(f"run {text!r}: expected NAME=PATH, e.g. qwen2.5-32b=path/to/run.jsonl")
    return name, Path(path.strip())


def parse_runs(specs: Sequence[str]) -> dict[str, Path]:
    """{name: path} from NAME=PATH strings. Names must be unique and must not be a reference name."""
    runs: dict[str, Path] = {}
    for spec in specs:
        name, path = parse_run_spec(spec)
        if name.lower() in REFERENCE_NAMES:
            raise ValueError(f"run name {name!r} is reserved for the reference assistants")
        if name in runs:
            raise ValueError(f"duplicate run name {name!r}")
        runs[name] = path
    return runs


def resolve_ids(users: Sequence[int] | None, sets: Sequence[int] | None) -> tuple[list[int], list[int]]:
    """Declared users and sets as lists; None means the full range."""
    users_l = list(range(config.N_USERS_EXPECTED)) if users is None else list(users)
    sets_l = list(range(config.N_FRESH_SETS)) if sets is None else list(sets)
    return users_l, sets_l


def reference_curves(
    interaction: dict[str, np.ndarray],
    heldout: dict[str, np.ndarray],
    users: Sequence[int],
    n_obs: Sequence[int],
    sets: Sequence[int],
) -> tuple[dict[str, dict[str, np.ndarray]], int, int]:
    """Expected success of the Bayesian and oracle assistants on the declared subset.

    Returns ({"bayesian": curve, "oracle": curve}, number of (user, set) cases whose oracle candidate set has a
    true tie, number of (user, set) cases); each curve {"mean": (R',), "se": (R',)}.
    """
    labels = heldout_labels(interaction, heldout)[np.ix_(list(users), list(sets))]
    bayes = subset_candidate_sets(bayesian_factory, interaction, heldout, users, n_obs, sets)
    oracle = subset_candidate_sets(oracle_factory, interaction, heldout, users, n_obs, sets)
    ref = {
        "bayesian": success_rate_curve(expected_success(bayes, labels)),
        "oracle": success_rate_curve(expected_success(oracle, labels)),
    }
    return ref, int((oracle[:, 0].sum(-1) > 1).sum()), int(labels.size)


def load_paper_approx(path: str | Path) -> dict[str, Any]:
    """Read and validate a paper-approx JSON file (format in the module docstring).

    Returns {"description": str, "values": {name: {n_obs (int): accuracy (float)}}, "sources": {name: str}}.
    Raises ValueError on any format problem.
    """
    path = Path(path)
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: invalid JSON ({e})") from None
    if not isinstance(obj, dict) or "values" not in obj or not set(obj) <= _PAPER_KEYS:
        raise ValueError(f"{path}: expected an object with key 'values' and optional 'description', 'sources'")
    if not isinstance(obj["values"], dict) or not obj["values"]:
        raise ValueError(f"{path}: 'values' must be a non-empty object {{name: {{n_obs: accuracy}}}}")
    values: dict[str, dict[int, float]] = {}
    for name, per_round in obj["values"].items():
        if not isinstance(per_round, dict) or not per_round:
            raise ValueError(f"{path}: values[{name!r}] must be a non-empty object {{n_obs: accuracy}}")
        values[name] = {}
        for key, v in per_round.items():
            try:
                n = int(key)
            except ValueError:
                raise ValueError(f"{path}: values[{name!r}]: n_obs key {key!r} is not an integer") from None
            if str(n) != key or n not in config.EVAL_OBS_ROUNDS:
                raise ValueError(f"{path}: values[{name!r}]: n_obs key {key!r} must be one of "
                                 f"{[str(r) for r in config.EVAL_OBS_ROUNDS]}")
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                raise ValueError(f"{path}: values[{name!r}][{key!r}] = {v!r} must be a number in [0, 1]")
            values[name][n] = float(v)
    sources = obj.get("sources", {})
    if not isinstance(sources, dict) or not all(isinstance(s, str) for s in sources.values()):
        raise ValueError(f"{path}: 'sources' must be an object {{name: text}}")
    description = obj.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"{path}: 'description' must be a string")
    return {"description": description, "values": values, "sources": dict(sources)}


def paper_deviation(ours: float, se: float, paper: float) -> tuple[float, float]:
    """(deviation, deviation / se) with deviation = ours - paper; deviation / se is NaN if se is NaN or
    <= config.DEVIATION_SE_MIN (float noise of an exactly constant per-user value)."""
    dev = float(ours) - float(paper)
    z = dev / se if (not math.isnan(se) and se > config.DEVIATION_SE_MIN) else float("nan")
    return dev, z


def run_comparison(
    runs: dict[str, Path],
    interaction: dict[str, np.ndarray],
    heldout: dict[str, np.ndarray],
    users: Sequence[int] | None = None,
    n_obs: Sequence[int] | None = None,
    sets: Sequence[int] | None = None,
    paper: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate every run on the declared subset and build the comparison table.

    Returns n_obs, users, sets, evals {name: evaluate_predictions result}, reference {"bayesian", "oracle":
    curve}, oracle_tied_cases, n_cases_per_n_obs, rows (one dict per (n_obs, name) with TABLE_COLUMNS keys, runs
    first, then the references; paper columns None without a paper value) and, with paper, paper_unmatched
    (names without a run) and paper_skipped ([name, n_obs] outside the declared n_obs).
    Raises ValueError (invalid subset or file) or OSError (unreadable file).
    """
    if not runs:
        raise ValueError("at least one run NAME=PATH is required")
    evals = {
        name: evaluate_predictions(path, interaction, heldout, users=users, n_obs=n_obs, sets=sets)
        for name, path in runs.items()
    }
    first = next(iter(evals.values()))
    obs = first["n_obs"]
    users_l, sets_l = resolve_ids(users, sets)
    ref, n_ties, n_cases = reference_curves(interaction, heldout, users_l, obs, sets_l)
    values = paper["values"] if paper else {}

    rows = []
    for i, n in enumerate(obs):
        entries = [(name, "run", ev["success"], ev["agreement"], ev["fail_rate"]) for name, ev in evals.items()]
        entries += [(r, "reference", ref[r], None, None) for r in REFERENCE_NAMES]
        for name, kind, succ, agree, fail in entries:
            row = {
                "n_obs": n, "name": name, "kind": kind,
                "success": float(succ["mean"][i]), "success_se": float(succ["se"][i]),
                "agreement": None if agree is None else float(agree["mean"][i]),
                "agreement_se": None if agree is None else float(agree["se"][i]),
                "fail_rate": None if fail is None else float(fail["mean"][i]),
                "fail_rate_se": None if fail is None else float(fail["se"][i]),
                "n_users": len(users_l), "n_sets": len(sets_l),
                "paper_approx": None, "deviation": None, "deviation_over_se": None,
            }
            if n in values.get(name, {}):
                row["paper_approx"] = values[name][n]
                row["deviation"], row["deviation_over_se"] = paper_deviation(
                    row["success"], row["success_se"], values[name][n])
            rows.append(row)
    out = {
        "n_obs": obs, "users": users_l, "sets": sets_l, "evals": evals, "reference": ref,
        "oracle_tied_cases": n_ties, "n_cases_per_n_obs": n_cases, "rows": rows,
    }
    if paper:
        known = set(evals) | set(REFERENCE_NAMES)
        out["paper_unmatched"] = sorted(name for name in values if name not in known)
        out["paper_skipped"] = sorted([name, n] for name, per in values.items() if name in known
                                      for n in per if n not in obs)
    return out


def _cell(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


def _pm(row: dict, key: str) -> str:
    return "-" if row[key] is None else fmt(row[key], row[f"{key}_se"]).replace(" ", "")


def write_csv(rows: Sequence[dict], columns: Sequence[str], path: str | Path) -> None:
    """Write rows (dicts) with the given columns as CSV; None and NaN become empty cells; LF line endings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        for row in rows:
            w.writerow(["" if row[c] is None or (isinstance(row[c], float) and math.isnan(row[c])) else row[c]
                        for c in columns])


def write_json(obj: Any, path: str | Path) -> None:
    """Write obj as indented JSON (NaN -> null) with LF line endings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(to_jsonable(obj), f, indent=1)
        f.write("\n")


def write_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def md_table(header: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    """Markdown table; columns after the first two are right-aligned."""
    align = [":--" if j < 2 else "--:" for j in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines) + "\n"


def text_table(header: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    """Fixed-width text table; first two columns left-aligned, the rest right-aligned."""
    widths = [max(len(str(x)) for x in col) for col in zip(header, *body)]

    def line(cells: Sequence[str]) -> str:
        return " | ".join(c.ljust(w) if j < 2 else c.rjust(w) for j, (c, w) in enumerate(zip(cells, widths)))

    head = line(header)
    return "\n".join([head, "-" * len(head), *(line(r) for r in body)]) + "\n"


def _display_name(row: dict) -> str:
    return REFERENCE_LABELS[row["name"]] if row["kind"] == "reference" else row["name"]


def main_table(rows: Sequence[dict]) -> tuple[list[str], list[list[str]]]:
    header = ["n_obs", "name", "success", "agreement", "fail_rate"]
    body = [[str(r["n_obs"]), _display_name(r), _pm(r, "success"), _pm(r, "agreement"), _cell(r["fail_rate"])]
            for r in rows]
    return header, body


def deviation_table(rows: Sequence[dict]) -> tuple[list[str], list[list[str]]]:
    header = ["n_obs", "name", "ours", "paper (approx.)", "deviation", "deviation/SE"]
    body = [[str(r["n_obs"]), _display_name(r), _pm(r, "success"), _cell(r["paper_approx"], 2),
             f"{r['deviation']:+.4f}", _cell(r["deviation_over_se"], 2)]
            for r in rows if r["paper_approx"] is not None]
    return header, body


def describe_lines(res: dict, args_text: dict[str, str]) -> list[str]:
    return [
        f"subset: {len(res['users'])} users ({args_text['users']}), n_obs {res['n_obs']}, "
        f"{len(res['sets'])} sets ({args_text['sets']})",
        "success = rec == stored label; agreement = 1/|A| if rec in the Bayesian candidate set A; "
        "fail_rate = share of \"rec\": null (counted as failed)",
        "references are expected success under uniform tie-breaking on the same subset",
        f"mean +- SE; {SE_TEXT}",
        f"Oracle < 1 because {res['oracle_tied_cases']:,} of {res['n_cases_per_n_obs']:,} held-out cases in this "
        "subset contain true ties.",
    ]


def deviation_lines(res: dict, paper_path: Path, paper: dict) -> list[str]:
    out = [f"Deviation from approximate paper values ({paper_path}): deviation = ours - paper, "
           "deviation/SE uses our SE over users only (the paper values carry reading error and their own, "
           "different error bars); a report, not a pass/fail test."]
    if paper["description"]:
        out.append(f"paper values: {paper['description']}")
    for name, src in paper["sources"].items():
        out.append(f"  source {name}: {src}")
    if res["paper_unmatched"]:
        out.append(f"unmatched paper names (no run with that name): {', '.join(res['paper_unmatched'])}")
    if res["paper_skipped"]:
        out.append("skipped (n_obs not declared): " + ", ".join(f"{n} n_obs={k}" for n, k in res["paper_skipped"]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="NAME=PATH of a predictions file, e.g. qwen2.5-32b=path/to/run.jsonl")
    ap.add_argument("--users", default="all")
    ap.add_argument("--n-obs", default="all")
    ap.add_argument("--sets", default="all")
    ap.add_argument("--table-md", type=Path, default=None)
    ap.add_argument("--table-csv", type=Path, default=None)
    ap.add_argument("--table-json", type=Path, default=None)
    ap.add_argument("--plot", type=Path, default=None, help="line plot of success vs n_obs (plot_curves)")
    ap.add_argument("--plot-rounds", type=Path, default=None, help="grouped bars at --rounds (plot_round_comparison)")
    ap.add_argument("--rounds", default="1,5", help="n_obs shown by --plot-rounds (default 1,5)")
    ap.add_argument("--note", default=None, help="caption note under the plots, e.g. 'single seed, non-thinking'")
    ap.add_argument("--paper-approx", type=Path, default=None, help="JSON with approximate paper values")
    ap.add_argument("--interaction", type=Path, default=ROOT / config.EVAL_INTERACTION_PATH)
    ap.add_argument("--heldout", type=Path, default=ROOT / config.EVAL_HELDOUT_PATH)
    args = ap.parse_args(argv)

    try:
        runs = parse_runs(args.runs)
        users = parse_ids(args.users, "users")
        n_obs = parse_ids(args.n_obs, "n-obs")
        sets = parse_ids(args.sets, "sets")
        rounds = parse_ids(args.rounds, "rounds")
        paper = load_paper_approx(args.paper_approx) if args.paper_approx else None
        interaction = load_eval_interactions(args.interaction)
        heldout = load_heldout(args.heldout)
        res = run_comparison(runs, interaction, heldout, users, n_obs, sets, paper)
        shown = [n for n in (rounds or res["n_obs"]) if n in res["n_obs"]]
        if args.plot_rounds and not shown:
            raise ValueError(f"--rounds {args.rounds}: none of these n_obs is declared ({res['n_obs']})")
    except (ValueError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    args_text = {"users": args.users, "sets": args.sets}
    for name, path in runs.items():
        print(f"run {name}: {path} ({res['evals'][name]['n_lines']} records)")
    print("\n".join(describe_lines(res, args_text)) + "\n")
    print(text_table(*main_table(res["rows"])))
    if paper:
        print("\n".join(deviation_lines(res, args.paper_approx, paper)) + "\n")
        print(text_table(*deviation_table(res["rows"])))

    warn = []
    for name, ev in res["evals"].items():
        d = ev["diagnostics"]
        if d["suspicious"]:
            warn += [f"WARNING: {name}: predictions look MISALIGNED (0/1-based rec or shifted user order?)",
                     *(f"  {r}" for r in d["reasons"])]
        high = [(n, f) for n, f in zip(res["n_obs"], ev["fail_rate"]["mean"]) if f > config.FAIL_RATE_WARN]
        if high:
            warn.append(f"WARNING: {name}: fail_rate (\"rec\": null) above {config.FAIL_RATE_WARN:.0%} at n_obs "
                        + ", ".join(f"{n} ({f:.1%})" for n, f in high))
    if warn:
        banner(warn)

    if args.table_md:
        text = "\n".join(f"- {line}" for line in describe_lines(res, args_text)) + "\n\n"
        text += md_table(*main_table(res["rows"]))
        if paper:
            text += "\n" + "\n".join(f"- {line}" for line in deviation_lines(res, args.paper_approx, paper))
            text += "\n\n" + md_table(*deviation_table(res["rows"]))
        write_text(text, args.table_md)
        print(f"wrote {args.table_md}")
    if args.table_csv:
        write_csv(res["rows"], TABLE_COLUMNS, args.table_csv)
        print(f"wrote {args.table_csv}")
    if args.table_json:
        out = {
            "se": SE_TEXT,
            "runs": {name: {"predictions": str(path), "n_lines": res["evals"][name]["n_lines"],
                            "diagnostics": res["evals"][name]["diagnostics"]} for name, path in runs.items()},
            **{k: res[k] for k in ("n_obs", "users", "sets", "oracle_tied_cases", "n_cases_per_n_obs", "rows")},
        }
        if paper:
            out["paper_approx"] = {"file": str(args.paper_approx), **paper,
                                   "values": {k: {str(n): v for n, v in per.items()} for k, per in paper["values"].items()},
                                   "unmatched": res["paper_unmatched"], "skipped": res["paper_skipped"]}
        write_json(out, args.table_json)
        print(f"wrote {args.table_json}")

    if args.plot or args.plot_rounds:
        from bayes_ref.plotting import plot_curves, plot_round_comparison

        obs = res["n_obs"]
        curves = {name: {"n_obs": obs, **ev["success"]} for name, ev in res["evals"].items()}
        curves.update({REFERENCE_LABELS[r]: {"n_obs": obs, **res["reference"][r]} for r in REFERENCE_NAMES})
        targets = paper["values"]["bayesian"] if paper and "bayesian" in paper["values"] else config.TARGET_ACCURACY
        user_note = args.note.strip() if args.note else ""
        if user_note and not user_note.endswith((".", "!", "?")):
            user_note += "."
        bars = "Error bars: \u00b11 SE of the mean over users (not comparable to the paper's)."
        if args.plot:
            note = " ".join(filter(None, [user_note, bars, "x = paper Bayesian Assistant (approx., read from figures)."]))
            plot_curves(curves, args.plot, targets=targets, ylabel="accuracy on held-out sets", note=note)
            print(f"wrote {args.plot}")
        if args.plot_rounds:
            plot_round_comparison(curves, args.plot_rounds, n_obs=shown, note=" ".join(filter(None, [user_note, bars])))
            print(f"wrote {args.plot_rounds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
