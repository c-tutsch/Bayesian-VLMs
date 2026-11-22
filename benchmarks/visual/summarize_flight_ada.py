"""Validate Ada prediction shards and create a small report for the repository."""
import argparse
import csv
import hashlib
from itertools import product
import json
from math import ceil, floor
from pathlib import Path

from run_flight_ada import DATA, FORMATS, read_rows, write_rows

LABELS = {
    "text": "Plain text",
    "ocr": "OCR images",
    "diagrams": "Diagrams",
    "diagrams-noise": "Diagrams with noise",
}


def load_shards(run, allow_dry_run):
    shards = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run.glob("shard-*.json"))
    ]
    if not shards:
        raise ValueError(f"No shard metadata under {run}")
    settings = {key: val for key, val in shards[0].items() if key not in ("users", "shard")}
    if settings["dry_run"] and not allow_dry_run:
        raise ValueError("Dry-run predictions are synthetic and cannot be reported as model results.")
    if [row["shard"] for row in shards] != list(range(settings["shards"])):
        raise ValueError("One or more shards are missing.")
    for row in shards:
        if {k: v for k, v in row.items() if k not in ("users", "shard")} != settings:
            raise ValueError("Shards use different settings.")
    users = sorted(user for row in shards for user in row["users"])
    expected = list(range(624)) if settings["sets"] == list(range(100)) else [0, 1]
    if users != expected:
        raise ValueError("User membership is incomplete or duplicated.")
    return shards, settings, users


def predictions(run, mode, shards):
    for shard in shards:
        base = run / mode / f"shard-{shard['shard']:03d}"
        for n_obs in shard["n_obs"]:
            path = base / f"eval-{n_obs}.jsonl"
            for (user, offer), row in zip(product(shard["users"], shard["sets"]), read_rows(path), strict=True):
                if (row["user"], row["n_obs"], row["set"]) != (user, n_obs, offer):
                    raise ValueError(f"Wrong case order or duplicate in {path}")
                if row["rec"] is not None and (type(row["rec"]) is not int or row["rec"] not in range(3)):
                    raise ValueError(f"Invalid flight index in {path}: {row}")
                yield row


def write_plot(metrics, output, smoke, dry_run, include_table=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if include_table:
        fig, (ax, table_ax) = plt.subplots(
            2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": (3, 1)},
        )
    else:
        fig, ax = plt.subplots(figsize=(8.0, 4.7))
    table_values = []
    styles = (("o", "-"), ("s", "--"), ("D", "-"), ("x", ":"))
    full_run = not (dry_run or smoke)
    observations = sorted({row["n_obs"] for row in metrics})
    for (mode, label), (marker, line) in zip(LABELS.items(), styles, strict=True):
        rows = [row for row in metrics if row["format"] == mode]
        values = [100 * row["accuracy"] for row in rows]
        legend = f"{label} ({values[-1]:.2f}%)" if full_run else label
        table_values.append([label, *(f"{value:.2f}" for value in values)])
        ax.plot([row["n_obs"] for row in rows], values,
                marker=marker, linestyle=line, linewidth=2, label=legend)
    limits = (0, 100)
    if dry_run:
        title = "Flight pipeline dry run — synthetic predictions"
    elif smoke:
        title = "Flight benchmark — Ada smoke test\n2 users × 2 offers per point"
    else:
        title = "Flight recommendation accuracy\nQwen3.5-9B BF16 · 624 profiles · 100 test offers per round"
        values = [100 * row["accuracy"] for row in metrics]
        limits = (max(0, floor(min(values)) - 1), min(100, ceil(max(values)) + 1))
    ax.set(title=title, xlabel="Observed user choices", ylabel="Accuracy (%)", ylim=limits)
    ax.set_xticks(observations)
    ax.grid(axis="y", alpha=0.25)
    if full_run:
        ax.legend(title=f"After {observations[-1]} choices", loc="upper left",
                  bbox_to_anchor=(1.02, 1), frameon=False)
    else:
        ax.legend(frameon=False)
    if include_table:
        table_ax.axis("off")
        table = table_ax.table(
            cellText=table_values,
            colLabels=["Input format", *observations],
            colWidths=[0.34] + [0.66 / len(observations)] * len(observations),
            cellLoc="center",
            bbox=(0, 0, 1, 1),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#d7e0eb")
            if row == 0 or col == len(observations):
                cell.set_facecolor("#eaf0f6")
                cell.set_text_props(weight="bold")
            if col == 0:
                cell.set_text_props(ha="left")
    fig.tight_layout()
    fig.savefig(output, dpi=200, metadata={"Author": "Julius Pantel", "Software": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--allow-dry-run", action="store_true")
    args = parser.parse_args()
    shards, settings, users = load_shards(args.run, args.allow_dry_run)
    for name, expected in settings["dataset_sha256"].items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Evaluation data changed since the run: {name}")
    observations, offers = settings["n_obs"], settings["sets"]
    labels = json.loads((DATA / "eval/heldout/flight.json").read_text(encoding="utf-8"))["user_idxs"]
    labels = [row["idxs"] for row in labels]

    args.out.mkdir(parents=True, exist_ok=True)
    merged = args.run / "merged"
    merged.mkdir(exist_ok=True)
    metrics, text_accuracy = [], {}
    for mode in FORMATS:
        counts = {n: dict(correct=0, invalid=0, records=0) for n in observations}

        def evaluated_rows():
            for row in predictions(args.run, mode, shards):
                user, n_obs, offer, rec = (row[key] for key in ("user", "n_obs", "set", "rec"))
                count = counts[n_obs]
                count["records"] += 1
                count["correct"] += rec == labels[user][offer]
                count["invalid"] += rec is None
                yield row

        write_rows(merged / f"{mode}.jsonl", evaluated_rows())
        for n_obs, count in counts.items():
            total = count["records"]
            accuracy = count["correct"] / total
            if mode == "text":
                text_accuracy[n_obs] = accuracy
            metrics.append(dict(
                format=mode, n_obs=n_obs, cases=total, correct=count["correct"], accuracy=accuracy,
                difference_to_text_pp=100 * (accuracy - text_accuracy[n_obs]),
                invalid=count["invalid"], invalid_rate=count["invalid"] / total,
            ))

    with (args.out / "metrics.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    (args.out / "summary.json").write_text(
        json.dumps(dict(settings=settings, users=users, metrics=metrics), indent=2) + "\n", encoding="utf-8",
    )
    if args.plot:
        smoke, dry_run = len(users) < 624, settings["dry_run"]
        write_plot(metrics, args.out / "accuracy.png", smoke, dry_run)
        if not (smoke or dry_run):
            write_plot(metrics, args.out / "comparison.png", smoke, dry_run, include_table=True)
    print(f"Checked {sum(row['cases'] for row in metrics):,} predictions and wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
