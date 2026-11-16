"""Read-only replay of the paper's Bayesian Assistant training data against bayes_ref.assistant.

a) For each tie rule, replays data/train/bayesian.jsonl and compares the paper's recommendation with ours.
b) Checks that the oracle assistant's recommendations in rounds 1-4 are all marked correct.
c) Informational accuracy preview on data/eval/{interaction,heldout} (expected accuracy under random tie-breaking).
Writes outputs/replay_check.log and prints the same text.
"""

from pathlib import Path

import numpy as np

from bayes_ref import config
from bayes_ref.assistant import BayesianAssistant, EmptyPosteriorError
from bayes_ref.choice import tied_best
from bayes_ref.data_io import load_eval_interactions, load_heldout
from bayes_ref.train_data import iter_interactions
from bayes_ref.world import build_theta

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "outputs" / "replay_check.log"
N_MISMATCH_SHOWN = 5

_lines: list[str] = []


def out(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def section(title: str) -> None:
    out("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def replay_bayesian(interactions: list[dict], Theta: np.ndarray) -> None:
    section(f"a) Replay of {config.TRAIN_BAYESIAN_PATH}: {len(interactions)} interactions")
    out("Turn r: scores computed BEFORE the update with choices[r]; eligible = our tied-best set has size 1.")
    R = config.N_ROUNDS
    for rule in config.TIE_RULE_CHOICES:
        n_inter = np.zeros(R, dtype=np.int64)
        n_eligible = np.zeros(R, dtype=np.int64)
        n_match = np.zeros(R, dtype=np.int64)
        n_in_cand = np.zeros(R, dtype=np.int64)
        n_empty = np.zeros(R, dtype=np.int64)
        mismatches = []
        for it in interactions:
            a = BayesianAssistant(rule, Theta)
            for r in range(R):
                scores = a.scores(it["options"][r])
                cand = tied_best(scores)
                rec = int(it["recs"][r])
                n_inter[r] += 1
                n_in_cand[r] += rec in cand
                if len(cand) == 1:
                    n_eligible[r] += 1
                    if rec == cand[0]:
                        n_match[r] += 1
                    elif len(mismatches) < N_MISMATCH_SHOWN:
                        mismatches.append((it["idx"], r + 1, scores.tolist(), cand.tolist(), rec,
                                           it["choices"][:r].tolist()))
                if r < R - 1:
                    try:
                        a.update(it["options"][r], int(it["choices"][r]))
                    except EmptyPosteriorError:
                        n_empty[r] += 1
                        break

        out(f"\nrule = {rule!r}")
        out(f"{'round':>5} {'n_inter':>8} {'eligible':>9} {'match|elig':>11} {'rec in cand':>12} {'empty_post':>11}")
        for r in range(R):
            match_rate = f"{n_match[r] / n_eligible[r]:.4f}" if n_eligible[r] else "n/a"
            in_cand = n_in_cand[r] / n_inter[r] if n_inter[r] else float("nan")
            out(f"{r + 1:>5} {n_inter[r]:>8} {n_eligible[r]:>9} {match_rate:>11} {in_cand:>12.4f} {n_empty[r]:>11}")
        tot_e, tot_m = int(n_eligible.sum()), int(n_match.sum())
        out(f"total eligible turns {tot_e}, matches {tot_m} ({tot_m / tot_e:.4f})" if tot_e else "no eligible turns")
        out("empty_post in row r = EmptyPosteriorError raised by the update with the round-r choice.")
        out(f"first {N_MISMATCH_SHOWN} mismatches on eligible turns (idx, round, our scores, candidates, paper rec, "
            f"choices so far; 0-based option indices):")
        if not mismatches:
            out("  none")
        for m in mismatches:
            out(f"  idx={m[0]} round={m[1]} scores={m[2]} candidates={m[3]} paper_rec={m[4]} choices={m[5]}")


def check_oracle() -> None:
    path = ROOT / config.TRAIN_ORACLE_PATH
    section(f"b) {config.TRAIN_ORACLE_PATH}: are the oracle's recommendations in rounds 1-4 marked correct?")
    n_inter = n_turns = n_correct = 0
    for it in iter_interactions(path):
        n_inter += 1
        n_turns += len(it["choices"])
        n_correct += int(np.sum(it["recs"][: len(it["choices"])] == it["choices"]))
    out(f"{n_inter} interactions, {n_turns} turns with feedback, {n_correct} marked correct, "
        f"{n_turns - n_correct} marked incorrect")


def accuracy_preview(Theta: np.ndarray) -> None:
    section("c) Informational accuracy preview on the eval files (expected accuracy under random tie-breaking)")
    inter = load_eval_interactions(ROOT / config.EVAL_INTERACTION_PATH)
    held = load_heldout(ROOT / config.EVAL_HELDOUT_PATH)
    held_row = {tuple(rf): i for i, rf in enumerate(held["reward_fn"])}
    rows = np.array([held_row[tuple(rf)] for rf in inter["reward_fn"]])
    out(f"users matched by reward_fn; heldout row order identical to interaction order: "
        f"{bool(np.array_equal(rows, np.arange(len(rows))))}")
    labels = held["user_idxs"][rows]  # (N, S), aligned with interaction users

    R = config.N_ROUNDS
    out(f"{'rule':>7} | " + " ".join(f"{'obs=' + str(n):>8}" for n in range(R + 1)) + " | empty posteriors")
    for rule in config.TIE_RULE_CHOICES:
        acc_sum = np.zeros(R + 1)
        n_users = np.zeros(R + 1, dtype=np.int64)
        n_empty = 0
        for u in range(len(rows)):
            a = BayesianAssistant(rule, Theta)
            for n_obs in range(R + 1):
                if n_obs > 0:
                    try:
                        a.update(inter["options"][u, n_obs - 1], int(inter["user_idx"][u, n_obs - 1]))
                    except EmptyPosteriorError:
                        n_empty += 1
                        break
                scores = a.scores(held["options"])  # (S, K)
                is_best = scores == scores.max(axis=1, keepdims=True)
                hit = is_best[np.arange(len(scores)), labels[u]]
                acc_sum[n_obs] += float(np.mean(hit / is_best.sum(axis=1)))
                n_users[n_obs] += 1
        acc = acc_sum / np.maximum(n_users, 1)
        out(f"{rule:>7} | " + " ".join(f"{x:>8.4f}" for x in acc) + f" | {n_empty}")
        if np.any(n_users != len(rows)):
            out(f"        users evaluated per obs count: {n_users.tolist()}")
    targets = " ".join(
        f"{config.TARGET_ACCURACY[n]:>8.2f}" if n in config.TARGET_ACCURACY else f"{'?':>8}" for n in range(R + 1)
    )
    out(f"{'paper':>7} | {targets} | (approximate targets, Fig. 2/3; '?' = not recorded in config)")


def main() -> None:
    Theta = build_theta()
    interactions = list(iter_interactions(ROOT / config.TRAIN_BAYESIAN_PATH))
    replay_bayesian(interactions, Theta)
    check_oracle()
    accuracy_preview(Theta)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {LOG_PATH}")


if __name__ == "__main__":
    main()
