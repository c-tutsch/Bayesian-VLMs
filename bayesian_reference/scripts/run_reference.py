"""Reference curves of the Bayesian and oracle assistants on the paper's eval files.

Writes outputs/reference_curves.png, outputs/reference_curves.log (same text as printed) and
outputs/example_predictions_bayes_seed0.jsonl (sampled_recs of the Bayesian candidates, seed 0).
"""

import time
from pathlib import Path

import numpy as np

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
    sampled_recs,
    success_rate_curve,
    write_predictions,
)
from bayes_ref.plotting import plot_curves
from bayes_ref.rng import make_rng

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
PLOT_PATH = OUT_DIR / "reference_curves.png"
LOG_PATH = OUT_DIR / "reference_curves.log"
PRED_PATH = OUT_DIR / "example_predictions_bayes_seed0.jsonl"
SEED = 0

_lines: list[str] = []


def out(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def row(label: str, values, fmt: str = "{:>15.4f}") -> None:
    out(f"{label:<34}" + "".join(fmt.format(v) for v in values))


def curve_row(label: str, curve: dict[str, np.ndarray]) -> None:
    out(f"{label:<34}" + "".join(f"{m:>8.4f}+-{s:.4f}" for m, s in zip(curve["mean"], curve["se"])))


def header() -> None:
    out(f"{'n_obs':<34}" + "".join(f"{n:>15}" for n in config.EVAL_OBS_ROUNDS))


def targets_row() -> None:
    out(f"{'paper target (approx.)':<34}" + "".join(
        f"{config.TARGET_ACCURACY[n]:>15.2f}" if n in config.TARGET_ACCURACY else f"{'-':>15}"
        for n in config.EVAL_OBS_ROUNDS
    ))


def main() -> None:
    t0 = time.perf_counter()
    rule = config.TIE_RULE_LIKELIHOOD
    interaction = load_eval_interactions(ROOT / config.EVAL_INTERACTION_PATH)
    heldout = load_heldout(ROOT / config.EVAL_HELDOUT_PATH)
    labels = heldout_labels(interaction, heldout)
    out(f"tie rule (config.TIE_RULE_LIKELIHOOD) = {rule!r}, SE ddof = {config.SE_DDOF}")
    out(f"heldout rows in interaction order: {np.array_equal(heldout['reward_fn'], interaction['reward_fn'])}; "
        f"interaction idx == row number: {np.array_equal(interaction['idx'], np.arange(len(interaction['idx'])))}")

    bayes = candidate_sets(bayesian_factory, interaction, heldout, rule)
    oracle = candidate_sets(oracle_factory, interaction, heldout, rule)
    n_cells = bayes.shape[0] * bayes.shape[2]

    out("\nSuccess = expected accuracy under uniform random tie-breaking among the candidates; mean +- se over users")
    header()
    curve_row("Bayesian", success_rate_curve(expected_success(bayes, labels)))
    curve_row("Oracle", success_rate_curve(expected_success(oracle, labels)))
    targets_row()

    out(f"\nTie counts: cells (user, set) with |candidates| > 1, out of {n_cells} per n_obs")
    header()
    row("Bayesian |A|>1", (bayes.sum(-1) > 1).sum(axis=(0, 2)), "{:>15d}")
    row("Bayesian |A|=3", (bayes.sum(-1) == 3).sum(axis=(0, 2)), "{:>15d}")
    row("Oracle |A|>1", (oracle.sum(-1) > 1).sum(axis=(0, 2)), "{:>15d}")

    label_in_oracle = in_candidates(labels, oracle)
    out(f"\nOracle: stored heldout labels NOT in the oracle candidate set: {int((~label_in_oracle).sum())} "
        f"of {label_in_oracle.size} (expected 0)")
    row("Oracle P(label in A)", label_in_oracle.mean(axis=(0, 2)))

    rng = make_rng(SEED)
    recs = sampled_recs(bayes, rng)
    out(f"\nBayesian self-agreement, sampled_recs(bayes, seed {SEED})")
    header()
    curve_row("expected agreement (1/|A|)", agreement_curve(agreement(recs, bayes)))
    row("candidate-set agreement (rec in A)", in_candidates(recs, bayes).mean(axis=(0, 2)))
    curve_row("sampled success (rec == label)", success_rate_curve(recs == labels[:, None, :]))

    write_predictions(recs, PRED_PATH)
    out(f"\nwrote {PRED_PATH.relative_to(ROOT)}; evaluate_predictions on it:")
    ev = evaluate_predictions(PRED_PATH, interaction, heldout, rule)
    header()
    curve_row("success", ev["success"])
    curve_row("agreement with Bayesian", ev["agreement"])

    plot_curves(
        {
            f"Bayesian ({rule}), expected": success_rate_curve(expected_success(bayes, labels)),
            "Oracle, expected": success_rate_curve(expected_success(oracle, labels)),
        },
        PLOT_PATH,
        note=f"Oracle < 1 because {int((oracle[:, 0].sum(-1) > 1).sum()):,} of {n_cells:,} held-out cases "
        "contain true ties",
    )
    out(f"\nwrote {PLOT_PATH.relative_to(ROOT)}")
    out(f"elapsed {time.perf_counter() - t0:.1f} s")
    LOG_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"wrote {LOG_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
