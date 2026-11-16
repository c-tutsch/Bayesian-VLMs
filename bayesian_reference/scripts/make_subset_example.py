"""Write outputs/example_subset_bayes.jsonl: Bayesian sampled recs (seed 0) for users 0-49, n_obs 1 and 5, all 100 sets.

The recs are sampled on the full candidate array with seed 0 (as in scripts/run_reference.py) and then sliced,
so this file is exactly the matching subset of outputs/example_predictions_bayes_seed0.jsonl.
Evaluate it with: python scripts/evaluate_file.py outputs/example_subset_bayes.jsonl --users 0-49 --n-obs 1,5
"""

from pathlib import Path

from bayes_ref import config
from bayes_ref.data_io import load_eval_interactions, load_heldout
from bayes_ref.evaluate import bayesian_factory, candidate_sets, sampled_recs, write_predictions
from bayes_ref.rng import make_rng

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "outputs" / "example_subset_bayes.jsonl"
SEED = 0
USERS = range(50)
N_OBS = (1, 5)


def main() -> None:
    interaction = load_eval_interactions(ROOT / config.EVAL_INTERACTION_PATH)
    heldout = load_heldout(ROOT / config.EVAL_HELDOUT_PATH)
    cands = candidate_sets(bayesian_factory, interaction, heldout)
    recs = sampled_recs(cands, make_rng(SEED))
    sub = recs[list(USERS)][:, [config.EVAL_OBS_ROUNDS.index(n) for n in N_OBS]]
    write_predictions(sub, OUT_PATH, USERS, N_OBS)
    print(f"wrote {OUT_PATH.relative_to(ROOT)}: {sub.size} records (users 0-49, n_obs {N_OBS}, sets 0-99, seed {SEED})")
    print(f"evaluate with: python scripts/evaluate_file.py {OUT_PATH.relative_to(ROOT).as_posix()} --users 0-49 --n-obs 1,5")


if __name__ == "__main__":
    main()
