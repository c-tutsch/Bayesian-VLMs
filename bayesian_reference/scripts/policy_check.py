"""Read-only diagnostic: which likelihood rule and recommendation policy generated data/train/bayesian.jsonl?

For every interaction and turn r = 0..4 the posterior weights w (U,) are built from choices[0..r-1] under a
likelihood rule, then each candidate policy assigns probabilities q (3,) to the options of round r; we score
the paper's recommendation recs[r] by log q[recs[r]].

Likelihood rules: 'simple', 'strict', 'exact' (bayes_ref.assistant.likelihood_weights) and the diagnostic-only
'first' / 'last' (theta' consistent iff the lowest / highest index among its tied-best options is the choice).
Policies: P1 sample-random, P2 sample-first, P3 mean (smoothed), P4 mean-of-k-samples (Monte Carlo, first
N_P4_INTERACTIONS interactions only).
Writes outputs/policy_check.log and prints the same text.
"""

import time
from pathlib import Path

import numpy as np

from bayes_ref import config
from bayes_ref.assistant import BayesianAssistant, likelihood_weights
from bayes_ref.choice import rewards_int_all, tied_best, to_int_features, to_int_theta
from bayes_ref.train_data import iter_interactions

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "outputs" / "policy_check.log"

RULES = ("simple", "strict", "exact", "first", "last")
K_VALUES = (1, 2, 3, 5, 10, 30)
N_MC = 300
Q_FLOOR = 0.003
SMOOTH_KEEP = 0.99
SMOOTH_EPS = 0.01
N_P4_INTERACTIONS = 1000
N_EXAMPLES = 3
# Calibration bin lower edges in tenths: [0,.1) [.1,.3) [.3,.5) [.5,.7) [.7,.9) [.9,1.0].
CAL_EDGES_TENTHS = (0, 1, 3, 5, 7, 9)
CAL_LABELS = ("[0.0,0.1)", "[0.1,0.3)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,0.9)", "[0.9,1.0]")
CAL_RULES = ("simple", "exact")
SEED = config.DEFAULT_SEED
R = config.N_ROUNDS
K = config.N_OPTIONS_PER_SET

_lines: list[str] = []


def out(text: str = "") -> None:
    print(text, flush=True)
    _lines.append(text)


def section(title: str) -> None:
    out("\n" + "=" * 110 + f"\n{title}\n" + "=" * 110)


# ---------------------------------------------------------------------------------------------------------
# Likelihoods and posterior weights
# ---------------------------------------------------------------------------------------------------------

def best_index_tables(Rw: np.ndarray) -> dict[str, np.ndarray]:
    """From integer rewards Rw (K, U): is_best (K, U) bool, n_best (U,), first (U,) and last (U,) tied-best index."""
    is_best = Rw == Rw.max(axis=0)
    first = np.argmax(is_best, axis=0)
    last = Rw.shape[0] - 1 - np.argmax(is_best[::-1], axis=0)
    return {"is_best": is_best, "n_best": is_best.sum(axis=0), "first": first, "last": last}


def rule_likelihood(rule: str, options: np.ndarray, chosen: int, Theta: np.ndarray, tab: dict) -> np.ndarray:
    """Integer likelihood (U,) of `chosen` among options (K, 4) for every theta in Theta (U, 4) under `rule`."""
    if rule in config.TIE_RULE_CHOICES:
        return likelihood_weights(options, chosen, Theta, rule)
    if rule in ("first", "last"):
        return (tab[rule] == chosen).astype(np.int64)
    raise ValueError(f"unknown rule {rule!r}")


def weight_sequence(rule: str, it: dict, Theta: np.ndarray, tabs: list[dict]) -> tuple[list[np.ndarray], int | None]:
    """Posterior weights (U,) int64 before each turn r = 0..4, updating with choices[r] for r <= 3.

    Returns (ws, empty_round): ws stops early and empty_round is the 0-based round whose update emptied the
    posterior (None if it never empties).
    """
    w = np.ones(len(Theta), dtype=np.int64)
    ws = [w]
    for r in range(R - 1):
        w = w * rule_likelihood(rule, it["options"][r], int(it["choices"][r]), Theta, tabs[r])
        if not w.any():
            return ws, r
        ws.append(w)
    return ws, None


# ---------------------------------------------------------------------------------------------------------
# Policies: each returns q (3,) over the options of the current turn
# ---------------------------------------------------------------------------------------------------------

def p1_sample_random_exact(w: np.ndarray, tab: dict) -> tuple[np.ndarray, int]:
    """P1 as exact integers: q_j = num[j] / den. Returns num (K,) int64 and den."""
    share = np.where(tab["is_best"], config.WEIGHT_SCALE // tab["n_best"], 0)  # (K, U), 6/|T| on T
    return share @ w, int(config.WEIGHT_SCALE * w.sum())


def p2_sample_first(w: np.ndarray, tab: dict) -> np.ndarray:
    """P2: posterior mass of thetas whose lowest tied-best index is j. Shape (K,)."""
    return np.array([w[tab["first"] == j].sum() for j in range(K)], dtype=np.float64) / w.sum()


def p3_mean(w: np.ndarray, feat_int: np.ndarray, theta_int: np.ndarray) -> np.ndarray:
    """P3 unsmoothed: posterior-mean argmax, ties uniform. feat_int (K, 4), theta_int (U, 4). Shape (K,)."""
    cand = tied_best(feat_int @ (w @ theta_int))
    q = np.zeros(K)
    q[cand] = 1.0 / len(cand)
    return q


def p4_mean_of_k(w: np.ndarray, feat_int: np.ndarray, theta_int: np.ndarray, k: int,
                 rng: np.random.Generator) -> np.ndarray:
    """P4 Monte Carlo estimate of q (K,), unfloored: N_MC repetitions of k posterior draws, summed theta, argmax.

    Ties among the argmax are split exactly (1/|tied| each) within a repetition instead of sampled.
    """
    draws = rng.choice(len(w), size=(N_MC, k), p=w / w.sum())
    scores = theta_int[draws].sum(axis=1) @ feat_int.T  # (N_MC, K)
    is_best = scores == scores.max(axis=1, keepdims=True)
    return (is_best / is_best.sum(axis=1, keepdims=True)).mean(axis=0)


# ---------------------------------------------------------------------------------------------------------
# Per-combination accumulator
# ---------------------------------------------------------------------------------------------------------

class Combo:
    """Per-turn records for one (rule, policy) combination."""

    def __init__(self, rule: str, policy: str) -> None:
        self.rule, self.policy = rule, policy
        self.pos: list[int] = []        # position of the interaction in the file (0-based)
        self.rnd: list[int] = []        # 0-based round
        self.logq: list[float] = []     # log of the q used for the likelihood
        self.zero: list[bool] = []      # unsmoothed / unfloored q[paper rec] == 0
        self.q_choice: list[float] = []  # unsmoothed q[user's choice], nan in round 4
        self.obs: list[float] = []      # paper rec == choice, nan in round 4
        self.examples: list[str] = []

    def add(self, pos: int, idx: int, r: int, q_raw: np.ndarray, q_used: np.ndarray, rec: int,
            choice: int | None, w: np.ndarray) -> None:
        self.pos.append(pos)
        self.rnd.append(r)
        with np.errstate(divide="ignore"):
            self.logq.append(float(np.log(q_used[rec])))
        is_zero = bool(q_raw[rec] == 0)
        self.zero.append(is_zero)
        self.q_choice.append(float(q_raw[choice]) if choice is not None else np.nan)
        self.obs.append(float(rec == choice) if choice is not None else np.nan)
        if is_zero and len(self.examples) < N_EXAMPLES:
            self.examples.append(
                f"idx={idx} round={r + 1} w=[n_pos={np.count_nonzero(w)} sum={int(w.sum())} max={int(w.max())}] "
                f"q_raw={np.round(q_raw, 4).tolist()} paper_rec={rec}"
                + (f" user_choice={choice}" if choice is not None else "")
            )

    def arrays(self, max_pos: int | None = None) -> dict[str, np.ndarray]:
        pos = np.array(self.pos)
        m = np.ones(len(pos), dtype=bool) if max_pos is None else pos < max_pos
        return {
            "rnd": np.array(self.rnd)[m],
            "logq": np.array(self.logq)[m],
            "zero": np.array(self.zero)[m],
            "q_choice": np.array(self.q_choice)[m],
            "obs": np.array(self.obs)[m],
        }


def fmt(x: float, width: int = 8) -> str:
    return f"{x:>{width}.4f}" if np.isfinite(x) or np.isinf(x) else f"{'n/a':>{width}}"


def ll_table(combos: list[Combo], max_pos: int | None, title: str) -> dict[tuple[str, str], tuple[float, float, int, int]]:
    """Print mean log q[paper rec] per round / overall; return {(rule, policy): (LL, LL|q>0, zeros, n)}."""
    out(f"\n{title}")
    out("LL = mean log q[paper rec] per turn (uses smoothed q for P3, floored q for P4; -inf if any q == 0).")
    out("LL|q>0 = same mean over turns with q[paper rec] > 0 only (supplementary).")
    out("zeros = turns with unsmoothed/unfloored q[paper rec] == 0.")
    hdr = (f"{'rule':>6} {'policy':>14} {'turns':>6} | " + " ".join(f"{'r' + str(r + 1):>8}" for r in range(R))
           + f" | {'LL':>8} {'LL|q>0':>8} {'zeros':>6}")
    out(hdr)
    res = {}
    for c in combos:
        a = c.arrays(max_pos)
        per_round = [a["logq"][a["rnd"] == r].mean() if np.any(a["rnd"] == r) else np.nan for r in range(R)]
        pos_mask = np.isfinite(a["logq"])
        ll = float(a["logq"].mean())
        ll_pos = float(a["logq"][pos_mask].mean()) if pos_mask.any() else np.nan
        nz = int(a["zero"].sum())
        res[(c.rule, c.policy)] = (ll, ll_pos, nz, len(a["logq"]))
        out(f"{c.rule:>6} {c.policy:>14} {len(a['logq']):>6} | " + " ".join(fmt(x) for x in per_round)
            + f" | {fmt(ll)} {fmt(ll_pos)} {nz:>6}")
    return res


def correct_rate_table(combos: list[Combo], max_pos: int | None, title: str) -> None:
    out(f"\n{title}")
    out("pred = mean unsmoothed/unfloored q[user's actual choice]; obs = fraction of turns with paper rec == choice "
        "(same turns).")
    out(f"{'rule':>6} {'policy':>14} | " + " ".join(f"{'pred r' + str(r + 1):>8}" for r in range(R - 1))
        + f" {'pred all':>8} | " + " ".join(f"{'obs r' + str(r + 1):>7}" for r in range(R - 1)) + f" {'obs all':>8}")
    for c in combos:
        a = c.arrays(max_pos)
        fb = a["rnd"] < R - 1
        pred = [a["q_choice"][a["rnd"] == r].mean() for r in range(R - 1)]
        obs = [a["obs"][a["rnd"] == r].mean() for r in range(R - 1)]
        out(f"{c.rule:>6} {c.policy:>14} | " + " ".join(fmt(x) for x in pred) + f" {fmt(a['q_choice'][fb].mean())} | "
            + " ".join(fmt(x, 7) for x in obs) + f" {fmt(a['obs'][fb].mean())}")


# ---------------------------------------------------------------------------------------------------------
# Calibration (P1, exact rationals)
# ---------------------------------------------------------------------------------------------------------

def cal_bin(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Bin index of q = num / den (exact integer comparison against CAL_EDGES_TENTHS / 10). Same shape as num."""
    b = np.zeros(np.shape(num), dtype=np.int64)
    for e in CAL_EDGES_TENTHS[1:]:
        b += 10 * num >= e * den
    return b


def calibration(rule: str, nums: np.ndarray, dens: np.ndarray, recs: np.ndarray) -> None:
    """nums (N, K) int, dens (N,) int, recs (N,) int: P1 q's and paper recs on every turn of `rule`."""
    q = nums / dens[:, None]
    n = len(recs)
    rows = np.arange(n)
    rng = np.random.default_rng([SEED, 7, RULES.index(rule)])
    synth = np.array([rng.choice(K, p=qi) for qi in q])
    b_paper = cal_bin(nums[rows, recs], dens)
    b_synth = cal_bin(nums[rows, synth], dens)
    b_all = cal_bin(nums, dens[:, None])  # (N, K)

    out(f"\nrule={rule!r}, P1 sample-random, {n} turns. Bins of q[rec]; n = turns in bin, mean q = mean q[rec] in bin.")
    out("synthetic = one draw rec ~ q per turn (seeded); expected = exact expectation under P1 "
        "(sum over turns and options of q_j, for q_j in the bin).")
    out(f"{'bin':>10} | {'paper n':>8} {'mean q':>7} {'frac':>6} | {'synth n':>8} {'mean q':>7} {'frac':>6} | "
        f"{'exp n':>9} {'mean q':>7} {'frac':>6}")
    q_paper, q_synth = q[rows, recs], q[rows, synth]
    for bi, label in enumerate(CAL_LABELS):
        mp, ms, ma = b_paper == bi, b_synth == bi, b_all == bi
        exp_n = float(q[ma].sum())
        exp_mean = float((q[ma] ** 2).sum() / exp_n) if exp_n > 0 else np.nan
        out(f"{label:>10} | {int(mp.sum()):>8} {fmt(q_paper[mp].mean() if mp.any() else np.nan, 7)} "
            f"{mp.mean():>6.3f} | {int(ms.sum()):>8} {fmt(q_synth[ms].mean() if ms.any() else np.nan, 7)} "
            f"{ms.mean():>6.3f} | {exp_n:>9.1f} {fmt(exp_mean, 7)} {exp_n / n:>6.3f}")
    with np.errstate(divide="ignore"):
        out(f"mean log q[rec]: paper {np.log(q_paper).mean():.4f}, synthetic {np.log(q_synth).mean():.4f}, "
            f"expected under P1 {float(np.sum(q * np.log(np.where(q > 0, q, 1.0)), axis=1).mean()):.4f}")


# ---------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------

def main() -> None:
    t_start = time.perf_counter()
    Theta = BayesianAssistant(Theta=None).Theta
    theta_int = to_int_theta(Theta)
    interactions = list(iter_interactions(ROOT / config.TRAIN_BAYESIAN_PATH))
    n_inter = len(interactions)
    n_p4 = min(N_P4_INTERACTIONS, n_inter)

    section("Policy / likelihood-rule diagnostic on " + config.TRAIN_BAYESIAN_PATH)
    out(f"{n_inter} interactions, {R} turns each. Turn r uses w from choices[0..r-1]; round labels below are r+1.")
    out(f"Rules: {RULES} ('first'/'last' are diagnostic only). Policies: P1 sample-random, P2 sample-first, "
        f"P3 mean (smoothed {SMOOTH_KEEP}*uniform-on-candidates + {SMOOTH_EPS}/3),")
    out(f"P4 mean-of-k-samples k in {K_VALUES}: {N_MC} MC repetitions per turn, q floored at {Q_FLOOR} (not "
        f"renormalised), seeded per (rule, k).")
    out(f"P4 is run ONLY on the first {n_p4} interactions ({n_p4 * R} turns per rule) because of cost.")
    out(f"Null baseline log(1/3) = {np.log(1 / 3):.4f}")

    combos: dict[tuple[str, str], Combo] = {}
    policies_all = ["P1 samp-rand", "P2 samp-first", "P3 mean"]
    policies_p4 = [f"P4 k={k}" for k in K_VALUES]
    for rule in RULES:
        for p in policies_all + policies_p4:
            combos[(rule, p)] = Combo(rule, p)
    rngs = {(rule, k): np.random.default_rng([SEED, RULES.index(rule), k]) for rule in RULES for k in K_VALUES}
    empties = {rule: np.zeros(R - 1, dtype=np.int64) for rule in RULES}
    cal_data = {rule: ([], [], []) for rule in CAL_RULES}
    t_p4 = 0.0

    for pos, it in enumerate(interactions):
        feats = [to_int_features(it["options"][r]) for r in range(R)]
        tabs = [best_index_tables(rewards_int_all(it["options"][r], Theta)) for r in range(R)]
        for rule in RULES:
            ws, empty_round = weight_sequence(rule, it, Theta, tabs)
            if empty_round is not None:
                empties[rule][empty_round] += 1
            for r, w in enumerate(ws):
                rec = int(it["recs"][r])
                choice = int(it["choices"][r]) if r < R - 1 else None
                add = lambda p, q_raw, q_used: combos[(rule, p)].add(pos, it["idx"], r, q_raw, q_used, rec, choice, w)

                num, den = p1_sample_random_exact(w, tabs[r])
                q1 = num / den
                add("P1 samp-rand", q1, q1)
                if rule in CAL_RULES:
                    cal_data[rule][0].append(num)
                    cal_data[rule][1].append(den)
                    cal_data[rule][2].append(rec)
                q2 = p2_sample_first(w, tabs[r])
                add("P2 samp-first", q2, q2)
                q3 = p3_mean(w, feats[r], theta_int)
                add("P3 mean", q3, SMOOTH_KEEP * q3 + SMOOTH_EPS / K)

                if pos < n_p4:
                    t0 = time.perf_counter()
                    for k in K_VALUES:
                        q4 = p4_mean_of_k(w, feats[r], theta_int, k, rngs[(rule, k)])
                        add(f"P4 k={k}", q4, np.maximum(q4, Q_FLOOR))
                    t_p4 += time.perf_counter() - t0
    t_loop = time.perf_counter() - t_start

    section("Posterior empties (count of interactions whose update with choices[r] emptied w; rest skipped)")
    out(f"{'rule':>6} | " + " ".join(f"{'upd r' + str(r + 1):>7}" for r in range(R - 1)) + f" | {'total':>6}")
    for rule in RULES:
        out(f"{rule:>6} | " + " ".join(f"{x:>7}" for x in empties[rule]) + f" | {int(empties[rule].sum()):>6}")

    main_combos = [combos[(rule, p)] for rule in RULES for p in policies_all]
    all_combos = [combos[(rule, p)] for rule in RULES for p in policies_all + policies_p4]

    section(f"A) Log-likelihood of paper recs, P1-P3, all {n_inter} interactions")
    res_all = ll_table(main_combos, None, "All interactions:")
    correct_rate_table(main_combos, None, "Predicted vs observed correct-rate, rounds 1-4, all interactions:")

    section(f"B) Log-likelihood of paper recs, all policies incl. P4, first {n_p4} interactions only")
    res_sub = ll_table(all_combos, n_p4, f"First {n_p4} interactions:")
    correct_rate_table(all_combos, n_p4, f"Predicted vs observed correct-rate, rounds 1-4, first {n_p4} interactions:")

    section("C) Hard impossibilities: first examples of q[paper rec] == 0 (unsmoothed / unfloored q)")
    out("w summary: n_pos = thetas with positive weight, sum / max of integer weights. Option indices 0-based.")
    out(f"P1-P3 examples from all interactions, P4 from the first {n_p4}.")
    for c in all_combos:
        nz = int(np.sum(c.zero))
        out(f"\n{c.rule} / {c.policy}: {nz} zero turns" + (":" if nz else ""))
        for e in c.examples:
            out("  " + e)

    section("D) Calibration of P1 sample-random (all interactions)")
    for rule in CAL_RULES:
        nums, dens, recs = cal_data[rule]
        calibration(rule, np.array(nums), np.array(dens), np.array(recs))

    section(f"E) Ranking of all (rule x policy) combos by mean log-likelihood on the first {n_p4} interactions")
    out("(Only subset on which every combo, incl. P4, was evaluated. Ties at -inf ordered by fewer zero turns, then "
        "LL|q>0. 'LL all' = all interactions, P1-P3 only.)")
    out(f"{'rank':>4} {'rule':>6} {'policy':>14} | {'LL sub':>8} {'LL|q>0':>8} {'zeros':>6} {'turns':>6} | "
        f"{'LL all':>8} {'zeros':>6} {'turns':>6}")
    order = sorted(res_sub, key=lambda key: (-res_sub[key][0], res_sub[key][2], -np.nan_to_num(res_sub[key][1], nan=-1e9)))
    for i, key in enumerate(order, 1):
        ll, ll_pos, nz, n = res_sub[key]
        full = res_all.get(key)
        tail = f"{fmt(full[0])} {full[2]:>6} {full[3]:>6}" if full else f"{'-':>8} {'-':>6} {'-':>6}"
        out(f"{i:>4} {key[0]:>6} {key[1]:>14} | {fmt(ll)} {fmt(ll_pos)} {nz:>6} {n:>6} | {tail}")
    out(f"null baseline log(1/3) = {np.log(1 / 3):.4f}")

    total = time.perf_counter() - t_start
    out(f"\nruntime: total {total:.1f} s (turn loop {t_loop:.1f} s, of which P4 {t_p4:.1f} s)")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {LOG_PATH}")


if __name__ == "__main__":
    main()
