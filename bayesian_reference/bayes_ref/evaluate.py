"""Evaluation of assistants on the paper's eval files: candidate sets, success and agreement curves, prediction files.

Protocol: for each user u (624, in interaction-file order) and each n_obs in config.EVAL_OBS_ROUNDS, the
assistant has seen the first n_obs rounds of u's interaction (options[u, :n_obs], user_idx[u, :n_obs]) and is
then asked about each of the 100 heldout option sets without feedback. The label of (u, set k) is the stored
heldout choice, matched to u by reward_fn.

Array conventions: N = 624 users, R = len(config.EVAL_OBS_ROUNDS) = 6, S = 100 heldout sets, K = 3 options.
Prediction files may cover a declared subset (users, n_obs, sets); their arrays then have shape
(len(users), len(n_obs), len(sets)) and axis positions map to the declared ids, not to the full ranges.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from bayes_ref import config
from bayes_ref.assistant import BayesianAssistant
from bayes_ref.data_io import load_eval_interactions, load_heldout
from bayes_ref.oracle import OracleAssistant
from bayes_ref.world import build_theta

_ROOT = Path(__file__).resolve().parent.parent
_PRED_KEYS = ("user", "n_obs", "set", "rec")
_OBS_INDEX = {n: i for i, n in enumerate(config.EVAL_OBS_ROUNDS)}


def _full_shape() -> tuple[int, int, int]:
    return config.N_USERS_EXPECTED, len(config.EVAL_OBS_ROUNDS), config.N_FRESH_SETS


def _describe(values: Sequence[int]) -> str:
    values = list(values)
    if values == list(range(values[0], values[-1] + 1)):
        return f"{values[0]}..{values[-1]}" if len(values) > 1 else str(values[0])
    return ",".join(map(str, values))


def _resolve_axis(name: str, values: Sequence[int] | None, full: Sequence[int]) -> tuple[int, ...]:
    if values is None:
        return tuple(full)
    if isinstance(values, (str, bytes)) or not isinstance(values, (Sequence, np.ndarray)):
        raise ValueError(f"{name}: expected a sorted sequence of ints, got {values!r}")
    out = []
    for v in values:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"{name}: expected ints, got {v!r}")
        out.append(int(v))
    if not out:
        raise ValueError(f"{name}: must not be empty")
    allowed = set(full)
    bad = [v for v in out if v not in allowed]
    if bad:
        raise ValueError(f"{name}: {len(bad)} value(s) out of range, first {bad[0]}; allowed {_describe(full)}")
    if len(set(out)) != len(out):
        dup = next(v for i, v in enumerate(out) if v in out[:i])
        raise ValueError(f"{name}: duplicate value {dup}")
    if out != sorted(out):
        raise ValueError(f"{name}: values must be sorted ascending, got {_describe(out)}")
    return tuple(out)


def _resolve_subset(
    users: Sequence[int] | None, n_obs: Sequence[int] | None, sets: Sequence[int] | None
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Validated (users, n_obs, sets); None means the full range. Raises ValueError."""
    return (
        _resolve_axis("users", users, range(config.N_USERS_EXPECTED)),
        _resolve_axis("n_obs", n_obs, config.EVAL_OBS_ROUNDS),
        _resolve_axis("sets", sets, range(config.N_FRESH_SETS)),
    )


class Assistant(Protocol):
    def update(self, options: np.ndarray, chosen: int) -> None: ...

    def scores(self, options: np.ndarray) -> np.ndarray: ...


AssistantFactory = Callable[[np.ndarray, str], Assistant]


def bayesian_factory(theta: np.ndarray, rule: str) -> BayesianAssistant:
    """Fresh BayesianAssistant with tie rule `rule`; the user's theta (4,) is ignored."""
    return BayesianAssistant(rule, build_theta())


def oracle_factory(theta: np.ndarray, rule: str) -> OracleAssistant:
    """OracleAssistant for the user's true theta (4,); `rule` is ignored."""
    return OracleAssistant(theta)


def heldout_labels(interaction: dict[str, np.ndarray], heldout: dict[str, np.ndarray]) -> np.ndarray:
    """Stored heldout choices aligned with the interaction users, matched by reward_fn. Shape (N, S) int64.

    Raises ValueError if the reward_fns of the two files are not the same set of N distinct vectors.
    """
    held_row = {tuple(rf): i for i, rf in enumerate(heldout["reward_fn"])}
    if len(held_row) != len(heldout["reward_fn"]):
        raise ValueError("heldout reward_fn rows are not distinct")
    try:
        rows = np.array([held_row[tuple(rf)] for rf in interaction["reward_fn"]], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"interaction reward_fn {e.args[0]} not found in heldout") from None
    if len(set(rows.tolist())) != len(rows) or len(rows) != len(held_row):
        raise ValueError("interaction and heldout reward_fns are not in one-to-one correspondence")
    return heldout["user_idxs"][rows]


def candidate_sets(
    assistant_factory: AssistantFactory,
    interaction: dict[str, np.ndarray],
    heldout: dict[str, np.ndarray],
    rule: str | None = None,
) -> np.ndarray:
    """Tied-best candidate sets on the heldout sets. Shape (N, R, S, K) bool.

    out[u, i, k, j] is True iff option j of heldout set k attains the max of the assistant's exact scores
    after it has observed the first config.EVAL_OBS_ROUNDS[i] rounds of user u. One fresh assistant
    assistant_factory(reward_fn[u], rule) per user, updated round by round. An empty posterior raises
    (EmptyPosteriorError from BayesianAssistant.update).
    """
    rule = config.TIE_RULE_LIKELIHOOD if rule is None else rule
    n_users, n_rounds = interaction["user_idx"].shape
    if list(config.EVAL_OBS_ROUNDS) != sorted(config.EVAL_OBS_ROUNDS) or max(config.EVAL_OBS_ROUNDS) > n_rounds:
        raise ValueError(f"EVAL_OBS_ROUNDS {config.EVAL_OBS_ROUNDS} must be sorted and <= {n_rounds}")
    held_options = heldout["options"]  # (S, K, 4)
    out = np.zeros((n_users, len(config.EVAL_OBS_ROUNDS), *held_options.shape[:2]), dtype=bool)
    for u in range(n_users):
        assistant = assistant_factory(interaction["reward_fn"][u], rule)
        n_seen = 0
        for i, n_obs in enumerate(config.EVAL_OBS_ROUNDS):
            while n_seen < n_obs:
                assistant.update(interaction["options"][u, n_seen], int(interaction["user_idx"][u, n_seen]))
                n_seen += 1
            scores = assistant.scores(held_options)  # (S, K)
            out[u, i] = scores == scores.max(axis=1, keepdims=True)
    return out


def _check_cands(cands: np.ndarray) -> None:
    if cands.dtype != bool or cands.ndim < 1:
        raise ValueError("candidate sets must be a bool array (..., K)")
    if not cands.any(axis=-1).all():
        raise ValueError("every candidate set must contain at least one option")


def in_candidates(idx: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """True where option index idx is in the candidate set. idx (N, R, S) or (N, S) int, cands (N, R, S, K). Shape (N, R, S)."""
    _check_cands(cands)
    idx = np.asarray(idx)
    if idx.ndim == cands.ndim - 2:
        idx = np.broadcast_to(idx[:, None, ...], cands.shape[:-1])
    return np.take_along_axis(cands, idx[..., None], axis=-1)[..., 0]


def expected_success(cands: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Expected accuracy under uniform random tie-breaking: 1/|A| if label in A else 0. Shape (N, R, S) float64.

    cands (N, R, S, K) bool candidate sets A; labels (N, S) int, the same for every n_obs.
    """
    return in_candidates(labels, cands) / cands.sum(axis=-1)


def sampled_recs(cands: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One uniform draw from each candidate set of cands (..., K) bool. Shape (...) int64.

    Draws one rng.integers per cell (also for singleton sets), in C order of cands.shape[:-1].
    """
    _check_cands(cands)
    size = cands.sum(axis=-1)
    pick = rng.integers(0, size)  # rank of the chosen option among the True entries
    rank = np.cumsum(cands, axis=-1) - 1
    return np.argmax(cands & (rank == pick[..., None]), axis=-1).astype(np.int64)


def agreement(recs: np.ndarray, bayes_cands: np.ndarray) -> np.ndarray:
    """Expected agreement with the Bayesian reference under its random tie-breaking. Shape (N, R, S) float64.

    1/|A| if recs (N, R, S) is in the Bayesian candidate set A (from bayes_cands (N, R, S, K)), else 0.
    """
    return in_candidates(recs, bayes_cands) / bayes_cands.sum(axis=-1)


def success_rate_curve(values: np.ndarray) -> dict[str, np.ndarray]:
    """Mean over users and sets per n_obs, with standard error over users. values (N, R, S) -> mean (R,), se (R,).

    se = std of the N per-user means (ddof=config.SE_DDOF) / sqrt(N): standard error of the mean over users (not
    comparable to the paper's error bars, see README). The heldout sets are fixed, so unlike
    the paper (which averages 3 random seeds; what is seeded is not specified) this has no set-level variance component.
    """
    per_user = np.asarray(values, dtype=np.float64).mean(axis=2)  # (N, R)
    n = per_user.shape[0]
    if n - config.SE_DDOF <= 0:  # e.g. a single user with ddof=1: se is undefined, NaN without a warning
        return {"mean": per_user.mean(axis=0), "se": np.full(per_user.shape[1], np.nan)}
    return {"mean": per_user.mean(axis=0), "se": per_user.std(axis=0, ddof=config.SE_DDOF) / np.sqrt(n)}


def agreement_curve(values: np.ndarray) -> dict[str, np.ndarray]:
    """Same aggregation as success_rate_curve, for agreement values (N, R, S)."""
    return success_rate_curve(values)


def write_predictions(
    recs: np.ndarray,
    path: str | Path,
    users: Sequence[int] | None = None,
    n_obs: Sequence[int] | None = None,
    sets: Sequence[int] | None = None,
) -> None:
    """Write recs (len(users), len(n_obs), len(sets)) int as JSONL, one {"user", "n_obs", "set", "rec"} record per cell.

    users / n_obs / sets: sorted id sequences of the declared subset; None means the full range (range(624),
    config.EVAL_OBS_ROUNDS, range(100)). The written ids are the real ids, not array positions: user = 0-based
    row in the interaction file, n_obs = observed rounds, set = heldout set index; rec = recommended option
    index (0-based). Records in user, n_obs, set ascending order; LF line endings, so output is deterministic.
    """
    users_t, obs_t, sets_t = _resolve_subset(users, n_obs, sets)
    shape = (len(users_t), len(obs_t), len(sets_t))
    recs = np.asarray(recs)
    if recs.shape != shape:
        raise ValueError(f"recs: expected shape {shape} for the declared subset, got {recs.shape}")
    if not np.all((recs >= 0) & (recs < config.N_OPTIONS_PER_SET)):
        raise ValueError("recs out of range")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for a, u in enumerate(users_t):
            for i, n in enumerate(obs_t):
                for b, k in enumerate(sets_t):
                    rec = {"user": u, "n_obs": n, "set": k, "rec": int(recs[a, i, b])}
                    f.write(json.dumps(rec) + "\n")


def read_predictions(
    path: str | Path,
    users: Sequence[int] | None = None,
    n_obs: Sequence[int] | None = None,
    sets: Sequence[int] | None = None,
    *,
    return_failed: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Read a predictions JSONL file for an explicitly declared subset. recs shape (len(users), len(n_obs), len(sets)) int64.

    users / n_obs / sets: sorted id sequences without duplicates; None means the full range, so the default
    call requires all 624 * 6 * 100 = 374,400 records. The subset is never inferred from the file.
    "rec": null marks an unparsable answer: the record counts as present and FAILED; recs holds -1 there.
    With return_failed=True returns (recs, failed), failed a bool array of the same shape.

    Raises ValueError listing, per problem kind, the count and the first offending record: invalid JSON,
    wrong keys, wrong types, rec not null and not in 0..K-1, records outside the declared subset (never
    silently ignored), duplicate records, missing records. Record order does not matter; blank lines are skipped.
    """
    users_t, obs_t, sets_t = _resolve_subset(users, n_obs, sets)
    u_pos = {u: i for i, u in enumerate(users_t)}
    o_pos = {n: i for i, n in enumerate(obs_t)}
    s_pos = {k: i for i, k in enumerate(sets_t)}
    shape = (len(users_t), len(obs_t), len(sets_t))
    recs = np.full(shape, -1, dtype=np.int64)
    failed = np.zeros(shape, dtype=bool)
    seen = np.zeros(shape, dtype=bool)
    problems: dict[str, list] = {}  # kind -> [count, first offending record]

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
            if seen[idx]:
                problem("duplicate", f"{where} (user={u} n_obs={n} set={k} already present)")
                continue
            seen[idx] = True
            if r is None:
                failed[idx] = True
            else:
                recs[idx] = r
    missing = np.argwhere(~seen)
    if len(missing):
        a, i, b = missing[0]
        problems["missing"] = [len(missing), f"user={users_t[a]} n_obs={obs_t[i]} set={sets_t[b]}"]
    if problems:
        declared = f"users={_describe(users_t)}, n_obs={_describe(obs_t)}, sets={_describe(sets_t)}"
        lines = [f"  {kind}: {count} record(s), first {first}" for kind, (count, first) in problems.items()]
        hint = ""
        if "missing" in problems:
            hint = "\n  (a file covering only a subset must be read with that subset declared via users/n_obs/sets)"
        raise ValueError(
            f"{path}: invalid predictions file for the declared subset ({declared}, {seen.size} records expected):\n"
            + "\n".join(lines) + hint
        )
    return (recs, failed) if return_failed else recs


def subset_candidate_sets(
    assistant_factory: AssistantFactory,
    interaction: dict[str, np.ndarray],
    heldout: dict[str, np.ndarray],
    users: Sequence[int] | None = None,
    n_obs: Sequence[int] | None = None,
    sets: Sequence[int] | None = None,
    rule: str | None = None,
) -> np.ndarray:
    """candidate_sets restricted to a declared subset. Shape (len(users), len(n_obs), len(sets), K) bool.

    Equals candidate_sets(...)[users][:, n_obs positions][:, :, sets], but only runs the declared users.
    """
    users_t, obs_t, sets_t = _resolve_subset(users, n_obs, sets)
    u_idx = np.array(users_t)
    sub_interaction = {key: interaction[key][u_idx] for key in ("reward_fn", "options", "user_idx")}
    sub_heldout = {"options": heldout["options"][np.array(sets_t)]}
    cands = candidate_sets(assistant_factory, sub_interaction, sub_heldout, rule)
    return cands[:, [_OBS_INDEX[n] for n in obs_t]]


def _shift_diagnostics(
    recs: np.ndarray, failed: np.ndarray, labels: np.ndarray, obs: Sequence[int]
) -> dict[str, Any]:
    """Core of shift_diagnostics on arrays: recs, failed (U, R', S'), labels (U, S'), obs the declared n_obs."""
    ok = ~failed
    obs = list(obs)
    pool = np.array([n >= config.SHIFT_DIAG_MIN_N_OBS for n in obs])
    if not pool.any():
        pool[:] = True

    def score(r: np.ndarray, lab: np.ndarray, cols: np.ndarray) -> float:
        sel = ok[:, cols]
        if not sel.any():
            return float("nan")
        return float((r[:, cols] == lab[:, None, :])[sel].mean())

    k = config.N_OPTIONS_PER_SET
    original = score(recs, labels, pool)
    variants = {
        "success_rec_plus_1": score((recs + 1) % k, labels, pool),
        "success_rec_minus_1": score((recs - 1) % k, labels, pool),
        "success_user_shift": score(recs, np.roll(labels, -1, axis=0), pool) if len(recs) > 1 else float("nan"),
    }
    by_obs = [score(recs, labels, np.arange(len(obs)) == i) for i in range(len(obs))]
    reasons = [
        f"{name} = {v:.4f} beats the file as-is ({original:.4f}) by more than {config.SHIFT_DIAG_MARGIN}"
        for name, v in variants.items()
        if v - original > config.SHIFT_DIAG_MARGIN
    ]
    reasons += [
        f"success at n_obs={n} is {s:.4f} < {config.SHIFT_DIAG_MIN_SUCCESS}"
        for n, s in zip(obs, by_obs)
        if n >= config.SHIFT_DIAG_MIN_N_OBS and s < config.SHIFT_DIAG_MIN_SUCCESS
    ]
    return {
        "success": original,
        **variants,
        "n_obs": obs,
        "success_by_n_obs": by_obs,
        "suspicious": bool(reasons),
        "reasons": reasons,
    }


def shift_diagnostics(
    path: str | Path,
    users: Sequence[int] | None = None,
    n_obs: Sequence[int] | None = None,
    sets: Sequence[int] | None = None,
    *,
    interaction: dict[str, np.ndarray] | None = None,
    heldout: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Heuristic check for misaligned prediction files. It cannot prove alignment, only flag likely mistakes.

    Success (rec == stored label, failed "rec": null records ignored) pooled over the declared n_obs >=
    config.SHIFT_DIAG_MIN_N_OBS (all declared n_obs if none), for the file as-is ("success") and as if
    every rec were (rec+1)%3 ("success_rec_plus_1") or (rec-1)%3 ("success_rec_minus_1"), the classic
    1-based/0-based mistake, and as if the user axis were shifted by one ("success_user_shift": predictions
    of the i-th declared user scored against the labels of the (i+1)-th, cyclic; NaN with one user).
    Also "success_by_n_obs" (file as-is, per declared n_obs). "suspicious" is True if a shifted variant beats
    the file as-is by more than config.SHIFT_DIAG_MARGIN, or success is below config.SHIFT_DIAG_MIN_SUCCESS
    at some n_obs >= config.SHIFT_DIAG_MIN_N_OBS; "reasons" lists why. All values are NaN if every record failed.
    """
    users_t, obs_t, sets_t = _resolve_subset(users, n_obs, sets)
    if interaction is None:
        interaction = load_eval_interactions(_ROOT / config.EVAL_INTERACTION_PATH)
    if heldout is None:
        heldout = load_heldout(_ROOT / config.EVAL_HELDOUT_PATH)
    recs, failed = read_predictions(path, users_t, obs_t, sets_t, return_failed=True)
    labels = heldout_labels(interaction, heldout)[np.ix_(users_t, sets_t)]
    return _shift_diagnostics(recs, failed, labels, obs_t)


def evaluate_predictions(
    path: str | Path,
    interaction: dict[str, np.ndarray] | None = None,
    heldout: dict[str, np.ndarray] | None = None,
    rule: str | None = None,
    *,
    users: Sequence[int] | None = None,
    n_obs: Sequence[int] | None = None,
    sets: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Success, agreement-with-Bayesian and fail-rate curves of a predictions file on a declared subset.

    users / n_obs / sets: sorted id sequences declaring the subset the file must cover exactly (see
    read_predictions); None means the full range, so the default call needs the full 374,400-record file.
    Labels are the stored heldout choices; the reference is the Bayesian assistant with tie rule `rule`
    (default config.TIE_RULE_LIKELIHOOD). interaction / heldout default to the config eval files.

    Returns:
        n_obs: list of the declared n_obs, R' = len(n_obs)
        success, agreement, fail_rate: each {"mean": (R',), "se": (R',)}; se over the per-user means with
            ddof=config.SE_DDOF, NaN when there are too few users (a single user with ddof=1); standard error of
            the mean over users (not comparable to the paper's error bars, see README)
        n_users, n_sets, n_lines: declared users, declared sets, records read
        diagnostics: shift_diagnostics of the same file and subset (a heuristic, not a proof of alignment)

    success is rec == label (concrete recs, not expected over ties). agreement is 1/|A| if rec is in the
    Bayesian candidate set A, else 0. Failed records ("rec": null) count as 0 for success and agreement and
    1 for fail_rate. At n_obs = 0 the posterior is uniform and every Bayesian candidate set has all 3
    options, so the expected agreement of any rec is exactly 1/3.
    """
    users_t, obs_t, sets_t = _resolve_subset(users, n_obs, sets)
    if interaction is None:
        interaction = load_eval_interactions(_ROOT / config.EVAL_INTERACTION_PATH)
    if heldout is None:
        heldout = load_heldout(_ROOT / config.EVAL_HELDOUT_PATH)
    recs, failed = read_predictions(path, users_t, obs_t, sets_t, return_failed=True)
    labels = heldout_labels(interaction, heldout)[np.ix_(users_t, sets_t)]  # (U, S')
    bayes_cands = subset_candidate_sets(bayesian_factory, interaction, heldout, users_t, obs_t, sets_t, rule)
    ok = ~failed
    hit = ((recs == labels[:, None, :]) & ok).astype(np.float64)
    agree = np.where(ok, agreement(np.where(ok, recs, 0), bayes_cands), 0.0)
    return {
        "n_obs": list(obs_t),
        "success": success_rate_curve(hit),
        "agreement": agreement_curve(agree),
        "fail_rate": success_rate_curve(failed),
        "n_users": len(users_t),
        "n_sets": len(sets_t),
        "n_lines": int(recs.size),
        "diagnostics": _shift_diagnostics(recs, failed, labels, obs_t),
    }
