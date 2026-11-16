"""Read-only loaders for the Zenodo flight evaluation files."""

import json
from pathlib import Path

import numpy as np

from bayes_ref import config

# Feature order as named inside the data files; maps 1:1 onto config.FEATURE_NAMES.
_FILE_FEATURE_NAMES = ["departure_time", "duration", "number_of_stops", "price"]


def _check_shape(name: str, arr: np.ndarray, expected: tuple[int, ...]) -> None:
    if arr.shape != expected:
        raise ValueError(f"{name}: expected shape {expected}, got {arr.shape}")


def _check_features(path: Path, features: list[str]) -> None:
    if features != _FILE_FEATURE_NAMES:
        raise ValueError(f"{path}: unexpected feature order {features}, expected {_FILE_FEATURE_NAMES}")


def load_eval_interactions(path: str | Path) -> dict[str, np.ndarray]:
    """Load data/eval/interaction/flight.jsonl.

    File structure (verified): 624 JSON lines, each with keys
    idx (int), reward_fn (4 floats), seed (int), features (4 names),
    rounds (5 x {"options": 3 strings, "user_idx": int}) and
    rounds_numpy (list of 5 rounds, each a list of 3 options, each a list of 4 floats),
    i.e. rounds_numpy has shape (5, 3, 4) per record. The numeric field is used directly.

    Returns:
        reward_fn: (624, 4) float64
        options:   (624, 5, 3, 4) float64, from rounds_numpy
        user_idx:  (624, 5) int64, index of the user's chosen option per round
        idx:       (624,) int64
    """
    path = Path(path)
    reward_fn, options, user_idx, idx = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            _check_features(path, rec["features"])
            idx.append(rec["idx"])
            reward_fn.append(rec["reward_fn"])
            options.append(rec["rounds_numpy"])
            user_idx.append([rnd["user_idx"] for rnd in rec["rounds"]])

    n, r, k, d = config.N_USERS_EXPECTED, config.N_ROUNDS, config.N_OPTIONS_PER_SET, config.N_FEATURES
    out = {
        "reward_fn": np.array(reward_fn, dtype=np.float64),
        "options": np.array(options, dtype=np.float64),
        "user_idx": np.array(user_idx, dtype=np.int64),
        "idx": np.array(idx, dtype=np.int64),
    }
    _check_shape("reward_fn", out["reward_fn"], (n, d))
    _check_shape("options", out["options"], (n, r, k, d))
    _check_shape("user_idx", out["user_idx"], (n, r))
    _check_shape("idx", out["idx"], (n,))
    return out


def load_heldout(path: str | Path) -> dict[str, np.ndarray]:
    """Load data/eval/heldout/flight.json.

    File structure (verified): a single JSON object with keys
    features (4 names), all_options (100 sets x 3 strings),
    all_options_numpy (100 sets x 3 options x 4 floats) and
    user_idxs (624 items, each exactly {"reward_fn": 4 floats, "idxs": 100 ints}).

    Returns:
        options:   (100, 3, 4) float64, from all_options_numpy
        reward_fn: (624, 4) float64, user_idxs[i]["reward_fn"]
        user_idxs: (624, 100) int64, user_idxs[i]["idxs"], the user's chosen option per set
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    _check_features(path, obj["features"])

    n, s, k, d = config.N_USERS_EXPECTED, config.N_FRESH_SETS, config.N_OPTIONS_PER_SET, config.N_FEATURES
    out = {
        "options": np.array(obj["all_options_numpy"], dtype=np.float64),
        "reward_fn": np.array([u["reward_fn"] for u in obj["user_idxs"]], dtype=np.float64),
        "user_idxs": np.array([u["idxs"] for u in obj["user_idxs"]], dtype=np.int64),
    }
    _check_shape("options", out["options"], (s, k, d))
    _check_shape("reward_fn", out["reward_fn"], (n, d))
    _check_shape("user_idxs", out["user_idxs"], (n, s))
    return out
