"""Global constants for the Bayesian Assistant reference module."""

# Number of flight features.
N_FEATURES = 4

# Feature names, in the canonical column order used throughout the module.
FEATURE_NAMES = ("departure_time", "duration", "stops", "price")

# Levels for departure_time, duration, price: np.linspace(0, 1, 11).
N_LEVELS_CONTINUOUS = 11

# Number of distinct values for the "stops" feature.
N_LEVELS_STOPS = 3

# Verified against Zenodo data: stops 0/1/2 map to 0.0/0.5/1.0.
STOPS_VALUES = (0.0, 0.5, 1.0)

# Possible per-feature preference weights theta.
THETA_VALUES = (-1.0, -0.5, 0.0, 0.5, 1.0)

# Number of simulated users: 5**4 - 1 (all theta combinations except all-zero).
N_USERS_EXPECTED = 624

# Number of flight options shown per round.
N_OPTIONS_PER_SET = 3

# Number of interaction rounds per user.
N_ROUNDS = 5

# Number of fresh held-out option sets used for evaluation.
N_FRESH_SETS = 100

# Default RNG seed.
DEFAULT_SEED = 0

# Best supported by replaying data/train/bayesian.jsonl (scripts/policy_check.py): 'exact' beats 'simple'
# by ~0.005 nats/turn; 'strict', 'first', 'last' produce impossible turns. Not proven (data is slightly
# sharper than our sampling model predicts).
TIE_RULE_LIKELIHOOD = "exact"
TIE_RULE_CHOICES = ("simple", "strict", "exact")

# lcm(1,2,3): lets the 'exact' tie rule keep integer weights (6, 3, 2 for k-way ties of 1, 2, 3).
WEIGHT_SCALE = 6

# Integer arithmetic: features in units of 0.1 (grid 0..10, stops 0/5/10), theta in units of 0.5 (-2..2).
# Integer reward = FEATURE_SCALE * THETA_SCALE * true reward = 20 * true reward.
FEATURE_SCALE = 10
THETA_SCALE = 2

# Approximate paper values for n_obs 0, 1 and 5 (read from figures); validation targets only.
# Intermediate rounds are not recorded here.
TARGET_ACCURACY = {0: 0.33, 1: 0.58, 5: 0.81}

# Numbers of observed interaction rounds after which the assistant is evaluated on the heldout sets.
EVAL_OBS_ROUNDS = (0, 1, 2, 3, 4, 5)

# Delta degrees of freedom of the std of per-user means in the standard error (1 = sample std).
# Not specified by the step description; 0 vs 1 changes the SE by a factor sqrt(624/623).
SE_DDOF = 1

# shift_diagnostics (a heuristic; it cannot prove alignment). A file is flagged "suspicious" if a shifted
# interpretation beats the file as-is by more than SHIFT_DIAG_MARGIN (absolute success), or if its success is
# below SHIFT_DIAG_MIN_SUCCESS at some n_obs >= 1. The four summary numbers pool the declared n_obs >=
# SHIFT_DIAG_MIN_N_OBS (all declared n_obs if none is), because at n_obs = 0 every interpretation scores ~1/3.
SHIFT_DIAG_MARGIN = 0.05
SHIFT_DIAG_MIN_SUCCESS = 0.30
SHIFT_DIAG_MIN_N_OBS = 1

# scripts/evaluate_file.py prints a WARNING if the fraction of "rec": null records exceeds this at any n_obs.
FAIL_RATE_WARN = 0.05

# Data locations, relative to the repository root (./data is gitignored).
DATA_DIR = "data"
EVAL_INTERACTION_PATH = "data/eval/interaction/flight.jsonl"
EVAL_HELDOUT_PATH = "data/eval/heldout/flight.json"
TRAIN_BAYESIAN_PATH = "data/train/bayesian.jsonl"
TRAIN_ORACLE_PATH = "data/train/oracle.jsonl"

# Raw-value grids behind the [0, 1] features, verified against the Zenodo text/numeric pairs.
# Departure time: 06:00 AM to 10:00 PM in 11 steps of 96 minutes.
DEPARTURE_MINUTES_START = 6 * 60
DEPARTURE_STEP_MIN = 96
# Duration: 30 min to 20 hr in 11 steps of 117 minutes.
DURATION_START_MIN = 30
DURATION_STEP_MIN = 117
# Price: $100 to $1000 in 11 steps of $90.
PRICE_START = 100
PRICE_STEP = 90
