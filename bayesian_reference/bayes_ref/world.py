"""The flight world: feature grids, the flight universe, user reward vectors, text parsing."""

import itertools
import re

import numpy as np

from bayes_ref import config

_DEPARTURE_SPAN_MIN = config.DEPARTURE_STEP_MIN * (config.N_LEVELS_CONTINUOUS - 1)
_DURATION_SPAN_MIN = config.DURATION_STEP_MIN * (config.N_LEVELS_CONTINUOUS - 1)
_PRICE_SPAN = config.PRICE_STEP * (config.N_LEVELS_CONTINUOUS - 1)

_OPTION_RE = re.compile(
    r"^\s*(?:Flight \d+:\s*)?"
    r"departure time: (?P<hour>\d{1,2}):(?P<minute>\d{2}) (?P<ampm>AM|PM), "
    r"duration: (?P<duration>(?:\d+ hr)?(?: ?\d+ min)?), "
    r"number of stops: (?P<stops>\d+), "
    r"price: \$(?P<price>\d+(?:\.\d+)?)\s*$"
)
_DURATION_RE = re.compile(r"^(?:(?P<hr>\d+) hr)?(?: ?(?P<min>\d+) min)?$")


def continuous_levels() -> np.ndarray:
    """Levels of departure_time, duration and price. Shape (11,)."""
    return np.linspace(0.0, 1.0, config.N_LEVELS_CONTINUOUS)


def stops_levels() -> np.ndarray:
    """Levels of the stops feature. Shape (3,)."""
    return np.array(config.STOPS_VALUES, dtype=np.float64)


def all_flights() -> np.ndarray:
    """Every possible flight, one row per feature combination. Shape (3 * 11**3, 4) = (3993, 4).

    Columns follow config.FEATURE_NAMES; rows follow itertools.product order over those columns.
    """
    cont = continuous_levels()
    per_feature = {
        "departure_time": cont,
        "duration": cont,
        "stops": stops_levels(),
        "price": cont,
    }
    grids = [per_feature[name] for name in config.FEATURE_NAMES]
    return np.array(list(itertools.product(*grids)), dtype=np.float64)


def build_theta() -> np.ndarray:
    """All user reward vectors: product of THETA_VALUES (repeat=4) minus the all-zero row. Shape (624, 4)."""
    rows = [
        row
        for row in itertools.product(config.THETA_VALUES, repeat=config.N_FEATURES)
        if any(v != 0.0 for v in row)
    ]
    theta = np.array(rows, dtype=np.float64)
    assert theta.shape == (config.N_USERS_EXPECTED, config.N_FEATURES), theta.shape
    return theta


def _parse_duration_minutes(text: str) -> int:
    """Parse "N hr M min", "N hr" or "N min" into total minutes."""
    m = _DURATION_RE.match(text)
    if m is None or (m["hr"] is None and m["min"] is None):
        raise ValueError(f"Unparseable duration: {text!r}")
    return int(m["hr"] or 0) * 60 + int(m["min"] or 0)


def text_to_features(text: str) -> np.ndarray:
    """Parse one flight description into its feature vector. Shape (4,), order config.FEATURE_NAMES.

    Accepts e.g. "departure time: 02:00 PM, duration: 30 min, number of stops: 1, price: $370",
    optionally prefixed by "Flight N: ". Values are rounded to 1e-9 so they land exactly on the grid.
    Raises ValueError if the text does not match this format.
    """
    m = _OPTION_RE.match(text)
    if m is None:
        raise ValueError(f"Unparseable flight text: {text!r}")

    hour, minute = int(m["hour"]), int(m["minute"])
    if not (1 <= hour <= 12 and 0 <= minute < 60):
        raise ValueError(f"Invalid clock time in: {text!r}")
    hour24 = hour % 12 + (12 if m["ampm"] == "PM" else 0)
    departure = (hour24 * 60 + minute - config.DEPARTURE_MINUTES_START) / _DEPARTURE_SPAN_MIN

    duration = (_parse_duration_minutes(m["duration"]) - config.DURATION_START_MIN) / _DURATION_SPAN_MIN
    stops = int(m["stops"]) / (config.N_LEVELS_STOPS - 1)
    price = (float(m["price"]) - config.PRICE_START) / _PRICE_SPAN

    by_name = {"departure_time": departure, "duration": duration, "stops": stops, "price": price}
    return np.array([round(by_name[name], 9) for name in config.FEATURE_NAMES], dtype=np.float64)
