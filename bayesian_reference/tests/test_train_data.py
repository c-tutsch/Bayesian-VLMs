"""Tests for bayes_ref.train_data (data tests are skipped when ./data is missing)."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from bayes_ref import config
from bayes_ref.choice import to_int_features
from bayes_ref.train_data import iter_interactions, parse_assistant, parse_feedback, parse_options
from bayes_ref.world import text_to_features

ROOT = Path(__file__).resolve().parent.parent
TRAIN_BAYESIAN = ROOT / config.TRAIN_BAYESIAN_PATH

needs_train = pytest.mark.skipif(not TRAIN_BAYESIAN.exists(), reason=f"missing {TRAIN_BAYESIAN}")

# Copied from data/train/bayesian.jsonl, idx 0.
FIRST_USER = (
    "Help me select the best flights for my trips. I have specific preferences for what I like and dislike in a "
    "flight, and these preferences remain the same. You need to figure out my preferences and select the best "
    "flights for me. Use your best judgment if you are unsure. Do not say you need more information.\n\n"
    "Which flight is the best option?\n\n"
    "Flight 1:\ndeparture time: 10:48 AM, duration: 14 hr 9 min, number of stops: 0, price: $190\n"
    "Flight 2:\ndeparture time: 06:00 AM, duration: 18 hr 3 min, number of stops: 1, price: $820\n"
    "Flight 3:\ndeparture time: 08:24 PM, duration: 8 hr 18 min, number of stops: 0, price: $1000"
)
INCORRECT_USER = (
    "Your option Flight 3 is incorrect. I prefer Flight 1.\n\nWhich flight is the best option?\n\n"
    "Flight 1:\ndeparture time: 05:12 PM, duration: 2 hr 27 min, number of stops: 0, price: $640\n"
    "Flight 2:\ndeparture time: 09:12 AM, duration: 12 hr 12 min, number of stops: 1, price: $820\n"
    "Flight 3:\ndeparture time: 10:48 AM, duration: 2 hr 27 min, number of stops: 1, price: $910"
)
CORRECT_USER = (
    "Your option Flight 1 is correct.\n\nWhich flight is the best option?\n\n"
    "Flight 1:\ndeparture time: 02:00 PM, duration: 20 hr, number of stops: 2, price: $280\n"
    "Flight 2:\ndeparture time: 06:00 AM, duration: 30 min, number of stops: 2, price: $550\n"
    "Flight 3:\ndeparture time: 05:12 PM, duration: 10 hr 15 min, number of stops: 1, price: $550"
)
ASSISTANT = "The best option is Flight 3."


def test_parse_options_first_message():
    expected = np.array([
        [0.3, 0.7, 0.0, 0.1],
        [0.0, 0.9, 0.5, 0.8],
        [0.9, 0.4, 0.0, 1.0],
    ])
    np.testing.assert_allclose(parse_options(FIRST_USER), expected, rtol=0, atol=1e-9)


def test_parse_options_feedback_messages():
    o = parse_options(INCORRECT_USER)
    assert o.shape == (3, 4)
    np.testing.assert_allclose(o[0], [0.7, 0.1, 0.0, 0.6], rtol=0, atol=1e-9)
    np.testing.assert_allclose(parse_options(CORRECT_USER)[1], [0.0, 0.0, 1.0, 0.5], rtol=0, atol=1e-9)


def test_parse_options_same_line():
    text = INCORRECT_USER.replace("Flight 2:\n", "Flight 2: ")
    np.testing.assert_array_equal(parse_options(text), parse_options(INCORRECT_USER))


@pytest.mark.parametrize("bad", [
    INCORRECT_USER.replace("Flight 3:", "Flight 4:"),
    INCORRECT_USER.split("Flight 3:")[0],
    INCORRECT_USER + "\nsomething else",
    INCORRECT_USER.replace("number of stops: 1, price: $910", "price: $910"),
])
def test_parse_options_fails_loudly(bad):
    with pytest.raises(ValueError):
        parse_options(bad)


def test_parse_feedback():
    assert parse_feedback(INCORRECT_USER) == (2, False, 0)
    assert parse_feedback(CORRECT_USER) == (0, True, None)
    assert parse_feedback("your option flight 2 is INCORRECT. i prefer flight 3.") == (1, False, 2)
    for bad in (FIRST_USER, "Your option Flight 1 is incorrect.", "Your option Flight 4 is correct."):
        with pytest.raises(ValueError):
            parse_feedback(bad)


def test_parse_assistant():
    assert parse_assistant(ASSISTANT) == 2
    for bad in ("The best option is Flight 4.", "Flight 1", "The best option is Flight 1. Because ..."):
        with pytest.raises(ValueError):
            parse_assistant(bad)


def _raw_full_records(n: int) -> list[dict]:
    out = []
    with open(TRAIN_BAYESIAN, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if len(rec["messages"]) == 2 * config.N_ROUNDS:
                out.append(rec)
                if len(out) == n:
                    return out
    return out


@needs_train
def test_first_50_interactions():
    parsed = list(itertools.islice(iter_interactions(TRAIN_BAYESIAN), 50))
    raw = _raw_full_records(50)
    assert len(parsed) == 50
    for it, rec in zip(parsed, raw, strict=True):
        assert it["idx"] == rec["idx"]
        assert it["options"].shape == (5, 3, 4)
        assert it["recs"].shape == (5,) and it["recs"].dtype == np.int64
        assert it["choices"].shape == (4,) and it["choices"].dtype == np.int64
        to_int_features(it["options"])  # on the grid

        user = [m["content"] for m in rec["messages"][0::2]]
        for r, text in enumerate(user):
            lines = [ln for ln in text.splitlines() if ln.startswith("departure time:")]
            expected = np.array([text_to_features(ln) for ln in lines])
            np.testing.assert_array_equal(it["options"][r], expected)
        for r in range(config.N_ROUNDS - 1):
            rec_fb, correct, preferred = parse_feedback(user[r + 1])
            assert it["recs"][r] == rec_fb
            assert it["choices"][r] == (rec_fb if correct else preferred)
