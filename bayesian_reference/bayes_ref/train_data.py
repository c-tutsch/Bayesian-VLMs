"""Streaming parser for the flight training files data/train/{bayesian,oracle}.jsonl.

File format (inspected on data/train/bayesian.jsonl):
Each line is a JSON object {"idx": int, "messages": [{"role": ..., "content": str}, ...]}.
One interaction is stored as 5 consecutive records with the same idx and 2, 4, 6, 8, 10 messages,
each shorter record being an exact prefix of the 10-message one. Roles alternate user, assistant.

User message 0 (instruction preamble, then the round-0 options):
    "Help me select the best flights for my trips. I have specific preferences for what I like and
    dislike in a flight, and these preferences remain the same. You need to figure out my preferences
    and select the best flights for me. Use your best judgment if you are unsure. Do not say you need
    more information.\\n\\nWhich flight is the best option?\\n\\n"
    "Flight 1:\\ndeparture time: 10:48 AM, duration: 14 hr 9 min, number of stops: 0, price: $190\\n"
    "Flight 2:\\n...\\nFlight 3:\\n..."
User messages 1..4 (feedback on the previous round, then the next options):
    "Your option Flight 3 is incorrect. I prefer Flight 1.\\n\\nWhich flight is the best option?\\n\\nFlight 1:\\n..."
    "Your option Flight 1 is correct.\\n\\nWhich flight is the best option?\\n\\nFlight 1:\\n..."
Assistant messages (one per round):
    "The best option is Flight 3."
Flight numbers in the text are 1-based; every function here returns 0-based indices.
The 10-message record carries no feedback on the round-4 recommendation.
"""

import json
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from bayes_ref import config
from bayes_ref.world import text_to_features

_FLIGHT_HEADER_RE = re.compile(r"Flight (\d+):")
_FEEDBACK_RE = re.compile(
    r"^\s*Your option Flight (?P<rec>\d+) is (?P<verdict>correct|incorrect)\."
    r"(?:\s*I prefer Flight (?P<pref>\d+)\.)?",
    re.IGNORECASE,
)
_ASSISTANT_RE = re.compile(r"^\s*The best option is Flight (\d+)\.\s*$")

_N_MESSAGES_FULL = 2 * config.N_ROUNDS
_N_PREFIX_CHECK = 200


def _flight_index(number: str, text: str) -> int:
    i = int(number) - 1
    if not 0 <= i < config.N_OPTIONS_PER_SET:
        raise ValueError(f"flight number {number} out of range in {text!r}")
    return i


def parse_options(text: str) -> np.ndarray:
    """Parse the "Flight 1:" .. "Flight 3:" blocks of a user message. Shape (3, 4) float64.

    Each block runs from its header to the next header (or the end of the text) and must parse with
    text_to_features. Raises ValueError unless the headers are exactly Flight 1, 2, 3 in order.
    """
    headers = list(_FLIGHT_HEADER_RE.finditer(text))
    numbers = [int(h.group(1)) for h in headers]
    expected = list(range(1, config.N_OPTIONS_PER_SET + 1))
    if numbers != expected:
        raise ValueError(f"expected flight headers {expected}, found {numbers} in {text!r}")
    rows = []
    for j, h in enumerate(headers):
        end = headers[j + 1].start() if j + 1 < len(headers) else len(text)
        rows.append(text_to_features(text[h.end():end].strip()))
    return np.array(rows, dtype=np.float64)


def parse_feedback(text: str) -> tuple[int, bool, int | None]:
    """Parse the leading feedback of a user message into (rec, correct, preferred), 0-based.

    preferred is None when the feedback says correct. Raises ValueError if the text does not start with
    feedback, if "incorrect" has no preferred flight, or if "correct" has one.
    """
    m = _FEEDBACK_RE.match(text)
    if m is None:
        raise ValueError(f"no feedback found at the start of {text!r}")
    rec = _flight_index(m["rec"], text)
    correct = m["verdict"].lower() == "correct"
    if correct:
        if m["pref"] is not None:
            raise ValueError(f"'correct' feedback names a preferred flight: {text!r}")
        return rec, True, None
    if m["pref"] is None:
        raise ValueError(f"'incorrect' feedback without a preferred flight: {text!r}")
    preferred = _flight_index(m["pref"], text)
    if preferred == rec:
        raise ValueError(f"'incorrect' feedback prefers the recommended flight: {text!r}")
    return rec, False, preferred


def parse_assistant(text: str) -> int:
    """Parse "The best option is Flight N." into the 0-based index N - 1. Raises ValueError otherwise."""
    m = _ASSISTANT_RE.match(text)
    if m is None:
        raise ValueError(f"unparseable assistant message: {text!r}")
    return _flight_index(m.group(1), text)


def _parse_full_record(rec: dict) -> dict:
    msgs = rec["messages"]
    roles = [m["role"] for m in msgs]
    if roles != ["user", "assistant"] * config.N_ROUNDS:
        raise ValueError(f"idx {rec['idx']}: unexpected role sequence {roles}")
    user = [m["content"] for m in msgs[0::2]]
    assistant = [m["content"] for m in msgs[1::2]]

    options = np.array([parse_options(t) for t in user], dtype=np.float64)
    recs = np.array([parse_assistant(t) for t in assistant], dtype=np.int64)
    choices = np.empty(config.N_ROUNDS - 1, dtype=np.int64)
    for r in range(config.N_ROUNDS - 1):
        fb_rec, correct, preferred = parse_feedback(user[r + 1])
        if fb_rec != recs[r]:
            raise ValueError(
                f"idx {rec['idx']} round {r}: assistant recommended {recs[r]}, feedback refers to {fb_rec}"
            )
        choices[r] = fb_rec if correct else preferred
    if _FEEDBACK_RE.match(user[0]):
        raise ValueError(f"idx {rec['idx']}: first user message unexpectedly starts with feedback")
    return {"idx": rec["idx"], "options": options, "recs": recs, "choices": choices}


def _check_prefix_group(group: list[dict]) -> None:
    lengths = [len(r["messages"]) for r in group]
    expected = list(range(2, _N_MESSAGES_FULL + 1, 2))
    if lengths != expected:
        raise ValueError(
            f"idx {group[0]['idx']}: consecutive records have message counts {lengths}, expected {expected}"
        )
    full = group[-1]["messages"]
    for r in group[:-1]:
        if r["messages"] != full[: len(r["messages"])]:
            raise ValueError(
                f"idx {group[0]['idx']}: the {len(r['messages'])}-message record is not a prefix of the full one"
            )


def iter_interactions(path: str | Path) -> Iterator[dict]:
    """Stream the full (10-message) interactions of a training file.

    Yields dicts with idx (int), options (5, 3, 4) float64, recs (5,) int64 (assistant recommendation per
    round) and choices (4,) int64 (the user's choice in rounds 0..3 from the feedback: the recommendation
    if correct, else the preferred flight). For the first 200 interactions, raises ValueError unless they
    are stored as consecutive prefix records with 2, 4, 6, 8, 10 messages.
    """
    group: list[dict] = []
    n_groups_checked = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            is_full = len(rec["messages"]) == _N_MESSAGES_FULL
            if n_groups_checked < _N_PREFIX_CHECK:
                if group and rec["idx"] != group[0]["idx"]:
                    _check_prefix_group(group)  # group ended without a 10-message record: raises
                group.append(rec)
                if is_full:
                    _check_prefix_group(group)
                    n_groups_checked += 1
                    group = []
            if is_full:
                yield _parse_full_record(rec)
    if group:
        _check_prefix_group(group)
