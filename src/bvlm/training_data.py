"""Validate the published teacher pairs and expose their offers to renderers."""

from itertools import zip_longest
import json
from pathlib import Path
import re
import zipfile

from bvlm.flight_cards import render_flight_card
from bvlm.flight_data import ARCHIVE, verify_archive

FLIGHTS = re.compile(r'^Flight ([1-3]):\n([^\n]+)', re.MULTILINE)
ANSWER = re.compile(r'The best option is Flight ([1-3])\.')
FEEDBACK = re.compile(r'Your option Flight ([1-3]) is (correct\.|incorrect\. I prefer Flight ([1-3])\.)')


def training_options(message):
    matches = FLIGHTS.findall(message['content'])
    if [label for label, _ in matches] != ['1', '2', '3']:
        raise ValueError('Expected Flight 1/2/3 in original order')
    options = [f'Flight {label}: {description}' for label, description in matches]
    render_flight_card(options)
    return options


def training_card(record, prefix_row, round_index):
    """Public offer only; prefix_row identifies the paired JSONL line (zero-based)."""
    if type(prefix_row) is not int or not 0 <= prefix_row < 31200:
        raise ValueError('Training prefix row must be 0..31199')
    if type(round_index) is not int or not 0 <= round_index < len(record['messages']) // 2:
        raise ValueError('Round is outside this dialogue prefix')
    options = training_options(record['messages'][2 * round_index])
    return {'prefix_row': prefix_row, 'profile_idx': record['idx'], 'round': round_index,
            'options': options, 'card_text': render_flight_card(options)}


def validate_pair(pair, rounds, previous=None):
    for record in pair:
        messages = record['messages']
        if len(messages) != rounds * 2 or type(record['idx']) is not int or not 0 <= record['idx'] < 624:
            raise ValueError('Unexpected dialogue prefix length or profile ID')
        if [message['role'] for message in messages] != ['user', 'assistant'] * rounds:
            raise ValueError('Unexpected message roles')
        if any(ANSWER.fullmatch(message['content']) is None for message in messages[1::2]):
            raise ValueError('Unexpected teacher recommendation')
    if pair[0]['idx'] != pair[1]['idx']:
        raise ValueError('Teacher profile IDs differ')
    for first, second in zip(pair[0]['messages'][::2], pair[1]['messages'][::2]):
        if training_options(first) != training_options(second):
            raise ValueError('Teacher flight offers differ')
    if rounds == 1:
        return
    if previous is None:
        raise ValueError('Missing preceding dialogue prefix')
    truths = []
    for old, new in zip(previous, pair):
        if old['idx'] != new['idx'] or old['messages'] != new['messages'][:-2]:
            raise ValueError('Dialogue prefixes are inconsistent')
        recommendation = ANSWER.fullmatch(old['messages'][-1]['content'])[1]
        feedback = FEEDBACK.fullmatch(new['messages'][-2]['content'].split('\n')[0])
        if feedback is None or feedback[1] != recommendation:
            raise ValueError('Feedback does not reference preceding recommendation')
        truths.append(feedback[3] or feedback[1])
    oracle_feedback = pair[0]['messages'][-2]['content'].split('\n')[0]
    if truths[0] != truths[1] or 'incorrect' in oracle_feedback:
        raise ValueError('Teachers disagree on actual user choice')


def prepare_training(output, archive=ARCHIVE):
    verify_archive(archive)
    output = Path(output)
    counts = {rounds: 0 for rounds in range(1, 6)}
    with zipfile.ZipFile(archive) as source:
        with source.open('data/train/oracle.jsonl') as oracle, source.open('data/train/bayesian.jsonl') as bayes:
            previous = None
            for row, lines in enumerate(zip_longest(oracle, bayes)):
                if None in lines:
                    raise ValueError('Teacher files have different lengths')
                pair = [json.loads(line) for line in lines]
                rounds = row % 5 + 1
                validate_pair(pair, rounds, previous)
                counts[rounds] += 1
                previous = pair
        if counts != {rounds: 6240 for rounds in range(1, 6)}:
            raise ValueError(f'Unexpected prefix counts: {counts}')
        payloads = {teacher: source.read(f'data/train/{teacher}.jsonl') for teacher in ('oracle', 'bayesian')}
    # Check both destinations before writing either file.
    for teacher, payload in payloads.items():
        target = output / f'{teacher}.jsonl'
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f'Refusing to replace different data at {target}')
    output.mkdir(parents=True, exist_ok=True)
    for teacher, payload in payloads.items():
        target = output / f'{teacher}.jsonl'
        if not target.exists():
            target.write_bytes(payload)
    return counts
