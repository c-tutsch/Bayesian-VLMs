"""Check the official pair boundary, source copies and image-card handoff."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from bvlm.flight_data import ARCHIVE
from bvlm.training_data import prepare_training, training_card, validate_pair


@unittest.skipUnless(ARCHIVE.exists(), 'Download the official archive for integration tests')
class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prefixes = []
        with zipfile.ZipFile(ARCHIVE) as source:
            for teacher in ('oracle', 'bayesian'):
                with source.open(f'data/train/{teacher}.jsonl') as stream:
                    cls.prefixes.append([json.loads(next(stream)) for _ in range(5)])

    def test_real_pair_and_shared_public_training_card(self):
        previous = None
        for index in range(5):
            pair = [records[index] for records in self.prefixes]
            validate_pair(pair, index + 1, previous)
            for round_index in range(index + 1):
                first = training_card(pair[0], index, round_index)
                self.assertEqual(first, training_card(pair[1], index, round_index))
                self.assertEqual(set(first), {'prefix_row', 'profile_idx', 'round', 'options', 'card_text'})
            previous = pair
        self.assertEqual(self.prefixes[0][0]['messages'][1]['content'], 'The best option is Flight 1.')
        self.assertEqual(self.prefixes[1][0]['messages'][1]['content'], 'The best option is Flight 3.')
        self.assertIn('I prefer Flight 1.', self.prefixes[1][1]['messages'][2]['content'])

    def test_corrupt_pair_is_rejected(self):
        previous = [records[0] for records in self.prefixes]
        good = [records[1] for records in self.prefixes]
        for corruption in ('offer', 'profile', 'role', 'feedback', 'prefix'):
            pair = copy.deepcopy(good)
            if corruption == 'offer':
                pair[1]['messages'][2]['content'] = pair[1]['messages'][2]['content'].replace('Flight 1:', 'Flight 3:')
            elif corruption == 'profile':
                pair[1]['idx'] = 1
            elif corruption == 'role':
                pair[1]['messages'][2]['role'] = 'assistant'
            elif corruption == 'feedback':
                pair[1]['messages'][2]['content'] = pair[1]['messages'][2]['content'].replace('I prefer Flight 1.', 'I prefer Flight 2.')
            else:
                pair[1]['messages'][0]['content'] += 'changed'
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                validate_pair(pair, 2, previous)

    def test_all_prefixes_and_byte_exact_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(prepare_training(directory), {rounds: 6240 for rounds in range(1, 6)})
            with zipfile.ZipFile(ARCHIVE) as source:
                for teacher in ('oracle', 'bayesian'):
                    self.assertEqual((Path(directory) / f'{teacher}.jsonl').read_bytes(), source.read(f'data/train/{teacher}.jsonl'))

    def test_training_card_bounds(self):
        record = self.prefixes[0][0]
        for row, round_index in ((-1, 0), (31200, 0), (0, -1), (0, 1)):
            with self.assertRaises(ValueError):
                training_card(record, row, round_index)


if __name__ == '__main__':
    unittest.main()
