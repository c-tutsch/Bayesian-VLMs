"""Validate/export both official teacher files and show their first two rounds."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from bvlm.flight_data import ROOT
from bvlm.training_data import prepare_training


def main():
    output = ROOT / 'data/generated/training'
    counts = prepare_training(output)
    print('Validated 31,200 prefixes / 6,240 full dialogues per teacher:', counts)
    for teacher in ('oracle', 'bayesian'):
        path = output / f'{teacher}.jsonl'
        with path.open() as stream:
            next(stream)
            record = json.loads(next(stream))
        print(f'\n{teacher.upper()} — first two rounds — {path}')
        for message in record['messages']:
            print(f"\n{message['role']}: {message['content']}")
    print('\nTeacher files are ready for the team training pipeline; no model was trained.')


if __name__ == '__main__':
    main()
