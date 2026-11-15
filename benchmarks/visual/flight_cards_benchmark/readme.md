# Flight Cards Benchmark

Generates PNG flight comparison cards from the Bayesian flight dataset. Each image displays three flight options, with the assistant-selected flight highlighted in green.

## Setup

Requires Python 3 and Pillow.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pillow
```

## Dataset

Place `bayesian.jsonl` inside `data/train/`, relative to the project folder.

## Generate PNGs

From the `flight_cards_benchmark` folder, run:

```bash
python generate.py --jsonl data/train/bayesian.jsonl
```

To generate only 10 samples:

```bash
python generate.py --jsonl data/train/bayesian.jsonl --count 10
```

## Output

Generated files are saved in `output/png/`:

```text
output/
└── png/
    ├── record_0001_round_01.png
    ├── record_0001_round_02.png
    └── ...
```

Each PNG contains three flight cards using the original JSONL values. The green **SELECTED** highlight indicates the assistant's choice, not necessarily the user's preferred option.

Generated PNGs are excluded from Git and can be recreated by running the script.