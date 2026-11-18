# Flight Cards Benchmark

Generates PNG flight comparison cards from the Bayesian flight recommendation dataset used in *Bayesian Teaching Enables Probabilistic Reasoning in Large Language Models*.

Each image displays three flight options with their departure time, duration, number of stops, and price.

Two generation scripts are available:

- **`generate.py`** — Generates plain flight cards without any selected option or preference highlighting. Intended for VLM benchmarking.
- **`selected_generate.py`** — Generates flight cards with the simulated user's preferred flight highlighted in green, based on feedback available in the dataset.

## Setup

Requires Python 3 and Pillow.

From the `flight_cards_benchmark` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install pillow
```

## Dataset

Place `bayesian.jsonl` inside `data/train/`, relative to the `flight_cards_benchmark` directory.

```text
flight_cards_benchmark/
├── data/
│   └── train/
│       └── bayesian.jsonl
├── generate.py
├── selected_generate.py
├── readme.md
└── output/
    ├── png/
    └── png_selected/
```

## Generate PNGs

Run the following commands from the `flight_cards_benchmark` directory.

### 1. Plain Flight Cards

Generate flight cards without highlighting any option:

```bash
python3 generate.py --jsonl data/train/bayesian.jsonl
```

Generate only 10 samples:

```bash
python3 generate.py --jsonl data/train/bayesian.jsonl --count 10
```

**Output directory:** `output/png/`

### 2. User-Selected Flight Cards

Generate cards highlighting the user's preferred flight when feedback is available:

```bash
python3 selected_generate.py --jsonl data/train/bayesian.jsonl
```

Generate only 10 samples:

```bash
python3 selected_generate.py --jsonl data/train/bayesian.jsonl --count 10
```

**Output directory:** `output/png_selected/`

The selected option is derived from the user's feedback rather than directly from the Bayesian Assistant's recommendation.

If feedback is unavailable for a particular question, the flight cards remain unselected.

## Output

Both scripts generate PNG images and a `manifest.json` file.

```text
output/
├── png/
│   ├── record_00001_round_01.png
│   ├── record_00002_round_02.png
│   └── manifest.json
└── png_selected/
    ├── record_0001_round_01.png
    ├── record_0002_round_02.png
    └── manifest.json
```

Each PNG contains three flight options populated using the original dataset values.

### Difference Between Outputs

| Feature | Plain cards | User-selected cards |
|---|---|---|
| Flight attributes | Yes | Yes |
| Three flight options | Yes | Yes |
| Green selected highlight | No | When feedback exists |
| Uses user feedback | No | Yes |
| Intended use | VLM benchmark inputs | Feedback visualization |

### Manifest

The scripts also save metadata in `manifest.json`, including:

- Generated image filename
- Original dataset record index
- Round number
- User choice (selected-card version only)

## Notes

- The scripts use the original flight information from the JSONL dataset.
- The configured generators produce one image per processed record, using its latest flight question.
- Plain cards do not reveal the correct answer and are suitable as model inputs.
- Selected cards are intended for visualizing user feedback and should not be shown to a VLM before it makes its prediction.
- Generated PNG files are excluded from Git through `.gitignore` and can be recreated using the generation scripts.