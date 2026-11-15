"""Run the same flight cases as plain text and OCR images and flight diagrams using Ollama."""

import argparse
import base64
from contextlib import ExitStack
import itertools
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bvlm.flight_data import PaperDataset

MODEL = "qwen3.5:4b"
CARD_ROOT = ROOT / "data/generated"
FORMATS = ("text", "ocr", "diagrams", "diagrams-noise")
CASE_KEYS = ("user", "n_obs", "set")
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 32}
FORMAT = {
    "type": "object",
    "properties": {"flight": {"type": "integer", "enum": [1, 2, 3]}},
    "required": ["flight"],
    "additionalProperties": False,
}
QUESTION = "Which flight would this user prefer in the test offer? Return JSON with the flight number (1, 2, or 3)."


def request_json(url, route, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(url.rstrip("/") + route, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=300) as response:
            return json.load(response)
    except HTTPError as error:
        raise ValueError(f"Ollama returned HTTP {error.code}: {error.read().decode(errors='replace')}") from error
    except URLError as error:
        raise ValueError(f"Cannot reach Ollama at {url}: {error.reason}") from error


def make_message(case, mode, cards, manifest):
    """Keep the same known earlier choices in both input formats."""
    user, set_index = case["user"], case["set"]
    offers = [
        (f"Earlier offer {index + 1}", round_["options"],
         f"The user chose Flight {round_['user_idx'] + 1}.",
         f"interaction/user-{user:03d}/round-{index}.png")
        for index, round_ in enumerate(case["history"])
    ]
    offers.append(("Test offer", case["options"], QUESTION, f"heldout/set-{set_index:03d}.png"))
    blocks, images = [], []
    for number, (heading, options, ending, relative_path) in enumerate(offers, start=1):
        if mode != "text":
            if manifest.get(relative_path) != options:
                raise ValueError(f"Card manifest does not match the source offer: {relative_path}")
            images.append(base64.b64encode((cards / relative_path).read_bytes()).decode("ascii"))
            display = f"See attached image {number}."
        else:
            display = "\n".join(options)
        blocks.append(f"{heading}\n{display}\n{ending}")
    message = {
        "role": "user",
        "content": "You are helping the same user choose a flight.\n\n" + "\n\n".join(blocks),
    }
    if images:
        message["images"] = images
    return message


def recommendation(content):
    try:
        flight = json.loads(content)["flight"]
    except (ValueError, TypeError, KeyError):
        return None
    return flight - 1 if type(flight) is int and flight in (1, 2, 3) else None


def score(pairs, formats):
    total = len(pairs)
    result = {"cases": total}
    if all("bayes_recs" in pair for pair in pairs):
        result["bayes_reference"] = {
            "expected_accuracy": sum(
                (pair["answer"] in pair["bayes_recs"]) / len(pair["bayes_recs"]) for pair in pairs
            ) / total,
            "unique_cases": sum(len(pair["bayes_recs"]) == 1 for pair in pairs),
        }
    for mode in formats:
        correct = sum(pair[mode] == pair["answer"] for pair in pairs)
        result[mode] = {
            "correct": correct,
            "accuracy": correct / total,
            "invalid": sum(pair[mode] is None for pair in pairs),
        }
        if mode != "text":
            result[mode]["same_as_text"] = sum(
                pair[mode] is not None and pair[mode] == pair["text"] for pair in pairs
            )
        if "bayes_reference" in result:
            result[mode]["same_as_unique_bayes"] = sum(
                len(pair["bayes_recs"]) == 1 and pair[mode] == pair["bayes_recs"][0]
                for pair in pairs
            )
    result["same_recommendation"] = result["ocr"]["same_as_text"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--users", type=int, nargs="+", default=[0])
    parser.add_argument("--sets", type=int, nargs="+", default=[1])
    parser.add_argument("--n-obs", type=int, nargs="+", default=[2])
    parser.add_argument("--all-formats", action="store_true",
                        help="Compare text, OCR, clean diagrams and diagrams with noise")
    parser.add_argument("--cards", type=Path, help="Override the image folder for the two-format pilot")
    parser.add_argument("--bayes-reference", type=Path,
                        help="Optional JSON list of cases with target and zero-based bayes_recs")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    try:
        if any(len(values) != len(set(values)) for values in (args.users, args.sets, args.n_obs)):
            raise ValueError("User, set and observation lists must each contain unique values.")
        if args.all_formats and args.cards is not None:
            raise ValueError("--cards is only available for the two-format pilot.")
        formats = FORMATS if args.all_formats else FORMATS[:2]
        dataset = PaperDataset(ROOT / "benchmarks/text/data")
        cases = [
            dataset.case(user, n_obs, set_index)
            for user, n_obs, set_index in itertools.product(args.users, args.n_obs, args.sets)
        ]
        card_inputs = {"text": (None, None)}
        for mode in formats[1:]:
            directory = args.cards or (CARD_ROOT / mode)
            with (directory / "manifest.jsonl").open(encoding="utf-8") as manifest_file:
                manifest = {row["path"]: row["options"] for row in map(json.loads, manifest_file)}
            card_inputs[mode] = (directory, manifest)

        references = {}
        if args.bayes_reference is not None:
            rows = json.loads(args.bayes_reference.read_text(encoding="utf-8"))
            references = {tuple(row[key] for key in CASE_KEYS): row for row in rows}
            if len(references) != len(rows):
                raise ValueError("Duplicate cases in the Bayesian reference.")
            for case in cases:
                ids = tuple(case[key] for key in CASE_KEYS)
                _, _, answer = dataset.offer(case["user"], case["set"])
                reference = references.get(ids)
                choices = reference["bayes_recs"] if reference else []
                if (not choices or any(type(rec) is not int or rec not in range(3) for rec in choices)
                        or len(choices) != len(set(choices)) or reference["target"] != answer):
                    raise ValueError(f"Bayesian reference does not match case {ids}.")
        models = request_json(args.url, "/api/tags")["models"]
        model = next((item for item in models if item["name"] == args.model), None)
        if model is None:
            raise ValueError(f"Model not found on Ollama. Run: ollama pull {args.model}")
        settings = {"model": args.model, "think": False, "options": OPTIONS, "format": FORMAT}
        metadata = {"ollama_version": request_json(args.url, "/api/version")["version"],
                    "model_details": model, "settings": settings, "formats": formats,
                    "case_grid": {"users": args.users, "sets": args.sets, "n_obs": args.n_obs},
                    "card_folders": {mode: str(card_inputs[mode][0].resolve()) for mode in formats[1:]},
                    "bayes_reference": str(args.bayes_reference.resolve()) if args.bayes_reference else None}
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        pairs = []
        with ExitStack() as files:
            raw = files.enter_context((args.out / "responses.jsonl").open("x", encoding="utf-8", buffering=1))
            predictions = {
                mode: files.enter_context((args.out / f"{mode}.jsonl").open("x", encoding="utf-8", buffering=1))
                for mode in formats
            }
            for index, case in enumerate(cases, start=1):
                ids = {key: case[key] for key in CASE_KEYS}
                pair = dict(ids)
                for mode in formats:
                    message = make_message(case, mode, *card_inputs[mode])
                    response = request_json(args.url, "/api/chat", {
                        **settings, "messages": [message], "stream": False, "keep_alive": "10m",
                    })
                    rec = recommendation(response["message"]["content"])
                    raw.write(json.dumps({**ids, "mode": mode, "rec": rec, "response": response}) + "\n")
                    if rec is not None:
                        predictions[mode].write(json.dumps({**ids, "rec": rec}) + "\n")
                    pair[mode] = rec
                # Use the stored choice only for scoring.
                _, _, pair["answer"] = dataset.offer(case["user"], case["set"])
                if references:
                    pair["bayes_recs"] = references[tuple(ids.values())]["bayes_recs"]
                pairs.append(pair)
                picks = " ".join(f"{mode}={pair[mode]}" for mode in formats)
                print(f"{index}/{len(cases)} user={ids['user']} n_obs={ids['n_obs']} set={ids['set']}"
                      f" rec (0–2) {picks} target={pair['answer']}", flush=True)
        summary = {
            "overall": score(pairs, formats),
            "by_n_obs": {n: score([pair for pair in pairs if pair["n_obs"] == n], formats) for n in args.n_obs},
            "pairs": pairs,
        }
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Saved results in {args.out.resolve()}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Flight test error: {error}\n")


if __name__ == "__main__":
    main()
