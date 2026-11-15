"""Run the same flight cases as plain text and OCR images using Ollama."""

import argparse
import base64
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
        if mode == "ocr":
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


def score(pairs):
    total = len(pairs)
    result = {"cases": total}
    for mode in ("text", "ocr"):
        correct = sum(pair[mode] == pair["answer"] for pair in pairs)
        result[mode] = {
            "correct": correct,
            "accuracy": correct / total,
            "invalid": sum(pair[mode] is None for pair in pairs),
        }
    result["same_recommendation"] = sum(
        pair["text"] is not None and pair["text"] == pair["ocr"] for pair in pairs
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--users", type=int, nargs="+", default=[0])
    parser.add_argument("--sets", type=int, nargs="+", default=[1])
    parser.add_argument("--n-obs", type=int, nargs="+", default=[2])
    parser.add_argument("--cards", type=Path, default=ROOT / "data/generated/ocr")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    try:
        if any(len(values) != len(set(values)) for values in (args.users, args.sets, args.n_obs)):
            raise ValueError("User, set and observation lists must each contain unique values.")
        dataset = PaperDataset(ROOT / "benchmarks/text/data")
        cases = [
            dataset.case(user, n_obs, set_index)
            for user, n_obs, set_index in itertools.product(args.users, args.n_obs, args.sets)
        ]
        with (args.cards / "manifest.jsonl").open(encoding="utf-8") as source:
            manifest = {row["path"]: row["options"] for row in map(json.loads, source)}
        models = request_json(args.url, "/api/tags")["models"]
        model = next((item for item in models if item["name"] == args.model), None)
        if model is None:
            raise ValueError(f"Model not found on Ollama. Run: ollama pull {args.model}")
        settings = {"model": args.model, "think": False, "options": OPTIONS, "format": FORMAT}
        metadata = {"ollama_version": request_json(args.url, "/api/version")["version"],
                    "model_details": model, "settings": settings}
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        pairs = []
        with (
            (args.out / "responses.jsonl").open("x", encoding="utf-8", buffering=1) as raw,
            (args.out / "text.jsonl").open("x", encoding="utf-8", buffering=1) as text,
            (args.out / "ocr.jsonl").open("x", encoding="utf-8", buffering=1) as ocr,
        ):
            for index, case in enumerate(cases, start=1):
                ids = {key: case[key] for key in ("user", "n_obs", "set")}
                pair = dict(ids)
                for mode, predictions in (("text", text), ("ocr", ocr)):
                    message = make_message(case, mode, args.cards, manifest)
                    response = request_json(args.url, "/api/chat", {
                        **settings, "messages": [message], "stream": False, "keep_alive": "10m",
                    })
                    rec = recommendation(response["message"]["content"])
                    raw.write(json.dumps({**ids, "mode": mode, "rec": rec, "response": response}) + "\n")
                    if rec is not None:
                        predictions.write(json.dumps({**ids, "rec": rec}) + "\n")
                    pair[mode] = rec
                # Read the stored choice only after both prompts have been answered.
                _, _, pair["answer"] = dataset.offer(case["user"], case["set"])
                pairs.append(pair)
                print(f"{index}/{len(cases)} user={ids['user']} n_obs={ids['n_obs']} set={ids['set']}"
                      f" rec (0–2) text={pair['text']} ocr={pair['ocr']} target={pair['answer']}", flush=True)
        summary = {
            "overall": score(pairs),
            "by_n_obs": {n: score([pair for pair in pairs if pair["n_obs"] == n]) for n in args.n_obs},
            "pairs": pairs,
        }
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Saved paired results in {args.out.resolve()}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Flight test error: {error}\n")


if __name__ == "__main__":
    main()
