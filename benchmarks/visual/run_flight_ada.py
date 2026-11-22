"""Run the interactive flight benchmark with text and PNG inputs using vLLM."""
import argparse
from collections.abc import Iterable
from contextlib import contextmanager
import fcntl
import hashlib
from itertools import islice
import json
from pathlib import Path
import re
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from bvlm.flight_data import PaperDataset

FORMATS = ("text", "ocr", "diagrams", "diagrams-noise")
MODEL = "Qwen/Qwen3.5-9B"
REFERENCE_COMMIT = "9128b3fc4ab31ac16972010eed8eff654d3c2fb7"
# Dialogue wording and decoding follow the text benchmark at REFERENCE_COMMIT.
INTRO = (
    "Help me select the best flights for my trips. I have specific preferences for what I like and dislike in a "
    "flight, and these preferences remain the same. You need to figure out my preferences and select the best "
    "flights for me. Use your best judgment if you are unsure. Do not say you need more information."
)
QUESTION = "Which flight is the best option?"
PREFILL = "The best option is Flight"
SAMPLING = dict(temperature=0, max_tokens=4, stop=["."], include_stop_str_in_output=True)
END_OF_THOUGHT = re.compile(r"</think>|<channel\|>")
DATA = ROOT / "benchmarks/text/data"


def recommendation(text):
    for pattern in (
        r"best option is:?\W*Flight\s*([123])",
        r"Flight\s*([123])\W*(?:is|would be|seems to be|appears to be)\s+(?:the\s+)?(?:best|most suitable)",
    ):
        if matches := re.findall(pattern, text, re.I):
            return int(matches[-1]) - 1
    matches = set(re.findall(r"Flight\s*([123])", text, re.I))
    return int(matches.pop()) - 1 if len(matches) == 1 else None


def feedback(rec, choice):
    if rec is None:
        return f"I prefer Flight {choice + 1}."
    if rec == choice:
        return f"Your option Flight {rec + 1} is correct."
    return f"Your option Flight {rec + 1} is incorrect. I prefer Flight {choice + 1}."


def user_message(prefix, options, image=None):
    text = f"{prefix}\n\n{QUESTION}\n\n"
    if image is None:
        content = text + "\n".join(options)
    else:
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image.as_uri()}},
        ]
    return {"role": "user", "content": content}


def read_rows(path):
    with path.open(encoding="utf-8") as rows:
        yield from map(json.loads, rows)


def write_rows(path, rows):
    """A finished file is the checkpoint. Incomplete files never replace it."""
    checkpoint = path.with_suffix(".part")
    with checkpoint.open("w", encoding="utf-8") as temp:
        for row in rows:
            temp.write(json.dumps(row, separators=(",", ":")) + "\n")
    checkpoint.replace(path)


@contextmanager
def shard_lock(path):
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def image_paths(directory, users, offers, last_round, dataset):
    manifest_path = directory / "manifest.jsonl"
    manifest = {row["path"]: row["options"] for row in read_rows(manifest_path)}
    paths = {}
    cases = [(f"heldout/set-{offer:03d}.png", dataset.heldout["all_options"][offer]) for offer in offers]
    cases.extend(
        (f"interaction/user-{user:03d}/round-{round_}.png", dataset.users[user]["rounds"][round_]["options"])
        for user in users for round_ in range(last_round)
    )
    for relative_path, options in cases:
        target = (directory / relative_path).resolve()
        if manifest.get(relative_path) != options or not target.is_file() or not target.is_relative_to(directory):
            raise ValueError(f"Missing or mismatched card {target}")
        paths[relative_path] = target
    return paths


def chunks(items: Iterable, size):
    stream = iter(items)
    while batch := list(islice(stream, size)):
        yield batch


def read_or_write_history(path, users, conversations, generate, batch_size):
    if not path.exists():
        answers = (
            {"user": user, "answer": text, "rec": recommendation(text)}
            for batch in chunks(zip(users, conversations), batch_size)
            for (user, _), text in zip(batch, generate([conv for _, conv in batch]), strict=True)
        )
        write_rows(path, answers)
    answers = list(read_rows(path))
    if [row["user"] for row in answers] != users:
        raise ValueError(f"Wrong user order or missing rows in {path}")
    return answers


def benchmark_format(dataset, users, offers, observations, directory, cards, generate, batch_size):
    histories = {user: [] for user in users}
    pending = dict.fromkeys(users, INTRO)
    for n_obs in range(max(observations) + 1):
        if n_obs in observations:
            path = directory / f"eval-{n_obs}.jsonl"
            if not path.exists():
                started = monotonic()
                # With no observations the prompt depends only on the shared offer.
                eval_users = users[:1] if n_obs == 0 else users
                cases = (
                    (user, offer, histories[user] + [user_message(
                        pending[user], dataset.heldout["all_options"][offer],
                        cards.get(f"heldout/set-{offer:03d}.png"),
                    )])
                    for user in eval_users for offer in offers
                )

                def replies():
                    for batch in chunks(cases, batch_size):
                        texts = generate([conversation for _, _, conversation in batch])
                        for (user, offer, _), text in zip(batch, texts, strict=True):
                            yield dict(user=user, n_obs=n_obs, set=offer, rec=recommendation(text))

                rows = replies()
                if n_obs == 0:
                    first_user_rows = list(rows)
                    rows = ({**row, "user": user} for user in users for row in first_user_rows)
                write_rows(path, rows)
                print(f"{directory.parent.name} n_obs={n_obs}: {len(users) * len(offers)} rows "
                      f"({len(eval_users) * len(offers)} prompts) in {monotonic() - started:.1f}s", flush=True)
        if n_obs == max(observations):
            break
        rounds = {user: dataset.users[user]["rounds"][n_obs] for user in users}
        conversations = [
            histories[user] + [user_message(
                pending[user], rounds[user]["options"],
                cards.get(f"interaction/user-{user:03d}/round-{n_obs}.png"),
            )]
            for user in users
        ]
        answers = read_or_write_history(
            directory / f"history-{n_obs}.jsonl", users, conversations, generate, batch_size,
        )
        for user, conversation, answer in zip(users, conversations, answers, strict=True):
            histories[user] = conversation + [{"role": "assistant", "content": answer["answer"]}]
            pending[user] = feedback(answer["rec"], rounds[user]["user_idx"])


def load_generator(args):
    if args.dry_run:
        return lambda conversations: [
            f"{PREFILL} {int.from_bytes(hashlib.sha256(json.dumps(conv).encode()).digest()[:2]) % 3 + 1}."
            for conv in conversations
        ], "dry-run"

    from vllm import LLM, SamplingParams, __version__

    llm = LLM(
        model=MODEL,
        revision=args.revision,
        tokenizer_revision=args.revision,
        dtype="bfloat16",
        max_model_len=args.context,
        max_num_seqs=args.batch_size,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        seed=0,
        allowed_local_media_path=str(args.cards),
        limit_mm_per_prompt={"image": 6, "video": 0},
    )
    sampling = SamplingParams(**SAMPLING)

    def generate(conversations):
        outputs = llm.chat(
            [conv + [{"role": "assistant", "content": PREFILL}] for conv in conversations],
            sampling, add_generation_prompt=False, continue_final_message=True,
            chat_template_kwargs={"enable_thinking": False}, use_tqdm=False,
        )
        return [END_OF_THOUGHT.split(PREFILL + output.outputs[0].text)[-1].strip() for output in outputs]

    return generate, __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cards", type=Path, default=ROOT / "data/generated")
    parser.add_argument("--revision", help="Pinned Hugging Face checkpoint revision")
    parser.add_argument("--full", action="store_true", help="Use all users, sets and observation counts")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true", help="Exercise the pipeline without model predictions")
    args = parser.parse_args()
    dataset = PaperDataset(DATA)
    all_users = list(range(len(dataset.users))) if args.full else [0, 1]
    offers = list(range(len(dataset.heldout["all_options"]))) if args.full else [0, 1]
    observations = list(range(6)) if args.full else [0, 2, 5]
    if not (0 <= args.shard < args.shards <= len(all_users)) or args.batch_size < 1 or args.context < 1:
        parser.error("Invalid shard, batch size or context size")
    if not args.dry_run and not args.revision:
        parser.error("Pin the model with --revision for a real run")
    users = all_users[args.shard::args.shards]
    args.cards = args.cards.resolve()
    manifests = {
        mode: image_paths(args.cards / mode, users, offers, max(observations), dataset)
        for mode in FORMATS[1:]
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with shard_lock(args.out / f"shard-{args.shard:03d}.lock"):
        generate, version = load_generator(args)
        meta = dict(
            users=users, sets=offers, n_obs=observations, shards=args.shards, shard=args.shard,
            model=MODEL, revision=args.revision, vllm=version, dtype="bfloat16", thinking=False,
            context=args.context, batch_size=args.batch_size, sampling=SAMPLING, dry_run=args.dry_run,
            reference_commit=REFERENCE_COMMIT,
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            dataset_sha256={
                name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
                for name in ("eval/interaction/flight.jsonl", "eval/heldout/flight.json")
            },
            manifests_sha256={
                mode: hashlib.sha256((args.cards / mode / "manifest.jsonl").read_bytes()).hexdigest()
                for mode in FORMATS[1:]
            },
        )
        meta_path = args.out / f"shard-{args.shard:03d}.json"
        if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != meta:
            parser.error(f"Settings changed for {meta_path}. Choose a new output folder.")
        if not meta_path.exists():
            write_rows(meta_path, [meta])
        for mode in FORMATS:
            directory = args.out / mode / f"shard-{args.shard:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            benchmark_format(dataset, users, offers, observations, directory,
                             manifests.get(mode, {}), generate, args.batch_size)


if __name__ == "__main__":
    main()
