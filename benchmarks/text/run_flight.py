"""Flight recommendation benchmark from Qiu et al. 2026 (Nat. Commun.), Fig. 2, over all 624 simulated users.

For each user, the model plays 5 interaction rounds and gets feedback on each one. Its own raw answers stay in the history.
After 0..5 rounds (n_obs), the conversation branches off once for each of the 100 held-out option sets.
These branches never go back into the main conversation (Supp. Fig. A1). Prompt wording follows Supp. Table H3.

Writes:
  <out>              {"user", "n_obs", "set", "rec"} per line, 624*6*100 = 374,400 lines (input for evaluate_predictions)
  <stem>.raw.jsonl   every model answer (interaction + eval), so the parsing can be audited
  <stem>.stats.json  parse failure rate and accuracy per n_obs
where <stem> is <out> without its .jsonl extension.

  python run_flight.py --model Qwen/Qwen2.5-7B-Instruct --out results/flight_qwen2.5-7b.jsonl
  python run_flight.py --limit-users 3 --out results/test-run.jsonl  # quick test on an interactive GPU node
  python run_flight.py --dry-run --out results/flight_dry.jsonl   # fake model, no GPU/vLLM: checks the pipeline
"""

import argparse
import json
import random
import re
from pathlib import Path

INTRO = (
    "Help me select the best flights for my trips. I have specific preferences for what I like and dislike in a "
    "flight, and these preferences remain the same. You need to figure out my preferences and select the best "
    "flights for me. Use your best judgment if you are unsure. Do not say you need more information."
)
END_OF_THOUGHT = re.compile(r"</think>|<channel\|>")  # Qwen / Gemma 4

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--out", required=True)
p.add_argument("--interaction", default="data/eval/interaction/flight.jsonl")
p.add_argument("--heldout", default="data/eval/heldout/flight.json")
p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
p.add_argument("--max-new-tokens", type=int, default=256)
p.add_argument("--max-model-len", type=int, default=8192)
p.add_argument(
    "--tp",
    type=int,
    default=1,
    help="tensor parallel GPUs, e.g. 2 for Qwen2.5-32B on L40S",
)
p.add_argument(
    "--seed", type=int, default=0, help="for the random fallback on unparseable answers"
)
p.add_argument(
    "--limit-users",
    type=int,
    help="first N users only (output then fails the 374,400-line check)",
)
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()


def question(options):
    return "Which flight is the best option?\n\n" + "\n".join(options)


def visible(text):
    return END_OF_THOUGHT.split(text)[-1].strip()


def parse(text):
    """Zero-based flight index, or None if the answer doesn't name exactly one flight."""
    for pat in (
        r"best option is:?\W*Flight\s*([123])",
        r"Flight\s*([123])\W*(?:is|would be|seems to be|appears to be)\s+(?:the\s+)?(?:best|most suitable)",
    ):
        if m := re.findall(pat, text, re.I):
            return int(m[-1]) - 1
    found = set(re.findall(r"Flight\s*([123])", text, re.I)) or set(
        re.findall(r"^\W*([123])\W*$", text)
    )
    return int(found.pop()) - 1 if len(found) == 1 else None


assert parse("The best option is Flight 3.") == 2
assert parse("The best option is **Flight 2**, because Flight 1 is too long.") == 1
assert parse("Flight 1 is the best choice for you.") == 0
assert parse("Given your preferences, **Flight 2** appears to be the best option. It has no stops.") == 1
assert parse("Flight 2 is cheap but Flight 3 is shorter.") is None
assert parse("I need more information.") is None


# Without thinking, the assistant turn is pre-filled with the answer format from Table H3 and the model only
# completes the flight number. Free generation (Qwen2.5-7B writes long analyses) was mostly unparseable.
# Thinking models need room to reason first, so they generate freely and the parser picks the answer.
PREFILL = "The best option is Flight"


def make_generate():
    if args.dry_run:
        fake_rng = random.Random(1)
        fakes = [" {}.", " {}.", " none."] if not args.thinking else [
            "The best option is Flight {}.", "I'd go with **Flight {}**.", "Hard to say."]
        prefix = "" if args.thinking else PREFILL
        return lambda convs: [prefix + fake_rng.choice(fakes).format(fake_rng.randint(1, 3)) for _ in convs]
    from vllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    # No system message: each model's chat template applies its own default (Qwen2.5 inserts one).
    kw = {"enable_thinking": args.thinking}  # templates without this variable ignore it
    if args.thinking:
        sp = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)  # greedy, as in the paper
        return lambda convs: [o.outputs[0].text for o in llm.chat(convs, sp, chat_template_kwargs=kw, use_tqdm=True)]
    sp = SamplingParams(temperature=0, max_tokens=4, stop=["."], include_stop_str_in_output=True)
    return lambda convs: [
        PREFILL + o.outputs[0].text
        for o in llm.chat(
            [c + [{"role": "assistant", "content": PREFILL}] for c in convs], sp,
            add_generation_prompt=False, continue_final_message=True, chat_template_kwargs=kw, use_tqdm=True,
        )
    ]


generate = make_generate()


def run(convs):
    # Greedy, so identical prompts give identical answers: n_obs=0 has only 100 unique prompts, not 62,400.
    keys = [json.dumps(c) for c in convs]
    uniq = list(dict.fromkeys(keys))
    outs = dict(zip(uniq, generate([json.loads(k) for k in uniq])))
    return [outs[k] for k in keys]


rows = [json.loads(line) for line in open(args.interaction)][: args.limit_users]
held = json.load(open(args.heldout))
sets, labels = held["all_options"], [u["idxs"] for u in held["user_idxs"]]
assert all(r["reward_fn"] == u["reward_fn"] for r, u in zip(rows, held["user_idxs"])), (
    "user order mismatch"
)

rng = random.Random(args.seed)
hist = [[] for _ in rows]  # main conversation per user (never contains eval branches)
pending = [INTRO + "\n\n"] * len(
    rows
)  # text that starts the next user message (intro or last feedback)
stats = {"model": args.model, "thinking": args.thinking, "eval": {}, "interaction": {}}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
stem = str(out).removesuffix(".jsonl")  # not with_suffix: model names contain dots (qwen2.5-7b)

with open(out, "w") as f, open(f"{stem}.raw.jsonl", "w") as fraw:
    for k in range(6):
        # Eval branches after k rounds: same prefix for all 100 sets of a user, so vLLM prefix caching reuses it.
        convs = [
            h + [{"role": "user", "content": pre + question(s)}]
            for h, pre in zip(hist, pending)
            for s in sets
        ]
        fails = correct = 0
        for i, text in enumerate(run(convs)):
            u, j = divmod(i, len(sets))
            rec = parse(visible(text))
            fraw.write(
                json.dumps(
                    {
                        "kind": "eval",
                        "user": u,
                        "n_obs": k,
                        "set": j,
                        "text": text,
                        "rec": rec,
                    }
                )
                + "\n"
            )
            if rec is None:
                fails += 1
                rec = rng.randrange(
                    3
                )  # counts as chance; the failure rate is reported in stats
            correct += rec == labels[u][j]
            f.write(json.dumps({"user": u, "n_obs": k, "set": j, "rec": rec}) + "\n")
        stats["eval"][k] = {
            "parse_fail": fails / len(convs),
            "accuracy": correct / len(convs),
        }
        print(f"n_obs={k}", stats["eval"][k], flush=True)
        if k == 5:
            break

        # Interaction round k+1 on the main conversation; feedback depends on the model's own answer.
        convs = [
            h + [{"role": "user", "content": pre + question(r["rounds"][k]["options"])}]
            for h, pre, r in zip(hist, pending, rows)
        ]
        fails = 0
        for u, (conv, text, r) in enumerate(zip(convs, run(convs), rows)):
            ans, choice = visible(text), r["rounds"][k]["user_idx"]
            rec = parse(ans)
            fraw.write(
                json.dumps(
                    {
                        "kind": "interaction",
                        "user": u,
                        "round": k + 1,
                        "text": text,
                        "rec": rec,
                    }
                )
                + "\n"
            )
            hist[u] = conv + [
                {"role": "assistant", "content": ans}
            ]  # raw visible answer, thinking stripped
            if (
                rec is None
            ):  # no answer to grade, so only the user's choice is revealed (not in the paper)
                fails += 1
                fb = f"I prefer Flight {choice + 1}."
            elif rec == choice:
                fb = f"Your option Flight {rec + 1} is correct."
            else:
                fb = f"Your option Flight {rec + 1} is incorrect. I prefer Flight {choice + 1}."
            pending[u] = fb + "\n\n"
        stats["interaction"][k + 1] = {"parse_fail": fails / len(rows)}

Path(f"{stem}.stats.json").write_text(json.dumps(stats, indent=2))
keys = {(d["user"], d["n_obs"], d["set"]) for d in map(json.loads, open(out))}
assert len(keys) == len(rows) * 6 * len(sets), "missing or duplicate predictions"
print("wrote", out, len(keys), "predictions")
