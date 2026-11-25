"""Flight recommendation benchmark from Qiu et al. 2026 (Nat. Commun.), Fig. 2, over all 624 simulated users.

For each user, the model plays 5 interaction rounds and gets feedback on each one. Its own raw answers stay in the history.
After 0..5 rounds (n_obs), the conversation branches off once for each of the 100 held-out option sets.
These branches never go back into the main conversation (Supp. Fig. A1). Prompt wording follows Supp. Table H3.

Every answer, thinking or not, ends in the same forced format: "The best option is Flight" + up to 4 generated tokens.
With --thinking, generation has two phases:
  1. The model thinks freely, up to --think-budget tokens, and stops at its end-of-thought tag.
  2. The script closes the thought itself (with a short "answer now" nudge if the budget ran out), pre-fills
     "The best option is Flight" and lets the model complete only the flight number, exactly as without thinking.
The rate of thoughts cut off by the budget is reported in the stats as think_truncated.
Thinking samples with the Qwen3.5 recommended settings (THINK_SAMPLING) and a fixed seed per request,
because greedy thinking tends to loop until the budget runs out. --think-greedy switches back to greedy.

A run can cover a seeded random subset of users and option sets (--n-users, --n-sets) and score only some n_obs
(--eval-n-obs); see plot_sampling.py for how large a subset must be. Output ids always refer to the full dataset.

Writes:
  <out>              {"user", "n_obs", "set", "rec"} per line, 624*6*100 = 374,400 lines (input for evaluate_predictions)
  <stem>.raw.jsonl   every model answer (interaction + eval), so the parsing can be audited
  <stem>.stats.json  parse failure rate, thought truncation rate and accuracy per n_obs
where <stem> is <out> without its .jsonl extension.

  python run_flight.py --model Qwen/Qwen2.5-7B-Instruct --out results/flight_qwen2.5-7b.jsonl
  python run_flight.py --model Qwen/Qwen3.5-9B --thinking --out results/flight_qwen3.5-9b-think.jsonl
  python run_flight.py --n-users 3 --out results/test-run.jsonl  # quick test on an interactive GPU node
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

# The assistant answer is always pre-filled with the answer format from Table H3 and the model only completes the
# flight number. Free generation (Qwen2.5-7B writes long analyses) was mostly unparseable.
PREFILL = "The best option is Flight"

# End-of-thought tag found in the tokenizer vocab -> text the script writes to close a thought.
THINK_CLOSE = {"</think>": "\n</think>\n\n", "<channel|>": "<channel|>"}
# n_obs=0 prompts are the same for every user. Greedy needs one answer per distinct prompt; sampling draws up to this
# many, so the n_obs=0 accuracy doesn't hinge on single samples (noise sd <= 0.5/sqrt(50 * n_sets)).
SAMPLES_PER_PROMPT = 50
# Thinking sampling, "thinking mode for general tasks" from the Qwen3.5 model card. The model ships no
# generation_config.json, so vLLM would fall back to plain temperature 1.0. One setting for all models,
# make it a flag when a model with different recommendations is benchmarked.
THINK_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
}
# Appended before the close when the budget runs out (budget forcing as described in the Qwen3 model cards).
NUDGE = "\n\nConsidering the limited time, I have to give the solution based on the thinking directly now."

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--out", required=True)
p.add_argument("--interaction", default="data/eval/interaction/flight.jsonl")
p.add_argument("--heldout", default="data/eval/heldout/flight.json")
p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
p.add_argument(
    "--think-budget",
    type=int,
    default=4096,
    help="max thinking tokens per answer; longer thoughts are cut off and closed by the script",
)
p.add_argument(
    "--think-greedy",
    action="store_true",
    help="greedy thinking instead of THINK_SAMPLING (prone to repetition loops)",
)
p.add_argument(
    "--think-close",
    help="text that closes a thought, if auto-detection from the tokenizer is wrong for a model",
)
p.add_argument(
    "--max-model-len",
    type=int,
    default=16384,
    help="must fit the longest prompt (n_obs=5) + think budget",
)
p.add_argument(
    "--tp",
    type=int,
    default=1,
    help="tensor parallel GPUs, e.g. 2 for Qwen2.5-32B on L40S",
)
p.add_argument(
    "--seed",
    type=int,
    default=0,
    help="for the user/set subset, thinking samples and the random fallback on unparseable answers",
)
p.add_argument(
    "--n-users", type=int, help="seeded random subset of N users (default all 624)"
)
p.add_argument(
    "--n-sets",
    type=int,
    help="seeded random subset of N held-out option sets (default all 100)",
)
p.add_argument(
    "--eval-n-obs",
    type=lambda s: sorted({int(k) for k in s.split(",")}),
    default=list(range(6)),
    help="comma-separated n_obs to score, e.g. 0,5 (default 0,1,2,3,4,5); interaction stops after the largest",
)
p.add_argument("--dry-run", action="store_true")


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
assert (
    parse(
        "Given your preferences, **Flight 2** appears to be the best option. It has no stops."
    )
    == 1
)
assert parse("Flight 2 is cheap but Flight 3 is shorter.") is None
assert parse("I need more information.") is None
assert (
    visible("<think>\nhmm\n</think>\n\nThe best option is Flight 2.")
    == "The best option is Flight 2."
)


def make_generate(args):
    """Returns generate(convs) -> [(raw_text, thought_truncated)]."""
    if args.dry_run:
        fake_rng = random.Random(1)
        fakes = [" {}.", " {}.", " none."]

        def fake(_):
            answer = PREFILL + fake_rng.choice(fakes).format(fake_rng.randint(1, 3))
            if not args.thinking:
                return answer, False
            cut = fake_rng.random() < 0.2
            thought = "Flight 1 is long, Flight 3 is cheap. Let me weigh this" + (
                NUDGE if cut else "."
            )
            return thought + THINK_CLOSE["</think>"] + answer, cut

        return lambda convs: [fake(c) for c in convs]

    from vllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        enable_prefix_caching=True,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        seed=args.seed,
        mamba_cache_mode="align",  # prefix caching for hybrid models (Qwen3.5); ignored by plain transformers
    )
    # No system message: each model's chat template applies its own default (Qwen2.5 inserts one).
    kw = {"enable_thinking": args.thinking}  # templates without this variable ignore it
    # Same answer sampling in both modes: greedy, as in the paper, and only the flight number after the pre-fill.
    answer_sp = SamplingParams(
        temperature=0, max_tokens=4, stop=["."], include_stop_str_in_output=True
    )

    if not args.thinking:
        return lambda convs: [
            (PREFILL + o.outputs[0].text, False)
            for o in llm.chat(
                [c + [{"role": "assistant", "content": PREFILL}] for c in convs],
                answer_sp,
                add_generation_prompt=False,
                continue_final_message=True,
                chat_template_kwargs=kw,
                use_tqdm=True,
            )
        ]

    # Thinking: work on raw token ids, so the script controls exactly how the thought is closed. Going through
    # llm.chat with a pre-filled assistant message would let the chat template rewrite or drop the thought.
    tok = llm.get_tokenizer()
    vocab = tok.get_vocab()
    end = next((t for t in THINK_CLOSE if t in vocab), None)
    if end is None:
        raise SystemExit(
            f"no end-of-thought tag {list(THINK_CLOSE)} in the vocab of {args.model}"
        )
    end_id = vocab[end]
    close = args.think_close or THINK_CLOSE[end]
    if args.think_greedy:
        sampling = {"temperature": 0}
    else:
        sampling = THINK_SAMPLING
    print(
        f"thinking: stop at {end!r}, close with {close!r}, budget {args.think_budget}, sampling {sampling}",
        flush=True,
    )

    def ids(text):  # the chat template already contains BOS where a model needs it
        return tok.encode(text, add_special_tokens=False)

    def generate(convs):
        prompts = [
            ids(
                tok.apply_chat_template(
                    c, tokenize=False, add_generation_prompt=True, **kw
                )
            )
            for c in convs
        ]
        room = [
            args.max_model_len - len(x) - 64 for x in prompts
        ]  # 64: nudge + close + pre-fill + answer
        if min(room) < args.think_budget:
            print(
                f"warning: longest prompt has {max(map(len, prompts))} tokens, leaving only {min(room)} "
                f"thinking tokens; raise --max-model-len",
                flush=True,
            )
        think = llm.generate(
            [{"prompt_token_ids": x} for x in prompts],
            [
                SamplingParams(
                    **sampling,
                    seed=args.seed
                    + i,  # reproducible, and different for repeats of the same prompt
                    max_tokens=max(1, min(args.think_budget, r)),
                    stop_token_ids=[end_id],
                )
                for i, r in enumerate(room)
            ],
            use_tqdm=True,
        )
        heads, cont, cuts = [], [], []
        for x, o in zip(prompts, think):
            out = o.outputs[0]
            thought = list(out.token_ids)
            if thought and thought[-1] == end_id:
                thought.pop()
            cut = out.finish_reason == "length"
            suffix = (NUDGE if cut else "") + close + PREFILL
            heads.append(out.text.split(end)[0] + suffix)
            cont.append({"prompt_token_ids": x + thought + ids(suffix)})
            cuts.append(cut)
        answers = llm.generate(cont, answer_sp, use_tqdm=True)
        return [(h + o.outputs[0].text, c) for h, o, c in zip(heads, answers, cuts)]

    return generate


def main():
    args = p.parse_args()
    generate = make_generate(args)

    copies = 1 if args.think_greedy or not args.thinking else SAMPLES_PER_PROMPT

    def run(convs):
        # Identical prompts give identical greedy answers, so each distinct prompt runs once (n_obs=0 has only
        # n_sets distinct prompts); with sampling, up to `copies` times.
        seen, keys = {}, []
        for c in convs:
            k = json.dumps(c)
            seen[k] = seen.get(k, -1) + 1
            keys.append((k, seen[k] % copies))
        uniq = list(dict.fromkeys(keys))
        outs = dict(zip(uniq, generate([json.loads(k) for k, _ in uniq])))
        return [outs[k] for k in keys]

    rows = [json.loads(line) for line in open(args.interaction)]
    held = json.load(open(args.heldout))
    assert all(
        r["reward_fn"] == u["reward_fn"]
        for r, u in zip(rows, held["user_idxs"], strict=True)
    ), "user order mismatch"
    pick = random.Random(args.seed)
    user_ids = sorted(pick.sample(range(len(rows)), args.n_users or len(rows)))
    set_ids = sorted(
        pick.sample(
            range(len(held["all_options"])), args.n_sets or len(held["all_options"])
        )
    )
    rows = [rows[u] for u in user_ids]
    sets = [held["all_options"][s] for s in set_ids]
    labels = [[held["user_idxs"][u]["idxs"][s] for s in set_ids] for u in user_ids]
    last = max(args.eval_n_obs)

    rng = random.Random(args.seed)
    hist = [
        [] for _ in rows
    ]  # main conversation per user (never contains eval branches)
    pending = [INTRO + "\n\n"] * len(
        rows
    )  # text that starts the next user message (intro or last feedback)
    stats = {
        "model": args.model,
        "thinking": args.thinking,
        "think_budget": args.think_budget if args.thinking else None,
        "think_greedy": args.think_greedy if args.thinking else None,
        "seed": args.seed,
        "users": user_ids,  # to compare other runs on exactly the same subset
        "sets": set_ids,
        "eval": {},
        "interaction": {},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = str(out).removesuffix(
        ".jsonl"
    )  # not with_suffix: model names contain dots (qwen2.5-7b)

    with open(out, "w") as f, open(f"{stem}.raw.jsonl", "w") as fraw:
        for k in range(last + 1):
            if k in args.eval_n_obs:
                # Eval branches after k rounds: same prefix for all sets of a user, so vLLM prefix caching reuses it.
                convs = [
                    h + [{"role": "user", "content": pre + question(s)}]
                    for h, pre in zip(hist, pending)
                    for s in sets
                ]
                fails = correct = cuts = 0
                for i, (text, cut) in enumerate(run(convs)):
                    u, j = divmod(i, len(sets))
                    uid, sid = user_ids[u], set_ids[j]
                    rec = parse(visible(text))
                    cuts += cut
                    fraw.write(
                        json.dumps(
                            {
                                "kind": "eval",
                                "user": uid,
                                "n_obs": k,
                                "set": sid,
                                "text": text,
                                "think_truncated": cut,
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
                    f.write(
                        json.dumps({"user": uid, "n_obs": k, "set": sid, "rec": rec})
                        + "\n"
                    )
                stats["eval"][k] = {
                    "parse_fail": fails / len(convs),
                    "think_truncated": cuts / len(convs),
                    "accuracy": correct / len(convs),
                }
                print(f"n_obs={k}", stats["eval"][k], flush=True)
                # Written every round so a job killed by the time limit still leaves stats behind.
                Path(f"{stem}.stats.json").write_text(json.dumps(stats, indent=2))
            if k == last:
                break

            # Interaction round k+1 on the main conversation; feedback depends on the model's own answer.
            convs = [
                h
                + [
                    {
                        "role": "user",
                        "content": pre + question(r["rounds"][k]["options"]),
                    }
                ]
                for h, pre, r in zip(hist, pending, rows)
            ]
            fails = cuts = 0
            for u, (conv, (text, cut), r) in enumerate(zip(convs, run(convs), rows)):
                ans, choice = visible(text), r["rounds"][k]["user_idx"]
                rec = parse(ans)
                cuts += cut
                fraw.write(
                    json.dumps(
                        {
                            "kind": "interaction",
                            "user": user_ids[u],
                            "round": k + 1,
                            "text": text,
                            "think_truncated": cut,
                            "rec": rec,
                        }
                    )
                    + "\n"
                )
                # Visible answer only, thinking stripped: "The best option is Flight N." in both modes.
                hist[u] = conv + [{"role": "assistant", "content": ans}]
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
            stats["interaction"][k + 1] = {
                "parse_fail": fails / len(rows),
                "think_truncated": cuts / len(rows),
            }

    Path(f"{stem}.stats.json").write_text(json.dumps(stats, indent=2))
    keys = {(d["user"], d["n_obs"], d["set"]) for d in map(json.loads, open(out))}
    assert len(keys) == len(rows) * len(args.eval_n_obs) * len(sets), (
        "missing or duplicate predictions"
    )
    print("wrote", out, len(keys), "predictions")


# vLLM may start its engine with multiprocessing "spawn", which re-imports this file.
# Everything with side effects (arg parsing, model loading, the run) must stay behind this guard.
if __name__ == "__main__":
    main()
