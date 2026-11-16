"""Throwaway read-only inspection of ./data. Never loads whole .jsonl files."""
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
INTERACTION = DATA / "eval/interaction/flight.jsonl"
HELDOUT = DATA / "eval/heldout/flight.json"
TRAIN = {n: DATA / f"train/{n}.jsonl" for n in ("bayesian", "oracle")}
TRUNC = 1500
N_SAMPLE = 2000

OPTION_RE = re.compile(
    r"Flight (\d+):\s*departure time: (.+?), duration: (.+?), number of stops: (.+?), price: (.+?)\s*$",
    re.M,
)
ANSWER_RE = re.compile(r"The best option is Flight (\d+)\.")
FEEDBACK_RE = re.compile(r"^Your option Flight (\d+) is (correct|incorrect)\.(?: I prefer Flight (\d+)\.)?")
FEATURES = ["departure time", "duration", "number of stops", "price"]


def trunc(obj, n=TRUNC):
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + f"... [truncated, total {len(s)} chars]"


def iter_jsonl(path, limit=None):
    with open(path, encoding="utf-8") as f:
        n = 0
        for line in f:
            if not line.strip():
                continue
            if limit is not None and n >= limit:
                return
            n += 1
            yield json.loads(line)


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def parse_options(text):
    return [m.groups() for m in OPTION_RE.finditer(text)]


def load_heldout():
    return json.loads(HELDOUT.read_text(encoding="utf-8"))  # single JSON document, ~0.8 MB


def q1_readme():
    section("Q1. data/README.md (full)")
    print((DATA / "README.md").read_text(encoding="utf-8"))


def q2_structure():
    section("Q2. Records, top-level keys, first record")
    for path in [INTERACTION, *TRAIN.values()]:
        n, keysets, first = 0, Counter(), None
        for rec in iter_jsonl(path):
            first = first or rec
            keysets[tuple(rec.keys())] += 1
            n += 1
        print(f"\n--- {path.relative_to(DATA)}: {n} records")
        for k, c in keysets.items():
            print(f"  top-level keys {list(k)}: {c} records")
        print("  first record:", trunc(first))

    obj = load_heldout()
    print(f"\n--- {HELDOUT.relative_to(DATA)}: single JSON object (not JSONL)")
    print(f"  top-level keys: {list(obj.keys())}")
    for k, v in obj.items():
        print(f"  len({k}) = {len(v)}")
    print("  features:", obj["features"])
    print("  all_options[0]:", trunc(obj["all_options"][0]))
    print("  all_options_numpy[0]:", trunc(obj["all_options_numpy"][0]))
    print("  user_idxs[0]:", trunc(obj["user_idxs"][0]))

    print("\n--- train/*.jsonl grouping by idx (full file)")
    for name, path in TRAIN.items():
        per_idx = Counter()
        lens_per_idx = defaultdict(list)
        order = []
        for rec in iter_jsonl(path):
            if not order or order[-1] != rec["idx"]:
                order.append(rec["idx"])
            per_idx[rec["idx"]] += 1
            lens_per_idx[rec["idx"]].append(len(rec["messages"]))
        contiguous = len(order) == len(per_idx)
        print(f"  {name}: distinct idx={len(per_idx)}, records per idx dist={sorted(Counter(per_idx.values()).items())}, "
              f"idx blocks contiguous={contiguous}, idx range={min(per_idx)}..{max(per_idx)}")
        print(f"    message-count patterns per idx: {Counter(tuple(v) for v in lens_per_idx.values()).most_common(3)}")

    print("\n--- train/bayesian: all 5 records of idx=0 (each truncated)")
    for rec in iter_jsonl(TRAIN["bayesian"], limit=5):
        print(f"  idx={rec['idx']} n_messages={len(rec['messages'])}: {trunc(rec, 300)}")
    full = [r for r in iter_jsonl(TRAIN["bayesian"], limit=5) if len(r["messages"]) == 10][0]
    print("\n  longest record of idx=0, messages:")
    for m in full["messages"]:
        print(f"    [{m['role']}] {trunc(m['content'], 700)!r}")


def q3_features():
    section(f"Q3. Option representation and distinct feature values (first {N_SAMPLE} records)")
    rec = next(iter_jsonl(INTERACTION))
    print("Interaction record has BOTH text options ('rounds[*].options') and numeric 'rounds_numpy'.")
    print("Example round 0 text   :", rec["rounds"][0]["options"])
    print("Example round 0 numeric:", rec["rounds_numpy"][0])
    print("Training records contain only text inside 'messages[*].content'.")

    def report(label, text_values, num_values=None):
        print(f"\n--- {label}")
        for i, f in enumerate(FEATURES):
            vals = text_values[i]
            print(f"  {f}: {len(vals)} distinct text values: {sorted(vals, key=sort_key(f))}")
            if num_values is not None:
                nv = num_values[i]
                print(f"  {f}: {len(nv)} distinct numeric values: {sorted(nv)}")

    # interaction (all records < N_SAMPLE), with text -> numeric mapping
    tv, nv = [set() for _ in FEATURES], [set() for _ in FEATURES]
    mapping = [defaultdict(set) for _ in FEATURES]
    n_opts = 0
    for rec in iter_jsonl(INTERACTION, limit=N_SAMPLE):
        for rnd, rnum in zip(rec["rounds"], rec["rounds_numpy"]):
            parsed = parse_options("\n".join(rnd["options"]))
            assert len(parsed) == len(rnd["options"]) == len(rnum), (rnd, rnum)
            for (_, *vals), num in zip(parsed, rnum):
                n_opts += 1
                for i in range(len(FEATURES)):
                    tv[i].add(vals[i])
                    nv[i].add(num[i])
                    mapping[i][vals[i]].add(num[i])
    report(f"eval/interaction/flight.jsonl ({n_opts} options parsed)", tv, nv)
    print("\n  text -> numeric mapping (interaction file):")
    for i, f in enumerate(FEATURES):
        m = {k: sorted(v) for k, v in sorted(mapping[i].items(), key=lambda kv: min(kv[1]))}
        one_to_one = all(len(v) == 1 for v in m.values())
        print(f"    {f} (each text maps to exactly one number: {one_to_one}): {m}")

    obj = load_heldout()
    tv, nv = [set() for _ in FEATURES], [set() for _ in FEATURES]
    for opts, onum in zip(obj["all_options"], obj["all_options_numpy"]):
        for (_, *vals), num in zip(parse_options("\n".join(opts)), onum):
            for i in range(len(FEATURES)):
                tv[i].add(vals[i])
                nv[i].add(num[i])
    report(f"eval/heldout/flight.json ({len(obj['all_options'])} option sets)", tv, nv)

    for name, path in TRAIN.items():
        tv = [set() for _ in FEATURES]
        n_opts = 0
        for rec in iter_jsonl(path, limit=N_SAMPLE):
            for m in rec["messages"]:
                if m["role"] != "user":
                    continue
                for _, *vals in parse_options(m["content"]):
                    n_opts += 1
                    for i in range(len(FEATURES)):
                        tv[i].add(vals[i])
        report(f"train/{name}.jsonl first {N_SAMPLE} records ({n_opts} option mentions parsed, incl. repeats "
               f"from prefix records)", tv)


def sort_key(feature):
    def k(s):
        if feature == "departure time":
            m = re.match(r"(\d+):(\d+) (AM|PM)", s)
            h = int(m[1]) % 12 + (12 if m[3] == "PM" else 0)
            return (h * 60 + int(m[2]), s)
        if feature == "duration":
            h = re.search(r"(\d+) hr", s)
            mi = re.search(r"(\d+) min", s)
            return ((int(h[1]) if h else 0) * 60 + (int(mi[1]) if mi else 0), s)
        num = re.sub(r"[^\d.]", "", s)
        return (float(num) if num else float("inf"), s)
    return k


def q4_theta():
    section("Q4. Is the hidden preference vector stored?")
    rec = next(iter_jsonl(INTERACTION))
    print("eval/interaction/flight.jsonl record 0: reward_fn =", rec["reward_fn"], "| features =", rec["features"])
    rfs = [tuple(r["reward_fn"]) for r in iter_jsonl(INTERACTION)]
    vals = sorted({v for rf in rfs for v in rf})
    print(f"  {len(rfs)} records, {len(set(rfs))} distinct reward_fn vectors, values used: {vals}, "
          f"lengths: {sorted(Counter(len(r) for r in rfs).items())}")
    obj = load_heldout()
    hrf = [tuple(u["reward_fn"]) for u in obj["user_idxs"]]
    print(f"eval/heldout/flight.json: user_idxs[i].reward_fn present for {len(hrf)} users, "
          f"{len(set(hrf))} distinct; same order as interaction file: {hrf == rfs}")
    keys = set()
    for path in TRAIN.values():
        for r in iter_jsonl(path):
            keys |= set(r.keys())
    print(f"train/*.jsonl: union of top-level keys over all records = {sorted(keys)} (no reward_fn / theta field)")
    hits = 0
    for r in iter_jsonl(TRAIN["bayesian"], limit=N_SAMPLE):
        hits += any(re.search(r"reward|theta|weight", m["content"], re.I) for m in r["messages"])
    print(f"  train/bayesian first {N_SAMPLE} records whose message text mentions reward/theta/weight: {hits}")


def q5_rounds():
    section("Q5. Rounds per interaction and options per round")
    rounds, opts, opts_num = Counter(), Counter(), Counter()
    for r in iter_jsonl(INTERACTION):
        rounds[len(r["rounds"])] += 1
        for rnd, rn in zip(r["rounds"], r["rounds_numpy"]):
            opts[len(rnd["options"])] += 1
            opts_num[len(rn)] += 1
    print(f"eval/interaction: rounds/interaction {sorted(rounds.items())}; options/round (text) {sorted(opts.items())}; "
          f"(numpy) {sorted(opts_num.items())}")
    obj = load_heldout()
    print(f"eval/heldout: options/set {sorted(Counter(len(o) for o in obj['all_options']).items())}; "
          f"idxs per user {sorted(Counter(len(u['idxs']) for u in obj['user_idxs']).items())}")
    for name, path in TRAIN.items():
        rounds, opts = Counter(), Counter()
        for r in full_interactions(path, float("inf")):
            rounds[sum(m["role"] == "assistant" for m in r["messages"])] += 1
            for m in r["messages"]:
                if m["role"] == "user":
                    opts[len(parse_options(m["content"]))] += 1
        print(f"train/{name} (longest record per idx, all idx): rounds per interaction {sorted(rounds.items())}; "
              f"options per user turn {sorted(opts.items())}")


def full_interactions(path, limit):
    """Yield the longest record per idx (= full interaction), for the first `limit` distinct idx."""
    cur_idx, cur = None, None
    seen = 0
    for rec in iter_jsonl(path):
        if rec["idx"] != cur_idx:
            if cur is not None:
                yield cur
                seen += 1
                if seen >= limit:
                    return
            cur_idx, cur = rec["idx"], rec
        elif len(rec["messages"]) > len(cur["messages"]):
            cur = rec
    if cur is not None and seen < limit:
        yield cur


def q6_round1():
    section(f"Q6. Round-1 assistant recommendation position (first {N_SAMPLE} interactions = distinct idx)")
    for name, path in TRAIN.items():
        pos = Counter()
        round1_sets = Counter()
        n = 0
        for rec in full_interactions(path, N_SAMPLE):
            n += 1
            m = ANSWER_RE.search(rec["messages"][1]["content"])
            pos[m[1] if m else "UNPARSED"] += 1
            round1_sets[rec["messages"][0]["content"]] += 1
        print(f"train/{name}: {n} interactions; round-1 recommended position counts: {dict(sorted(pos.items()))}; "
              f"distinct round-1 prompts: {len(round1_sets)}")

    pos, n = Counter(), 0
    for rec in iter_jsonl(TRAIN["bayesian"], limit=N_SAMPLE):
        if len(rec["messages"]) == 2:
            n += 1
            pos[ANSWER_RE.search(rec["messages"][1]["content"])[1]] += 1
    print(f"(alt. reading) train/bayesian first {N_SAMPLE} RECORDS -> {n} round-1-only records: {dict(sorted(pos.items()))}")

    print("\nExamples (bayesian, first 3 interactions, round 1):")
    for rec in full_interactions(TRAIN["bayesian"], 3):
        print(f"  idx={rec['idx']}: user={trunc(rec['messages'][0]['content'][-350:], 350)!r}")
        print(f"          assistant={rec['messages'][1]['content']!r}")


def dot(theta, phi):
    return sum(Fraction(str(t)) * Fraction(str(p)) for t, p in zip(theta, phi))


def tie_stats(label, items):
    """items: iterable of (theta, options_numpy, user_idx, context)."""
    c = Counter()
    examples = []
    for theta, opts, uidx, ctx in items:
        r = [dot(theta, o) for o in opts]
        rf = [sum(t * p for t, p in zip(theta, o)) for o in opts]
        c["rounds"] += 1
        best = max(r)
        argmax = [i for i, v in enumerate(r) if v == best]
        if len(set(r)) < len(r):
            c["rounds with any tie (exact rational)"] += 1
        if len(set(rf)) < len(rf):
            c["rounds with any tie (raw float ==)"] += 1
        if uidx in argmax:
            c["user_idx is an argmax"] += 1
        else:
            c["user_idx NOT an argmax"] += 1
        if len(argmax) >= 2:
            c["tie at max"] += 1
            c[f"tie at max, size {len(argmax)}"] += 1
            if uidx == argmax[0]:
                c["tie at max -> user picked FIRST tied index"] += 1
            elif uidx == argmax[-1]:
                c["tie at max -> user picked LAST tied index"] += 1
            elif uidx in argmax:
                c["tie at max -> user picked MIDDLE tied index"] += 1
            else:
                c["tie at max -> user picked non-tied index"] += 1
            c[f"tie at max, tied={argmax}, user_idx={uidx}"] += 1
            if len(examples) < 3:
                examples.append((ctx, theta, opts, [float(x) for x in r], uidx))
    print(f"\n--- {label}")
    for k in sorted(c, key=lambda k: (k.startswith("tie at max, tied"), k)):
        print(f"  {k}: {c[k]}")
    for ctx, theta, opts, r, uidx in examples:
        print(f"  example {ctx}: theta={theta} options_numpy={opts} rewards={r} user_idx={uidx}")


def q7_ties():
    section("Q7. Ties in theta . phi (reward_fn . options_numpy)")

    def interaction_items():
        for rec in iter_jsonl(INTERACTION):
            for k, (rnd, rn) in enumerate(zip(rec["rounds"], rec["rounds_numpy"])):
                yield rec["reward_fn"], rn, rnd["user_idx"], f"idx={rec['idx']} round={k}"

    tie_stats("eval/interaction/flight.jsonl (all records, all rounds)", interaction_items())

    obj = load_heldout()

    def heldout_items():
        for u, user in enumerate(obj["user_idxs"]):
            for s, (opts, uidx) in enumerate(zip(obj["all_options_numpy"], user["idxs"])):
                yield user["reward_fn"], opts, uidx, f"user={u} set={s}"

    tie_stats("eval/heldout/flight.json (all users x all option sets)", heldout_items())

    print("\n--- train/*.jsonl: no theta stored. User choice per round is only visible via the next user "
          "turn's feedback text (rounds 1-4; round 5 has no following feedback).")

    runs = []
    for rec in iter_jsonl(TRAIN["bayesian"]):
        if runs and runs[-1][0] == rec["idx"]:
            runs[-1][1] += 1
        else:
            runs.append([rec["idx"], 1])
    seq = [r[0] for r in runs]
    print(f"  bayesian idx run lengths: {sorted(Counter(r[1] for r in runs).items())}; number of runs: {len(runs)}; "
          f"first 8 run idx: {seq[:8]}; runs at 624..631: {seq[624:632]}; "
          f"sequence == list(range(624))*10: {seq == list(range(624)) * 10}")

    print("\n  HYPOTHESIS TEST (not stated in README): train idx i uses eval/interaction reward_fn of record idx=i, "
          "with text->numeric mapping from the eval files.")
    rf_by_idx = {r["idx"]: r["reward_fn"] for r in iter_jsonl(INTERACTION)}
    enc = text_to_numeric()
    for name, path in TRAIN.items():
        c = Counter()
        for rec in full_interactions(path, float("inf")):
            theta = rf_by_idx[rec["idx"]]
            msgs = rec["messages"]
            for k in range(0, len(msgs) - 2, 2):
                opts = [[enc[i][v] for i, v in enumerate(vals)] for _, *vals in parse_options(msgs[k]["content"])]
                fb = FEEDBACK_RE.match(msgs[k + 2]["content"])
                said = int(fb[1]) - 1
                user = said if fb[2] == "correct" else int(fb[3]) - 1
                r = [dot(theta, o) for o in opts]
                argmax = [i for i, v in enumerate(r) if v == max(r)]
                c["rounds with revealed user choice"] += 1
                c["user choice is argmax under hypothesised theta"] += user in argmax
                if len(argmax) >= 2:
                    c["tie at max"] += 1
                    c[f"tie at max -> user picked {'FIRST' if user == argmax[0] else 'LAST' if user == argmax[-1] else 'MIDDLE' if user in argmax else 'non-tied'}"] += 1
        print(f"  train/{name}: {dict(c)}")


def text_to_numeric():
    obj = load_heldout()
    enc = [dict() for _ in FEATURES]
    for opts, onum in zip(obj["all_options"], obj["all_options_numpy"]):
        for (_, *vals), num in zip(parse_options("\n".join(opts)), onum):
            for i in range(len(FEATURES)):
                enc[i][vals[i]] = num[i]
    for rec in iter_jsonl(INTERACTION):
        for rnd, rn in zip(rec["rounds"], rec["rounds_numpy"]):
            for (_, *vals), num in zip(parse_options("\n".join(rnd["options"])), rn):
                for i in range(len(FEATURES)):
                    enc[i][vals[i]] = num[i]
    return enc


if __name__ == "__main__":
    q1_readme()
    q2_structure()
    q3_features()
    q4_theta()
    q5_rounds()
    q6_round1()
    q7_ties()
