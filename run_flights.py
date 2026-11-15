"""Inspect one original flight case; optionally export public cases as JSONL."""

import argparse
import json
from pathlib import Path
import sys

# Enables the VS Code Run Python File button without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from bvlm.flight_data import PaperDataset, Flight, utility_scores


def export_cases(data, path, user=0, n_obs=0):
    cases = [data.case(user, n_obs, index) for index in range(100)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise ValueError(f"Choose a new export path; different data exists at {path}")
    path.write_text(payload, encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", type=int, default=0)
    parser.add_argument("--set", type=int, default=1, dest="set_index")
    parser.add_argument("--n-obs", type=int, default=0)
    parser.add_argument("--answers", action="store_true", help="Inspect private weights and stored solution locally")
    parser.add_argument("--export", type=Path, help="Export all 100 public test cases for this user/n_obs")
    args = parser.parse_args(argv)
    try:
        data = PaperDataset()
        case = data.case(args.user, args.n_obs, args.set_index)
        print("Original paper data: 624 users, 5 observed rounds per user, 100 heldout sets.")
        print("Each decision has three flights. --set selects a different offer.\n")
        print(f"User {args.user}, observations {args.n_obs}, heldout set {args.set_index}")
        for index, observed in enumerate(case["history"]):
            print(f"\nObserved round {index}:")
            print("\n".join(observed["options"]))
            print(f"Observed user choice: Flight {observed['user_idx'] + 1}")
        print("\n" + case["card_text"])
        if args.answers:
            _, values, chosen = data.offer(args.user, args.set_index)
            weights = data.users[args.user]["reward_fn"]
            scores = utility_scores([Flight.from_values(v) for v in values], weights)
            print("\nINSPECTION ONLY — excluded from exported model input")
            print("Preference weights:", weights)
            print("Utilities:", [value / 20 for value in scores])
            print(f"Stored user choice / Oracle: Flight {chosen + 1} (rec={chosen})")
        if args.export:
            path = export_cases(data, args.export, args.user, args.n_obs)
            print(f"\n100 public cases exported to {path.resolve()}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Data error: {error}\n")


if __name__ == "__main__":
    main()
