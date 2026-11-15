"""Paper flight values, exact utility scores and original evaluation data."""

from dataclasses import dataclass
import hashlib
import json
import zipfile
from pathlib import Path

FEATURES = ["departure_time", "duration", "number_of_stops", "price"]
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/zenodo-17677329/data"
ARCHIVE = ROOT / "data/raw/zenodo-17677329.zip"
ARCHIVE_MD5 = "a341ca74418ed9152a31cc3f91e261dd"


def verify_archive(archive=ARCHIVE):
    if hashlib.md5(Path(archive).read_bytes()).hexdigest() != ARCHIVE_MD5:
        raise ValueError("Archive checksum differs from Zenodo record 17677329")


def prepare_evaluation(directory=DATA, archive=ARCHIVE):
    """Copy only the published README and flight evaluation members."""
    verify_archive(archive)
    names = ("README.md", "eval/interaction/flight.jsonl", "eval/heldout/flight.json")
    with zipfile.ZipFile(archive) as source:
        for name in names:
            payload = source.read("data/" + name)
            target = Path(directory) / name
            if target.exists() and target.read_bytes() != payload:
                raise ValueError(f"Refusing to replace different original data: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(payload)
    return Path(directory)


def scaled(values, factor):
    result = tuple(round(value * factor) for value in values)
    if len(result) != 4 or any(abs(value * factor - integer) > 1e-8 for value, integer in zip(values, result)):
        raise ValueError("Expected four values on the documented feature/weight grid")
    return result


@dataclass(frozen=True)
class Flight:
    # Features are multiplied by ten, so utility comparisons stay exact.
    levels: tuple

    def __post_init__(self):
        if len(self.levels) != 4 or any(type(x) is not int or x not in range(11) for x in self.levels) or self.levels[2] not in (0, 5, 10):
            raise ValueError("Invalid four-feature flight vector")

    @classmethod
    def from_values(cls, values):
        return cls(scaled(values, 10))

    @property
    def fields(self):
        departure, duration, stops, price = self.levels
        minutes = 360 + 96 * departure
        return {"departure": f"{minutes // 60:02d}:{minutes % 60:02d}",
                "duration_minutes": 30 + 117 * duration,
                "stops": stops // 5, "price": 100 + 90 * price}

    def description(self, index):
        data = self.fields
        hour, minute = map(int, data["departure"].split(":"))
        clock = f"{hour % 12 or 12:02d}:{minute:02d} {'AM' if hour < 12 else 'PM'}"
        hours, minutes = divmod(data["duration_minutes"], 60)
        duration = f"{hours} hr {minutes} min" if hours else f"{minutes} min"
        return (f"Flight {index + 1}: departure time: {clock}, duration: {duration}, "
                f"number of stops: {data['stops']}, price: ${data['price']}")


def utility_scores(options, weights):
    # Scores are 20 times the paper utility: features x10, weights x2.
    integers = scaled(weights, 2)
    if any(x not in (-2, -1, 0, 1, 2) for x in integers) or not any(integers):
        raise ValueError("Invalid preference profile")
    return [sum(w * x for w, x in zip(integers, option.levels)) for option in options]


def best_choices(options, weights):
    scores = utility_scores(options, weights)
    return [i for i, score in enumerate(scores) if score == max(scores)]


class PaperDataset:
    def __init__(self, directory=DATA):
        directory = Path(directory)
        if directory == DATA:
            prepare_evaluation()
        self.users = [json.loads(line) for line in (directory / "eval/interaction/flight.jsonl").read_text().splitlines() if line.strip()]
        self.heldout = json.loads((directory / "eval/heldout/flight.json").read_text())
        self.validate()

    def validate(self):
        if len(self.users) != 624 or len(self.heldout["all_options"]) != 100 or self.heldout["features"] != FEATURES:
            raise ValueError("Expected the original 624 users and 100 test sets")
        if len(self.heldout["user_idxs"]) != 624 or len(self.heldout["all_options_numpy"]) != 100:
            raise ValueError("Test labels/numeric options do not match original counts")
        for index, user in enumerate(self.users):
            labels = self.heldout["user_idxs"][index]
            if user["idx"] != index or user["features"] != FEATURES or user["reward_fn"] != labels["reward_fn"] or len(user["rounds"]) != 5 or len(user["rounds_numpy"]) != 5 or len(labels["idxs"]) != 100:
                raise ValueError("Original user/round/label order is inconsistent")
            for r, history in enumerate(user["rounds"]):
                self.check_offer(history["options"], user["rounds_numpy"][r])
                self.check_choice(user["rounds_numpy"][r], user["reward_fn"], history["user_idx"])
            for option_set, choice in zip(self.heldout["all_options_numpy"], labels["idxs"]):
                self.check_choice(option_set, user["reward_fn"], choice)
        for texts, values in zip(self.heldout["all_options"], self.heldout["all_options_numpy"]):
            self.check_offer(texts, values)

    @staticmethod
    def check_choice(values, weights, choice):
        if type(choice) is not int or choice not in best_choices([Flight.from_values(x) for x in values], weights):
            raise ValueError("Stored choice is not a utility maximum")

    @staticmethod
    def check_offer(texts, values):
        from bvlm.flight_cards import render_flight_card
        if len(values) != 3:
            raise ValueError("Expected three options")
        converted = [Flight.from_values(value).description(i) for i, value in enumerate(values)]
        if render_flight_card(texts) != render_flight_card(converted):
            raise ValueError("Original text and normalized features disagree")

    def offer(self, user, index, interaction=False):
        if type(user) is not int or not 0 <= user < len(self.users):
            raise ValueError("User must be 0..623")
        limit = 5 if interaction else 100
        if type(index) is not int or not 0 <= index < limit:
            raise ValueError(f"Offer index must be 0..{limit - 1}")
        record = self.users[user]
        if interaction:
            return record["rounds"][index]["options"], record["rounds_numpy"][index], record["rounds"][index]["user_idx"]
        return self.heldout["all_options"][index], self.heldout["all_options_numpy"][index], self.heldout["user_idxs"][user]["idxs"][index]

    def prompt_history(self, user, n_obs):
        """Only already observed options and feedback; private weights stay local."""
        if type(user) is not int or type(n_obs) is not int or not 0 <= user < 624 or not 0 <= n_obs <= 5:
            raise ValueError("Invalid user or observation count")
        return [{"options": list(r["options"]), "user_idx": r["user_idx"]}
                for r in self.users[user]["rounds"][:n_obs]]

    def case(self, user, n_obs, set_index):
        """Public evaluation input shared by text models and image renderers."""
        from bvlm.flight_cards import render_flight_card
        history = self.prompt_history(user, n_obs)
        options, _, _ = self.offer(user, set_index)
        return {"source": "eval/heldout/flight.json", "user": user,
                "n_obs": n_obs, "set": set_index, "options": list(options),
                "card_text": render_flight_card(options), "history": history}
