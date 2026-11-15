"""Check the published labels and the boundary between inputs and answers."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from bvlm.flight_data import ARCHIVE, PaperDataset, Flight, best_choices, utility_scores, prepare_evaluation
from run_flights import export_cases, main


class FlightDataTests(unittest.TestCase):
    def test_all_maxima_survive_exact_ties(self):
        flights = [Flight((0, 0, 0, 0)), Flight((0, 1, 0, 0)), Flight((0, 2, 0, 0))]
        self.assertEqual(best_choices(flights, [-1, 0, 0, 0]), [0, 1, 2])

    def test_invalid_feature_and_weight_grids(self):
        for values in ([0, 0, .3, 0], [0, .12, 0, 0], [0, 0, 0, 2]):
            with self.assertRaises(ValueError):
                Flight.from_values(values)
        with self.assertRaises(ValueError):
            utility_scores([Flight((0, 0, 0, 0))], [0, 0, 0, 0])


@unittest.skipUnless(ARCHIVE.exists(), "Download the official archive for integration tests")
class OriginalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = PaperDataset()

    def test_known_source_case_and_history_boundary(self):
        _, values, choice = self.data.offer(0, 1)
        scores = utility_scores([Flight.from_values(v) for v in values], self.data.users[0]["reward_fn"])
        self.assertEqual(scores, [-46, -24, -58])
        self.assertEqual(choice, 1)
        for count in range(6):
            case = self.data.case(0, count, 1)
            self.assertEqual(len(case["history"]), count)
            self.assertEqual(set(case), {"source", "user", "n_obs", "set", "options", "card_text", "history"})
            self.assertEqual(case["options"], self.data.heldout["all_options"][1])
            self.assertTrue(all(set(item) == {"options", "user_idx"} for item in case["history"]))
        case["options"].clear()
        case["history"][0]["options"].clear()
        self.assertEqual(len(self.data.case(0, 5, 1)["options"]), 3)
        self.assertEqual(len(self.data.case(0, 5, 1)["history"][0]["options"]), 3)

    def test_bounds_and_solution_independence(self):
        for args in ((-1, 0, 0), (624, 0, 0), (0, -1, 0), (0, 6, 0), (0, 0, -1), (0, 0, 100)):
            with self.assertRaises(ValueError):
                self.data.case(*args)
        before = self.data.case(0, 0, 1)
        labels = self.data.heldout["user_idxs"][0]["idxs"]
        original = labels[1]
        try:
            labels[1] = 2
            self.assertEqual(before, self.data.case(0, 0, 1))
        finally:
            labels[1] = original

    def test_export_all_sets_and_answer_is_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            path = export_cases(self.data, Path(directory) / "cases.jsonl", n_obs=2)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 100)
            self.assertEqual([row["set"] for row in rows], list(range(100)))
            self.assertTrue(all(row["user"] == 0 and row["n_obs"] == 2 for row in rows))
            self.assertNotIn("reward_fn", path.read_text())
            self.assertNotIn('"rec"', path.read_text())
            with self.assertRaises(ValueError):
                export_cases(self.data, path, user=1)
        for answers in (False, True):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(["--answers"] if answers else [])
            self.assertEqual("Stored user choice" in output.getvalue(), answers)
            self.assertEqual("Preference weights" in output.getvalue(), answers)

    def test_fresh_extraction_and_original_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = prepare_evaluation(directory)
            self.assertEqual(PaperDataset(root).case(0, 0, 1), self.data.case(0, 0, 1))
            self.assertEqual(len(list(root.rglob("*.*"))), 3)
            target = root / "eval/heldout/flight.json"
            target.write_text("changed")
            with self.assertRaises(ValueError):
                prepare_evaluation(root)
            self.assertEqual(target.read_text(), "changed")


if __name__ == "__main__":
    unittest.main()
