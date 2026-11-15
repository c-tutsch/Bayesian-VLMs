"""Check flight data placement and the meaning of the noise condition."""

from importlib.util import find_spec
import unittest

from generate_diagram_cards import NOISE_LEFT, diagram_fields, diagram_template, render_diagram
from generate_ocr_cards import card_rows
from bvlm.flight_data import PaperDataset, ROOT


@unittest.skipUnless(find_spec("PIL"), "Install Pillow to test card rendering")
class DiagramCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = PaperDataset(ROOT / "benchmarks/text/data")
        rows = list(card_rows(cls.data))
        cls.heldout = next(row for row in rows if row["path"] == "heldout/set-001.png")
        cls.interaction = next(row for row in rows if row["path"] == "interaction/user-000/round-0.png")
        cls.template = diagram_template()

    def test_numeric_values_preserve_option_order(self):
        flights = diagram_fields(self.data, self.heldout)
        self.assertEqual([flight["price"] for flight in flights], [550, 730, 460])
        self.assertEqual([flight["departure_minutes"] for flight in flights], [840, 360, 936])
        self.assertEqual([flight["duration_minutes"] for flight in flights], [966, 30, 1083])
        self.assertEqual([flight["stops"] for flight in flights], [1, 1, 2])
        self.assertEqual([flight["price"] for flight in diagram_fields(self.data, self.interaction)],
                         [370, 730, 1000])

    def test_noise_is_seeded_and_only_changes_the_sidebar(self):
        from PIL import ImageChops

        for row in (self.heldout, self.interaction):
            with self.subTest(card=row["path"]):
                render_options = {"data": self.data, "template": self.template}
                clean = render_diagram(row, with_noise=False, **render_options)
                noisy = render_diagram(row, with_noise=True, **render_options)
                diff = ImageChops.difference(clean, noisy)
                self.assertIsNotNone(diff.crop((NOISE_LEFT, 101, 1264, 336)).getbbox())
                self.assertIsNotNone(diff.crop((NOISE_LEFT, 353, 1264, 600)).getbbox())
                self.assertGreaterEqual(diff.getbbox()[0], NOISE_LEFT)
                again = render_diagram(row, with_noise=True, **render_options)
                self.assertEqual(noisy.tobytes(), again.tobytes())
