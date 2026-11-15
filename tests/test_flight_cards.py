"""Synthetic fixtures in the verified original schema; no original solutions."""

import unittest

from bvlm.flight_cards import render_flight_card


# Synthetic combinations, deliberately not sorted by time, duration, or price.
OPTIONS = [
    'Flight 1: departure time: 12:00 PM, duration: 30 min, number of stops: 1, price: $370',
    'Flight 2: departure time: 12:00 AM, duration: 4 hr 24 min, number of stops: 0, price: $190',
    'Flight 3: departure time: 11:45 PM, duration: 20 hr, number of stops: 2, price: $1000',
]
EXPECTED = '''Flight options
Flight 1: Departure: 12:00 (24-hour) | Duration: 0 h 30 min | Stops: 1 | Price: $370
Flight 2: Departure: 00:00 (24-hour) | Duration: 4 h 24 min | Stops: 0 | Price: $190
Flight 3: Departure: 23:45 (24-hour) | Duration: 20 h 0 min | Stops: 2 | Price: $1000'''


class FlightCardTests(unittest.TestCase):
    def test_order_and_all_four_features_including_zero_stops(self):
        self.assertEqual(render_flight_card(OPTIONS), EXPECTED)

    def test_am_and_pm_conversion(self):
        options = list(OPTIONS)
        options[0] = options[0].replace('12:00 PM', '02:36 PM')
        options[1] = options[1].replace('12:00 AM', '07:36 AM')
        card = render_flight_card(options)
        self.assertIn('Departure: 14:36', card)
        self.assertIn('Departure: 07:36', card)

    def test_does_not_mutate_options(self):
        options = list(OPTIONS)
        render_flight_card(options)
        self.assertEqual(options, OPTIONS)

    def test_exactly_three_options(self):
        for options in ([], OPTIONS[:2], OPTIONS + OPTIONS[:1], 'abc'):
            with self.subTest(options=options), self.assertRaises(ValueError):
                render_flight_card(options)

    def test_reordered_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            render_flight_card([OPTIONS[1], OPTIONS[0], OPTIONS[2]])

    def test_rejects_missing_features_and_unproven_units(self):
        for replacement in (
            OPTIONS[0].replace(', number of stops: 1', ''),
            OPTIONS[0].replace('price: $370', 'price: EUR 370'),
            OPTIONS[0].replace('12:00 PM', '00:00 PM'),
            OPTIONS[0].replace('stops: 1', 'stops: -1'),
            {'departure_time': 0.5, 'duration': 0.0, 'number_of_stops': 0.5, 'price': 0.3},
        ):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                render_flight_card([replacement, *OPTIONS[1:]])

    def test_rejects_solution_text_inside_an_option(self):
        for secret in ('user_idx: 1', 'reward_fn: SECRET', 'Bayes recommendation: Flight 2'):
            with self.subTest(secret=secret), self.assertRaises(ValueError):
                render_flight_card([OPTIONS[0] + ', ' + secret, *OPTIONS[1:]])


if __name__ == '__main__':
    unittest.main()
