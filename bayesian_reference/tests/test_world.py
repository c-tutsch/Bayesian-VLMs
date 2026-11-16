"""Tests for bayes_ref.world (no data files needed)."""

import numpy as np
import pytest

from bayes_ref import config
from bayes_ref.world import (
    all_flights,
    build_theta,
    continuous_levels,
    stops_levels,
    text_to_features,
)


def test_levels():
    np.testing.assert_array_equal(continuous_levels(), np.linspace(0, 1, 11))
    np.testing.assert_array_equal(stops_levels(), [0.0, 0.5, 1.0])


def test_build_theta_shape_and_dtype():
    theta = build_theta()
    assert theta.shape == (624, 4)
    assert theta.dtype == np.float64
    assert len(config.THETA_VALUES) ** config.N_FEATURES - 1 == theta.shape[0]


def test_build_theta_rows():
    theta = build_theta()
    assert len({tuple(r) for r in theta}) == 624
    assert not np.any(np.all(theta == 0.0, axis=1))
    assert np.all(np.isin(theta, config.THETA_VALUES))


def test_build_theta_is_full_product_minus_zero_row():
    full = {tuple(r) for r in np.array(np.meshgrid(*[config.THETA_VALUES] * 4)).reshape(4, -1).T}
    assert len(full) == 5**4
    missing = full - {tuple(r) for r in build_theta()}
    assert missing == {(0.0, 0.0, 0.0, 0.0)}


def test_build_theta_keeps_product_order():
    theta = build_theta()
    np.testing.assert_array_equal(theta[0], [-1.0, -1.0, -1.0, -1.0])
    np.testing.assert_array_equal(theta[-1], [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(theta[311], [0.0, 0.0, 0.0, -0.5])
    np.testing.assert_array_equal(theta[312], [0.0, 0.0, 0.0, 0.5])


def test_all_flights_shape_unique_on_grid():
    flights = all_flights()
    assert flights.shape == (3993, 4)
    assert len({tuple(r) for r in flights}) == 3993
    cont, stops = continuous_levels(), stops_levels()
    for j, name in enumerate(config.FEATURE_NAMES):
        grid = stops if name == "stops" else cont
        assert set(flights[:, j]) == set(grid)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("departure time: 02:00 PM, duration: 30 min, number of stops: 1, price: $370", [0.5, 0.0, 0.5, 0.3]),
        ("departure time: 06:00 AM, duration: 20 hr, number of stops: 0, price: $100", [0.0, 1.0, 0.0, 0.0]),
        ("Flight 2: departure time: 07:36 AM, duration: 2 hr 27 min, number of stops: 2, price: $1000",
         [0.1, 0.1, 1.0, 1.0]),
        ("departure time: 12:24 PM, duration: 10 hr 15 min, number of stops: 1, price: $640", [0.4, 0.5, 0.5, 0.6]),
        ("departure time: 10:00 PM, duration: 18 hr 3 min, number of stops: 0, price: $730", [1.0, 0.9, 0.0, 0.7]),
    ],
)
def test_text_to_features_examples(text, expected):
    out = text_to_features(text)
    assert out.shape == (4,)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "departure time: 2 PM, duration: 30 min, number of stops: 1, price: $370",
        "departure time: 02:00 PM, duration: , number of stops: 1, price: $370",
        "departure time: 02:00 PM, duration: 30 min, number of stops: one, price: $370",
        "departure time: 02:00 PM, duration: 30 min, number of stops: 1, price: 370",
    ],
)
def test_text_to_features_rejects_garbage(text):
    with pytest.raises(ValueError):
        text_to_features(text)
