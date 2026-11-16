"""Tests for bayes_ref.plotting: both plot functions write a non-empty PNG for synthetic curves."""

import warnings

import numpy as np

from bayes_ref.plotting import plot_curves, plot_round_comparison

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _curves() -> dict[str, dict]:
    return {
        "full": {"mean": np.linspace(0.33, 0.81, 6), "se": np.full(6, 0.01)},
        "nan se": {"mean": np.linspace(0.3, 0.7, 6), "se": np.array([np.nan, 0.02, 0.02, np.nan, 0.01, 0.01])},
        "subset (1, 5)": {"n_obs": [1, 5], "mean": np.array([0.5, 0.7]), "se": np.array([np.nan, 0.03])},
    }


def _is_png(path) -> bool:
    data = path.read_bytes()
    return len(data) > 1000 and data.startswith(PNG_MAGIC)


def test_plot_curves_png(tmp_path):
    path = tmp_path / "sub" / "curves.png"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_curves(_curves(), path, note="Oracle < 1 because 3,315 of 62,400 held-out cases contain true ties")
    assert _is_png(path)
    plot_curves({"only subset": _curves()["subset (1, 5)"]}, tmp_path / "subset_only.png", ylabel="accuracy")
    assert _is_png(tmp_path / "subset_only.png")


def test_plot_round_comparison_png(tmp_path):
    path = tmp_path / "bars.png"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_round_comparison(_curves(), path, n_obs=(1, 5))
    assert _is_png(path)
    plot_round_comparison(_curves(), tmp_path / "bars0.png", n_obs=(0, 1, 5), note="note")
    assert _is_png(tmp_path / "bars0.png")
