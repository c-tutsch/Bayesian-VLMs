"""Plots of evaluation curves. The only module that imports matplotlib.

A curve is {"mean": (R,), "se": (R,)} with an optional "n_obs" list of length R (default config.EVAL_OBS_ROUNDS),
so curves evaluated on a subset of n_obs can be drawn next to full ones. NaN se values draw no error bar.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bayes_ref import config  # noqa: E402

RANDOM_LEVEL = 1 / config.N_OPTIONS_PER_SET
EXPECTED_YLABEL = "accuracy on held-out sets (expected under uniform tie-breaking)"


def _curve_x(curve: dict) -> np.ndarray:
    return np.array(curve.get("n_obs", config.EVAL_OBS_ROUNDS))


def _save(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _random_line(ax: plt.Axes) -> None:
    ax.axhline(RANDOM_LEVEL, color="gray", linestyle="--", linewidth=1, label=f"random (1/{config.N_OPTIONS_PER_SET})")


def plot_curves(
    curves: dict[str, dict],
    path: str | Path,
    targets: dict[int, float] | None = config.TARGET_ACCURACY,
    ylabel: str = EXPECTED_YLABEL,
    note: str | None = None,
) -> None:
    """Line plot of curves {name: {"mean": (R,), "se": (R,), optional "n_obs": R ints}} against n_obs.

    Error bars are +- 1 se (none where se is NaN). A dashed line marks random choice (1/3). targets
    {n_obs: accuracy} are drawn as markers only at n_obs present in some curve (None to omit). note is
    printed under the axes. Saves a PNG to path.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5 if note else 4))
    xs = set()
    for name, curve in curves.items():
        x = _curve_x(curve)
        xs.update(x.tolist())
        ax.errorbar(x, curve["mean"], yerr=np.asarray(curve["se"], dtype=np.float64), marker="o", markersize=4,
                    capsize=3, label=name)
    _random_line(ax)
    if targets:
        tx = sorted(n for n in targets if n in xs)
        if tx:
            ax.plot(tx, [targets[n] for n in tx], "kx", markersize=9, mew=2, linestyle="none", label="paper (approx.)")
    ax.set_xticks(sorted(xs))
    ax.set_xlabel("observed interaction rounds")
    ax.set_ylabel(ylabel, fontsize=8 if len(ylabel) > 40 else None)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if note:
        fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8, wrap=True)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    _save(fig, path)


def plot_round_comparison(
    curves: dict[str, dict],
    path: str | Path,
    n_obs: Sequence[int] = (1, 5),
    ylabel: str = "accuracy on held-out sets",
    note: str | None = None,
) -> None:
    """Grouped bar chart in the style of the paper's Fig. 2: one group per n_obs, one bar per curve.

    curves as in plot_curves; a curve without a value at some n_obs leaves an empty slot there. Error bars
    are +- 1 se (none where se is NaN); a dashed line marks random choice (1/3). Saves a PNG to path.
    """
    n_obs = list(n_obs)
    names = list(curves)
    width = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(max(5.0, 1.2 + 0.5 * len(names) * len(n_obs)), 4.5 if note else 4))
    group = np.arange(len(n_obs))
    for j, name in enumerate(names):
        x_of = {n: i for i, n in enumerate(_curve_x(curves[name]).tolist())}
        mean = np.asarray(curves[name]["mean"], dtype=np.float64)
        se = np.asarray(curves[name]["se"], dtype=np.float64)
        sel = [g for g, n in enumerate(n_obs) if n in x_of]
        idx = [x_of[n_obs[g]] for g in sel]
        pos = group[sel] - 0.4 + width * (j + 0.5)
        ax.bar(pos, mean[idx], width=width * 0.95, yerr=se[idx], capsize=3, label=name)
    _random_line(ax)
    ax.set_xticks(group)
    ax.set_xticklabels([f"after {n} round{'s' if n != 1 else ''}" for n in n_obs])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.3)  # headroom above 1 for the legend
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper center", ncol=min(len(names) + 1, 3))
    if note:
        fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8, wrap=True)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    _save(fig, path)
