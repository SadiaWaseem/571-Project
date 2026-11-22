"""Matplotlib figures for the notebook and the IEEE paper (single-column width)."""
from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import ALG_LABEL, ALG_ORDER
from .instance import Instance

COLORS = {"vns-fixed": "#8c8c8c", "ms-vns": "#1f5fa8", "ms-vns-code": "#7aa6d6",
          "vns-random": "#d08c2a", "vns-n4": "#6a9a3a", "vns-qmix": "#9b59b6",
          "q-vns": "#b3261e"}
COLUMN_W = 3.5  # inches, IEEE single column


def _order(algs: Sequence[str]) -> List[str]:
    return [a for a in ALG_ORDER if a in algs] + [a for a in algs if a not in ALG_ORDER]


def plot_instance(inst: Instance, ax=None):
    ax = ax or plt.subplots(figsize=(COLUMN_W, COLUMN_W))[1]
    share = np.array([z / max(1, f + z) for f, z in zip(inst.fresh, inst.frozen)])[1:]
    sc = ax.scatter(inst.x[1:], inst.y[1:], c=share, cmap="coolwarm", s=12, vmin=0, vmax=1)
    ax.scatter([inst.x[0]], [inst.y[0]], marker="s", c="k", s=40, label="depot")
    ax.set_title(f"{inst.name}: {inst.n} customers", fontsize=9)
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, fraction=0.046, label="frozen share of demand")
    ax.legend(fontsize=7, loc="upper right")
    return ax


def plot_routes(inst: Instance, routes: Sequence[Sequence[int]], vehicles: Sequence[str] = (),
                ax=None, title: Optional[str] = None):
    """``routes`` use original customer labels (as stored in RunResult.routes)."""
    ax = ax or plt.subplots(figsize=(COLUMN_W, COLUMN_W))[1]
    cmap = plt.get_cmap("tab20")
    for k, r in enumerate(routes):
        idx = [0] + [inst.index_of(c) for c in r] + [0]
        ax.plot([inst.x[i] for i in idx], [inst.y[i] for i in idx], "-", lw=0.8,
                color=cmap(k % 20), marker="o", ms=2)
    ax.scatter([inst.x[0]], [inst.y[0]], marker="s", c="k", s=40, zorder=5)
    ax.set_aspect("equal")
    ax.set_title(title or f"{inst.name}: {len(routes)} routes", fontsize=9)
    return ax


def plot_convergence(conv: pd.DataFrame, ax=None, logy: bool = False):
    """Median (line) and inter-quartile band of the best-so-far gap over the budget."""
    ax = ax or plt.subplots(figsize=(COLUMN_W, 2.4))[1]
    for alg in _order(conv["algorithm"].unique()):
        g = conv[conv["algorithm"] == alg].groupby("t")["gap"]
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        ax.plot(med.index, med.values, color=COLORS.get(alg), label=ALG_LABEL.get(alg, alg), lw=1.4)
        ax.fill_between(med.index, q1.values, q3.values, color=COLORS.get(alg), alpha=0.15, lw=0)
    ax.set_xlabel("share of time budget")
    ax.set_ylabel("gap to reference (%)")
    if logy:
        ax.set_yscale("symlog")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return ax


def plot_gap_box(df: pd.DataFrame, ax=None, col: str = "gap"):
    ax = ax or plt.subplots(figsize=(COLUMN_W, 2.4))[1]
    algs = _order(df["algorithm"].unique())
    data = [df.loc[df["algorithm"] == a, col].values for a in algs]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True)
    for patch, a in zip(bp["boxes"], algs):
        patch.set_facecolor(COLORS.get(a, "#cccccc"))
        patch.set_alpha(0.6)
    ax.set_xticks(range(1, len(algs) + 1))
    ax.set_xticklabels([ALG_LABEL.get(a, a).replace(" (", "\n(") for a in algs], fontsize=7)
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_ylabel("gap to reference (%)")
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_usage(usage: pd.DataFrame, ax=None):
    """Share of shaking steps per neighborhood and the share that were infeasible."""
    ax = ax or plt.subplots(figsize=(COLUMN_W, 2.4))[1]
    algs = _order(usage["algorithm"].unique())
    hoods = sorted(usage["neighborhood"].unique())
    w = 0.8 / len(algs)
    for k, a in enumerate(algs):
        m = usage[usage["algorithm"] == a].groupby("neighborhood")["share"].mean()
        ax.bar(np.arange(len(hoods)) + k * w, [m.get(h, 0) for h in hoods], w,
               color=COLORS.get(a), label=ALG_LABEL.get(a, a))
    ax.set_xticks(np.arange(len(hoods)) + 0.4 - w / 2)
    ax.set_xticklabels(hoods)
    ax.set_ylabel("share of shaking steps")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_q_policy(q_table: dict, ax=None,
                  features=("outcome", "stagnation", "phase", "boxes")):
    """Heat map of the learned Q-values, one row per visited state."""
    ax = ax or plt.subplots(figsize=(COLUMN_W, 4))[1]
    keys = sorted(q_table)
    mat = np.array([q_table[k] for k in keys])
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(keys, fontsize=5)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([f"N{h + 1}" for h in range(mat.shape[1])])
    ax.set_ylabel("state (" + ",".join(features) + ")", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Q")
    return ax
