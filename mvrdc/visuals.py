"""Visual analysis for the notebook: problem anatomy, solution anatomy, result
overviews (heat map, performance profile, critical-difference diagram, radar),
learning dynamics, and an interactive explorer.

All functions work from stored results (``results/main``) except
:func:`trace_run`, which records one short live run decision by decision.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from .analysis import ALG_LABEL
from .insertion import start_times
from .instance import FRESH, FROZEN, MIXED, Instance
from .published import to_solution
from .solution import Evaluator, Solution

KIND_COLOR = {FRESH: "#2a9d8f", FROZEN: "#1f5fa8", MIXED: "#9b59b6"}
KIND_LABEL = {FRESH: "fresh vehicle", FROZEN: "frozen vehicle", MIXED: "mixed-reconfigurable"}
METHOD_ORDER = ["ms-vns", "q-vns", "vns-n4", "ops-random", "ops-alns", "ops-q"]
SHORT = {"ms-vns": "MS-VNS [1]", "q-vns": "Q-VNS", "vns-n4": "N4 only", "ops-random": "ops-random",
         "ops-alns": "ops-ALNS", "ops-q": "ops-Q"}
METHOD_COLOR = {"ms-vns": "#1f5fa8", "q-vns": "#7aa6d6", "vns-n4": "#6a9a3a",
                "ops-random": "#c9a227", "ops-alns": "#2a9d8f", "ops-q": "#b3261e"}


# ======================================================== problem anatomy
def plot_instance_gallery(instances: Sequence[Instance], axes=None):
    """Customers of several instances side by side: colour = frozen share of the
    customer's demand, size = total demand; the fleet is summarized in the title."""
    if axes is None:
        _, axes = plt.subplots(1, len(instances), figsize=(3.4 * len(instances), 3.4))
    axes = np.atleast_1d(axes)
    sc = None
    for ax, inst in zip(axes, instances):
        tot = np.array([f + z for f, z in zip(inst.fresh, inst.frozen)])[1:]
        share = np.array([z / max(1, f + z) for f, z in zip(inst.fresh, inst.frozen)])[1:]
        sc = ax.scatter(inst.x[1:], inst.y[1:], c=share, s=4 + 1.2 * tot, cmap="coolwarm",
                        vmin=0, vmax=1, edgecolor="k", linewidth=0.2)
        ax.scatter([inst.x[0]], [inst.y[0]], marker="s", c="k", s=50)
        caps = [u.capacity for u in inst.units if u.kind != MIXED]
        s = inst.summary()
        ax.set_title(f"{inst.name}\n{s['fresh_units']} fresh, {s['frozen_units']} frozen, "
                     f"{s['mixed_units']} mixed vehicles\ncapacity {min(caps)}–{max(caps)}, "
                     f"F = {inst.total_frigo} frigoboxes", fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    cb = plt.colorbar(sc, ax=list(axes), fraction=0.02, pad=0.01)
    cb.set_label("frozen share of demand")
    return axes


def plot_tw_gantt(inst: Instance, route: Sequence[int], ax=None, title: str = ""):
    """Time-window chart of one route: grey bar = window [e_i, l_i], dot = arrival,
    coloured bar = service, hatched = waiting. Labels are internal indices."""
    ax = ax or plt.subplots(figsize=(7, 0.35 * len(route) + 1))[1]
    d = inst.dist
    t, prev = 0.0, 0
    for k, c in enumerate(route):
        arr = t + d[prev][c]
        start = max(arr, inst.ready[c])
        ax.add_patch(Rectangle((inst.ready[c], k - 0.3), inst.due[c] - inst.ready[c], 0.6,
                               color="#d9d9d9", lw=0))
        if start > arr:
            ax.add_patch(Rectangle((arr, k - 0.12), start - arr, 0.24, fill=False, hatch="////",
                                   edgecolor="#999999", lw=0.5))
        ax.add_patch(Rectangle((start, k - 0.18), inst.service[c], 0.36, color="#b3261e", lw=0))
        ax.plot([arr], [k], "ko", ms=3)
        t, prev = start + inst.service[c], c
    back = t + d[prev][0]
    ax.axvline(back, color="k", ls="--", lw=0.8)
    ax.text(back, -0.6, "back at depot ", fontsize=7, va="bottom", ha="right")
    ax.set_yticks(range(len(route)))
    ax.set_yticklabels([f"customer {inst.ids[c]}" for c in route], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, max(back, max(inst.due[c] + inst.service[c] for c in route)) * 1.03)
    ax.set_xlabel("time")
    ax.legend(handles=[Patch(color="#d9d9d9", label="time window"),
                       Patch(color="#b3261e", label="service"),
                       Patch(fill=False, hatch="////", edgecolor="#999999", label="waiting")],
              fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    ax.set_title(title or "Schedule of one route (service must start inside the window)", fontsize=9)
    return ax


# ======================================================= solution anatomy
def solution_from_run(inst: Instance, run: dict, ev: Optional[Evaluator] = None) -> Solution:
    """Rebuild a Solution (with frigobox counts) from a stored RunResult dict."""
    ev = ev or Evaluator(inst)
    return to_solution(ev, list(zip(run["vehicles_used"], run["routes"])))


def plot_solution_anatomy(inst: Instance, sol: Solution, ev: Evaluator, axes=None, title: str = ""):
    """Left: routes coloured by vehicle type. Right: every route's load split into
    fresh, frozen, frigobox padding and empty space, relative to its capacity."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [1, 1.1]})
    ax, bx = axes
    order = sorted(range(len(sol.routes)), key=lambda k: (inst.units[sol.assign[k].unit].kind,
                                                          -len(sol.routes[k])))
    for k in order:
        r, a = sol.routes[k], sol.assign[k]
        kind = inst.units[a.unit].kind
        idx = [0] + list(r) + [0]
        ax.plot([inst.x[i] for i in idx], [inst.y[i] for i in idx], "-", color=KIND_COLOR[kind],
                lw=0.9, alpha=0.85)
        ax.scatter([inst.x[i] for i in r], [inst.y[i] for i in r], s=6, color=KIND_COLOR[kind],
                   zorder=3)
        if a.boxes:
            mx = np.mean([inst.x[i] for i in r]); my = np.mean([inst.y[i] for i in r])
            ax.text(mx, my, "■" * a.boxes, fontsize=5, color="#e07a5f", ha="center", va="center")
    ax.scatter([inst.x[0]], [inst.y[0]], marker="s", c="k", s=60, zorder=4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=[Patch(color=KIND_COLOR[k], label=KIND_LABEL[k]) for k in KIND_COLOR]
              + [Patch(color="#e07a5f", label="■ = one frigobox")], fontsize=6.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    ax.set_title(title or f"{len(sol.routes)} routes", fontsize=9)

    bw = inst.box_space
    for row, k in enumerate(order):
        r, a = sol.routes[k], sol.assign[k]
        u = inst.units[a.unit]
        w1, w2 = ev.load(r)
        if u.kind in (FRESH, FROZEN):
            cap = u.capacity
            native, foreign = (w1, w2) if u.kind == FRESH else (w2, w1)
            parts = [(native, "#2a9d8f" if u.kind == FRESH else "#1f5fa8"),
                     (foreign, "#1f5fa8" if u.kind == FRESH else "#2a9d8f"),
                     (a.boxes * bw - foreign, "#f2c6b4")]
        else:
            _, q1, q2 = u.configs[a.config]
            cap = q1 + q2
            parts = [(w1, "#2a9d8f"), (q1 - w1, "#e9e9e9"), (w2, "#1f5fa8")]
        left = 0.0
        for val, colr in parts:
            bx.barh(row, val / cap, left=left, color=colr, height=0.7, lw=0)
            left += val / cap
        if u.kind == MIXED:
            bx.plot([q1 / cap] * 2, [row - 0.4, row + 0.4], color="k", lw=0.8)
        bx.barh(row, 1 - left, left=left, color="#ffffff", edgecolor="#cccccc", height=0.7, lw=0.3)
        bx.text(1.01, row, f"{u.configs[a.config][0]} ({cap})", fontsize=5.5, va="center")
    bx.set_yticks([]); bx.set_xlim(0, 1.18); bx.invert_yaxis()
    bx.set_xlabel("share of vehicle capacity")
    bx.legend(handles=[Patch(color="#2a9d8f", label="fresh load"), Patch(color="#1f5fa8", label="frozen load"),
                       Patch(color="#f2c6b4", label="unused frigobox space"),
                       Patch(color="#e9e9e9", label="unused fresh compartment"),
                       Patch(facecolor="white", edgecolor="#cccccc", label="empty")],
              fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    bx.set_title("load of every route (black tick = bulkhead of a mixed vehicle)", fontsize=8)
    return axes


# ======================================================== result overviews
def plot_gap_heatmap(df: pd.DataFrame, ax=None, methods: Sequence[str] = METHOD_ORDER):
    """Instance × method heat map of the mean gap (%), with the best method per row framed."""
    m = df.groupby(["instance", "algorithm"])["gap"].mean().unstack()[list(methods)]
    m = m.loc[sorted(m.index, key=lambda s: (s.split("-")[0], s.split("-")[2].rstrip("0123456789"), s))]
    ax = ax or plt.subplots(figsize=(5.5, 0.22 * len(m) + 1.2))[1]
    lim = np.nanmax(np.abs(m.values))
    im = ax.imshow(m.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(m.shape[0]):
        j_best = int(np.nanargmin(m.values[i]))
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(v) > 0.6 * lim else "black",
                    fontweight="bold" if j == j_best else "normal")
        ax.add_patch(Rectangle((j_best - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="k", lw=1))
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels([SHORT.get(c, c) for c in m.columns], rotation=30,
                                                         ha="right", fontsize=7)
    ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(m.index, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="mean gap to [1] (%)  (blue = better)")
    return ax


def plot_performance_profile(df: pd.DataFrame, ax=None, methods: Sequence[str] = METHOD_ORDER,
                             tau_max: float = 1.25):
    """Dolan-Moré performance profile: share of instances on which a method's mean
    cost is within a factor tau of the best method's mean cost."""
    m = df.groupby(["instance", "algorithm"])["best_cost"].mean().unstack()[list(methods)]
    ratio = m.div(m.min(axis=1), axis=0)
    ax = ax or plt.subplots(figsize=(4.5, 2.8))[1]
    taus = np.linspace(1, tau_max, 400)
    for c in methods:
        ax.step(taus, [(ratio[c] <= t).mean() for t in taus], where="post", color=METHOD_COLOR[c],
                label=SHORT[c], lw=1.4)
    ax.set_xlabel(r"$\tau$ (cost / best method's cost)")
    ax.set_ylabel(r"share of instances within $\tau$")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7, frameon=False, loc="lower right")
    return ax


# Studentized range / sqrt(2) for the Nemenyi test at alpha = 0.05 (Demsar, 2006)
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def plot_cd_diagram(df: pd.DataFrame, ax=None, methods: Sequence[str] = METHOD_ORDER):
    """Critical-difference diagram (Demsar, 2006): average rank of each method over
    the instances; methods joined by a bar are not significantly different
    (Nemenyi test, alpha = 0.05)."""
    m = df.groupby(["instance", "algorithm"])["best_cost"].mean().unstack()[list(methods)]
    ranks = m.rank(axis=1).mean().sort_values()
    k, n = len(methods), len(m)
    cd = _Q05[k] * math.sqrt(k * (k + 1) / (6 * n))
    ax = ax or plt.subplots(figsize=(6.5, 2.4))[1]
    ax.set_xlim(-0.6, k + 1.6); ax.set_ylim(0, 1)
    y_axis = 0.82
    ax.hlines(y_axis, 1, k, color="k", lw=0.8)
    for r in range(1, k + 1):
        ax.vlines(r, y_axis - 0.02, y_axis + 0.02, color="k", lw=0.8)
        ax.text(r, y_axis + 0.04, str(r), ha="center", fontsize=7)
    names = list(ranks.index)
    vals = list(ranks.values)
    cliques = []
    for i in range(len(vals)):
        j = i
        while j + 1 < len(vals) and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c, (i, j) in enumerate(cliques):
        y = y_axis - 0.07 - 0.045 * c
        ax.hlines(y, vals[i] - 0.04, vals[j] + 0.04, color="k", lw=2.5)
    top = y_axis - 0.07 - 0.045 * len(cliques) - 0.04
    half = (len(names) + 1) // 2
    for i, name in enumerate(names):
        r = ranks[name]
        left = i < half
        y = top - 0.12 * (i if left else len(names) - 1 - i)
        xt = 0.85 if left else k + 0.15
        colr = METHOD_COLOR.get(name, "k")
        ax.plot([r, r, xt], [y_axis, y, y], color=colr, lw=1)
        ax.text(xt - 0.05 if left else xt + 0.05, y, f"{SHORT.get(name, name)} ({r:.2f})", fontsize=7,
                ha="right" if left else "left", va="center", color=colr)
    ax.hlines(0.97, 1, 1 + cd, color="#b3261e", lw=1.5)
    ax.text(1 + cd + 0.05, 0.97, f"critical difference = {cd:.2f}", ha="left", va="center", fontsize=7,
            color="#b3261e")
    ax.axis("off")
    ax.set_title(f"Average rank over {n} instances (1 = best); thick bars join methods that are "
                 f"not significantly different", fontsize=8)
    return ranks, cd


def plot_radar(df: pd.DataFrame, conv: Optional[pd.DataFrame] = None, ax=None,
               methods: Sequence[str] = METHOD_ORDER):
    """Six criteria scaled per criterion to [0, 1], 1 = best method on that criterion.
    ``conv`` (from analysis.convergence) adds anytime quality: the gap after 20% of the budget."""
    g = df.groupby("algorithm")
    sd = df.groupby(["instance", "algorithm"])["gap"].std().groupby("algorithm").mean()
    crit = pd.DataFrame({
        "solution quality\n(avg gap)": -g["gap"].mean(),
        "fewer vehicles": -g["vehicles"].mean(),
        "shorter distance": -g["distance"].mean(),
        "lower emissions": -g["emissions"].mean(),
        "robustness\n(low seed spread)": -sd,
    })
    if conv is not None:
        early = conv[np.isclose(conv["t"], conv["t"].unique()[np.argmin(np.abs(conv["t"].unique() - 0.2))])]
        crit["anytime quality\n(gap after 20% of time)"] = -early.groupby("algorithm")["gap"].mean()
    crit = crit.loc[list(methods)]
    norm = (crit - crit.min()) / (crit.max() - crit.min()).replace(0, 1)
    ang = np.linspace(0, 2 * np.pi, len(norm.columns), endpoint=False).tolist()
    ang += ang[:1]
    if ax is None:
        ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw={"polar": True})[1]
    for c in methods:
        v = norm.loc[c].tolist()
        v += v[:1]
        ax.plot(ang, v, color=METHOD_COLOR[c], lw=1.3, label=SHORT[c])
        ax.fill(ang, v, color=METHOD_COLOR[c], alpha=0.06)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(norm.columns, fontsize=6.5)
    ax.set_yticks([0.25, 0.5, 0.75]); ax.set_yticklabels([]); ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", pad=8)
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.08, 1.05), frameon=False)
    return norm


# ======================================================= learning dynamics
def plot_alns_weights(run: dict, op_names: Sequence[str], ax=None):
    """Evolution of the roulette-wheel weights of ALNS over the segments of one run."""
    W = np.array(run["selector_stats"]["weight_history"])
    W = W / W.sum(axis=1, keepdims=True)
    ax = ax or plt.subplots(figsize=(4.5, 2.6))[1]
    cmap = plt.get_cmap("tab10")
    ax.stackplot(np.arange(len(W)) * 100, W.T, colors=[cmap(i) for i in range(W.shape[1])],
                 labels=op_names, alpha=0.85)
    ax.set_xlabel("shaking step"); ax.set_ylabel("selection probability")
    ax.set_xlim(0, (len(W) - 1) * 100); ax.set_ylim(0, 1)
    ax.set_title(f"ALNS roulette: {run['instance']}", fontsize=8)
    return ax


def trace_run(inst: Instance, algorithm: str, seconds: float, seed: int = 1):
    """Run one method and record every shaking decision: (time share, action, outcome, reward)."""
    from .operators import operator_shaking
    from .selectors import OPS_ALGORITHMS, Outcome, make_selector
    from .vns import VNSConfig, solve
    sel = make_selector(algorithm)
    log: List[tuple] = []
    orig = sel.feedback
    rewards = (5.0, 2.0, 0.5, 0.0, -1.0)

    def feedback(h, outcome, s):
        log.append((s.elapsed_frac, h, int(outcome), rewards[int(outcome)]))
        orig(h, outcome, s)
    sel.feedback = feedback
    if algorithm in OPS_ALGORITHMS:
        with operator_shaking():
            res = solve(inst, sel, VNSConfig(time_limit=seconds), seed)
    else:
        res = solve(inst, sel, VNSConfig(time_limit=seconds), seed)
    return res, pd.DataFrame(log, columns=["t", "action", "outcome", "reward"])


def plot_learning_trace(trace: pd.DataFrame, op_names: Sequence[str], ax=None, window: int = 150,
                        title: str = ""):
    """Rolling share of each action over the run (what the policy is choosing now)."""
    ax = ax or plt.subplots(figsize=(4.5, 2.6))[1]
    onehot = np.eye(len(op_names))[trace["action"].values]
    roll = pd.DataFrame(onehot).rolling(window, min_periods=20).mean().values
    cmap = plt.get_cmap("tab10")
    ax.stackplot(np.arange(len(roll)), np.nan_to_num(roll).T, colors=[cmap(i) for i in range(len(op_names))],
                 labels=op_names, alpha=0.85)
    ax.set_xlim(0, len(roll)); ax.set_ylim(0, 1)
    ax.set_xlabel("shaking step"); ax.set_ylabel(f"share of last {window} steps")
    ax.set_title(title, fontsize=8)
    return ax


# =================================================== interactive explorer
def explorer(runs: List[dict], data_root: str, refs: Optional[pd.DataFrame] = None):
    """Dropdown explorer of the stored solutions (works in Colab / Jupyter)."""
    import ipywidgets as W
    from IPython.display import display
    from .instance import load_instance

    by = {(r["instance"], r["algorithm"], r["seed"]): r for r in runs}
    insts = sorted({k[0] for k in by})
    algs = [a for a in METHOD_ORDER if any(k[1] == a for k in by)]
    seeds = sorted({k[2] for k in by})
    refmap = refs.set_index("instance")["ref_cost"].to_dict() if refs is not None else {}
    w_inst = W.Dropdown(options=insts, value=insts[0], description="instance")
    w_alg = W.Dropdown(options=[(SHORT[a], a) for a in algs], value=algs[-1], description="method")
    w_seed = W.Dropdown(options=seeds, value=seeds[0], description="seed")
    out = W.Output()
    cache: Dict[str, Instance] = {}

    def draw(*_):
        out.clear_output(wait=True)
        name = w_inst.value
        if name not in cache:
            size = "200 customers" if "-50-" in name else f"{int(name.split('-')[1])} customers"
            cache[name] = load_instance(f"{data_root}/Instances/{size}/{name}.txt")
        inst = cache[name]
        run = by.get((name, w_alg.value, w_seed.value))
        with out:
            if run is None:
                print("no run stored for this combination")
                return
            ev = Evaluator(inst)
            sol = solution_from_run(inst, run, ev)
            ref = refmap.get(name)
            gap = f", gap to [1] {100 * (run['best_cost'] - ref) / ref:+.2f}%" if ref else ""
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
            plot_solution_anatomy(inst, sol, ev, axes,
                                  title=f"{SHORT[w_alg.value]}, seed {w_seed.value}: cost {run['best_cost']:.1f}"
                                        f"{gap}\n{run['vehicles']} vehicles, distance {run['distance']:.0f}, "
                                        f"{run['boxes']} frigoboxes")
            plt.tight_layout(); plt.show()

    for w in (w_inst, w_alg, w_seed):
        w.observe(draw, names="value")
    display(W.HBox([w_inst, w_alg, w_seed]), out)
    draw()
