"""Regenerate the two figures of the paper from results/main (python experiments/make_figures.py)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mvrdc import analysis as A  # noqa: E402
from mvrdc.operators import OP_NAMES  # noqa: E402

DATA, RES, OUT = "data/MVRDC", "results/main", "paper/figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.6, "pdf.fonttype": 42})
LAB = {"ms-vns": "MS-VNS [1]", "q-vns": "Q-VNS", "vns-n4": "$N_4$ only", "ops-random": "ops-random",
       "ops-alns": "ops-ALNS", "ops-q": "ops-Q (ours)"}
COL = {"ms-vns": "#1f5fa8", "q-vns": "#7aa6d6", "vns-n4": "#6a9a3a", "ops-random": "#c9a227",
       "ops-alns": "#2a9d8f", "ops-q": "#b3261e"}

refs = A.published_references(DATA)
runs = A.load_runs(RES)

# Fig. 1: convergence of the best-so-far gap
conv = A.convergence(runs, refs.set_index("instance")["ref_cost"].to_dict(), np.linspace(0, 1, 121))
fig, ax = plt.subplots(figsize=(3.5, 2.1))
for a in LAB:
    g = conv[conv.algorithm == a].groupby("t")["gap"]
    med = g.median()
    ax.plot(med.index, med.values, color=COL[a], lw=1.2, label=LAB[a])
    ax.fill_between(med.index, g.quantile(.25).values, g.quantile(.75).values, color=COL[a], alpha=.12, lw=0)
ax.set_xlabel("share of the 60 s budget")
ax.set_ylabel(r"best-so-far gap $\Delta$ (%)")
ax.set_ylim(-20, 22)
ax.axhline(0, color="k", lw=.5, ls="--")
ax.grid(alpha=.3)
ax.legend(ncol=2, fontsize=6, frameon=False, loc="upper right")
plt.tight_layout(pad=0.3)
plt.savefig(f"{OUT}/fig_results.pdf")

# Fig. 2: operator shares and infeasibility by fleet class
u = A.neighborhood_usage([r for r in runs if r["algorithm"] in ("ops-q", "ops-alns")])
u["op"] = u.neighborhood.str[1:].astype(int).map(lambda h: OP_NAMES[h - 1])
u["cls"] = np.where(u.instance.str.startswith("A0"), "A0", "A1/A2")
sh = u.groupby(["cls", "algorithm", "op"])["share"].mean()
inf = u.groupby(["cls", "op"])[["picks", "infeasible"]].sum()
inf = 100 * inf.infeasible / inf.picks
fig, ax = plt.subplots(1, 2, figsize=(7.16, 1.75), sharey=True)
x, w = np.arange(len(OP_NAMES)), .38
for k, c in enumerate(["A0", "A1/A2"]):
    for j, a in enumerate(["ops-alns", "ops-q"]):
        ax[k].bar(x + (j - .5) * w, [sh.get((c, a, o), 0) for o in OP_NAMES], w, color=COL[a], label=LAB[a])
    ax[k].axhline(1 / 8, color="k", lw=.5, ls=":")
    ax2 = ax[k].twinx()
    ax2.plot(x, [inf.get((c, o), np.nan) for o in OP_NAMES], "kx", ms=4)
    ax2.set_ylim(0, 70)
    if k == 1:
        ax2.set_ylabel("infeasible (%)")
    else:
        ax2.set_yticklabels([])
    ax[k].set_title({"A0": "A0: small vehicles (~23 routes)", "A1/A2": "A1/A2: large vehicles (~6 routes)"}[c],
                    fontsize=7, pad=2)
    ax[k].set_xticks(x)
    ax[k].set_xticklabels([o.replace("+", "+\n") for o in OP_NAMES], fontsize=5.5)
    ax[k].grid(axis="y", alpha=.3)
ax[0].set_ylabel("share of steps")
ax[0].legend(fontsize=6, frameon=False, loc="upper left")
plt.tight_layout(pad=0.3)
plt.savefig(f"{OUT}/fig_operators.pdf")
print("figures written to", OUT)
