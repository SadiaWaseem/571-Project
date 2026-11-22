"""Result analysis: reference values, gap tables, statistical tests and figures."""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .instance import load_instance
from .published import read_published, to_solution
from .solution import Evaluator

ALG_ORDER = ["vns-fixed", "ms-vns", "ms-vns-code", "vns-random", "vns-n4", "vns-qmix", "q-vns"]
ALG_LABEL = {
    "vns-fixed": "VNS (fixed order)",
    "ms-vns": "MS-VNS [1]",
    "ms-vns-code": "MS-VNS (code scores)",
    "vns-random": "VNS (random)",
    "vns-n4": "VNS (N4 only)",
    "vns-qmix": "VNS (static Q mix)",
    "q-vns": "Q-VNS (ours)",
}


# --------------------------------------------------------------- references
def published_references(data_root: str, sizes: Iterable[str] = ("10", "200")) -> pd.DataFrame:
    """Cost of the authors' published routes, recomputed with our evaluator."""
    rows = []
    for size in sizes:
        sdir = os.path.join(data_root, "Detailed_route_solutions", f"Solutions {size} customers")
        for p in sorted(glob.glob(os.path.join(sdir, "*.txt"))):
            ip = os.path.join(data_root, "Instances", f"{size} customers", os.path.basename(p))
            if not os.path.exists(ip):
                continue
            inst = load_instance(ip)
            vehicles, reported = read_published(p)
            if not {c for _, r in vehicles for c in r} <= set(inst.ids):
                continue
            ev = Evaluator(inst)
            sol = to_solution(ev, vehicles)
            rows.append({"instance": inst.name, "ref_cost": sol.cost, "ref_reported": reported,
                         "ref_vehicles": sol.n_vehicles, "ref_distance": sol.distance(ev)})
    return pd.DataFrame(rows)


def published_cpu(data_root: str) -> pd.DataFrame:
    """Best/Avg/CPU of the authors' MS-VNS (Online Table E1, configuration C0)."""
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(data_root, "Detailed Results.xlsx"), data_only=True)
    ws = wb["Online Figure E1"]
    rows = []
    for r in ws.iter_rows(min_row=3, values_only=True):
        if not r or not r[0]:
            continue
        try:
            rows.append({"instance": str(r[0]).strip(), "pub_best": float(r[1]),
                         "pub_avg": float(r[2]), "pub_cpu": float(r[4])})
        except (TypeError, ValueError):
            continue
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ tables
def _ensure_runs(d: str) -> None:
    """Unpack ``<d>/runs.zip`` (how stored results are kept in the repository)."""
    import zipfile
    zp = os.path.join(d, "runs.zip")
    if not os.path.isdir(os.path.join(d, "runs")) and os.path.exists(zp):
        with zipfile.ZipFile(zp) as z:
            z.extractall(d)


def load_results(out_dirs: Sequence[str] | str) -> pd.DataFrame:
    """One row per run, read from ``<out>/runs/*.json`` (the source of truth;
    results.csv is only a convenience copy)."""
    if isinstance(out_dirs, str):
        out_dirs = [out_dirs]
    rows = []
    skip = ("history", "selector_stats", "routes", "vehicles_used", "config")
    for d in out_dirs:
        _ensure_runs(d)
        for p in sorted(glob.glob(os.path.join(d, "runs", "*.json"))):
            with open(p) as fh:
                r = json.load(fh)
            row = {k: v for k, v in r.items() if k not in skip}
            row["time_limit"] = r.get("config", {}).get("time_limit")
            rows.append(row)
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["instance", "algorithm", "seed"], keep="last").reset_index(drop=True)


def add_gaps(df: pd.DataFrame, refs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Gap (%) to the published solution if available, else to the best found by any run."""
    df = df.copy()
    best_any = df.groupby("instance")["best_cost"].min().rename("best_any")
    df = df.merge(best_any, on="instance")
    if refs is not None and len(refs):
        df = df.merge(refs[["instance", "ref_cost"]], on="instance", how="left")
        df["reference"] = df["ref_cost"].fillna(df["best_any"])
        df["ref_type"] = np.where(df["ref_cost"].notna(), "published", "best found")
    else:
        df["reference"] = df["best_any"]
        df["ref_type"] = "best found"
    df["gap"] = 100.0 * (df["best_cost"] - df["reference"]) / df["reference"]
    df["family"] = df["instance"].str.extract(r"^(A\d)-\d+-([A-Z]+)\d")[[0, 1]].agg("-".join, axis=1)
    return df


def summary_table(df: pd.DataFrame, by: Sequence[str] = ()) -> pd.DataFrame:
    """Per algorithm (and optional grouping): best-run and average gaps, vehicles, times."""
    per_inst = df.groupby(list(by) + ["algorithm", "instance"]).agg(
        best_gap=("gap", "min"), avg_gap=("gap", "mean"), sd_gap=("gap", "std"),
        vehicles=("vehicles", "mean"), ttb=("time_to_best", "mean"),
        restarts=("restarts", "mean"), feasible=("feasible", "all"))
    tab = per_inst.groupby(list(by) + ["algorithm"]).agg(
        instances=("best_gap", "size"), best_gap=("best_gap", "mean"),
        avg_gap=("avg_gap", "mean"), sd_gap=("sd_gap", "mean"),
        vehicles=("vehicles", "mean"), time_to_best=("ttb", "mean"),
        restarts=("restarts", "mean"), all_feasible=("feasible", "all"))
    tab = tab.reset_index()
    tab["order"] = tab["algorithm"].map({a: i for i, a in enumerate(ALG_ORDER)}).fillna(99)
    return tab.sort_values(list(by) + ["order"]).drop(columns="order").round(3)


def wins_table(df: pd.DataFrame, target: str = "q-vns") -> pd.DataFrame:
    """Instances where ``target`` has a lower / equal / higher mean cost than each rival."""
    m = df.groupby(["instance", "algorithm"])["best_cost"].mean().unstack()
    rows = []
    for alg in m.columns:
        if alg == target:
            continue
        d = m[target] - m[alg]
        rows.append({"vs": alg, "better": int((d < -1e-6).sum()),
                     "equal": int((d.abs() <= 1e-6).sum()), "worse": int((d > 1e-6).sum())})
    return pd.DataFrame(rows)


def significance(df: pd.DataFrame, target: str = "q-vns", metric: str = "best_cost") -> pd.DataFrame:
    """Paired Wilcoxon signed-rank tests on per-instance means, Holm-corrected,
    plus a Friedman test across all algorithms."""
    from scipy.stats import friedmanchisquare, wilcoxon

    m = df.groupby(["instance", "algorithm"])[metric].mean().unstack().dropna()
    rows = []
    for alg in m.columns:
        if alg == target:
            continue
        d = m[target] - m[alg]
        if (d == 0).all():
            p, stat = 1.0, 0.0
        else:
            stat, p = wilcoxon(m[target], m[alg], zero_method="wilcox", alternative="two-sided")
        rows.append({"vs": alg, "n_instances": len(d), "median_diff": float(d.median()),
                     "mean_diff": float(d.mean()), "W": float(stat), "p": float(p)})
    out = pd.DataFrame(rows).sort_values("p")
    k = len(out)
    adj, running = [], 0.0
    for i, p in enumerate(out["p"]):  # Holm step-down
        running = max(running, min(1.0, (k - i) * p))
        adj.append(running)
    out["p_holm"] = adj
    if m.shape[1] >= 3 and m.shape[0] >= 2:
        fr = friedmanchisquare(*[m[c] for c in m.columns])
        out.attrs["friedman"] = {"chi2": float(fr.statistic), "p": float(fr.pvalue),
                                 "algorithms": list(m.columns), "instances": int(m.shape[0])}
    return out.reset_index(drop=True)


# ------------------------------------------------------------- run details
def load_runs(out_dir: str, pattern: str = "*") -> List[dict]:
    _ensure_runs(out_dir)
    runs = []
    for p in sorted(glob.glob(os.path.join(out_dir, "runs", pattern + ".json"))):
        with open(p) as fh:
            runs.append(json.load(fh))
    return runs


def convergence(runs: List[dict], reference: Dict[str, float], grid: np.ndarray) -> pd.DataFrame:
    """Best-so-far gap (%) on a common time grid (share of the budget)."""
    rows = []
    for r in runs:
        ref = reference.get(r["instance"])
        if ref is None:
            continue
        tl = r["config"]["time_limit"]
        hist = np.array(r["history"])  # time, restart, cost
        t = hist[:, 0] / tl
        c = hist[:, 2]
        idx = np.searchsorted(t, grid, side="right") - 1
        vals = np.where(idx >= 0, c[np.clip(idx, 0, None)], np.nan)
        for g, v in zip(grid, vals):
            rows.append({"algorithm": r["algorithm"], "instance": r["instance"],
                         "seed": r["seed"], "t": g, "gap": 100 * (v - ref) / ref})
    return pd.DataFrame(rows)


def neighborhood_usage(runs: List[dict]) -> pd.DataFrame:
    """Share of shaking steps and success rates per neighborhood and algorithm."""
    rows = []
    for r in runs:
        st = r["selector_stats"]
        picks = st["picks"]
        tot = max(1, sum(picks))
        for h, (n, oc) in enumerate(zip(picks, st["outcomes"])):
            rows.append({"algorithm": r["algorithm"], "instance": r["instance"], "seed": r["seed"],
                         "neighborhood": f"N{h + 1}", "share": n / tot,
                         "new_best": oc[0], "improved": oc[1], "accepted": oc[2],
                         "rejected": oc[3], "infeasible": oc[4], "picks": n})
    return pd.DataFrame(rows)
