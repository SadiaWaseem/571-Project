"""Resumable experiment runner.

Every (instance, algorithm, seed) run appends one row to ``<out>/results.csv``
and writes full details (convergence history, selector statistics, routes) to
``<out>/runs/<instance>__<algorithm>__<seed>.json``. Runs whose JSON already
exists are skipped, so an interrupted Colab session can simply be restarted.

Example
-------
python experiments/run.py --data data/MVRDC --sizes "200 customers" \
    --algorithms ms-vns q-vns vns-random --seeds 1-10 --time-limit 60 \
    --out results/main --workers 2
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mvrdc.instance import list_instances, load_instance  # noqa: E402
from mvrdc.selectors import make_selector  # noqa: E402
from mvrdc.vns import VNSConfig, solve  # noqa: E402

FIELDS = ["instance", "algorithm", "seed", "best_cost", "vehicles", "distance", "unused",
          "boxes", "time_to_best", "runtime", "restarts", "steps", "route_evals", "feasible",
          "emissions"]


def parse_seeds(text: str):
    seeds = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run_one(path: str, algorithm: str, seed: int, cfg_dict: dict, sel_kw: dict, out: str):
    inst = load_instance(path)
    cfg = VNSConfig(**cfg_dict)
    sel = make_selector(algorithm, **sel_kw.get(algorithm, {}))
    res = solve(inst, sel, cfg, seed)
    res.algorithm = algorithm  # keep the CLI label (e.g. variants of q-vns)
    tag = f"{inst.name}__{algorithm}__{seed}"
    with open(os.path.join(out, "runs", tag + ".json"), "w") as fh:
        json.dump(asdict(res), fh)
    return res.to_row()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="clone of github.com/Masmoudi-Amine/MVRDC-")
    ap.add_argument("--sizes", nargs="+", default=["200 customers"])
    ap.add_argument("--instances", nargs="*", default=["*"], help="glob patterns on instance names")
    ap.add_argument("--algorithms", nargs="+", default=["ms-vns", "q-vns"])
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--max-restarts", type=int, default=None)
    ap.add_argument("--out", default="results/run")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override VNSConfig fields, e.g. ls_moves=30")
    ap.add_argument("--q", nargs="*", default=[], metavar="KEY=VALUE",
                    help="Q-learning parameters, e.g. alpha=0.2 gamma=0.5")
    args = ap.parse_args(argv)

    cfg = {"time_limit": args.time_limit, "max_restarts": args.max_restarts}
    for kv in args.set:
        k, v = kv.split("=")
        cfg[k] = type(getattr(VNSConfig(), k))(v) if getattr(VNSConfig(), k) is not None else float(v)
    q_kw = {}
    for kv in args.q:
        k, v = kv.split("=")
        q_kw[k] = json.loads(v) if v.startswith(("[", "(")) else float(v)
    sel_kw = {a: q_kw for a in args.algorithms if a.startswith("q")}

    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    csv_path = os.path.join(args.out, "results.csv")
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump({"vns": cfg, "q": q_kw, "argv": sys.argv}, fh, indent=1)

    jobs = []
    for size in args.sizes:
        for path in list_instances(args.data, size):
            name = os.path.splitext(os.path.basename(path))[0]
            if not any(fnmatch.fnmatch(name, pat) for pat in args.instances):
                continue
            for alg in args.algorithms:
                for seed in parse_seeds(args.seeds):
                    tag = f"{name}__{alg}__{seed}"
                    if os.path.exists(os.path.join(args.out, "runs", tag + ".json")):
                        continue
                    jobs.append((path, alg, seed))
    print(f"{len(jobs)} runs to do -> {csv_path}", flush=True)
    rebuild_csv(args.out)
    new_file = not os.path.exists(csv_path)
    t0 = time.time()
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        if args.workers <= 1:
            for k, (p, a, s) in enumerate(jobs, 1):
                row = run_one(p, a, s, cfg, sel_kw, args.out)
                w.writerow(row)
                fh.flush()
                print(f"[{k}/{len(jobs)} {time.time() - t0:7.0f}s] {row['instance']} {a} seed={s} "
                      f"cost={row['best_cost']:.2f} veh={row['vehicles']}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(run_one, p, a, s, cfg, sel_kw, args.out): (p, a, s)
                        for (p, a, s) in jobs}
                for k, f in enumerate(as_completed(futs), 1):
                    row = f.result()
                    w.writerow(row)
                    fh.flush()
                    print(f"[{k}/{len(jobs)} {time.time() - t0:7.0f}s] {row['instance']} "
                          f"{row['algorithm']} seed={row['seed']} cost={row['best_cost']:.2f} "
                          f"veh={row['vehicles']}", flush=True)
    rebuild_csv(args.out)


def rebuild_csv(out: str) -> None:
    """Rewrite results.csv from the per-run JSON files (keeps it consistent
    when runs from several sessions or code versions are merged)."""
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "runs", "*.json"))):
        with open(p) as fh:
            r = json.load(fh)
        rows.append({k: r.get(k) for k in FIELDS})
    with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
