"""Constructive heuristic (Masmoudi et al. 2026, Algorithm 2; authors' constructive.cpp).

Phase 1: physical vehicles are taken in random order (a mixed vehicle starts
in a random configuration). Each vehicle is filled completely: unserved
customers are scanned in random order and inserted at their cheapest
time-feasible position if the load still fits the vehicle (frigoboxes are
added on demand, subject to F_k and to the depot stock F). A vehicle is
closed when a full pass inserts nobody.

Phase 2: every customer left over is inserted at its cheapest feasible
position in any open route; a mixed vehicle may switch to another
configuration if the whole route load fits it (Algorithm 2, lines 23-33).
If a customer still cannot be placed, an unused vehicle is opened for it.
If that fails too, construction is retried (the authors' code would return an
infeasible solution here; see docs/DEVIATIONS.md).
"""
from __future__ import annotations

import random
from typing import List, Optional

from .fleet import Candidate
from .insertion import best_insertion, start_times
from .instance import MIXED
from .solution import Evaluator, Solution


def construct(ev: Evaluator, rng: random.Random, max_tries: int = 50) -> Solution:
    for _ in range(max_tries):
        sol = _construct_once(ev, rng)
        if sol is not None:
            return sol
    raise RuntimeError(f"{ev.inst.name}: construction failed {max_tries} times")


def _construct_once(ev: Evaluator, rng: random.Random) -> Optional[Solution]:
    inst = ev.inst
    fresh, frozen = inst.fresh, inst.frozen
    order = list(range(len(inst.units)))
    rng.shuffle(order)
    unserved = list(inst.customers)
    boxes_left = inst.total_frigo
    routes: List[List[int]] = []
    cands: List[Candidate] = []
    used = set()

    for k in order:
        if not unserved:
            break
        u = inst.units[k]
        cfg = rng.randrange(len(u.configs)) if u.kind == MIXED else 0
        route: List[int] = []
        w1 = w2 = 0
        cur = ev.fit_loads(k, cfg, 0, 0)
        if cur is None:
            continue
        while True:
            rng.shuffle(unserved)
            inserted = False
            departs = start_times(inst, route)
            for c in list(unserved):
                nw1, nw2 = w1 + fresh[c], w2 + frozen[c]
                cand = ev.fit_loads(k, cfg, nw1, nw2)
                if cand is None or cand.boxes - cur.boxes > boxes_left:
                    continue
                ins = best_insertion(inst, route, c, departs)
                if ins is None:
                    continue
                route.insert(ins[1], c)
                boxes_left -= cand.boxes - cur.boxes
                cur, w1, w2 = cand, nw1, nw2
                unserved.remove(c)
                departs = start_times(inst, route)
                inserted = True
            if not inserted:
                break
        if route:
            routes.append(route)
            cands.append(cur)
            used.add(k)

    # Phase 2: leftover customers
    for c in list(unserved):
        best = None  # (delta, route index, position, new candidate)
        for r, route in enumerate(routes):
            w1, w2 = ev.load(route)
            nw1, nw2 = w1 + fresh[c], w2 + frozen[c]
            cur = cands[r]
            u = inst.units[cur.unit]
            configs = range(len(u.configs)) if u.kind == MIXED else (0,)
            fit = None
            for j in configs:
                cand = ev.fit_loads(cur.unit, j, nw1, nw2)
                if cand is not None and cand.boxes - cur.boxes <= boxes_left:
                    if fit is None or cand.residual < fit.residual:
                        fit = cand
            if fit is None:
                continue
            ins = best_insertion(inst, route, c)
            if ins is not None and (best is None or ins[0] < best[0]):
                best = (ins[0], r, ins[1], fit)
        if best is None:
            # open an unused vehicle for this customer
            for k in order:
                if k in used:
                    continue
                u = inst.units[k]
                for j in range(len(u.configs)):
                    cand = ev.fit_loads(k, j, fresh[c], frozen[c])
                    if (cand is not None and cand.boxes <= boxes_left
                            and ev.time_feasible([c])):
                        routes.append([c])
                        cands.append(cand)
                        used.add(k)
                        boxes_left -= cand.boxes
                        break
                else:
                    continue
                break
            else:
                return None
        else:
            _, r, pos, fit = best
            routes[r].insert(pos, c)
            boxes_left -= fit.boxes - cands[r].boxes
            cands[r] = fit
        unserved.remove(c)

    return Solution(routes, cands, ev.objective(routes, cands))
