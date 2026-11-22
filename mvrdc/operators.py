"""Destroy-and-repair operator portfolio (Option 2 of the project).

Every shaking step is a remove-insert move, as in N4 of Masmoudi et al. (2026),
but the policy chooses one of 8 (removal, insertion) pairs:

  removal  : random | worst (largest detour, randomized p=3) |
             related (Shaw: distance + ready-time similarity, p=6) |
             route (all customers of one of the shortest routes)
  insertion: greedy (globally cheapest feasible insertion, Ropke & Pisinger 2006) |
             regret2 (largest gap between best and second-best route first)

lambda (number of customers removed) adapts as in [1]: it is reset to
lambda_min after a new best solution and incremented otherwise. A customer
that cannot be re-inserted into any open route makes the move infeasible.
Routes emptied by the removal are closed, which is how a move saves a vehicle.
"""
from __future__ import annotations

import contextlib
from typing import List, Optional, Tuple

from . import vns as _vns
from .insertion import best_insertion, start_times
from .neighborhoods import Shaker

REMOVALS = ("random", "worst", "related", "route")
REPAIRS = ("greedy", "regret2")
OPS: List[Tuple[str, str]] = [(r, i) for r in REMOVALS for i in REPAIRS]
OP_NAMES = tuple(f"{r}+{i}" for r, i in OPS)


class OperatorShaker(Shaker):
    """Shaking by destroy-and-repair; action h = OPS[h]."""

    def __init__(self, ev, tries: int = 3):
        super().__init__(ev, tries)
        inst = ev.inst
        self.max_d = max(max(row) for row in inst.dist) or 1.0
        self.max_tw = (max(inst.ready) - min(inst.ready)) or 1.0
        self._moves = tuple((self._make(r, i),) for r, i in OPS)

    def _make(self, removal: str, repair: str):
        def move(sol, rng):
            return self.destroy_repair(sol, rng, removal, repair)
        return move

    # ------------------------------------------------------------- destroy
    def remove(self, sol, rng, rule: str):
        inst = self.ev.inst
        d = inst.dist
        routes = [r[:] for r in sol.routes]
        if rule == "route":
            order = sorted(range(len(routes)), key=lambda k: (len(routes[k]), rng.random()))
            k = order[int(rng.random() ** 3 * min(3, len(order)))]
            removed = routes[k][:]
            routes[k] = []
            return routes, removed
        pool = [c for r in routes for c in r]
        lam = min(self.lam, len(pool))
        if rule == "random":
            removed = rng.sample(pool, lam)
        elif rule == "worst":
            removed = []
            for _ in range(lam):
                cost = []
                for r in routes:
                    for i, c in enumerate(r):
                        p = r[i - 1] if i else 0
                        n = r[i + 1] if i + 1 < len(r) else 0
                        cost.append((d[p][c] + d[c][n] - d[p][n], c))
                cost.sort(reverse=True)
                c = cost[int(rng.random() ** 3 * len(cost))][1]
                removed.append(c)
                for r in routes:
                    if c in r:
                        r.remove(c)
                        break
            return routes, removed
        elif rule == "related":
            ready = inst.ready
            left = set(pool)
            c0 = rng.choice(pool)
            removed = [c0]
            left.discard(c0)
            while len(removed) < lam and left:
                c = rng.choice(removed)
                cand = sorted(left, key=lambda j: d[c][j] / self.max_d
                              + abs(ready[c] - ready[j]) / self.max_tw)
                j = cand[int(rng.random() ** 6 * len(cand))]
                removed.append(j)
                left.discard(j)
        else:
            raise ValueError(rule)
        gone = set(removed)
        return [[c for c in r if c not in gone] for r in routes], removed

    # -------------------------------------------------------------- repair
    def destroy_repair(self, sol, rng, removal: str, repair: str):
        ev, inst = self.ev, self.ev.inst
        routes, removed = self.remove(sol, rng, removal)
        if not removed:
            return None
        keep = [k for k, r in enumerate(routes) if r]
        routes = [routes[k] for k in keep]
        assign = [ev.fits(routes[i], sol.assign[k]) for i, k in enumerate(keep)]
        if any(a is None for a in assign):
            return None
        boxes_used = sum(a.boxes for a in assign)
        fresh, frozen = inst.fresh, inst.frozen

        def option(c, r, dep):
            w1, w2 = ev.load(routes[r])
            cand = ev.fit_loads(assign[r].unit, assign[r].config, w1 + fresh[c], w2 + frozen[c])
            if cand is None or boxes_used - assign[r].boxes + cand.boxes > inst.total_frigo:
                return None
            ins = best_insertion(inst, routes[r], c, dep)
            return None if ins is None else (ins[0], ins[1], cand)

        deps = [start_times(inst, r) for r in routes]
        table = {c: [option(c, r, deps[r]) for r in range(len(routes))] for c in removed}
        pending = set(removed)
        while pending:
            pick: Optional[tuple] = None
            for c in pending:
                opts = sorted((o[0], r) for r, o in enumerate(table[c]) if o is not None)
                if not opts:
                    return None
                best_cost, r = opts[0]
                if repair == "greedy":
                    key = -best_cost
                else:
                    second = opts[1][0] if len(opts) > 1 else 1e12
                    key = (second - best_cost) - 1e-6 * best_cost
                if pick is None or key > pick[0]:
                    o = table[c][r]
                    pick = (key, c, r, o[1], o[2])
            _, c, r, pos, cand = pick
            routes[r].insert(pos, c)
            boxes_used += cand.boxes - assign[r].boxes
            assign[r] = cand
            pending.discard(c)
            deps[r] = start_times(inst, routes[r])
            for c2 in pending:
                table[c2][r] = option(c2, r, deps[r])
                for r2, o in enumerate(table[c2]):
                    if o is not None and boxes_used - assign[r2].boxes + o[2].boxes > inst.total_frigo:
                        table[c2][r2] = None
        return routes, assign


@contextlib.contextmanager
def operator_shaking(tries: int = 3):
    """Run :func:`mvrdc.vns.solve` with OperatorShaker in place of N1-N4."""
    old = (_vns.Shaker, _vns.NAMES)
    _vns.Shaker = lambda ev, _tries: OperatorShaker(ev, tries)
    _vns.NAMES = OP_NAMES
    try:
        yield
    finally:
        _vns.Shaker, _vns.NAMES = old
