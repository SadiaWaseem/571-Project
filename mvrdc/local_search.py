"""Local search (Masmoudi et al. 2026, Sec. 4.4).

Four operators, applied with the vehicle assignment fixed (the paper: "the
local search further adjusts routes but keeps the previously assigned vehicle
types fixed"):

  * 2-opt (Lin, 1965)                : reverse a segment inside one route
  * relocate (Savelsbergh, 1992)     : move one customer to its cheapest
                                       feasible position in another route
  * 2-opt* (Potvin & Rousseau, 1995) : exchange the tails of two routes of
                                       the same vehicle type
  * remove-two-insert-one (Xiang et al., 2006): remove two customers of a
                                       route and re-insert each at its cheapest
                                       position in routes of the same type

As in the authors' code, one operator is drawn at random per call and applied
``moves`` times (50) at random positions. Unlike the authors' code, which keeps
every *feasible* random move whatever its cost, we keep a move only if it
lowers f(x): the paper describes this phase as intensification that
"improves x' to obtain x''". See docs/DEVIATIONS.md.
"""
from __future__ import annotations

import random
from typing import List, Optional

from .insertion import best_insertion
from .solution import Evaluator, Solution

EPS = 1e-9
LS_NAMES = ("2-opt", "relocate", "2-opt*", "remove2-insert1")


class LocalSearch:
    def __init__(self, ev: Evaluator, moves: int = 50):
        self.ev = ev
        self.moves = moves
        self.ops = (self._two_opt, self._relocate, self._two_opt_star, self._remove_two)

    def run(self, sol: Solution, rng: random.Random) -> Solution:
        sol = sol.copy()
        op = self.ops[rng.randrange(len(self.ops))]
        for _ in range(self.moves):
            if not sol.routes:
                break
            op(sol, rng)
        sol.routes, sol.assign = (
            [r for r in sol.routes if r],
            [a for r, a in zip(sol.routes, sol.assign) if r],
        )
        sol.cost = self.ev.objective(sol.routes, sol.assign)
        return sol

    # ------------------------------------------------------------- helpers
    def _cost(self, route: List[int], cand) -> float:
        return 0.0 if not route else self.ev.route_cost(route, cand)

    def _try(self, sol: Solution, changes: dict) -> bool:
        """Apply ``changes`` {route index: new route} if feasible and improving."""
        ev = self.ev
        new_assign = {}
        old = new = 0.0
        boxes = sol.boxes_used()
        for r, route in changes.items():
            a = sol.assign[r]
            old += self._cost(sol.routes[r], a)
            if route:
                if not ev.time_feasible(route):
                    return False
                fit = ev.fits(route, a)
                if fit is None:
                    return False
                new_assign[r] = fit
                new += self._cost(route, fit)
                boxes += fit.boxes - a.boxes
            else:
                new_assign[r] = a._replace(boxes=0)
                boxes -= a.boxes
        if boxes > ev.inst.total_frigo or new >= old - EPS:
            return False
        for r, route in changes.items():
            sol.routes[r] = route
            sol.assign[r] = new_assign[r]
        return True

    def _kind(self, sol: Solution, r: int) -> str:
        return self.ev.inst.units[sol.assign[r].unit].kind

    # ------------------------------------------------------------ operators
    def _two_opt(self, sol: Solution, rng: random.Random) -> None:
        idx = [r for r, rt in enumerate(sol.routes) if len(rt) >= 3]
        if not idx:
            return
        r = rng.choice(idx)
        route = sol.routes[r]
        i, j = sorted(rng.sample(range(len(route)), 2))
        new = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
        self._try(sol, {r: new})

    def _relocate(self, sol: Solution, rng: random.Random) -> None:
        nonempty = [r for r, rt in enumerate(sol.routes) if rt]
        if len(nonempty) < 2:
            return
        r1, r2 = rng.sample(nonempty, 2)
        a = sol.routes[r1][:]
        c = a.pop(rng.randrange(len(a)))
        ins = best_insertion(self.ev.inst, sol.routes[r2], c)
        if ins is None:
            return
        b = sol.routes[r2][:]
        b.insert(ins[1], c)
        self._try(sol, {r1: a, r2: b})

    def _two_opt_star(self, sol: Solution, rng: random.Random) -> None:
        nonempty = [r for r, rt in enumerate(sol.routes) if len(rt) >= 1]
        if len(nonempty) < 2:
            return
        r1 = rng.choice(nonempty)
        same = [r for r in nonempty if r != r1 and self._kind(sol, r) == self._kind(sol, r1)]
        if not same:
            return
        r2 = rng.choice(same)
        a, b = sol.routes[r1], sol.routes[r2]
        i, j = rng.randrange(len(a) + 1), rng.randrange(len(b) + 1)
        self._try(sol, {r1: a[:i] + b[j:], r2: b[:j] + a[i:]})

    def _remove_two(self, sol: Solution, rng: random.Random) -> None:
        idx = [r for r, rt in enumerate(sol.routes) if len(rt) >= 2]
        if not idx:
            return
        src = rng.choice(idx)
        kind = self._kind(sol, src)
        work = {r: rt[:] for r, rt in enumerate(sol.routes)
                if rt and self._kind(sol, r) == kind}
        i, j = sorted(rng.sample(range(len(work[src])), 2))
        c2 = work[src].pop(j)
        c1 = work[src].pop(i)
        ev = self.ev
        for c in (c1, c2):
            best: Optional[tuple] = None
            for r, rt in work.items():
                ins = best_insertion(ev.inst, rt, c)
                if ins is None:
                    continue
                w1, w2 = ev.load(rt)
                a = sol.assign[r]
                if ev.fit_loads(a.unit, a.config, w1 + ev.inst.fresh[c],
                                w2 + ev.inst.frozen[c]) is None:
                    continue
                if best is None or ins[0] < best[0]:
                    best = (ins[0], r, ins[1])
            if best is None:
                return
            work[best[1]].insert(best[2], c)
        changed = {r: rt for r, rt in work.items() if rt != sol.routes[r]}
        if changed:
            self._try(sol, changed)
