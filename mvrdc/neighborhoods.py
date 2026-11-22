"""Shaking neighborhoods N1-N4 (Masmoudi et al. 2026, Sec. 4.3; neighborhoods.cpp).

Every neighborhood picks one of its moves uniformly at random and applies it
at random positions. As in the authors' code, a move is kept only if every
modified route is time-feasible and its load still fits the vehicle it is
currently assigned to; vehicles are then re-assigned by SelectBestVehicle in
the main loop (Algorithm 1, line 9). Inter-route moves are restricted to
routes served by the same vehicle type (fresh, frozen, mixed), as in the paper.

If a random application is infeasible we retry up to ``tries`` times (the
authors' code tries once and otherwise returns the unchanged solution, which
then wastes a full local-search call). See docs/DEVIATIONS.md.

N1 Swap        : intra swap | inter swap | adjacent-pair inter swap |
                 non-adjacent exchange with random re-insertion
N2 Relocate    : one customer inter | two adjacent customers inter |
                 one customer intra | two adjacent customers intra
N3 Segment     : shuffle a random segment of d in [0.3 n_r, 0.7 n_r] customers
N4 Remove-insert: remove lambda random customers, re-insert with the basic
                 greedy heuristic of Ropke & Pisinger (2006); lambda in
                 [max(5, 0.1n), min(100, 0.3n)], reset to lambda_min after a
                 new best solution and incremented otherwise.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from .fleet import Candidate
from .insertion import best_insertion, start_times
from .solution import Evaluator, Solution

Routes = List[List[int]]
NAMES = ("N1 swap", "N2 relocate", "N3 segment", "N4 remove-insert")


class Shaker:
    def __init__(self, ev: Evaluator, tries: int = 10):
        self.ev = ev
        self.tries = tries
        n = ev.inst.n
        self.lambda_min = max(5, int(0.1 * n))
        self.lambda_max = max(self.lambda_min, min(100, int(0.3 * n)))
        self.lam = self.lambda_min
        self._moves = (
            (self._intra_swap, self._inter_swap, self._adjacent_swap, self._nonadjacent_swap),
            (self._relocate, self._relocate_two, self._intra_relocate, self._intra_relocate_two),
            (self._segment_shift,),
            (self._remove_insert,),
        )

    # ---------------------------------------------------------------- public
    def shake(self, sol: Solution, h: int, rng: random.Random
              ) -> Optional[Tuple[Routes, List[Candidate]]]:
        """Apply neighborhood ``h`` (0-based). None if no feasible move found."""
        moves = self._moves[h]
        for _ in range(self.tries):
            move = moves[rng.randrange(len(moves))] if len(moves) > 1 else moves[0]
            out = move(sol, rng)
            if out is not None:
                return out
        return None

    def on_new_best(self) -> None:
        self.lam = self.lambda_min

    def on_no_new_best(self) -> None:
        self.lam = min(self.lam + 1, self.lambda_max)

    # --------------------------------------------------------------- helpers
    def _kind(self, sol: Solution, r: int) -> str:
        return self.ev.inst.units[sol.assign[r].unit].kind

    def _pick(self, sol: Solution, rng: random.Random, min_len: int) -> Optional[int]:
        idx = [r for r, route in enumerate(sol.routes) if len(route) >= min_len]
        return rng.choice(idx) if idx else None

    def _pick_pair(self, sol: Solution, rng: random.Random, len1: int, len2: int
                   ) -> Optional[Tuple[int, int]]:
        r1 = self._pick(sol, rng, len1)
        if r1 is None:
            return None
        k = self._kind(sol, r1)
        others = [r for r, route in enumerate(sol.routes)
                  if r != r1 and len(route) >= len2 and self._kind(sol, r) == k]
        if not others:
            return None
        return r1, rng.choice(others)

    def _commit(self, sol: Solution, changes: Dict[int, List[int]]
                ) -> Optional[Tuple[Routes, List[Candidate]]]:
        ev = self.ev
        assign = list(sol.assign)
        for r, route in changes.items():
            if route:
                if not ev.time_feasible(route):
                    return None
                fit = ev.fits(route, assign[r])
                if fit is None:
                    return None
                assign[r] = fit
        routes = [changes.get(r, route) for r, route in enumerate(sol.routes)]
        return routes, assign

    # -------------------------------------------------------------- N1 swap
    def _intra_swap(self, sol, rng):
        r = self._pick(sol, rng, 2)
        if r is None:
            return None
        route = sol.routes[r][:]
        i, j = rng.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
        return self._commit(sol, {r: route})

    def _inter_swap(self, sol, rng):
        p = self._pick_pair(sol, rng, 1, 1)
        if p is None:
            return None
        r1, r2 = p
        a, b = sol.routes[r1][:], sol.routes[r2][:]
        i, j = rng.randrange(len(a)), rng.randrange(len(b))
        a[i], b[j] = b[j], a[i]
        return self._commit(sol, {r1: a, r2: b})

    def _adjacent_swap(self, sol, rng):
        p = self._pick_pair(sol, rng, 2, 2)
        if p is None:
            return None
        r1, r2 = p
        a, b = sol.routes[r1][:], sol.routes[r2][:]
        i, j = rng.randrange(len(a) - 1), rng.randrange(len(b) - 1)
        a[i:i + 2], b[j:j + 2] = b[j:j + 2], a[i:i + 2]
        return self._commit(sol, {r1: a, r2: b})

    def _nonadjacent_swap(self, sol, rng):
        p = self._pick_pair(sol, rng, 1, 1)
        if p is None:
            return None
        r1, r2 = p
        a, b = sol.routes[r1][:], sol.routes[r2][:]
        c1 = a.pop(rng.randrange(len(a)))
        c2 = b.pop(rng.randrange(len(b)))
        b.insert(rng.randrange(len(b) + 1), c1)
        a.insert(rng.randrange(len(a) + 1), c2)
        return self._commit(sol, {r1: a, r2: b})

    # ---------------------------------------------------------- N2 relocate
    def _relocate(self, sol, rng):
        p = self._pick_pair(sol, rng, 1, 1)
        if p is None:
            return None
        r1, r2 = p
        a, b = sol.routes[r1][:], sol.routes[r2][:]
        c = a.pop(rng.randrange(len(a)))
        b.insert(rng.randrange(len(b) + 1), c)
        return self._commit(sol, {r1: a, r2: b})

    def _relocate_two(self, sol, rng):
        p = self._pick_pair(sol, rng, 2, 1)
        if p is None:
            return None
        r1, r2 = p
        a, b = sol.routes[r1][:], sol.routes[r2][:]
        i = rng.randrange(len(a) - 1)
        seg = a[i:i + 2]
        del a[i:i + 2]
        if rng.random() < 0.5:
            seg.reverse()
        j = rng.randrange(len(b) + 1)
        b[j:j] = seg
        return self._commit(sol, {r1: a, r2: b})

    def _intra_relocate(self, sol, rng):
        r = self._pick(sol, rng, 2)
        if r is None:
            return None
        route = sol.routes[r][:]
        i, j = rng.sample(range(len(route)), 2)
        c = route.pop(i)
        route.insert(j, c)
        return self._commit(sol, {r: route})

    def _intra_relocate_two(self, sol, rng):
        r = self._pick(sol, rng, 3)
        if r is None:
            return None
        route = sol.routes[r][:]
        i = rng.randrange(len(route) - 1)
        seg = route[i:i + 2]
        del route[i:i + 2]
        j = rng.randrange(len(route) + 1)
        route[j:j] = seg
        return self._commit(sol, {r: route})

    # ------------------------------------------------------- N3 segment shift
    def _segment_shift(self, sol, rng):
        r = self._pick(sol, rng, 2)
        if r is None:
            return None
        route = sol.routes[r][:]
        n = len(route)
        d_min = max(1, int(0.3 * n))
        d_max = max(d_min, min(n, int(0.7 * n)))
        d = rng.randint(d_min, d_max)
        pos = rng.randint(0, n - d)
        seg = route[pos:pos + d]
        rng.shuffle(seg)
        route[pos:pos + d] = seg
        return self._commit(sol, {r: route})

    # ------------------------------------------------------ N4 remove-insert
    def _remove_insert(self, sol, rng):
        ev, inst = self.ev, self.ev.inst
        routes = [r[:] for r in sol.routes]
        assign = list(sol.assign)
        removed: List[int] = []
        for _ in range(self.lam):
            r = rng.randrange(len(routes))
            if len(routes[r]) <= 1:  # authors' code: never empty a route here
                continue
            removed.append(routes[r].pop(rng.randrange(len(routes[r]))))
        if not removed:
            return None
        for r in range(len(routes)):
            if not ev.time_feasible(routes[r]):  # removal never breaks TW, but be safe
                return None
            assign[r] = ev.fits(routes[r], assign[r])
        boxes_used = sum(a.boxes for a in assign)
        fresh, frozen = inst.fresh, inst.frozen

        def option(c: int, r: int, departs) -> Optional[Tuple[float, int, Candidate]]:
            w1, w2 = ev.load(routes[r])
            cand = ev.fit_loads(assign[r].unit, assign[r].config, w1 + fresh[c], w2 + frozen[c])
            if cand is None or boxes_used - assign[r].boxes + cand.boxes > inst.total_frigo:
                return None
            ins = best_insertion(inst, routes[r], c, departs)
            if ins is None:
                return None
            return ins[0], ins[1], cand

        departs = [start_times(inst, rt) for rt in routes]
        table = {c: [option(c, r, departs[r]) for r in range(len(routes))] for c in removed}
        pending = set(removed)
        while pending:
            best = None
            for c in pending:
                for r, opt in enumerate(table[c]):
                    if opt is not None and (best is None or opt[0] < best[0]):
                        best = (opt[0], c, r, opt[1], opt[2])
            if best is None:
                return None
            _, c, r, pos, cand = best
            routes[r].insert(pos, c)
            boxes_used += cand.boxes - assign[r].boxes
            assign[r] = cand
            pending.discard(c)
            departs[r] = start_times(inst, routes[r])
            for c2 in pending:
                table[c2][r] = option(c2, r, departs[r])
            # box stock changed: options on other routes may have become invalid
            for c2 in pending:
                for r2, opt in enumerate(table[c2]):
                    if opt is not None and (
                            boxes_used - assign[r2].boxes + opt[2].boxes > inst.total_frigo):
                        table[c2][r2] = None
        return routes, assign
