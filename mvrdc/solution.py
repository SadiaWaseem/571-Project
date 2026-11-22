"""Route evaluation, solutions and the MVRDC objective.

Objective (Masmoudi et al. 2026, Eq. (A.1), Sec. 5.1; identical to Eq. (1) of
our Assignment 2):

    f(x) = gamma1 * (#vehicles) + gamma2 * (routing cost) + theta * (unused space)

with gamma1 = 1000, gamma2 = 1 and theta = 1 / sum_i (q1_i + q2_i), which
enforces the lexicographic order vehicles > distance > unused space.

Unused space of a route is the residual rho(c) of the candidate the route was
assigned to (Algorithm 3): for single-temperature vehicles the boxes count as
occupied space, ``Q_k - (w_native + O * w)``, as in the MILP of Assignment 2.
The authors' C++ instead uses ``Q_k - (w1 + w2)`` (box rounding not counted).
Both are available through ``unused_definition``. The default is ``"code"``
because it reproduces the objective values the authors report: on the 74
published solutions that match a published instance, our recomputed cost
differs from theirs by <= 0.05 on 73 and by 1.0 on one (tests/test_published.py).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from .fleet import Candidate, FleetAssigner
from .instance import FRESH, FROZEN, Instance

GAMMA_VEHICLE = 1000.0
GAMMA_DISTANCE = 1.0


class Evaluator:
    """Per-instance evaluation services with memoisation."""

    def __init__(self, inst: Instance, unused_definition: str = "code",
                 cache_limit: int = 400_000):
        if unused_definition not in ("paper", "code"):
            raise ValueError("unused_definition must be 'paper' or 'code'")
        self.inst = inst
        self.fleet = FleetAssigner(inst)
        self.unused_definition = unused_definition
        self._route_cache: Dict[Tuple[int, ...], Tuple[float, bool]] = {}
        self._cache_limit = cache_limit
        self.route_evals = 0  # counts uncached evaluations (effort measure)

    # ----------------------------------------------------------------- routes
    def route_info(self, route: Sequence[int]) -> Tuple[float, bool]:
        """(distance, time-feasible) of a depot-to-depot route."""
        key = tuple(route)
        hit = self._route_cache.get(key)
        if hit is not None:
            return hit
        self.route_evals += 1
        inst = self.inst
        d = inst.dist
        ready, due, serv = inst.ready, inst.due, inst.service
        t = 0.0
        length = 0.0
        prev = 0
        ok = True
        for c in key:
            leg = d[prev][c]
            length += leg
            t += leg
            if t > due[c] + 1e-9:
                ok = False
                break
            if t < ready[c]:
                t = ready[c]
            t += serv[c]
            prev = c
        if ok:
            back = d[prev][0]
            length += back
            if t + back > inst.t_max + 1e-9:
                ok = False
        else:
            length = self.route_length(key)
        if len(self._route_cache) > self._cache_limit:
            self._route_cache.clear()
        res = (length, ok)
        self._route_cache[key] = res
        return res

    def route_length(self, route: Sequence[int]) -> float:
        d = self.inst.dist
        prev, s = 0, 0.0
        for c in route:
            s += d[prev][c]
            prev = c
        return s + d[prev][0]

    def time_feasible(self, route: Sequence[int]) -> bool:
        return self.route_info(route)[1]

    def load(self, route: Sequence[int]) -> Tuple[int, int]:
        f, z = self.inst.fresh, self.inst.frozen
        return (sum(f[c] for c in route), sum(z[c] for c in route))

    def route_ok(self, route: Sequence[int]) -> bool:
        """Time-feasible and loadable on at least one vehicle type."""
        if not route:
            return True
        if not self.route_info(route)[1]:
            return False
        w1, w2 = self.load(route)
        return self.fleet.loadable(w1, w2)

    def fit_loads(self, unit: int, config: int, w1: int, w2: int) -> Optional[Candidate]:
        """Candidate for loads (w1, w2) on a given unit/configuration, or None."""
        inst = self.inst
        u = inst.units[unit]
        if u.kind == FRESH or u.kind == FROZEN:
            native, foreign = (w1, w2) if u.kind == FRESH else (w2, w1)
            o = -(-foreign // inst.box_cap)
            rho = u.capacity - (native + o * inst.box_space)
            if o > u.max_frigo or rho < 0:
                return None
            return Candidate(unit, 0, o, rho)
        _, q1, q2 = u.configs[config]
        if w1 <= q1 and w2 <= q2:
            return Candidate(unit, config, 0, (q1 - w1) + (q2 - w2))
        return None

    def fits(self, route: Sequence[int], cand: Candidate) -> Optional[Candidate]:
        """Re-check a route against a fixed vehicle (boxes/residual recomputed)."""
        w1, w2 = self.load(route)
        return self.fit_loads(cand.unit, cand.config, w1, w2)

    def unused(self, route: Sequence[int], cand: Candidate) -> float:
        if self.unused_definition == "paper":
            return cand.residual
        u = self.inst.units[cand.unit]
        w1, w2 = self.load(route)
        if u.kind in (FRESH, FROZEN):
            return max(0, u.capacity - (w1 + w2))
        return cand.residual

    # --------------------------------------------------------------- objective
    def route_cost(self, route: Sequence[int], cand: Candidate) -> float:
        """Contribution of one non-empty route to f(x)."""
        return (GAMMA_VEHICLE + GAMMA_DISTANCE * self.route_info(route)[0]
                + self.inst.theta * self.unused(route, cand))

    def objective(self, routes: Sequence[Sequence[int]],
                  assign: Sequence[Candidate]) -> float:
        dist = sum(self.route_info(r)[0] for r in routes)
        unused = sum(self.unused(r, a) for r, a in zip(routes, assign))
        return GAMMA_VEHICLE * len(routes) + GAMMA_DISTANCE * dist + self.inst.theta * unused


class Solution:
    """A set of non-empty routes, each assigned to a distinct vehicle unit."""

    __slots__ = ("routes", "assign", "cost")

    def __init__(self, routes: List[List[int]], assign: List[Candidate], cost: float):
        self.routes = routes
        self.assign = assign
        self.cost = cost

    def copy(self) -> "Solution":
        return Solution([r[:] for r in self.routes], list(self.assign), self.cost)

    @property
    def n_vehicles(self) -> int:
        return len(self.routes)

    def boxes_used(self) -> int:
        return sum(a.boxes for a in self.assign)

    def distance(self, ev: Evaluator) -> float:
        return sum(ev.route_info(r)[0] for r in self.routes)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Solution(vehicles={self.n_vehicles}, cost={self.cost:.2f})"


def assign_and_evaluate(ev: Evaluator, routes: List[List[int]],
                        rng: random.Random) -> Optional[Solution]:
    """Drop empty routes, run SelectBestVehicle and evaluate. None if infeasible."""
    routes = [r for r in routes if r]
    loads = [ev.load(r) for r in routes]
    assign = ev.fleet.assign(loads, rng)
    if assign is None:
        return None
    return Solution(routes, assign, ev.objective(routes, assign))


def check_solution(ev: Evaluator, sol: Solution) -> List[str]:
    """Independent feasibility audit. Returns a list of violations (empty = OK)."""
    inst = ev.inst
    errs: List[str] = []
    seen: List[int] = [c for r in sol.routes for c in r]
    if sorted(seen) != list(inst.customers):
        missing = set(inst.customers) - set(seen)
        dup = {c for c in seen if seen.count(c) > 1}
        errs.append(f"coverage: missing={sorted(missing)[:10]} duplicated={sorted(dup)[:10]}")
    units = [a.unit for a in sol.assign]
    if len(set(units)) != len(units):
        errs.append("a physical vehicle is used by more than one route")
    if len(sol.routes) != len(sol.assign):
        errs.append("routes and assignments differ in length")
    boxes = 0
    for k, (r, a) in enumerate(zip(sol.routes, sol.assign)):
        if not ev.route_info(r)[1]:
            errs.append(f"route {k}: time windows / T_max violated")
        chk = ev.fits(r, a)
        if chk is None:
            errs.append(f"route {k}: load does not fit unit {inst.units[a.unit].uid}")
        else:
            boxes += chk.boxes
    if boxes > inst.total_frigo:
        errs.append(f"frigoboxes {boxes} > F={inst.total_frigo}")
    cost = ev.objective(sol.routes, sol.assign)
    if abs(cost - sol.cost) > 1e-6 * max(1.0, abs(cost)):
        errs.append(f"stored cost {sol.cost:.4f} != recomputed {cost:.4f}")
    return errs
