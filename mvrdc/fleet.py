"""Route-to-vehicle assignment (SelectBestVehicle), Masmoudi et al. (2026), Sec. 4.5.

Phase 1 (Algorithm 3, ``PerRouteCandidateSet``) builds, for every route, the
vehicles that can carry its fresh/frozen load ``(w1, w2)``:

  * fresh unit k : O = ceil(w2 / box_cap) boxes, residual = Q_k - (w1 + O*w)
  * frozen unit k: O = ceil(w1 / box_cap) boxes, residual = Q_k - (w2 + O*w)
    (feasible if O <= F_k and residual >= 0)
  * mixed unit k : best configuration j with w1 <= Q1_kj, w2 <= Q2_kj,
    residual = (Q1_kj - w1) + (Q2_kj - w2)

Phase 2 (Algorithm 4, ``AssignmentRouteToVehicle``) assigns each route to a
distinct physical unit:

  (i)  routes that need boxes are served first, in ascending order of the
       fewest boxes they can be served with, each by its minimum-residual
       boxed candidate, while depot boxes remain;
  (ii) the remaining routes are taken in random order and given their
       minimum-residual box-free candidate (mixed units, or single-temperature
       units whose load needs no box).

Note: Algorithm 3 in the paper evaluates mixed units only when no fresh/frozen
unit fits. The authors' code evaluates them always (comment: "Removed the
restrictive condition"), and the text of Algorithm 4 ("attempts to assign the
remaining routes using candidates with a mixed-reconfigurable vehicle") only
makes sense if mixed candidates exist for box-needing routes. We follow the
code. See docs/DEVIATIONS.md.
"""
from __future__ import annotations

import random
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from .instance import FRESH, FROZEN, MIXED, Instance


class Candidate(NamedTuple):
    unit: int        # index into Instance.units
    config: int      # configuration index (0 for single-temperature units)
    boxes: int       # O(c), frigoboxes needed
    residual: int    # rho(c), unused space


class FleetAssigner:
    """Caches candidate sets per demand pair and performs Algorithm 4."""

    def __init__(self, inst: Instance):
        self.inst = inst
        self._cache: Dict[Tuple[int, int], Tuple[Candidate, ...]] = {}
        self.fresh_idx = [k for k, u in enumerate(inst.units) if u.kind == FRESH]
        self.frozen_idx = [k for k, u in enumerate(inst.units) if u.kind == FROZEN]
        self.mixed_idx = [k for k, u in enumerate(inst.units) if u.kind == MIXED]

    # ---------------------------------------------------------------- Alg. 3
    def candidates(self, w1: int, w2: int) -> Tuple[Candidate, ...]:
        key = (w1, w2)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        inst = self.inst
        bc, bs = inst.box_cap, inst.box_space
        out: List[Candidate] = []
        o_fresh = -(-w2 // bc)  # ceil
        for k in self.fresh_idx:
            u = inst.units[k]
            rho = u.configs[0][1] - (w1 + o_fresh * bs)
            if o_fresh <= u.max_frigo and rho >= 0:
                out.append(Candidate(k, 0, o_fresh, rho))
        o_frozen = -(-w1 // bc)
        for k in self.frozen_idx:
            u = inst.units[k]
            rho = u.configs[0][2] - (w2 + o_frozen * bs)
            if o_frozen <= u.max_frigo and rho >= 0:
                out.append(Candidate(k, 0, o_frozen, rho))
        for k in self.mixed_idx:
            best: Optional[Candidate] = None
            for j, (_, q1, q2) in enumerate(inst.units[k].configs):
                if w1 <= q1 and w2 <= q2:
                    rho = (q1 - w1) + (q2 - w2)
                    if best is None or rho < best.residual:
                        best = Candidate(k, j, 0, rho)
            if best is not None:
                out.append(best)
        res = tuple(out)
        self._cache[key] = res
        return res

    def loadable(self, w1: int, w2: int) -> bool:
        """True if at least one vehicle type can carry the load (ignoring the fleet)."""
        return bool(self.candidates(w1, w2))

    # ---------------------------------------------------------------- Alg. 4
    def assign(
        self, loads: Sequence[Tuple[int, int]], rng: random.Random
    ) -> Optional[List[Candidate]]:
        """Assign one distinct unit to every route; None if impossible."""
        cand_sets = [self.candidates(w1, w2) for (w1, w2) in loads]
        if any(not c for c in cand_sets):
            return None
        n_routes = len(loads)
        result: List[Optional[Candidate]] = [None] * n_routes
        used = set()
        boxes_left = self.inst.total_frigo
        unserved = set(range(n_routes))

        # Phase (i): routes with box-using candidates, fewest boxes first.
        while unserved:
            best_o = None
            tied: List[int] = []
            for r in unserved:
                m = None
                for c in cand_sets[r]:
                    if c.boxes > 0 and c.unit not in used and c.boxes <= boxes_left:
                        if m is None or c.boxes < m:
                            m = c.boxes
                if m is None:
                    continue
                if best_o is None or m < best_o:
                    best_o, tied = m, [r]
                elif m == best_o:
                    tied.append(r)
            if not tied:
                break
            r = rng.choice(tied) if len(tied) > 1 else tied[0]
            feas = [c for c in cand_sets[r]
                    if c.boxes > 0 and c.unit not in used and c.boxes <= boxes_left]
            rho = min(c.residual for c in feas)
            best = [c for c in feas if c.residual == rho]
            c = rng.choice(best) if len(best) > 1 else best[0]
            result[r] = c
            used.add(c.unit)
            boxes_left -= c.boxes
            unserved.discard(r)

        # Phase (ii): box-free candidates, routes in random order.
        while unserved:
            ready = [r for r in unserved
                     if any(c.boxes == 0 and c.unit not in used for c in cand_sets[r])]
            if not ready:
                break
            r = rng.choice(ready) if len(ready) > 1 else ready[0]
            feas = [c for c in cand_sets[r] if c.boxes == 0 and c.unit not in used]
            rho = min(c.residual for c in feas)
            best = [c for c in feas if c.residual == rho]
            c = rng.choice(best) if len(best) > 1 else best[0]
            result[r] = c
            used.add(c.unit)
            unserved.discard(r)

        if unserved:
            return None
        return result  # type: ignore[return-value]
