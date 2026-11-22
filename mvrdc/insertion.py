"""Time-window-aware cheapest insertion (used by construction, N4 and local search)."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .instance import Instance


def start_times(inst: Instance, route: Sequence[int]) -> List[float]:
    """Departure time from each node of the route (index -1 = depot at time 0)."""
    d, ready, serv = inst.dist, inst.ready, inst.service
    out: List[float] = []
    t, prev = 0.0, 0
    for c in route:
        t += d[prev][c]
        if t < ready[c]:
            t = ready[c]
        t += serv[c]
        out.append(t)
        prev = c
    return out


def best_insertion(inst: Instance, route: Sequence[int], c: int,
                   departs: Optional[List[float]] = None) -> Optional[Tuple[float, int]]:
    """Cheapest time-feasible position for customer ``c`` in ``route``.

    Returns ``(distance increase, position)`` or None. Capacity is NOT checked
    here; callers check loads (they are position independent).
    """
    d, ready, due, serv = inst.dist, inst.ready, inst.due, inst.service
    if departs is None:
        departs = start_times(inst, route)
    n = len(route)
    best: Optional[Tuple[float, int]] = None
    for i in range(n + 1):
        prev = route[i - 1] if i > 0 else 0
        nxt = route[i] if i < n else 0
        delta = d[prev][c] + d[c][nxt] - d[prev][nxt]
        if best is not None and delta >= best[0]:
            continue
        t = (departs[i - 1] if i > 0 else 0.0) + d[prev][c]
        if t > due[c] + 1e-9:
            continue
        if t < ready[c]:
            t = ready[c]
        t += serv[c]
        # propagate the shift through the rest of the route
        ok = True
        p = c
        for j in range(i, n):
            q = route[j]
            t += d[p][q]
            if t > due[q] + 1e-9:
                ok = False
                break
            if t < ready[q]:
                t = ready[q]
            t += serv[q]
            # once we are no later than the original schedule nothing changes
            if t <= departs[j] + 1e-9:
                t = None  # type: ignore[assignment]
                break
            p = q
        if not ok:
            continue
        if t is not None:
            # the delay propagated to the end of the route: check T_max
            last = route[n - 1] if i < n else c
            if t + d[last][0] > inst.t_max + 1e-9:
                continue
        best = (delta, i)
    return best
