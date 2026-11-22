"""Read the route solutions published with Masmoudi et al. (2026).

Used to validate our objective and feasibility code against the authors'
reported values (Detailed_route_solutions/ in their repository).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .fleet import Candidate
from .instance import FRESH, FROZEN, Instance
from .solution import Evaluator, Solution

_VEH = re.compile(r"^Vehicle\s+(\S+)\s+\((\w+)\)")
_ROUTE = re.compile(r"^Route:\s*(.*)$")
_BEST = re.compile(r"Best Cost:\s*([-\d.eE+]+)")


def read_published(path: str) -> Tuple[List[Tuple[str, List[int]]], Optional[float]]:
    vehicles: List[Tuple[str, List[int]]] = []
    best = None
    current = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            ln = raw.strip()
            m = _VEH.match(ln)
            if m:
                current = m.group(1)
                continue
            m = _ROUTE.match(ln)
            if m and current is not None:
                nodes = [int(t) for t in re.findall(r"\d+", m.group(1))]
                vehicles.append((current, [c for c in nodes if c != 0]))
                current = None
                continue
            m = _BEST.search(ln)
            if m:
                best = float(m.group(1))
    return vehicles, best


def to_solution(ev: Evaluator, vehicles: List[Tuple[str, List[int]]]) -> Solution:
    inst: Instance = ev.inst
    routes, assign = [], []
    for vid, labels in vehicles:
        if not labels:
            continue
        route = [inst.index_of(c) for c in labels]
        unit = config = None
        for k, u in enumerate(inst.units):
            for j, (cid, _, _) in enumerate(u.configs):
                if cid == vid:
                    unit, config = k, j
        if unit is None:
            raise KeyError(f"vehicle {vid} not in instance {inst.name}")
        cand = ev.fits(route, Candidate(unit, config, 0, 0))
        if cand is None:  # keep it so that check_solution reports the violation
            u = inst.units[unit]
            w1, w2 = ev.load(route)
            o = 0
            if u.kind == FRESH:
                o = -(-w2 // inst.box_cap)
            elif u.kind == FROZEN:
                o = -(-w1 // inst.box_cap)
            cand = Candidate(unit, config, o, -1)
        routes.append(route)
        assign.append(cand)
    return Solution(routes, assign, ev.objective(routes, assign))
