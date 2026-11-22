"""Estimated CO2 emissions of a solution (reported next to routing cost).

As stated in our Assignment 3, emissions are *reported*, not optimised: in the
lexicographic objective they would rank after vehicles, distance and unused
space, so they could never change the selected solution.

Model (load-dependent, after the linear fuel model used in pollution-routing
work, e.g. Bektas & Laporte, 2011):

    E(route) = sum over arcs (i,j) of  d_ij * (e_empty + (e_full - e_empty) * L_ij / Q_k)
             + e_ref * (route duration) * [route carries frozen goods or frigoboxes]

where L_ij is the load on board when travelling (i,j) and Q_k the capacity of
the assigned vehicle (configuration). Instance distances are abstract units,
so the factors are per distance / time unit.

IMPORTANT: the default factors below are placeholders that only fix the
*shape* of the model. Before reporting absolute emission figures in the
paper, replace them with values from a cited source and state the units.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .insertion import start_times
from .instance import FRESH, FROZEN
from .solution import Evaluator, Solution


@dataclass
class EmissionFactors:
    e_empty: float = 1.0    # per distance unit, empty vehicle   (placeholder)
    e_full: float = 1.3     # per distance unit, fully loaded    (placeholder)
    e_ref: float = 0.0      # per time unit of refrigeration     (placeholder, off)


def solution_emissions(ev: Evaluator, sol: Solution,
                       f: EmissionFactors = EmissionFactors()) -> Dict[str, float]:
    inst = ev.inst
    d = inst.dist
    travel = refrig = 0.0
    for route, cand in zip(sol.routes, sol.assign):
        u = inst.units[cand.unit]
        _, q1, q2 = u.configs[cand.config]
        cap = u.capacity if u.kind in (FRESH, FROZEN) else q1 + q2
        load = sum(inst.fresh[c] + inst.frozen[c] for c in route)
        prev = 0
        for c in route + [0]:
            travel += d[prev][c] * (f.e_empty + (f.e_full - f.e_empty) * load / cap)
            if c:
                load -= inst.fresh[c] + inst.frozen[c]
            prev = c
        carries_cold = any(inst.frozen[c] for c in route) or cand.boxes > 0
        if f.e_ref and carries_cold:
            dep = start_times(inst, route)
            refrig += f.e_ref * (dep[-1] + d[route[-1]][0])
    return {"emissions": travel + refrig, "emissions_travel": travel,
            "emissions_refrigeration": refrig}
