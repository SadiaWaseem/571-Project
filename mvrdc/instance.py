"""MVRDC instance model and parser.

Reads the plain-text instance format published by Masmoudi et al. (2026),
https://github.com/Masmoudi-Amine/MVRDC- (see Instances/README.md there).

Conventions (taken from the authors' C++ reader, input_data.cpp):
  * Customer columns are ``CUST_NO X Y READY DUE SERVICE FROZEN FRESH``
    (frozen comes *before* fresh).
  * Node 0 is the depot; its DUE value is the maximum route duration T_max.
  * Travel time equals Euclidean distance (no rounding).
  * Mixed-reconfigurable vehicles are listed one line per configuration,
    e.g. ``A1, A2, A3`` are three configurations of the physical vehicle ``A``.
  * ``CAPACITY_PER_FRIGOBOX`` is the load a box holds; ``WEIGHT_PER_FRIGOBOX``
    is the space a box occupies in the host vehicle (both 40 in all instances).
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

FRESH, FROZEN, MIXED = "fresh", "frozen", "mixed"


@dataclass(frozen=True)
class Unit:
    """A physical vehicle. Fresh/frozen units have one configuration."""

    uid: str                      # "7" for single-temperature, "A" for a mixed group
    kind: str                     # FRESH | FROZEN | MIXED
    configs: Tuple[Tuple[str, int, int], ...]  # (config_id, cap_fresh, cap_frozen)
    max_frigo: int                # F_k (0 for mixed)

    @property
    def capacity(self) -> int:
        """Native capacity of a single-temperature unit."""
        _, qf, qz = self.configs[0]
        return qf if self.kind == FRESH else qz


@dataclass
class Instance:
    name: str
    n: int                         # number of customers
    x: List[float]
    y: List[float]
    ready: List[float]
    due: List[float]
    service: List[float]
    fresh: List[int]               # q^1_i
    frozen: List[int]              # q^2_i
    units: List[Unit]
    total_frigo: int               # F
    box_cap: int                   # load held by one frigobox
    box_space: int                 # space consumed by one frigobox (w)
    t_max: float
    dist: List[List[float]] = field(repr=False, default_factory=list)
    ids: List[int] = field(default_factory=list)  # original label of internal node k

    def index_of(self, label: int) -> int:
        """Internal index of an original customer label (small instances keep
        Homberger-Gehring ids such as 91..100)."""
        if not hasattr(self, "_rev"):
            self._rev = {lab: k for k, lab in enumerate(self.ids)}
        return self._rev[label]

    # ------------------------------------------------------------------ helpers
    @property
    def customers(self) -> range:
        return range(1, self.n + 1)

    @property
    def total_demand(self) -> int:
        return sum(self.fresh) + sum(self.frozen)

    @property
    def theta(self) -> float:
        """Scaling factor of the unused-space term, 1 / sum_i (q1_i + q2_i)."""
        return 1.0 / self.total_demand

    def units_of(self, kind: str) -> List[Unit]:
        return [u for u in self.units if u.kind == kind]

    def summary(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "customers": self.n,
            "fresh_units": len(self.units_of(FRESH)),
            "frozen_units": len(self.units_of(FROZEN)),
            "mixed_units": len(self.units_of(MIXED)),
            "mixed_configs": sum(len(u.configs) for u in self.units_of(MIXED)),
            "frigoboxes": self.total_frigo,
            "box_capacity": self.box_cap,
            "fresh_demand": sum(self.fresh),
            "frozen_demand": sum(self.frozen),
            "t_max": self.t_max,
        }


_MIXED_ID = re.compile(r"^([A-Za-z]+)(\d+)$")


def _group_of(vehicle_id: str) -> str:
    m = _MIXED_ID.match(vehicle_id)
    return m.group(1) if m else vehicle_id


def load_instance(path: str) -> Instance:
    """Parse one MVRDC instance file."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh.read().replace("\r", "").split("\n")]

    i = 0
    while i < len(lines) and "VEHICLE_ID" not in lines[i]:
        i += 1
    i += 1
    raw_vehicles = []
    while i < len(lines) and lines[i]:
        vid, vtype, qf, qz, mf = lines[i].split()[:5]
        raw_vehicles.append((vid, vtype.lower(), int(qf), int(qz), int(mf)))
        i += 1

    total_frigo = box_cap = box_space = None
    while i < len(lines) and "CUST" not in lines[i]:
        if ":" in lines[i]:
            key, val = (s.strip() for s in lines[i].split(":", 1))
            if key == "TOTAL_AVAILABLE_FRIGOBOXES":
                total_frigo = int(val)
            elif key == "CAPACITY_PER_FRIGOBOX":
                box_cap = int(val)
            elif key == "WEIGHT_PER_FRIGOBOX":
                box_space = int(val)
        i += 1
    i += 1
    rows = []
    for ln in lines[i:]:
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 8:
            continue
        rows.append([float(p) for p in parts[:8]])
    rows.sort(key=lambda r: r[0])
    if not rows or int(rows[0][0]) != 0:
        raise ValueError(f"{path}: depot (node 0) not found")
    ids = [int(r[0]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicated customer ids")
    if None in (total_frigo, box_cap, box_space):
        raise ValueError(f"{path}: frigobox section incomplete")

    # Build physical units; mixed configurations are grouped by letter prefix.
    units: List[Unit] = []
    groups: Dict[str, List[Tuple[str, int, int]]] = {}
    order: List[str] = []
    for vid, vtype, qf, qz, mf in raw_vehicles:
        if vtype == MIXED:
            g = _group_of(vid)
            if g not in groups:
                groups[g] = []
                order.append(g)
            groups[g].append((vid, qf, qz))
        else:
            if vtype not in (FRESH, FROZEN):
                raise ValueError(f"{path}: unknown vehicle type {vtype}")
            units.append(Unit(vid, vtype, ((vid, qf, qz),), mf))
    for g in order:
        units.append(Unit(g, MIXED, tuple(groups[g]), 0))

    x = [r[1] for r in rows]
    y = [r[2] for r in rows]
    n = len(rows) - 1
    dist = [[math.hypot(x[a] - x[b], y[a] - y[b]) for b in range(n + 1)] for a in range(n + 1)]
    return Instance(
        name=os.path.splitext(os.path.basename(path))[0],
        n=n,
        x=x,
        y=y,
        ready=[r[3] for r in rows],
        due=[r[4] for r in rows],
        service=[r[5] for r in rows],
        frozen=[int(r[6]) for r in rows],
        fresh=[int(r[7]) for r in rows],
        units=units,
        total_frigo=total_frigo,
        box_cap=box_cap,
        box_space=box_space,
        t_max=rows[0][4],
        dist=dist,
        ids=ids,
    )


def list_instances(root: str, size: str) -> List[str]:
    """Return sorted instance paths for a size folder, e.g. ``"10 customers"``."""
    folder = os.path.join(root, "Instances", size)
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")
    )
