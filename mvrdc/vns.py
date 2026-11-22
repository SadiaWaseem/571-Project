"""Multi-start VNS (Masmoudi et al. 2026, Algorithm 1) with a pluggable
neighborhood-selection policy.

Per restart: shake x with the neighborhood chosen by the policy, re-assign
vehicles (SelectBestVehicle), apply local search, then accept x'' if it uses no
more vehicles than x and either improves f or passes the Cauchy criterion
beta < T^2 / (T^2 + dE^2). After the restart a new solution is constructed
(multi-start) and the temperature is lowered.

Temperature. T_init = (-0.05 / ln 0.5) f(x0) as in the paper. The paper cools
T <- 0.999 T after every restart over 100,000 restarts (the code uses 0.99 and
also re-initialises T at every restart). A Python run completes far fewer
restarts within a time budget, so by default we cool geometrically with the
elapsed share of the budget, T = T_init * r^(t / t_max) with r = ``t_end_ratio``,
which completes the same kind of schedule within the budget. ``cooling="restart"``
reproduces the paper's per-restart rule. See docs/DEVIATIONS.md.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .construct import construct
from .emissions import solution_emissions
from .instance import Instance
from .local_search import LocalSearch
from .neighborhoods import NAMES, Shaker
from .selectors import Outcome, SearchState, Selector
from .solution import Evaluator, Solution, assign_and_evaluate, check_solution

EPS = 1e-9


@dataclass
class VNSConfig:
    time_limit: float = 60.0           # seconds (stops at whichever limit comes first)
    max_restarts: Optional[int] = None
    n_it: int = 5                      # attempts per neighborhood (paper: 5)
    shake_tries: int = 10
    ls_moves: int = 50                 # local-search moves per call (authors' code: 50)
    cooling: str = "time"              # "time" | "restart"
    t_end_ratio: float = 1e-3          # T_end / T_init for time-based cooling
    alpha: float = 0.999               # per-restart cooling factor (paper)
    unused_definition: str = "code"


@dataclass
class RunResult:
    instance: str
    algorithm: str
    seed: int
    best_cost: float
    vehicles: int
    distance: float
    unused: float
    boxes: int
    time_to_best: float
    runtime: float
    restarts: int
    steps: int
    route_evals: int
    feasible: bool
    emissions: float = 0.0
    history: List[List[float]] = field(default_factory=list)  # [time, restart, cost]
    selector_stats: Dict[str, object] = field(default_factory=dict)
    routes: List[List[int]] = field(default_factory=list)      # original customer labels
    vehicles_used: List[str] = field(default_factory=list)
    config: Dict[str, object] = field(default_factory=dict)

    def to_row(self) -> Dict[str, object]:
        """Flat record for CSV output."""
        return {k: v for k, v in asdict(self).items()
                if k not in ("history", "selector_stats", "routes", "vehicles_used", "config")}


def solve(inst: Instance, selector: Selector, cfg: VNSConfig, seed: int,
          verbose: bool = False) -> RunResult:
    rng = random.Random(seed)
    ev = Evaluator(inst, cfg.unused_definition)
    shaker = Shaker(ev, cfg.shake_tries)
    ls = LocalSearch(ev, cfg.ls_moves)
    n_h = len(NAMES)
    t0 = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - t0

    x = construct(ev, rng)
    best = x.copy()
    t_best = elapsed()
    history = [[round(t_best, 3), 0, best.cost]]
    T_init = (-0.05 / math.log(0.5)) * x.cost
    T = T_init
    selector.start_run(n_h, cfg.n_it, rng)
    st = SearchState()
    restart = steps = 0
    done = False

    while not done:
        selector.start_restart()
        st.last_outcome, st.step_in_restart = Outcome.REJECTED, 0
        while True:
            el = elapsed()
            if el >= cfg.time_limit:
                done = True
                break
            st.elapsed_frac = el / cfg.time_limit
            st.boxes_frac = x.boxes_used() / max(1, inst.total_frigo)
            h = selector.next(st)
            if h is None:
                break
            steps += 1
            st.step_in_restart += 1
            outcome = Outcome.INFEASIBLE
            shaken = shaker.shake(x, h, rng)
            if shaken is not None:
                xs = assign_and_evaluate(ev, shaken[0], rng)
                if xs is not None:
                    xss = ls.run(xs, rng)
                    if xss.n_vehicles <= x.n_vehicles:
                        dE = xss.cost - x.cost
                        better = dE < -EPS
                        if better or rng.random() < T * T / (T * T + dE * dE):
                            x = xss
                            if x.cost < best.cost - EPS:
                                best = x.copy()
                                t_best = elapsed()
                                history.append([round(t_best, 3), restart, best.cost])
                                outcome = Outcome.NEW_BEST
                                if verbose:
                                    print(f"[{t_best:7.1f}s] restart {restart:5d} "
                                          f"N{h + 1} best {best.cost:.2f} ({best.n_vehicles} veh)")
                            else:
                                outcome = Outcome.IMPROVED if better else Outcome.ACCEPTED
                        else:
                            outcome = Outcome.REJECTED
                    else:
                        outcome = Outcome.REJECTED
            if outcome == Outcome.NEW_BEST:
                shaker.on_new_best()
                st.steps_since_best = 0
            else:
                shaker.on_no_new_best()
                st.steps_since_best += 1
            st.last_outcome = outcome
            selector.feedback(h, outcome, st)
        selector.end_restart(restart)
        restart += 1
        if done or (cfg.max_restarts is not None and restart >= cfg.max_restarts):
            break
        x = construct(ev, rng)  # multi-start
        if cfg.cooling == "time":
            T = T_init * cfg.t_end_ratio ** min(1.0, elapsed() / cfg.time_limit)
        else:
            T = cfg.alpha * T

    errs = check_solution(ev, best)
    labels = inst.ids
    return RunResult(
        instance=inst.name,
        algorithm=selector.name,
        seed=seed,
        best_cost=round(best.cost, 4),
        vehicles=best.n_vehicles,
        distance=round(best.distance(ev), 4),
        unused=sum(ev.unused(r, a) for r, a in zip(best.routes, best.assign)),
        boxes=best.boxes_used(),
        time_to_best=round(t_best, 3),
        runtime=round(elapsed(), 3),
        restarts=restart,
        steps=steps,
        route_evals=ev.route_evals,
        feasible=not errs,
        emissions=round(solution_emissions(ev, best)["emissions"], 4),
        history=history,
        selector_stats=selector.stats(),
        routes=[[labels[c] for c in r] for r in best.routes],
        vehicles_used=[inst.units[a.unit].configs[a.config][0] for a in best.assign],
        config=asdict(cfg),
    )
