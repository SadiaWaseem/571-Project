"""Correctness tests for the MVRDC core.

The strongest checks compare against the authors' own published solutions:
our objective must reproduce their reported costs and our feasibility audit
must accept their routes.
"""
import glob
import itertools
import os
import random

import pytest

from mvrdc.construct import construct
from mvrdc.fleet import FleetAssigner
from mvrdc.insertion import best_insertion
from mvrdc.instance import FRESH, FROZEN, MIXED, Instance, Unit, list_instances, load_instance
from mvrdc.published import read_published, to_solution
from mvrdc.selectors import Outcome, QLearningSelector, SearchState
from mvrdc.solution import Evaluator, check_solution
from mvrdc.vns import VNSConfig, solve
from mvrdc.selectors import make_selector


# --------------------------------------------------------------------- fleet
def paper_example_instance() -> Instance:
    """Fleet of Tables 2-3 in Masmoudi et al. (2026); one dummy customer."""
    units = [
        Unit("1", FRESH, (("1", 200, 0),), 1),
        Unit("2", FRESH, (("2", 150, 0),), 2),
        Unit("3", FROZEN, (("3", 0, 200),), 2),
        Unit("4", FROZEN, (("4", 0, 250),), 1),
        Unit("E", MIXED, (("E1", 70, 160), ("E2", 125, 125), ("E3", 170, 80), ("E4", 150, 100)), 0),
        Unit("F", MIXED, (("F1", 44, 106), ("F2", 106, 44), ("F3", 75, 75)), 0),
    ]
    return Instance("example", 1, [0, 1], [0, 0], [0, 0], [1e9, 1e9], [0, 0], [0, 1], [0, 1],
                    units, 10, 40, 40, 1e9, [[0, 1], [1, 0]], [0, 1])


def test_candidates_route1_of_paper_example():
    fa = FleetAssigner(paper_example_instance())
    # Route 1 of Fig. 1: w1 = 70 fresh, w2 = 72 frozen
    c = {(fa.inst.units[x.unit].uid, x.config): (x.boxes, x.residual) for x in fa.candidates(70, 72)}
    assert ("1", 0) not in c                 # needs 2 boxes, F_1 = 1
    assert c[("2", 0)] == (2, 0)             # 150 - (70 + 2*40)
    assert c[("3", 0)] == (2, 48)            # the paper's choice: frozen vehicle 3, 2 boxes
    assert ("4", 0) not in c                 # F_4 = 1
    assert c[("E", 0)] == (0, 88)            # best E configuration (70,160)
    assert c[("F", 2)] == (0, 8)             # F3 (75,75)


def test_assignment_respects_units_and_boxes():
    inst = paper_example_instance()
    fa = FleetAssigner(inst)
    rng = random.Random(0)
    loads = [(70, 72), (40, 35), (66, 64)]  # the three routes of Fig. 1
    for _ in range(50):
        a = fa.assign(loads, rng)
        assert a is not None
        assert len({x.unit for x in a}) == 3
        assert sum(x.boxes for x in a) <= inst.total_frigo
    assert fa.assign([(500, 500)], rng) is None


# ----------------------------------------------------------------- insertion
def test_best_insertion_matches_brute_force(data_root):
    inst = load_instance(list_instances(data_root, "20 customers")[0])
    ev = Evaluator(inst)
    rng = random.Random(3)
    for _ in range(300):
        custs = rng.sample(list(inst.customers), 6)
        route, c = custs[:-1], custs[-1]
        if not ev.time_feasible(route):
            continue
        brute = None
        for i in range(len(route) + 1):
            r2 = route[:i] + [c] + route[i:]
            if ev.time_feasible(r2):
                delta = ev.route_length(r2) - ev.route_length(route)
                if brute is None or delta < brute[0] - 1e-9:
                    brute = (delta, i)
        fast = best_insertion(inst, route, c)
        if brute is None:
            assert fast is None
        else:
            assert fast is not None and abs(fast[0] - brute[0]) < 1e-6


# ------------------------------------------------------------- published data
def _published_pairs(root, size):
    sdir = os.path.join(root, "Detailed_route_solutions", f"Solutions {size} customers")
    for p in sorted(glob.glob(os.path.join(sdir, "*.txt"))):
        ip = os.path.join(root, "Instances", f"{size} customers", os.path.basename(p))
        if not os.path.exists(ip):
            continue
        inst = load_instance(ip)
        vehicles, best = read_published(p)
        if not {c for _, r in vehicles for c in r} <= set(inst.ids):
            continue  # A0-10-C127: solution ids do not match the instance file
        yield inst, vehicles, best


@pytest.mark.parametrize("size", ["10", "200"])
def test_objective_reproduces_published_costs(data_root, size):
    diffs = []
    for inst, vehicles, best in _published_pairs(data_root, size):
        ev = Evaluator(inst)  # 'code' definition of unused space
        sol = to_solution(ev, vehicles)
        assert check_solution(ev, sol) == [], inst.name
        diffs.append(abs(sol.cost - best))
    assert diffs
    assert sorted(diffs)[-2] <= 0.06          # all but one within rounding
    assert max(diffs) <= 1.0 + 1e-6           # A0-10-R127 reported 1.0 higher


# ---------------------------------------------------------------- algorithms
@pytest.mark.parametrize("size", ["10 customers", "15 customers", "20 customers"])
def test_construction_is_feasible(data_root, size):
    rng = random.Random(1)
    for path in list_instances(data_root, size):
        ev = Evaluator(load_instance(path))
        for _ in range(3):
            assert check_solution(ev, construct(ev, rng)) == []


def test_construction_large(data_root):
    rng = random.Random(1)
    for path in list_instances(data_root, "200 customers")[::12]:
        ev = Evaluator(load_instance(path))
        assert check_solution(ev, construct(ev, rng)) == []


@pytest.mark.parametrize("alg", ["vns-fixed", "ms-vns", "ms-vns-code", "vns-random", "q-vns"])
def test_all_algorithms_reach_published_optimum_small(data_root, alg):
    path = os.path.join(data_root, "Instances", "10 customers", "A0-10-C121.txt")
    res = solve(load_instance(path), make_selector(alg), VNSConfig(time_limit=3.0), seed=7)
    assert res.feasible
    assert res.best_cost == pytest.approx(3584.14, abs=0.01)  # CPLEX optimum per [1]


# ----------------------------------------------------------------- Q-learning
def test_q_update_terminal_and_bootstrap():
    q = QLearningSelector(alpha=0.5, gamma=0.9, eps_start=0.0, eps_end=0.0, q_init=0.0)
    q.start_run(4, 5, random.Random(0))
    q.start_restart()
    s = SearchState()
    a = q.next(s)
    q.feedback(a, Outcome.NEW_BEST, s)          # r = 5
    key = q.encode(s)
    a2 = q.next(s)                               # bootstraps: 0 + 0.5*(5 + 0.9*max(Q) - 0)
    assert q.Q[key][a] == pytest.approx(2.5)
    q.feedback(a2, Outcome.INFEASIBLE, s)        # r = -1
    q.end_restart(0)                             # terminal: Q += 0.5*(-1 - Q)
    expected = 2.5 if a2 == a else 0.0
    assert q.Q[key][a2] == pytest.approx(expected + 0.5 * (-1 - expected))


def test_q_state_space_size():
    q = QLearningSelector()
    q.start_run(4, 5, random.Random(0))
    states = set()
    for o, k, f, b in itertools.product(Outcome, [0, 50, 500], [0.1, 0.5, 0.9], [0.1, 0.6, 0.95]):
        states.add(q.encode(SearchState(f, k, b, o)))
    assert len(states) == 81
