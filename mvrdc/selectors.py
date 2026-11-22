"""Neighborhood-selection policies for the MS-VNS.

The only component that differs between the compared algorithms is the
policy that decides which shaking neighborhood N_h is applied next:

* ``SweepSelector(score=None)``   : plain VNS order N1 -> N2 -> N3 -> N4,
  ``n_it`` attempts per neighborhood (no adaptation).
* ``SweepSelector(score="paper")``: MS-VNS of Masmoudi et al. (2026). Scores
  phi_h += phi1 (new best), phi2 (improves current), phi3 (accepted by the
  annealing criterion); the order is re-sorted by descending score every
  ``reorder_every`` restarts and the scores are reset (authors' code).
  Defaults phi = (1, 0, 5) from Sec. 5.2 of the paper.
* ``SweepSelector(score="code")`` : the scoring actually implemented in the
  authors' main.cpp: +phi1 on a new best, +phi2 when accepted without a new
  best, and +phi3 on *every* attempt that does not give a new best. With
  phi3 = 5 this ranks first the neighborhood that failed most often.
* ``RandomSelector``             : uniform random choice, same step budget as Q.
* ``QLearningSelector``          : tabular Q-learning (proposed method).

Within a restart the sweep policies follow Algorithm 1: after a new best the
search stays in the same neighborhood with its attempt counter unchanged;
otherwise the counter is incremented and after ``n_it`` attempts the next
neighborhood is used. Random and Q-learning policies receive a budget of
``n_h * n_it`` shaking steps per restart (20 with the paper's settings).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple


class Outcome(IntEnum):
    NEW_BEST = 0     # x'' accepted and better than x_best
    IMPROVED = 1     # x'' accepted, better than current x
    ACCEPTED = 2     # x'' accepted by the annealing (Cauchy) criterion
    REJECTED = 3     # x'' not accepted
    INFEASIBLE = 4   # no feasible shake / vehicle assignment failed


@dataclass
class SearchState:
    """Information the policies may observe before choosing N_h."""
    elapsed_frac: float = 0.0      # share of the time budget used
    steps_since_best: int = 0      # shaking steps since the last new best
    boxes_frac: float = 0.0        # frigoboxes used by x / F
    last_outcome: Outcome = Outcome.REJECTED
    step_in_restart: int = 0


class Selector:
    name = "base"

    def start_run(self, n_h: int, n_it: int, rng: random.Random) -> None:
        self.n_h, self.n_it, self.rng = n_h, n_it, rng
        self.picks = [0] * n_h
        self.outcomes = [[0] * len(Outcome) for _ in range(n_h)]

    def start_restart(self) -> None:
        pass

    def next(self, s: SearchState) -> Optional[int]:
        raise NotImplementedError

    def feedback(self, h: int, outcome: Outcome, s: SearchState) -> None:
        self.picks[h] += 1
        self.outcomes[h][outcome] += 1

    def end_restart(self, restart: int) -> None:
        pass

    def stats(self) -> Dict[str, object]:
        return {"picks": list(self.picks), "outcomes": [list(o) for o in self.outcomes]}


class SweepSelector(Selector):
    def __init__(self, score: Optional[str] = None,
                 phi: Tuple[float, float, float] = (1.0, 0.0, 5.0),
                 reorder_every: int = 100):
        if score not in (None, "paper", "code"):
            raise ValueError("score must be None, 'paper' or 'code'")
        self.score, self.phi, self.reorder_every = score, phi, reorder_every
        self.name = {None: "VNS-fixed", "paper": "MS-VNS", "code": "MS-VNS-code"}[score]

    def start_run(self, n_h, n_it, rng):
        super().start_run(n_h, n_it, rng)
        self.order = list(range(n_h))
        self.scores = [0.0] * n_h
        self.order_history: List[List[int]] = [list(self.order)]

    def start_restart(self):
        self.pos, self.j = 0, 0

    def next(self, s):
        return self.order[self.pos] if self.pos < self.n_h else None

    def feedback(self, h, outcome, s):
        super().feedback(h, outcome, s)
        p1, p2, p3 = self.phi
        if self.score == "paper":
            if outcome == Outcome.NEW_BEST:
                self.scores[h] += p1
            elif outcome == Outcome.IMPROVED:
                self.scores[h] += p2
            elif outcome == Outcome.ACCEPTED:
                self.scores[h] += p3
        elif self.score == "code":
            if outcome == Outcome.NEW_BEST:
                self.scores[h] += p1
            else:
                if outcome in (Outcome.IMPROVED, Outcome.ACCEPTED):
                    self.scores[h] += p2
                self.scores[h] += p3
        if outcome != Outcome.NEW_BEST:
            self.j += 1
            if self.j >= self.n_it:
                self.pos, self.j = self.pos + 1, 0

    def end_restart(self, restart):
        if self.score is not None and (restart + 1) % self.reorder_every == 0:
            sc = self.scores
            self.order = sorted(range(self.n_h), key=lambda h: -sc[h])  # stable
            self.scores = [0.0] * self.n_h
            self.order_history.append(list(self.order))

    def stats(self):
        out = super().stats()
        out["order_history"] = self.order_history
        return out


class RandomSelector(Selector):
    name = "VNS-random"

    def start_restart(self):
        self.k = 0

    def next(self, s):
        if self.k >= self.n_h * self.n_it:
            return None
        self.k += 1
        return self.rng.randrange(self.n_h)


class WeightedRandomSelector(Selector):
    """Static (state-independent) allocation: N_h drawn with fixed probabilities.

    Used as an ablation control. ``vns-n4`` applies only N4; ``vns-qmix`` draws
    neighborhoods with the average shares Q-VNS ended up using in the pilot study
    (an *oracle* static mix: it borrows knowledge learned by Q-VNS). If Q-VNS
    beats ``vns-qmix``, the state-dependent part of the policy matters, not
    only the overall allocation.
    """

    def __init__(self, weights: Sequence[float], name: str = "VNS-static"):
        tot = float(sum(weights))
        self.weights = [w / tot for w in weights]
        self.name = name

    def start_restart(self):
        self.k = 0

    def next(self, s):
        if self.k >= self.n_h * self.n_it:
            return None
        self.k += 1
        return self.rng.choices(range(self.n_h), weights=self.weights)[0]


QMIX_PILOT = (0.055, 0.087, 0.294, 0.564)  # mean Q-VNS shares, pilot study (6 instances x 3 seeds)


class QLearningSelector(Selector):
    """Tabular Q-learning over neighborhoods (one episode = one restart).

    State (3^4 = 81 states):
      last outcome  : improved (new best or better than x) | accepted worse | rejected/infeasible
      stagnation    : shaking steps since the last new best  < b1 | < b2 | >= b2
      search phase  : elapsed share of the time budget       < 1/3 | < 2/3 | >= 2/3
      frigobox use  : boxes used by x / F                    < 0.5 | < 0.85 | >= 0.85
    Action: the neighborhood N_h. Reward per outcome (``rewards``).
    Update: Q(s,a) += alpha [r + gamma max_a' Q(s',a') - Q(s,a)]; the last step
    of a restart is terminal. Exploration: epsilon-greedy, epsilon decays
    linearly with the elapsed share of the budget from eps_start to eps_end.
    """

    name = "Q-VNS"

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9,
                 eps_start: float = 0.3, eps_end: float = 0.05,
                 rewards: Sequence[float] = (5.0, 2.0, 0.5, 0.0, -1.0),
                 q_init: float = 1.0,
                 stagnation_bins: Tuple[int, int] = (20, 100),
                 box_bins: Tuple[float, float] = (0.5, 0.85),
                 features: Sequence[str] = ("outcome", "stagnation", "phase", "boxes")):
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end = eps_start, eps_end
        self.rewards = tuple(rewards)
        self.q_init = q_init
        self.stag_bins, self.box_bins = stagnation_bins, box_bins
        self.features = tuple(features)

    def start_run(self, n_h, n_it, rng):
        super().start_run(n_h, n_it, rng)
        self.Q: Dict[Tuple[int, ...], List[float]] = {}
        self.pending: Optional[Tuple[Tuple[int, ...], int, float]] = None
        self.visits: Dict[Tuple[int, ...], int] = {}

    # state -------------------------------------------------------------
    def encode(self, s: SearchState) -> Tuple[int, ...]:
        parts = []
        for f in self.features:
            if f == "outcome":
                o = s.last_outcome
                parts.append(0 if o <= Outcome.IMPROVED else (1 if o == Outcome.ACCEPTED else 2))
            elif f == "stagnation":
                b1, b2 = self.stag_bins
                k = s.steps_since_best
                parts.append(0 if k < b1 else (1 if k < b2 else 2))
            elif f == "phase":
                parts.append(min(2, int(3 * s.elapsed_frac)))
            elif f == "boxes":
                b1, b2 = self.box_bins
                parts.append(0 if s.boxes_frac < b1 else (1 if s.boxes_frac < b2 else 2))
            else:
                raise ValueError(f"unknown feature {f}")
        return tuple(parts)

    def _q(self, key):
        q = self.Q.get(key)
        if q is None:
            q = [self.q_init] * self.n_h
            self.Q[key] = q
        return q

    # policy ------------------------------------------------------------
    def start_restart(self):
        self.k = 0

    def next(self, s):
        if self.k >= self.n_h * self.n_it:
            return None  # the pending transition is closed as terminal in end_restart
        key = self.encode(s)
        q = self._q(key)
        if self.pending is not None:  # bootstrap the previous transition
            ps, pa, pr = self.pending
            pq = self._q(ps)
            pq[pa] += self.alpha * (pr + self.gamma * max(q) - pq[pa])
            self.pending = None
        self.k += 1
        eps = self.eps_start + (self.eps_end - self.eps_start) * min(1.0, s.elapsed_frac)
        if self.rng.random() < eps:
            a = self.rng.randrange(self.n_h)
        else:
            m = max(q)
            best = [h for h, v in enumerate(q) if v == m]
            a = best[0] if len(best) == 1 else self.rng.choice(best)
        self._last = (key, a)
        self.visits[key] = self.visits.get(key, 0) + 1
        return a

    def feedback(self, h, outcome, s):
        super().feedback(h, outcome, s)
        key, a = self._last
        self.pending = (key, a, self.rewards[outcome])

    def end_restart(self, restart):
        if self.pending is not None:  # terminal transition
            ps, pa, pr = self.pending
            pq = self._q(ps)
            pq[pa] += self.alpha * (pr - pq[pa])
            self.pending = None

    def stats(self):
        out = super().stats()
        out["q_table"] = {",".join(map(str, k)): [round(v, 4) for v in q]
                          for k, q in sorted(self.Q.items())}
        out["state_visits"] = {",".join(map(str, k)): v for k, v in sorted(self.visits.items())}
        return out


def make_selector(name: str, **kw) -> Selector:
    name = name.lower()
    if name in ("vns-fixed", "fixed"):
        return SweepSelector(None)
    if name in ("ms-vns", "score", "paper"):
        return SweepSelector("paper", **kw)
    if name in ("ms-vns-code", "code"):
        return SweepSelector("code", **kw)
    if name in ("vns-random", "random"):
        return RandomSelector()
    if name == "vns-n4":
        return WeightedRandomSelector((0, 0, 0, 1), "VNS-N4-only")
    if name == "vns-qmix":
        return WeightedRandomSelector(QMIX_PILOT, "VNS-static-mix")
    if name in ("q-vns", "q", "qlearning"):
        return QLearningSelector(**kw)
    raise ValueError(f"unknown selector {name}")
