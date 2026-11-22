# Adaptive Destroy-and-Repair VNS with Q-Learning Operator Selection for Multi-Compartment Routing with Reconfigurable Vehicles

ISE 571 Heuristic Search Methods, KFUPM, Term Project, Group 6:
Maha Abd Elhakeem, Sadia Waseem, Saphiah Bajba.

This repository implements, in Python:

* the **MVRDC** (multi-compartment VRP with time windows, mixed-reconfigurable vehicles and frigoboxes) of Masmoudi et al. (2026) [1];
* their **multi-start variable neighborhood search (MS-VNS)**, as described in the paper (construction, neighborhoods N1–N4, local search, SelectBestVehicle, Cauchy annealing acceptance, score-based neighborhood re-ordering);
* **Experiment 1, Q-VNS:** a tabular Q-learning agent that chooses the shaking *neighborhood* (N1–N4) at every step, in place of the score-based re-ordering of [1];
* **Experiment 2 (main), N4-ops:** every shaking step is a destroy-and-repair move, and the policy chooses one of 8 operator pairs, (random | worst | related | route removal) × (greedy | regret-2 insertion). It is chosen by Q-learning (**ours**), by the ALNS roulette wheel of Ropke & Pisinger (2006), or at random;
* control policies (fixed order, random, N4 only, the authors' code scoring) so that the effect of *learning* can be isolated;
* a resumable experiment runner, statistical analysis and a Colab notebook.

## Quick start (Colab or local)

```bash
git clone https://github.com/SadiaWaseem/571-Project.git && cd 571-Project
pip install -r requirements.txt
bash scripts/fetch_data.sh          # clones the benchmark of [1] into data/MVRDC
python -m pytest -q                 # 21 tests, ~25 s
```

```python
from mvrdc.instance import load_instance
from mvrdc.selectors import make_selector
from mvrdc.vns import VNSConfig, solve

inst = load_instance("data/MVRDC/Instances/200 customers/A0-50-C121.txt")
res = solve(inst, make_selector("q-vns"), VNSConfig(time_limit=60), seed=1, verbose=True)
print(res.best_cost, res.vehicles, res.distance, res.feasible)
```

The notebook `notebooks/MVRDC_QVNS.ipynb` walks through the data, the validation against the published results, a single run, the full experiment and the analysis.

## Algorithms compared

All policies share every component (construction, vehicle assignment, local search, acceptance, cooling). Only the shaking step differs. Per-step policies get 20 shaking steps per restart, the nominal budget of the sweep of [1].

| Label | Experiment | Policy |
|---|---|---|
| `ms-vns` | 1, 2 | MS-VNS of [1]: sweep N1→N4, 5 attempts each, re-sorted every 100 restarts by scores (1, 0, 5) |
| `vns-fixed`, `ms-vns-code` | 1 | sweep without re-ordering / with the scoring found in the authors' C++ |
| `vns-random`, `vns-qmix` | 1 | random neighborhood / static mix equal to Q-VNS's average shares |
| `q-vns` | 1, 2 | Q-learning over N1–N4 (81 states) |
| `vns-n4` | 1, 2 | N4 (remove-insert of [1]) only |
| `ops-random` | 2 | random destroy-repair pair |
| `ops-alns` | 2 | ALNS roulette wheel, scores (33, 9, 13), reaction 0.1, segments of 100 steps |
| `ops-q` | 2 | **ours**: Q-learning over 8 destroy-repair pairs; state = (last outcome, stagnation, phase), 27 states |

Q-learning: rewards (5, 2, 0.5, 0, −1) for new best / improved / accepted / rejected / infeasible, α = 0.1, γ = 0.9, ε decays from 0.3 to 0.05.

## Running the experiments

The notebook runs everything (Sections 5–7). From the command line:

```bash
# Experiment 2 (main): 30 instances (every 2nd), 6 methods, 5 seeds, 60 s  -> ~7.5 h on 2 cores
python experiments/run.py --data data/MVRDC --sizes "200 customers" --every 2 \
    --algorithms ms-vns q-vns vns-n4 ops-random ops-alns ops-q --seeds 1-5 \
    --time-limit 60 --workers 2 --out results/main

# Sensitivity: one Q-learning variant (repeat per variant)
python experiments/run.py --data data/MVRDC --instances A0-50-C121 A0-50-R121 A0-50-RC121 \
    A1-50-C221 A2-50-R221 A2-50-RC221 --algorithms ops-q --q alpha=0.3 \
    --label "ops-q@alpha=0.3" --seeds 1-3 --time-limit 60 --workers 2 --out results/sens
```

* **Resumable.** Each run writes `<out>/runs/<instance>__<alg>__<seed>.json`, and existing runs are skipped. On Colab, write to Google Drive and rerun after a disconnect.
* **Stored results** (as `runs.zip`, unpacked automatically by `mvrdc.analysis`):
  * `results/main`: Experiment 2, 900 runs (30 instances × 6 methods × 5 seeds × 60 s), produced on an Apple MacBook Pro (4 performance cores, Python 3.9) with `--workers 3`;
  * `results/sens`: sensitivity study, 144 runs (8 Q-learning variants × 6 instances × 3 seeds × 60 s), same machine;
  * `results/pilot` and `results/opt2_pilot`: 6-instance pilot studies (3 seeds, 30 s) of Experiments 1 and 2.

  Every stored solution was re-checked by an independent feasibility checker.

## Validation

* Our objective reproduces the costs the authors report for their own published solutions. We check the 74 published solutions whose instance file is also published: 73 are within 0.05, and the remaining one is reported exactly 1.0 above the cost of its own routes.
* All published routes pass our independent feasibility audit (`solution.check_solution`), and so does every solution our code returns (`RunResult.feasible`).
* All policies reach the CPLEX optimum of [1] on `A0-10-C121` (3584.14) within 3 s.

See `docs/DEVIATIONS.md` for where the authors' code differs from their paper, and which version we follow.

## Repository layout

```
mvrdc/            instance.py      parser, physical vehicles, distance matrix
                  fleet.py         Algorithms 3–4 (candidate sets, route-to-vehicle assignment)
                  solution.py      route evaluation, objective, feasibility audit
                  insertion.py     time-window-aware cheapest insertion
                  construct.py     Algorithm 2
                  neighborhoods.py N1–N4 (shaking)
                  local_search.py  2-opt, relocate, 2-opt*, remove-two-insert-one
                  selectors.py     selection policies: sweep/scores of [1], random, ALNS roulette, Q-learning
                  operators.py     destroy-and-repair operator portfolio (Experiment 2)
                  vns.py           Algorithm 1 driver, RunResult
                  emissions.py     load-dependent CO2 estimate (reported only)
                  analysis.py      gaps, Wilcoxon/Friedman tests, convergence, usage
                  published.py     reader for the authors' published routes
experiments/run.py                 resumable experiment runner (CLI)
experiments/make_figures.py        regenerates the paper figures from results/main
paper/                             IEEE paper source (main.tex, refs.bib, figs/)
notebooks/MVRDC_QVNS.ipynb         Colab notebook
tests/                             pytest suite
docs/DEVIATIONS.md                 paper vs. code vs. this implementation
```

## Data and references

Instances and published solutions belong to the authors of [1] and are downloaded from their repository by `scripts/fetch_data.sh`. They are not redistributed here.

[1] M. A. Masmoudi, R. Baldacci, P. Fontaine, E. Khayya, "A novel multi-compartment vehicle routing problem with time windows: Reconfigurable vehicles and dynamic compartments," *European Journal of Operational Research*, 335(3), 787–801, 2026. Data/code: https://github.com/Masmoudi-Amine/MVRDC-

**AI assistance.** Parts of this code were written with the help of an AI assistant (Claude, Anthropic), as permitted in ISE 571. The group reviewed the code, chose the Q-learning design and is responsible for the results.
