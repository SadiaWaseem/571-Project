# Q-VNS: Learned Neighborhood Selection for Multi-Compartment Routing with Reconfigurable Vehicles

ISE 571 Heuristic Search Methods, KFUPM, Term Project, Group 6:
Maha Abd Elhakeem, Sadia Waseem, Saphiah Bajba.

This repository implements, in Python:

* the **MVRDC** (multi-compartment VRP with time windows, mixed-reconfigurable vehicles and frigoboxes) of Masmoudi et al. (2026) [1];
* their **multi-start variable neighborhood search (MS-VNS)**, as described in the paper (construction, neighborhoods N1–N4, local search, SelectBestVehicle, Cauchy annealing acceptance, score-based neighborhood re-ordering);
* **Q-VNS** (our proposal): a tabular Q-learning agent that chooses the shaking neighborhood at every step, in place of the score-based re-ordering;
* control policies (fixed order, uniform random, the authors' code scoring) so that the effect of *learning* can be isolated;
* a resumable experiment runner, statistical analysis and a Colab notebook.

## Quick start (Colab or local)

```bash
git clone https://github.com/SadiaWaseem/571-Project.git && cd 571-Project
pip install -r requirements.txt
bash scripts/fetch_data.sh          # clones the benchmark of [1] into data/MVRDC
python -m pytest -q                 # 16 tests, ~20 s
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

All policies share every component (construction, N1–N4, local search, vehicle assignment, acceptance, cooling). Only the neighborhood-selection policy differs.

| Label | Policy |
|---|---|
| `vns-fixed` | N1 → N2 → N3 → N4, `n_it` = 5 attempts each per restart |
| `ms-vns` | MS-VNS of [1]: same sweep, order re-sorted every 100 restarts by scores φ = (1, 0, 5) for new best / improvement / annealing acceptance |
| `ms-vns-code` | Same, with the scoring found in the authors' C++ (φ3 added on every non-improving attempt) |
| `vns-random` | Uniform random neighborhood, 20 shaking steps per restart |
| `vns-n4` | Ablation: N4 (remove–insert) only, 20 steps per restart |
| `vns-qmix` | Ablation: static random mix with the average shares Q-VNS learned in the pilot (an oracle static allocation) |
| `q-vns` | **Ours.** ε-greedy tabular Q-learning, 20 steps per restart; state = (last outcome, stagnation, search phase, frigobox use), 81 states; rewards (5, 2, 0.5, 0, −1) for new best / improved / accepted / rejected / infeasible |

## Running the experiments

```bash
python experiments/run.py --data data/MVRDC --sizes "200 customers" \
    --algorithms vns-fixed ms-vns vns-random q-vns --seeds 1-10 \
    --time-limit 60 --workers 2 --out results/main
```

* **Resumable.** Each run writes `results/main/runs/<instance>__<alg>__<seed>.json`, and existing runs are skipped. On Colab, point `--out` to Google Drive and simply rerun the cell after a disconnect.
* **Splitting the work.** Use `--instances` (glob patterns) to give each group member a share, e.g. `--instances "A0-50-C*"`, `"A0-50-R*" "A0-50-RC*"`, `"A1-*" "A2-*"`. Then merge the `results.csv` files.
* **Q-learning settings.** For example `--q alpha=0.2 gamma=0.5 eps_start=0.2`. **VNS settings.** For example `--set n_it=5 ls_moves=50`.
* **Small instances (vs. CPLEX in [1]).** `--sizes "10 customers" "15 customers" "20 customers" --time-limit 10`.

## Validation

* Our objective reproduces the costs the authors report for their own published solutions. We check the 73 solutions whose instance file is published: all are within 0.05, except one reported value that is exactly 1.0 above the cost of its own routes.
* All published routes pass our independent feasibility audit (`solution.check_solution`), and so does every solution our code returns (`RunResult.feasible`).
* All five algorithms reach the CPLEX optimum of [1] on `A0-10-C121` (3584.14) within 3 s.

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
                  selectors.py     neighborhood-selection policies incl. Q-learning
                  vns.py           Algorithm 1 driver, RunResult
                  emissions.py     load-dependent CO2 estimate (reported only)
                  analysis.py      gaps, Wilcoxon/Friedman tests, convergence, usage
                  published.py     reader for the authors' published routes
experiments/run.py                 resumable experiment runner (CLI)
notebooks/MVRDC_QVNS.ipynb         Colab notebook
tests/                             pytest suite
docs/DEVIATIONS.md                 paper vs. code vs. this implementation
```

## Data and references

Instances and published solutions belong to the authors of [1] and are downloaded from their repository by `scripts/fetch_data.sh`. They are not redistributed here.

[1] M. A. Masmoudi, R. Baldacci, P. Fontaine, E. Khayya, "A novel multi-compartment vehicle routing problem with time windows: Reconfigurable vehicles and dynamic compartments," *European Journal of Operational Research*, 335(3), 787–801, 2026. Data/code: https://github.com/Masmoudi-Amine/MVRDC-

**AI assistance.** Parts of this code were written with the help of an AI assistant (Claude, Anthropic), as permitted in ISE 571. The group reviewed the code, chose the Q-learning design and is responsible for the results.
