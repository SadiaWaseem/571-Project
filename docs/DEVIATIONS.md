# Paper vs. published code vs. this implementation

Our baseline is the MS-VNS **as described in the paper**:

> M. A. Masmoudi, R. Baldacci, P. Fontaine, E. Khayya, "A novel multi-compartment vehicle routing problem with time windows: Reconfigurable vehicles and dynamic compartments", *EJOR* 335(3):787–801, 2026.

We read the authors' public C++ code (<https://github.com/Masmoudi-Amine/MVRDC->, folder `MS-VNS code/`) and the paper side by side. In several places they disagree. The table below records each case and which version we follow. File and line references point to the authors' repository as cloned on 21 Nov 2026.

## A. Algorithmic differences

| # | Component | Paper | Authors' code | This implementation |
|---|---|---|---|---|
| A1 | Acceptance probability | Cauchy: β < T²/(T²+ΔE²) (Sec. 4.1 (ii)) | `T / (T*T + dE*dE)` (main.cpp l. 353, 369). For T in the thousands this is ≈ 1/T, so a worse or even equal solution is almost never accepted. | Paper |
| A2 | Where acceptance is applied | Once, to x'' after local search | Twice: first to the shaken solution (before local search), then to x'' (l. 353–371) | Paper |
| A3 | Cooling rate α | 0.999 (Sec. 5.2) | 0.99 (l. 18) | Paper value for `cooling="restart"`; time-based by default (see C1) |
| A4 | Temperature | T_init computed once (Alg. 1 l. 3), T ← αT after every restart (l. 36) | T recomputed from the new initial solution at every restart (l. 284) and cooled only after a new best (l. 404) | Paper (or C1) |
| A5 | Neighborhood scores | φ1 for a new best, φ2 for improving the current solution, φ3 for acceptance by the annealing criterion (Sec. 4.1 (iv)); φ = (1, 0, 5) | φ1 for a new best, φ2 for an accepted non-best solution, **φ3 for every attempt that does not produce a new best** (l. 389–413). With φ3 = 5 the re-ordering puts first the neighborhood that *failed* most often. | Both: `ms-vns` (paper) and `ms-vns-code` (code). The paper version is our main baseline. |
| A6 | Re-ordering | Every 100 iterations, by cumulative scores | Every 100 restarts; scores reset after each re-ordering (l. 424–448) | Code (it settles the ambiguity) |
| A7 | After a new best | Stay in the same neighborhood; counter unchanged (Alg. 1 l. 14–16) | Leave the neighborhood (`improved = true` ends the attempt loop) | Paper |
| A8 | Local search | Intensification that "improves x' to obtain x''" (Sec. 4.4) | One operator drawn at random and applied 50 times at random positions; **every feasible move is kept whatever its cost** (neighborhoods.cpp l. 998–1020). The "inter-route relocate" calls an intra-route relocate (l. 562–571). | Same operators and 50 moves per call, but a move is kept only if it lowers f(x); relocate is inter-route |
| A9 | Remove–insert (N4) re-insertion | Basic greedy of Ropke & Pisinger (2006): repeatedly insert the customer with the globally cheapest insertion | Customers are re-inserted in removal order, each at its cheapest position; the same-type restriction is commented out (l. 662–731) | Paper (global greedy), any route whose vehicle still fits the load |
| A10 | Same-type restriction of inter-route moves | Routes served by the same vehicle type | Type of `vehicles[0]` (the first vehicle in the list), not of the selected route; `relocateTwoCustomers` indexes the full list with positions from the filtered list (l. 463–508) | Type of the selected route |
| A11 | Candidate sets (Alg. 3) | Mixed vehicles evaluated only if no fresh/frozen vehicle fits (l. 21) | Always evaluated (VehicleOptimizer.cpp l. 123, "Removed the restrictive condition") | Code. The text of Alg. 4 assumes mixed candidates exist for routes that need boxes. |
| A12 | Construction, phase 2 | Leftover customers go to a mixed vehicle that can be reconfigured, else to a random mixed vehicle (may be infeasible) | Leftover customers may be inserted into a second configuration of an already used mixed vehicle (constructive.cpp l. 294–336) | Reconfigure an open route if the whole load fits; else open an unused vehicle; else retry the construction |
| A13 | Infeasible shake | not specified | `continue` without incrementing the attempt counter (l. 303–305), so the loop can repeat indefinitely | Counted as an attempt (outcome INFEASIBLE) |
| A14 | Shake attempts | one move | one random application; if infeasible, the solution is unchanged | up to 10 random applications to find a feasible move |
| A15 | Vehicle count | accept only if n(x'') ≤ n(x) | also accepts any x'' with fewer vehicles | Paper. Fewer vehicles already lowers f by ≥ 1000. |

## B. Data and reporting issues found while validating

* **B1.** The published 15- and 20-customer route solutions (`Detailed_route_solutions/Solutions 15|20 customers`) are for instances (`A0_15_C121`, …) that are **not** in `Instances/` (which contains `A0-15-C101`, …). They cannot be checked.
* **B2.** `A0-10-C127`: the solution file uses customers 91–100, but the instance file contains customers 1–10.
* **B3.** `A0-10-R127`: the reported cost (2528.82) is exactly 1.0 above the cost of the reported routes (2527.82).
* **B4.** The other 73 published solutions (14 with 10 customers, 60 with 200) pass our feasibility audit, and our objective reproduces the reported costs to within 0.05 (`tests/test_core.py::test_objective_reproduces_published_costs`).
* **B5.** The "main" best values differ between the online tables. For example, A0-50-C121 has 33,994.50 in Table D1 but 35,618.30 in Table D4.
* **B6.** The Chen–Shi (2019) MCmpt-VRPTW instances used in Tables 6–8 are not in the repository. Only the authors' solutions are.

## C. Adaptations needed to run in Python within a time budget

* **C1. Cooling schedule.** The paper's 100,000 restarts take 5,000+ s in C++ (Table 4). We stop on a wall-clock limit, and by default cool geometrically with the elapsed share of the budget: T = T_init · r^(t/t_max), r = 10⁻³. With `cooling="restart"`, the paper's T ← 0.999·T is used instead.
* **C2. Budget of the adaptive policies.** Random and Q-learning selection get n_h · n_it = 20 shaking steps per restart, the nominal budget of the sweep. The sweep can use more steps when it finds new bests (A7).
* **C3. Objective.** We use the authors' code definition of unused space for single-temperature vehicles (Q_k − w1 − w2) because it reproduces their reported values (B4). The paper/MILP definition (Q_k − w_native − O·w) is available through `Evaluator(unused_definition="paper")`. The two differ by less than θ·F·w (well under one distance unit).
