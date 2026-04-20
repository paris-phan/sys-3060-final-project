# NYC Citi Bikes — Analytical Modeling of Station Capacity Reallocation

## Team Members

[TBD]

## Background & Motivation

Citi Bike is the largest bike-share system in North America, with roughly 2,000 stations and 25,000+ bikes deployed across NYC. The system's value proposition is simple: a user walks to the nearest station, grabs a bike, rides to their destination, and returns the bike at the station nearest to where they're going. That promise breaks when users arrive at a station and find either no bikes (**stockout**) or no open docks (**dockblock**). Both failures force users to walk, wait, or abandon the trip — in the worst case producing a worse outcome than not using the system at all.

During commuter peaks, directional demand creates predictable imbalances: bikes pile up at destination stations near business districts while source stations in residential areas run dry. Because stations have fixed physical dock counts, the system's ability to absorb this imbalance is determined — before any operational intervention — by how capacity is allocated across the network. Our project asks whether the current allocation is optimal under the statistical properties of real demand, and quantifies the improvement available from a pure reallocation with no new docks added.

## The System

- A network of **N** stations. Station *n* has fixed capacity **cₙ** (total docks).
- At any moment, station *n* holds **sₙ ∈ {0, 1, …, cₙ}** bikes; available docks = cₙ − sₙ.
- Users interact with stations in two ways:
  - **Withdrawal**: a user arrives wanting a bike. Succeeds if sₙ > 0, fails (stockout) if sₙ = 0.
  - **Deposit**: a user arrives wanting to return a bike. Succeeds if sₙ < cₙ, fails (dockblock) if sₙ = cₙ.
- Bikes in transit, plus bikes currently at stations, form a closed subsystem on the timescale of interest.

## Data Available

Public Citi Bike trip data from [citibikenyc.com/system-data](https://citibikenyc.com/system-data), including: Ride ID, Rideable type, Started at, Ended at, Start station name/ID, End station name/ID, Start/End latitude/longitude, Member/casual flag. Station capacities (cₙ) are obtained from the Citi Bike GBFS feed.

## Analytical Framework

To keep the project fully analytical (no simulation), we model each station as a **continuous-time Markov chain (CTMC)** — specifically, a finite-capacity birth-death process on the state space {0, 1, …, cₙ}, where the state sₙ is the number of bikes currently at the station.

**Transition rates at station n:**

- Deposit (birth): sₙ → sₙ + 1 at rate **μₙ** (Poisson arrivals of users returning a bike), valid for sₙ < cₙ
- Withdrawal (death): sₙ → sₙ − 1 at rate **λₙ** (Poisson arrivals of users taking a bike), valid for sₙ > 0

This is structurally identical to an **M/M/1/K queue** with K = cₙ, where "items in system" = bikes at the station, "arrivals" = deposits at rate μₙ, and "service completions" = withdrawals at rate λₙ. The stationary distribution has a closed form:

Let $\rho_n = \mu_n / \lambda_n$ (dimensionless "fill intensity"). Then for k ∈ {0, …, cₙ}:

$$
\pi_k(n) = \frac{\rho_n^k \,(1 - \rho_n)}{1 - \rho_n^{c_n + 1}} \quad (\rho_n \ne 1), \qquad \pi_k(n) = \frac{1}{c_n + 1} \quad (\rho_n = 1)
$$

Because we have closed-form stationary probabilities, every quantity of interest — stockout probability, dockblock probability, expected fill level, long-run failure rate — is computable without simulation.

**Rate estimation from data.** Within the chosen time window, for each station n in the cluster:

- $\lambda_n$ = empirical rate of withdrawals (trips starting at station n) per unit time
- $\mu_n$ = empirical rate of deposits (trips ending at station n) per unit time

Both are computed across 6–8 weeks of data to smooth noise and ensure the CTMC is approximately stationary on the chosen window.

## Failure Cases

For station *n*, in steady state:

- **Stockout probability**: P(sₙ = 0) = π₀(n)
- **Dockblock probability**: P(sₙ = cₙ) = π_{cₙ}(n)

**Long-run failure rate at station n** (expected failed user events per unit time):

$$
F_n(c_n) = \lambda_n \,\pi_0(n; c_n) \;+\; \mu_n \,\pi_{c_n}(n; c_n)
$$

This is the core per-station objective — a single scalar, in closed form, depending only on (λₙ, μₙ, cₙ).

## Scope

- **Spatial**: A cluster of **3–4 stations** in a single high-imbalance corridor (candidate: Midtown East near Grand Central, where morning commuters create a strong directional flow). Final cluster selected during data exploration.
- **Temporal**: Weekday morning rush, **7:00–10:00 AM**, across 6–8 weeks of 2024–2025 data.
- **Granularity**: Each station is modeled independently as a CTMC. Inter-station coupling enters only through rate estimation — deposits at station j already reflect withdrawals routed from elsewhere, so the single-station CTMC captures the network effect in reduced form.

## Assumptions

1. Withdrawal and deposit inter-arrival times at each station are exponentially distributed within the chosen time window. Validated via Q-Q plots and chi-squared goodness-of-fit on the empirical data.
2. Rates λₙ and μₙ are constant within the 7:00–10:00 AM window (stationarity).
3. Bikes and docks form a closed system on the modeling horizon: total docks Σcₙ is fixed; bikes not in use reside at some station.
4. Users experiencing a failure at station n exit the system there (no balking to neighboring stations in the base model; balking is a natural extension if time permits).
5. Cross-station routing is stationary within the window and absorbed into μₙ; we do not need to model the routing matrix explicitly for the independent-station CTMC.

## Research Question

Given a fixed total capacity **C_total = Σcₙ** across the selected cluster, what capacity vector **(c₁*, c₂*, …, c_N*)** minimizes the total expected long-run failure rate?

Formally:

$$
\min_{c_1, \dots, c_N} \;\; F(c_1, \dots, c_N) = \sum_{n=1}^{N} \Big[\, \lambda_n \,\pi_0(n; c_n) \;+\; \mu_n \,\pi_{c_n}(n; c_n) \,\Big]
$$

$$
\text{subject to} \quad \sum_{n=1}^{N} c_n = C_{\text{total}}, \quad c_n \ge 1,\; c_n \in \mathbb{Z}_+
$$

## Base Model

Model each station with its **current, real-world cₙ** from the GBFS feed. Using estimated (λₙ, μₙ), compute πₖ(n) analytically for all k, then the per-station failure rates Fₙ and the total baseline

$$F_{\text{base}} = \sum_n F_n(c_n)$$

This is the incumbent benchmark the improvement must beat.

## Proposed Improvement: Capacity Reallocation

Hold C_total = Σcₙ fixed (no new docks built) and search over integer allocations for the optimal (c₁*, …, c_N*). This reflects a real operational lever: Citi Bike periodically reconfigures stations, and a data-driven reallocation is a cheap intervention relative to running rebalancing trucks.

**Solution method (fully analytical).** Because F is **separable** — each Fₙ depends only on cₙ given fixed λₙ, μₙ — the optimization reduces to a classical discrete resource allocation problem solvable exactly by **marginal analysis**:

1. Initialize cₙ = 1 for all n. Budget remaining: C_total − N docks.
2. For each station, compute the marginal benefit of one more dock: Δₙ = Fₙ(cₙ) − Fₙ(cₙ + 1).
3. Assign the next dock to argmaxₙ Δₙ. Decrement budget.
4. Repeat (2)–(3) until the budget is exhausted.

Provided Fₙ has non-increasing marginal returns in cₙ (which we verify numerically for the estimated parameters — a standard property of M/M/1/K in the regimes of interest), this greedy procedure is provably optimal. The entire solution is closed-form π evaluations plus discrete bookkeeping — no simulation, no heuristic.

## Comparison (Base vs. Improvement)

Report:

- Per-station failure rates Fₙ before and after reallocation.
- Stockout vs. dockblock decomposition at each station: does reallocation primarily fix empty-station problems, full-station problems, or both?
- Total system failure rate: F_base vs. F_optimal, and the percent reduction.
- The optimal allocation vector and its delta from the current deployment — which stations gain docks, which lose.
- **Sensitivity analysis**: how the optimal allocation shifts if ρₙ estimates are perturbed by ±10%, to assess robustness of the recommendation to estimation error.

## Expected Deliverables

- **Data pipeline** (Python / pandas): filters Citi Bike trip data to the selected cluster and time window; produces empirical λₙ, μₙ, and goodness-of-fit diagnostics.
- **Analytical model**: closed-form computation of πₖ(n) and Fₙ(cₙ) for any (c₁, …, c_N), vectorized over the cluster.
- **Optimization routine**: marginal-analysis solver returning the optimal allocation and its optimality proof (via verification of the non-increasing-returns property).
- **Validation**: exponential-fit diagnostics on inter-arrival data; comparison of model-predicted failure rates against observed failure proxies in held-out data.
- **SIEDS paper** (exam week).
- **5-minute presentation + slide deck** (last full week of class).