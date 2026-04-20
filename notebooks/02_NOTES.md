# 02 — Rebalancing-rate optimization notes

**Last rebuild:** 2026-04-20.

## Why this notebook exists

Notebook 01's base model showed `F_base = 41.87 events/hour` on the Dec-2021 Midtown cluster, and a quick pre-check showed that redistributing integer docks between the 5 stations moves the total by effectively zero — all stations are in the asymptotic regime where $\pi$ is saturated and adding/removing a dock changes the boundary probabilities by $< 10^{-10}$. Capacity reallocation is a dead lever for this cluster. So this notebook swaps the optimization target: **rebalancing rate $\theta_n$** instead of capacity $c_n$.

## Model extension

Same finite-state CTMC per station, plus a **truck-reset** transition $k \to t_n$ at rate $\theta_n$, valid for all $k \ne t_n$. That breaks birth-death structure (multi-state jump), so we drop the closed-form geometric $\pi$ and solve $\pi Q = 0$ with $\sum \pi = 1$ as a small dense linear system. At $\theta = 0$ we recover the base-model geometric distribution exactly (tested to `atol = 1e-10`).

`src/ctmc.py` gains three new functions; the base-model API (`stationary_distribution`, `failure_rate`) is untouched.

`src/optimization.py` is new: response curves, a greedy marginal-analysis optimizer on a $\theta$ grid, and a Pareto-frontier wrapper that exploits the nested-prefix structure of the greedy (running at the max budget and sampling the trace is equivalent to re-running at each budget, because the greedy allocation at budget $B + \Delta B$ extends the allocation at budget $B$ by one step).

## Target-level choice

Fixed $t_n = \lfloor c_n / 2 \rfloor$ ("half-full after a truck visit"). This is the operationally sensible default but it is a modelling assumption, not something we estimate from data. Sensitivity-checked in §5 (see below — and the check was useful: the answer is non-trivially sensitive to the target).

## Θ_total range and knee

Grid: linear `np.linspace(0.5, 50.0, 20)` events/hour. Why not log-spaced: the knee lives at $\Theta_\text{total} \approx 5$ — mid-grid — and the interesting structure is around the knee, not on either asymptote.

- **Knee**: $\Theta_\text{total} = 5.71$ events/hour → $F_\text{total} = 1.20$, a $-97.1\%$ reduction vs. $F_\text{base}$.
- **Chosen operating point**: $\Theta_\text{total} = 5.0$ events/hour (cleanly "one truck-visit per station per hour on average"). Gives $F_\text{opt} = 1.77$, a $-95.8\%$ reduction.
- The chosen budget is within $1.1\times$ of the knee — the operational sweet spot.

The reduction is enormous because the cluster has three near-pure extremes: a ρ = 14.4 sink (Madison & 51), a ρ = 0.47 source (8 Ave & 31), and a ρ = 2.45 sink (W 55 & 6). Even a small reset rate collapses the tail probabilities: for the ρ = 14.4 sink, $\pi_{c_n}$ at $\theta = 0$ is $0.93$; at $\theta = 2.4$ (its share of the $\Theta = 5$ budget) it's $\approx 0.03$. That's where the 95% reduction comes from — multiplicatively, three stations each giving a 20-50× reduction in their dominant failure mode.

## Allocation at the chosen budget

| station | type | ρ | $c_n$ | $F_\text{base}$ | $\theta^*$ | $F_\text{opt}$ | % reduction |
|---|---|--:|--:|--:|--:|--:|--:|
| Madison Ave & E 51 St | sink | 14.36 | 43 | 16.65 | **2.42** | 0.87 | 94.8% |
| W 42 St & 6 Ave       | balanced (mild sink) | 1.47 | 53 | 4.13  | 0.59 | 0.26 | 93.8% |
| 8 Ave & W 31 St       | source | 0.47 | 97 | 12.62 | **1.09** | 0.32 | 97.5% |
| Broadway & W 58 St    | balanced | 0.96 | 79 | 0.54  | 0.11 | 0.07 | 87.1% |
| W 55 St & 6 Ave       | sink | 2.45 | 77 | 7.93  | 0.79 | 0.25 | 96.8% |

The biggest imbalance (Madison & 51) takes ~half the budget. The balanced station (Broadway & 58) gets almost none — its $F_\text{base}$ is already $0.54$, so any $\theta$ dollars spent here are wasted versus the same dollars on a station with $F_\text{base} = 16$.

## Sensitivity findings

Two perturbation families at the $\Theta_\text{total} = 5$ operating point:

### 1. $\pm 10\%$ rate perturbations (one station × one param at a time)

20 perturbation runs. Max shifts:

- $|F_\text{opt} - F_\text{opt,\text{base}}|$ max: **0.37 events/hour** ($+20.7\%$ of $F_\text{opt}$, or $+0.88\%$ of $F_\text{base}$)
- $L_1$ shift of $\theta^*$ vector max: **0.32** (out of budget 5.0 → 6.4%)

**Interpretation:** in absolute terms these are small (sub-half-failure/hour swings on an optimum that's already $< 2$). In relative terms $F_\text{opt}$ can swing 20% under a $\pm 10\%$ rate change, so the *exact* post-optimization number should be reported with the band, not as a point estimate. The allocation vector $\theta^*$ is much more stable — the biggest single shift is 6.4% of the budget.

### 2. Target-level swap: $t_n \in \{\lfloor c/3 \rfloor, \lfloor c/2 \rfloor, \lfloor 2c/3 \rfloor\}$

| target rule | $F_\text{opt}$ | $\Delta$ vs $c/2$ | $L_1(\theta^*)$ shift |
|---|--:|--:|--:|
| $c/3$  | 1.30 | $-0.47$ | moderate |
| $c/2$  | 1.77 |   —     | —     |
| $2c/3$ | 3.58 | $+1.82$ | large |

**This is the most important sensitivity finding in the notebook.** Setting the target to $\lfloor 2c/3 \rfloor$ (trucks restore stations to 2/3-full) roughly **doubles $F_\text{opt}$** — from 1.77 → 3.58. Setting it to $\lfloor c/3 \rfloor$ shaves ~25% off $F_\text{opt}$ by putting the target closer to the stockout boundary for source-heavy stations. *Target level is not a knife-edge assumption, but it is meaningfully load-bearing.*

For the paper: do NOT present $F_\text{opt} = 1.77$ as the clean result. Present the **range across the three target rules** (1.30 – 3.58) and argue that $c/2$ is the operationally defensible default because (a) it symmetrically defends against both failure modes and (b) it matches what real rebalancing crews aim for. If a reviewer asks "why $c/2$?" we can point at the sensitivity table and say "we checked; here's what happens at the other rules."

This is exactly the kind of thing the prior-project feedback called out: **no tautological validation, quantify tradeoffs**. The rule of thumb "half-full target" sounds obvious but it's doing real work in the numbers, and we should be honest about that.

## Gotchas

1. **Broadway & 58 gets almost zero budget.** It's balanced ($\rho = 0.96$) and already has $F_\text{base} = 0.54$, two orders of magnitude below the other four stations. At the chosen budget it gets $\theta^* = 0.11$ and contributes essentially nothing to the improvement story. This is not a bug — it's the greedy correctly recognising that rebalancing a near-balanced station has terrible returns. Worth mentioning in the paper as a built-in robustness check: the optimizer is *not* fooled into spreading effort evenly.

2. **At very high budgets ($\Theta_\text{total} \gtrsim 30$), the allocation keeps growing on all stations but returns flatten essentially to machine precision.** The Pareto rows past index ~10 all show $F_\text{total} < 0.03$, effectively zero. Above the knee the decision is operational (can we actually dispatch that many trucks?), not analytical.

3. **The greedy's non-increasing-returns invariant passed at every step in every run** — no monotonicity violations on any perturbation, with any target rule. That's the evidence that the greedy is actually solving the discrete separable problem exactly, not just approximating it.

4. **Closure fraction is still ~2%** (from notebook 01). This entire analysis treats the 5 stations as independent queues with exogenous $(\lambda, \mu)$. In principle, aggressive rebalancing at one station shouldn't perturb the others' rates much at our cluster's closure level — but it's worth flagging as a limitation: a truly closed-loop simulation might catch second-order effects (a reset at station A instantly increases bikes-in-transit to B, which transiently raises $\mu_B$). The closed-form model can't see that.

## Reproducibility

- `uv run pytest tests/` → 47 passing (32 CTMC + 15 optimization).
- `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_rebalancing_optimization.ipynb` → clean, ~20 s end-to-end.
- Outputs: `data/processed/rebalancing_results.csv` (5 rows, per-station optima) and `data/processed/pareto_frontier.csv` (21 rows, budget sweep).
- Notebook is rebuilt from `notebooks/_build_notebook_02.py` — same pattern as notebook 01.
