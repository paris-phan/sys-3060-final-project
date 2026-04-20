# 01 — Notebook decisions, gotchas, things to double-check

**Last rebuild:** 2026-04-19.

## Decisions

- **Data window:** September 2024 only (per project memory). Zip is ~930 MB, cached in `data/raw/`. The zip contains five inner CSVs of ~195 MB each; `src.data.load_trips` handles both flat and nested-zip layouts.
- **Morning window:** `[07:00, 10:00)` local time, weekdays only (Mon–Fri). Exposure time = (# distinct weekdays in the month) × 3 h. For Sept 2024 that's 21 × 3 = 63 h.
- **Eligibility:** station must have ≥100 events (withdraws + deposits) in the window *and* capacity ≥ 1 in GBFS *and* both `lam > 0` and `mu > 0`. Drops ~550 of ~2,100 GBFS-matched stations — low-volume peripheral stations.
- **Seed ranking:** `|log(rho_hat)|` where `rho_hat = mu / lam`. Symmetric in source vs. sink.
- **Cluster construction:** each of the top 40 seeds expanded with its 2–3 nearest geographic neighbours within 500 m (haversine). Clusters with fewer than 3 members are dropped. Duplicates removed on the frozen station-id tuple.
- **Cluster score:** `mean(|log rho|) × total events` — balances how imbalanced the cluster is against how much of it is going on.
- **CTMC math:** `src/ctmc.py::stationary_distribution` uses log-sum-exp normalization instead of the direct `rho^k (1-rho) / (1 - rho^{c+1})` formula. The direct form overflows for the rho values we actually see (one cluster station has `rho ≈ 26`, so `rho^{c+1}` is ~10^30 which is fine in float64, but at larger c or more extreme rho it blows up; log-sum-exp is free and keeps the code safe).
- **Inter-arrivals for GOF:** same-day only. A gap from 09:58 Mon to 07:03 Tue is *not* a sample from the morning-rush arrival process; including it would bias the tail and guarantee rejection on spurious grounds.
- **Chi-squared:** equal-probability bins (5–10 depending on n), `ddof=1` because the rate is estimated from the sample. Skipped for n < 30.

## Gotchas I hit

1. **Trip `station_id` ≠ GBFS `station_id`.** The trip CSVs use strings like `5500.07`; GBFS calls that field `short_name` and reserves `station_id` for an opaque internal id (UUID or 19-digit number). Merge on GBFS's `short_name`, not `station_id`. `src.data.fetch_station_info` now renames `short_name → station_id` (and the opaque id → `gbfs_station_id`) so the rest of the pipeline can merge cleanly.
2. **Capacity = 0 in GBFS.** A handful of GBFS rows have `capacity=0` — ghost/placeholder stations. Filtered out; otherwise they produce a degenerate 1-state chain.
3. **Trip span crosses the month boundary.** Sept 2024 zip contains trips that *ended* as late as Sept 30 23:59 but some started Aug 31 late evening (a trip started in August ends in September → included in the zip). Not a problem for rate estimation (we only use timestamps that fall in the weekday 7–10 am window), but worth knowing — the total row count exceeds the notional "September" volume.

## Things you should double-check

- **The top cluster failed the exponential fit on every stream** (chi² p < 0.01, KS p ≤ 0.02 for almost all station × event-type pairs). This is the most important methodological finding in Part 2. Options:
  - Narrow the window to `[08:00, 09:30)` — the peak of the peak — and see whether stationarity holds there. Edit `START_HOUR`/`END_HOUR` and rerun. This trades sample size for less within-window drift.
  - Accept that Poisson is an approximation, present the CTMC results anyway, and flag the mis-specification in the paper's limitations section. The CTMC's long-run failure rate still makes sense as a *model* of the system under the stationarity assumption; we just can't claim the data supports it cleanly. Given the rigor rule against tautological validation, this is a legitimate thing to report rather than hide.
  - Pick a different cluster where rates are less extreme. Cluster rank 0 is the *most* imbalanced, so it's the worst case for stationarity. Try `SELECTED_CLUSTER_RANK = 3` or similar — intermediate imbalance with high volume may fit Exp better.
  - My recommendation: run at least one sensitivity pass across ranks 0–2 and narrow window, and report the three as a robustness check. Keep the headline cluster pinned once you've chosen it.
- **GBFS capacities are present-day, not Sept-2024-historical** (see `reference_data_sources.md`). For stations whose capacity has changed since Sept 2024 the base model is using a slightly-off `c`. Not fixable with public data — worth a one-line limitation in the paper.
- **One station in the selected cluster has n=14 withdrawals** (5 St & Market St) — passes the aggregate 100-event filter because deposits dominate. That is real — it's a pure sink station during the morning. But n=14 means the withdrawal-rate MLE is barely meaningful. If this station ends up in the final headline cluster, consider reporting `lam` with a bootstrap CI.
- **`F_base = 42.0 events/hour` over the selected 4-station cluster** → ~126 failed user events per weekday morning, driven mostly by two sink stations running at `rho` ≈ 25. This is the benchmark the reallocation in notebook 02 has to beat. Watch for that number to appear in the capacity-reallocation results.

## Reproducibility

- `uv run pytest tests/test_ctmc.py` — 14 passing.
- `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploration_and_base_model.ipynb` — runs clean from a fresh kernel.
- Notebook is rebuilt from `notebooks/_build_notebook.py`; if you edit cells interactively and want the changes permanent, re-run the builder or drop the builder and edit the .ipynb directly. Keeping both is fine for now since the notebook source is relatively stable.

## Routing-aware re-rank (2026-04-20)

### Why we re-ranked

The original Part 1 ranked stations by `|log(mu/lam)| × volume` and expanded each seed via its 2–3 nearest geographic neighbours within 500 m. The top cluster came back as **Dock 72 Way & Market St + Flushing Ave & Vanderbilt Ave + Clinton Ave & Flushing Ave + 5 St & Market St** — four stations in the Brooklyn Navy Yard, all with `rho ∈ [1.3, 26]`, i.e. **all sinks**. No source stations. That's a dead cluster for the reallocation story: every member wants more dock capacity, there's nobody to pull from, and the interesting tradeoff (source stockouts vs sink dockblocks) disappears. The previous `F_base = 42.0/h` was ~100% dockblocks, 0% stockouts — a flat benchmark.

### What we changed

1. **Three-way station classification** by `rho = mu / lam`: `source` if `rho < 0.7`, `sink` if `rho > 1.5`, `balanced` otherwise. Of the 1,542 eligible stations, Sept 2024 gives us 675 sources, 311 sinks, 556 balanced.
2. **Trip-routing matrix** built by `groupby(start_id, end_id)` on the 7–10am weekday slice: 171,647 non-zero flows over 595,507 intra-universe trips.
3. **Seed on sinks only** (20 biggest by `(lam+mu) × |log rho|`), then for each seed pull the **top-3 origins by trip count into the seed**. Require at least one of those origins to be a real source (`rho < 0.7`); else skip.
4. Optionally add one nearby (`≤ 1 km`) secondary sink that shares origin inflow with the seed. Cap at 5 members.
5. Score by `total_events × balance_factor × mean(|log rho|)` where `balance_factor = min(V_src, V_snk) / max(V_src, V_snk) ∈ (0, 1]` — the critical fix. The old scorer did not penalize monocultures of sinks or sources.
6. Hard gates: `n_source ≥ 1` **and** `n_sink ≥ 1` **and** `total_events ≥ 2000`.

### Threshold choices and why

- `SOURCE_RHO_MAX = 0.7`, `SINK_RHO_MIN = 1.5`. Symmetric on the log-scale-ish (`|log 0.7| ≈ 0.36`, `|log 1.5| ≈ 0.41`), and leaves a fat-enough `balanced` middle (36% of stations) that the tails are clearly unbalanced rather than just noisy.
- `MIN_CLUSTER_EVENTS = 2000`. Over the 63-hour exposure window, that's ≥ ~32 events/hour across the cluster — enough that per-station rate MLEs aren't dominated by shot noise.
- `K_ORIGINS = 3`. With the seed that gives a 4-station cluster, inside the "3–4 station" target in `Project_Specifications.md`. The optional nearby-sink step can push to 5.
- `N_SINK_SEEDS = 20`. Didn't need to sweep — 11 clusters passed all gates at the default thresholds, no relaxation required.

### Top 3 clusters from the new ranking

All three are in **Midtown West, around 10th–11th Ave between W 44th and W 50th**. That's the morning commuter funnel from the west-side residential blocks into the theatre-district/Times Square corridor — which is exactly the "high-imbalance corridor" the project spec hypothesised.

| rank | members (n_src / n_sink / n_bal) | seed sink | score | balance | events |
|------|--|--|--:|--:|--:|
| 0 | 3 src + 2 sink | Broadway & W 48 St | 7,710 | 0.79 | 11,947 |
| 1 | similar Midtown West cluster | (see top table) | — | — | — |
| 2 | similar Midtown West cluster | (see top table) | — | — | — |

(Concrete numbers for ranks 1–2 live in the notebook's `clusters_summary.head(10)` cell; they shift if thresholds are tweaked.)

### Cluster picked

Rank 0. Members:

| station | type | λ (/h) | μ (/h) | ρ | capacity |
|---|---|--:|--:|--:|--:|
| Broadway & W 48 St | sink   | 19.0 | 38.7 | 2.04 | 103 |
| W 47 St & 10 Ave   | source | 14.4 |  7.8 | 0.54 |  39 |
| W 44 St & 11 Ave   | source | 35.8 | 17.9 | 0.50 |  79 |
| W 50 St & 10 Ave   | source | 20.7 |  9.1 | 0.44 |  55 |
| W 44 St & 5 Ave    | sink   |  5.9 | 20.3 | 3.42 |  68 |

Total capacity `sum c_n = 344`. `F_base = 70.2 events/hour` ≈ 211 failed user events per weekday morning. Unlike the old cluster, the stockout column is now substantial: the three sources contribute 6.6 + 17.9 + 11.6 = 36.1 stockouts/h, alongside 19.7 + 14.4 = 34.1 dockblocks/h from the sinks. That 50/50 mix is exactly what reallocation can work on.

### Thresholds that had to be relaxed

**None.** 11 clusters passed at the default thresholds. The relaxation branch (`source_rho_max → 0.8`, then `min_events → 1500`) is unused; kept in the code as a guardrail.

### Gotcha discovered in the re-rank

**Closure fraction is low (~7% per sink).** In Midtown, each sink station draws deposits from a huge number of origins — the 3 cluster sources account for only 168/2437 (6.9%) of deposits into Broadway & W 48 St and 97/1279 (7.6%) into W 44 St & 5 Ave. That does not invalidate the independent-station CTMC, but it does mean the "network effect" baked into each `mu_n` is driven mostly by origins *outside* the cluster. For the paper: this is a good thing to explicitly acknowledge — the CTMC is treating each station as a black-box queue with exogenous rates, and the exogeneity is *real* here (97% of deposits do come from outside the cluster). It is also a limitation: a policy experiment that involves *moving docks between the cluster members* implicitly assumes those outside-cluster arrivals keep their current rate, which is a clean counterfactual under the independence assumption but would need revisiting for any larger-scale reallocation.

### Changes to downstream parts

- Part 2 now reads the new cluster via the same `SELECTED_CLUSTER_RANK` / `SELECTED_STATION_IDS` interface — no edits required except swapping the display column `direction` → `type`.
- Part 3 is unchanged (it always read `(lam, mu, c)` off the selected table) and writes the same CSV schema.
- All 14 `test_ctmc.py` tests still pass.
- Exponential GOF still rejects on most (station, event-type) streams in the new cluster. Same Poisson-misspecification story as before; Midtown morning-rush non-stationarity is genuinely present. Keep this as the paper's headline limitation and present the CTMC results as a model-under-assumption, not data-validated-distribution.
