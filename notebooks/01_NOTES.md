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
