# SYS 3060 Final Project — Citi Bike Station Occupancy Optimisation

**Authors.** Spencer Arnold, Paris Phan, Garrett Uthlaut.
**Course.** SYS 3060 — Stochastic Decision Models, University of Virginia, Spring 2026.

# Citation
This readme was written with assistance from Claude Opus 4.7 to document the codebase

## What this project does

We model a five-station Midtown Manhattan Citi Bike cluster as a set of independent finite-capacity continuous-time Markov chains (M/M/1/K queues) extended with a Poisson truck-rebalancing reset, then solve the budget-constrained allocation problem of distributing a total truck-visit-rate budget across the cluster to minimise long-run user failures (stockouts + dockblocks). At an operating budget of five truck-visits per hour, the optimal allocation reduces cluster failures from **41.87 → 1.77 events per hour (a 95.8% reduction)**; the Pareto knee falls at 5.71 truck-visits per hour.

Full methodology, results, sensitivity analysis, validation, and limitations are in the report.

## Submitted deliverables

| File | What it is |
|---|---|
| `report/report.pdf` | **Final report** (SIEDS IEEE conference format). Compile from `report/report.tex` on Overleaf. |
| `report/Presentation.pdf` | Slide deck used for the in-class presentation. |
| `report/report.tex` | LaTeX source for the report. |
| `report/figures/` | Figures used in the report (extracted from notebooks 01 and 02). |
| `Project_Specifications.md` | Original problem brief from the course. |

## Repository layout

```
sys-3060-final-project/
├── README.md                     ← you are here
├── Project_Specifications.md     ← original problem brief
├── pyproject.toml, uv.lock       ← Python environment (managed by uv)
├── src/                          ← reusable modules (imported by notebooks + tests)
│   ├── ctmc.py                       ← M/M/1/K closed form + truck-reset extension
│   ├── optimization.py               ← greedy budget allocator + Pareto frontier + knee detector
│   ├── data.py                       ← Citi Bike S3 + GBFS loaders, rate estimation
│   └── geo.py                        ← haversine distance, nearest-neighbour utilities
├── notebooks/                    ← analysis pipeline (run in order)
│   ├── 01_exploration_and_base_model.ipynb   ← data load, cluster selection, GOF, base-model F_n
│   ├── 02_rebalancing_optimization.ipynb     ← rebalancing extension, optimisation, Pareto, sensitivity
│   ├── 01_NOTES.md, 02_NOTES.md              ← decisions, gotchas, reviewer-facing context
│   └── _build_notebook*.py                   ← regenerate the .ipynb files from script form
├── tests/                        ← 47 pytest tests (see Validation below)
│   ├── test_ctmc.py                  ← 32 tests on the CTMC math (closed form, generator, monotonicity)
│   └── test_optimization.py          ← 15 tests on the greedy + Pareto solver
├── data/
│   ├── raw/                          ← cached Citi Bike trip zip + GBFS JSON (auto-downloaded; not checked in)
│   └── processed/                    ← base_model_results.csv, rebalancing_results.csv, pareto_frontier.csv
└── report/                       ← LaTeX report + supporting assets
    ├── report.tex                    ← SIEDS-format manuscript (8 sections)
    ├── IEEEtran.cls                  ← IEEE conference template
    ├── figures/                      ← four PNGs extracted from the notebooks
    ├── Presentation.pdf              ← slide deck
    └── course_requirements.md        ← submission rubric
```

## Reproducing the results from a clean clone

The data is **not** checked into the repo (the December 2021 trip zip is ~930 MB); both the trip data and the GBFS station info are auto-downloaded on first run from public sources.

```sh
# 1. Install dependencies (one-time)
uv sync

# 2. Run the test suite (validates the math + the solver)
uv run pytest                            # → 47 passed

# 3. Execute the analysis pipeline (regenerates data/processed/*.csv and figure outputs)
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_exploration_and_base_model.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/02_rebalancing_optimization.ipynb

# 4. Rebuild the report (Overleaf or local TeX)
cd report && pdflatex report.tex && pdflatex report.tex
```

Notebook 01 takes a few minutes on first run (downloads the 930 MB Dec 2021 trip zip into `data/raw/`); subsequent runs are cached and complete in under a minute. Notebook 02 takes about 20 seconds.

## Headline numbers (for quick reference)

| Quantity | Value |
|---|---|
| Cluster size | 5 stations, total capacity 349 docks |
| Data window | December 2021 weekdays, 07:00–10:00, 69 hours of exposure |
| Events in window | 8,093 |
| Base failure rate F_base | 41.87 events/hour (~126 per weekday morning) |
| Operating budget Θ_total | 5.0 truck-visits per hour |
| Optimal failure rate F_opt | 1.77 events/hour (95.8% reduction) |
| Pareto knee | Θ = 5.71, F = 1.20 (97.1% reduction) |
| Rate-perturbation sensitivity | F_opt swings ±0.37/h under ±10% λ, μ shocks |
| Target-rule sensitivity | F_opt ∈ [1.41, 3.58] across t = ⌊c/3⌋, ⌊c/2⌋, ⌊2c/3⌋ |

## Data sources

- Citi Bike trip history: <https://citibikenyc.com/system-data> (December 2021 archive)
- General Bikeshare Feed Specification (per-station capacities): <https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_information.json>


## Citations
This readme was generated by claude :)