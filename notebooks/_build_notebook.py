"""Build notebooks/01_exploration_and_base_model.ipynb programmatically.

Run with:  uv run python notebooks/_build_notebook.py
"""

from pathlib import Path

import nbformat as nbf

NB = nbf.v4.new_notebook()
cells = []

def md(src): cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
md(r"""# 01 — Exploration and Base CTMC Model

SYS 3060 final project — **NYC Citi Bike station capacity reallocation.**

This notebook does three things:

1. **Surface candidate clusters** of 3–4 neighbouring stations with high directional imbalance during the weekday morning rush.
2. **Validate the exponential inter-arrival assumption** that the birth–death CTMC requires, using Q-Q plots, chi-squared, and KS tests on each station × event-type stream.
3. **Compute the base model** — per-station stationary distributions $\pi_k(n)$, stockout/dockblock probabilities, failure rate $F_n$, and the total baseline $F_{\text{base}} = \sum_n F_n$.

All modelling math lives in `src/ctmc.py`; IO and rate estimation in `src/data.py`; geographic helpers in `src/geo.py`. The next notebook (`02_capacity_reallocation.ipynb`) imports from the same package and starts from the CSV this notebook writes.

Data window: **December 2021, weekday mornings 07:00–10:00 local time.**
""")


code(r"""%load_ext autoreload
%autoreload 2

import logging
import sys
from pathlib import Path

# Make the project root importable regardless of CWD.
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")

from src import ctmc, data as dataio, geo

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- global knobs -----------------------------------------------------------
YEAR_MONTH       = "202112"   # December 2021
START_HOUR       = 7          # inclusive
END_HOUR         = 10         # exclusive -> 3-hour window
MIN_EVENTS       = 100        # per-station minimum; drops noisy low-volume stations

# Imbalance thresholds for station classification (rho = mu / lam).
SOURCE_RHO_MAX   = 0.7        # rho below this -> 'source' (runs dry)
SINK_RHO_MIN     = 1.5        # rho above this -> 'sink'   (overfills)

# Cluster-assembly knobs.
N_SINK_SEEDS     = 20         # seed count: top sinks by volume * |log rho|
K_ORIGINS        = 3          # top-K origin stations feeding into each seed sink
MAX_CLUSTER_SIZE = 5
MIN_CLUSTER_EVENTS = 2000     # gate: total events across cluster members
NEARBY_SINK_KM   = 1.0        # radius for optional extra-sink inclusion

TOP_CLUSTERS     = 10         # rows to show in the ranked candidate table

print("root:", ROOT)
print("raw :", RAW_DIR)
""")

# ---------------------------------------------------------------------------
# Part 1 — Candidate clusters
# ---------------------------------------------------------------------------
md(r"""## Part 1 — Surface candidate clusters (routing-aware)

The previous revision picked clusters by geographic nearest-neighbours of the most-imbalanced seed. That surfaced four sinks clustered at the Brooklyn Navy Yard with no source stations — a cluster that can't tell a reallocation story because there's no one running dry for the extra docks to help. See `01_NOTES.md` section "Routing-aware re-rank" for the post-mortem.

Replacement pipeline:

1. **Classify** every eligible station by $\rho_n = \hat\mu_n / \hat\lambda_n$:
   * `source` if $\rho < $ `SOURCE_RHO_MAX` ($0.7$) — runs dry, withdrawals dominate.
   * `sink`   if $\rho > $ `SINK_RHO_MIN` ($1.5$) — overfills, deposits dominate.
   * `balanced` in between — not seed material but available as filler.
2. **Build the trip-routing matrix** on the already-filtered 7–10 AM weekday trips: counts of trips for every observed `(start_id, end_id)` pair. This is what makes the clustering "routing-aware" — cluster members are stations *actually connected by commuter flow*, not just geographically close.
3. **Seed on top sinks** by volume $\times$ imbalance: $(\hat\lambda + \hat\mu) \cdot |\log\rho|$ for all `sink` stations; take the top `N_SINK_SEEDS` = 20.
4. **Assemble each cluster** by picking the top `K_ORIGINS` = 3 stations feeding the seed (by trip count into the seed), requiring at least one of them to have $\rho < 0.7$. Optionally add 1 nearby sink within 1 km that shares source inflow. Cap at `MAX_CLUSTER_SIZE` = 5. Require $\geq 1$ source, $\geq 1$ sink, and total cluster events $\geq$ `MIN_CLUSTER_EVENTS` = 2000.
5. **Score clusters** with a balance-aware composite
$$
\text{score} = (\text{total cluster events}) \cdot \underbrace{\frac{\min(V_{\text{src}}, V_{\text{snk}})}{\max(V_{\text{src}}, V_{\text{snk}})}}_{\text{balance factor} \in (0, 1]} \cdot \overline{|\log\rho_n|}.
$$
The balance factor kills the old failure mode where a cluster is 95% sink traffic: for reallocation to do anything, there must be docks to move *from* under-used sinks *to* over-used sources (or the reverse). If sources and sinks are mismatched 10:1 in volume, the reallocation upside shrinks.
""")

code(r"""trips = dataio.load_trips(YEAR_MONTH, RAW_DIR)
stations_gbfs = dataio.fetch_station_info(RAW_DIR)
print(f"trips      : {len(trips):,} rows")
print(f"trip span  : {trips['started_at'].min()}  ->  {trips['ended_at'].max()}")
print(f"stations   : {len(stations_gbfs):,} in GBFS")
print(f"capacities : median={stations_gbfs['capacity'].median()}, "
      f"p90={int(stations_gbfs['capacity'].quantile(.9))}, "
      f"max={int(stations_gbfs['capacity'].max())}")
""")

code(r"""# Per-station rates in the 7-10am weekday window.
rates = dataio.station_rates(trips, START_HOUR, END_HOUR)
print(f"exposure       : {rates.attrs['exposure_hours']} hours "
      f"({rates.attrs['n_weekdays']} weekdays x {END_HOUR-START_HOUR}h)")
print(f"stations w/ trips in window : {len(rates):,}")

# Join capacities / coords from GBFS. Trip CSVs use the GBFS `short_name`
# as their station_id; `src.data.fetch_station_info` standardizes on that
# encoding so a direct merge on `station_id` is one-to-one.
stations = (
    rates.merge(stations_gbfs, on="station_id", how="left", validate="one_to_one")
          .dropna(subset=["lat", "lng", "capacity"])
          .astype({"capacity": "int64"})
          .reset_index(drop=True)
)
print(f"after joining GBFS         : {len(stations):,}")

# Capacity >= 1 (GBFS includes a handful of cap=0 'ghost' stations that aren't
# physically docked; they'd produce a degenerate 1-state chain).
stations = stations.loc[stations["capacity"] >= 1].reset_index(drop=True)
print(f"after capacity >= 1        : {len(stations):,}")

# Volume filter.
stations = stations.loc[stations["n_events"] >= MIN_EVENTS].reset_index(drop=True)
print(f"after n_events >= {MIN_EVENTS}     : {len(stations):,}")

# Require lam and mu both > 0 so log(rho) is finite.
stations = stations.loc[(stations["lam"] > 0) & (stations["mu"] > 0)].reset_index(drop=True)
print(f"after lam,mu > 0           : {len(stations):,}")

# Imbalance metric on the log scale -> symmetric in source vs. sink.
stations["log_imbalance"] = np.log(stations["rho_hat"])
stations["abs_log_imbalance"] = stations["log_imbalance"].abs()

# Three-way classification by rho. `balanced` stations are kept in the
# universe (they can appear as origins in the routing matrix and contribute
# to volume) but they are not seed material.
def _classify(rho):
    if rho < SOURCE_RHO_MAX:
        return "source"
    if rho > SINK_RHO_MIN:
        return "sink"
    return "balanced"

stations["type"] = stations["rho_hat"].apply(_classify).astype("category")
counts = stations["type"].value_counts().to_dict()
print(f"classification: source={counts.get('source',0)}, "
      f"balanced={counts.get('balanced',0)}, sink={counts.get('sink',0)}")

stations.sort_values("abs_log_imbalance", ascending=False).head(10)[
    ["station_id", "name", "capacity", "lam", "mu", "rho_hat", "log_imbalance", "n_events", "type"]
]
""")

code(r"""# ---- Build the trip-routing matrix on the 7-10am weekday slice. ----
# This is a groupby on the already-in-memory `trips` DataFrame -- no re-read.
s = trips["started_at"]
win = (s.dt.dayofweek < 5) & (s.dt.hour >= START_HOUR) & (s.dt.hour < END_HOUR)
trips_win = trips.loc[win, ["start_station_id", "end_station_id"]]

routing = (
    trips_win.groupby(["start_station_id", "end_station_id"], observed=True)
             .size().rename("trip_count").reset_index()
             .rename(columns={"start_station_id": "start_id",
                              "end_station_id":   "end_id"})
)
# Restrict to our filtered universe of eligible stations. Trips to/from
# dropped stations still represent real demand but we can't build a CTMC
# model of them, so they don't belong in cluster assembly.
universe = set(stations["station_id"])
routing = routing.loc[routing["start_id"].isin(universe)
                       & routing["end_id"].isin(universe)].reset_index(drop=True)
print(f"routing matrix: {len(routing):,} nonzero (start, end) pairs "
      f"over {len(universe):,} stations")
print(f"total intra-universe trips in window: {int(routing['trip_count'].sum()):,}")
""")

code(r"""# ---- Cluster assembly: seed on top sinks, pull in top origins via routing. ----
# Returns either a cluster dict or None. A cluster is a list of station_ids
# containing the seed sink plus up to K_ORIGINS top origin stations, and
# optionally one nearby secondary sink.
stations_by_id = stations.set_index("station_id")

def _assemble_cluster(seed_id, routing, stations_by_id,
                      k_origins=K_ORIGINS, max_size=MAX_CLUSTER_SIZE,
                      nearby_sink_km=NEARBY_SINK_KM,
                      source_rho_max=SOURCE_RHO_MAX):
    '''Build a cluster around `seed_id` (a sink) using routing data.

    Returns None if no origin has rho < source_rho_max or the cluster ends
    up without a proper source (ensures >= 1 source guaranteed).
    '''
    # Top k_origins stations that feed this seed sink by trip count.
    inflows = routing.loc[routing["end_id"] == seed_id]
    if inflows.empty:
        return None
    inflows = inflows.sort_values("trip_count", ascending=False)

    # Filter to origins that exist in our universe (they do, since routing is
    # already restricted) and that aren't the seed itself.
    inflows = inflows.loc[inflows["start_id"] != seed_id]
    top_origins = inflows["start_id"].head(k_origins).tolist()

    # Require at least one top origin to be a proper source (rho < 0.7).
    origin_types = stations_by_id.loc[top_origins, "type"]
    if not (origin_types == "source").any():
        return None

    members = [seed_id] + top_origins
    # Optional: one nearby sink (within nearby_sink_km) that shares an origin
    # with the seed. Helps bulk up clusters that are too thin.
    if len(members) < max_size:
        seed_row = stations_by_id.loc[seed_id]
        other_sinks = stations_by_id.loc[
            (stations_by_id["type"] == "sink")
            & (~stations_by_id.index.isin(members))
        ].copy()
        if len(other_sinks):
            other_sinks["dist_m"] = geo.haversine_m(
                float(seed_row["lat"]), float(seed_row["lng"]),
                other_sinks["lat"].to_numpy(), other_sinks["lng"].to_numpy(),
            )
            nearby = other_sinks.loc[other_sinks["dist_m"] <= nearby_sink_km * 1000]
            if len(nearby):
                # Score by how many trips the already-picked origins send
                # into each candidate nearby sink.
                overlap = routing.loc[
                    routing["start_id"].isin(top_origins)
                    & routing["end_id"].isin(nearby.index)
                ]
                if len(overlap):
                    overlap_count = (overlap.groupby("end_id")["trip_count"]
                                            .sum().sort_values(ascending=False))
                    for nid in overlap_count.index:
                        if nid in members:
                            continue
                        members.append(nid)
                        if len(members) >= max_size:
                            break

    return members


def _cluster_record(members, stations_by_id, routing):
    sub = stations_by_id.loc[members].reset_index()
    src_mask = sub["type"] == "source"
    snk_mask = sub["type"] == "sink"
    n_source = int(src_mask.sum())
    n_sink   = int(snk_mask.sum())
    total_events = int(sub["n_events"].sum())
    # Volumes (events/hour) split by type.
    v_source = float(sub.loc[src_mask, ["lam", "mu"]].sum(axis=1).sum())
    v_sink   = float(sub.loc[snk_mask, ["lam", "mu"]].sum(axis=1).sum())
    # Balance factor in (0, 1]. By construction n_source >= 1 and n_sink >= 1,
    # but a balanced-only filler could still leave v_source or v_sink == 0, so
    # guard.
    balance = (min(v_source, v_sink) / max(v_source, v_sink)
               if v_source > 0 and v_sink > 0 else 0.0)
    mean_abs_log_rho = float(sub["abs_log_imbalance"].mean())
    score = total_events * balance * mean_abs_log_rho
    return {
        "cluster_id":       "|".join(sorted(members)),
        "seed_id":          members[0],
        "n_stations":       len(sub),
        "n_source":         n_source,
        "n_sink":           n_sink,
        "member_station_ids": list(sub["station_id"]),
        "member_names":       list(sub["name"]),
        "member_types":       list(sub["type"].astype(str)),
        "sum_events":         total_events,
        "source_volume":      v_source,
        "sink_volume":        v_sink,
        "balance_factor":     balance,
        "mean_abs_log_rho":   mean_abs_log_rho,
        "score":              score,
        "sum_capacity":       int(sub["capacity"].sum()),
        "min_rho":            float(sub["rho_hat"].min()),
        "max_rho":            float(sub["rho_hat"].max()),
    }


def _build_clusters(source_rho_max, min_events, n_sink_seeds, stations, routing):
    # Recompute classification if source_rho_max was relaxed.
    s_local = stations.copy()
    s_local["type"] = s_local["rho_hat"].apply(
        lambda r: "source" if r < source_rho_max
                  else ("sink" if r > SINK_RHO_MIN else "balanced")
    ).astype("category")
    sbid = s_local.set_index("station_id")

    sinks_local = s_local.loc[s_local["type"] == "sink"].copy()
    sinks_local["sink_score"] = (
        (sinks_local["lam"] + sinks_local["mu"]) * sinks_local["abs_log_imbalance"]
    )
    seeds = (sinks_local.sort_values("sink_score", ascending=False)
                        .head(n_sink_seeds)["station_id"].tolist())

    raw = []
    for seed in seeds:
        members = _assemble_cluster(seed, routing, sbid,
                                    source_rho_max=source_rho_max)
        if members is None:
            continue
        sub = sbid.loc[members]
        if not ((sub["type"] == "source").any() and (sub["type"] == "sink").any()):
            continue
        if int(sub["n_events"].sum()) < min_events:
            continue
        raw.append(members)

    # Deduplicate on the sorted station-id set.
    seen, unique = set(), []
    for m in raw:
        key = tuple(sorted(m))
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)
    records = [_cluster_record(m, sbid, routing) for m in unique]
    return pd.DataFrame(records)


# First try at the stated thresholds. Fall back with explicit warning if empty.
clusters_summary = _build_clusters(
    source_rho_max=SOURCE_RHO_MAX,
    min_events=MIN_CLUSTER_EVENTS,
    n_sink_seeds=N_SINK_SEEDS,
    stations=stations, routing=routing,
)
relaxations = []
if clusters_summary.empty:
    print(f"WARNING: no clusters at default thresholds "
          f"(source_rho<{SOURCE_RHO_MAX}, min_events>={MIN_CLUSTER_EVENTS}). "
          f"Relaxing source threshold to 0.8.")
    clusters_summary = _build_clusters(0.8, MIN_CLUSTER_EVENTS, N_SINK_SEEDS,
                                        stations, routing)
    relaxations.append("source_rho_max 0.7 -> 0.8")
if clusters_summary.empty:
    print(f"WARNING: still empty after relaxing rho; lowering min_events to 1500.")
    clusters_summary = _build_clusters(0.8, 1500, N_SINK_SEEDS, stations, routing)
    relaxations.append("min_events 2000 -> 1500")
if clusters_summary.empty:
    raise RuntimeError(
        "No clusters found even after relaxing thresholds. "
        "Check the trip/GBFS data or lower thresholds further."
    )

clusters_summary = (clusters_summary.sort_values("score", ascending=False)
                                    .reset_index(drop=True))
if relaxations:
    print("relaxations applied:", "; ".join(relaxations))
print(f"{len(clusters_summary)} clusters pass gates; top {TOP_CLUSTERS} by composite score:")
clusters_summary.head(TOP_CLUSTERS).drop(columns=["member_station_ids", "member_names", "member_types"])
""")

code(r"""# Per-station detail for the top candidates, so the numbers aren't opaque.
def expand_cluster_rows(row, stations):
    sub = stations.set_index("station_id").loc[row["member_station_ids"], :].reset_index()
    sub.insert(0, "cluster_rank", row.name)
    return sub[["cluster_rank", "station_id", "name", "type", "capacity",
                "lam", "mu", "rho_hat", "n_withdraw", "n_deposit", "n_events"]]

top = clusters_summary.head(TOP_CLUSTERS).reset_index(drop=True)
per_station_detail = pd.concat(
    [expand_cluster_rows(row, stations) for _, row in top.iterrows()],
    ignore_index=True,
)
per_station_detail.round(3)
""")

code(r"""# Folium map of top-3 clusters. Stations coloured by type; within each
# cluster we draw the top-5 observed trip flows as AntPath arrows whose
# thickness is proportional to trip count.
import folium
from folium.plugins import AntPath

TYPE_COLORS = {"source": "#2b6cb0",   # blue   = needs bikes
               "sink":   "#c53030",   # red    = overfills
               "balanced": "#718096"} # gray   = filler / passthrough

top3 = clusters_summary.head(3).reset_index(drop=True)
all_pts = stations.set_index("station_id").loc[
    [sid for row in top3["member_station_ids"] for sid in row]
]
center_lat = float(all_pts["lat"].mean())
center_lng = float(all_pts["lng"].mean())
m = folium.Map(location=[center_lat, center_lng], zoom_start=14, tiles="cartodbpositron")

border_palette = ["#1b9e77", "#d95f02", "#7570b3"]  # one border hue per cluster

for rank, row in top3.iterrows():
    members = row["member_station_ids"]
    sub = stations.set_index("station_id").loc[members, :].reset_index()
    border = border_palette[rank]

    # --- station markers (fill color by type, border color by cluster) ---
    for _, s in sub.iterrows():
        folium.CircleMarker(
            location=[s["lat"], s["lng"]],
            radius=5 + 1.0 * np.sqrt(s["n_events"] / 100.0),
            color=border, weight=3,
            fill=True, fill_color=TYPE_COLORS[str(s["type"])], fill_opacity=0.9,
            popup=(
                f"<b>{s['name']}</b><br>"
                f"cluster rank {rank} &middot; type={s['type']}<br>"
                f"cap={int(s['capacity'])}, lam={s['lam']:.1f}/h, "
                f"mu={s['mu']:.1f}/h, rho={s['rho_hat']:.2f}<br>"
                f"events={int(s['n_events'])}"
            ),
        ).add_to(m)

    # --- top trip flows within this cluster ---
    intra = routing.loc[routing["start_id"].isin(members)
                         & routing["end_id"].isin(members)
                         & (routing["start_id"] != routing["end_id"])]
    top_flows = intra.sort_values("trip_count", ascending=False).head(5)
    if len(top_flows):
        max_count = float(top_flows["trip_count"].max())
        by_id = sub.set_index("station_id")
        for _, f in top_flows.iterrows():
            if f["start_id"] not in by_id.index or f["end_id"] not in by_id.index:
                continue
            a = by_id.loc[f["start_id"]]
            b = by_id.loc[f["end_id"]]
            # AntPath is an animated dashed polyline; direction is obvious
            # because dashes flow from start to end. Weight scales with count.
            AntPath(
                locations=[[float(a["lat"]), float(a["lng"])],
                           [float(b["lat"]), float(b["lng"])]],
                color=border, weight=2 + 6 * (float(f["trip_count"]) / max_count),
                opacity=0.85, delay=900, dash_array=[10, 20],
                tooltip=f"{a['name']} -> {b['name']}: {int(f['trip_count'])} trips",
            ).add_to(m)

# Simple legend.
legend_html = '''
<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
            background: white; padding: 8px 12px; font: 12px sans-serif;
            border: 1px solid #888; border-radius: 4px; line-height: 1.6;">
  <b>Station type</b><br>
  <span style="color:#2b6cb0;">&#9679;</span> source (rho &lt; 0.7)<br>
  <span style="color:#c53030;">&#9679;</span> sink  (rho &gt; 1.5)<br>
  <span style="color:#718096;">&#9679;</span> balanced (filler)<br>
  <b>Cluster outline</b><br>
  <span style="color:#1b9e77;">&#9632;</span> rank 0 (top)<br>
  <span style="color:#d95f02;">&#9632;</span> rank 1<br>
  <span style="color:#7570b3;">&#9632;</span> rank 2<br>
  <em>arrows: top-5 trip flows, thickness &prop; count</em>
</div>'''
m.get_root().html.add_child(folium.Element(legend_html))
m
""")

code(r"""# --- sanity check on the top-ranked cluster ---
# Closure fraction: what share of each sink's deposits come from sources
# already inside this cluster? Higher = the CTMC assumption that each
# station is an independent queue with exogenous arrivals is less of a lie
# (the sources really are the ones driving the sinks).
top_row = clusters_summary.iloc[0]
top_ids = top_row["member_station_ids"]
top_sub = stations.set_index("station_id").loc[top_ids]
top_sources = [sid for sid, t in zip(top_ids, top_row["member_types"]) if t == "source"]
top_sinks   = [sid for sid, t in zip(top_ids, top_row["member_types"]) if t == "sink"]

intra_src_to_snk = routing.loc[
    routing["start_id"].isin(top_sources)
    & routing["end_id"].isin(top_sinks)
]["trip_count"].sum()
total_into_sinks_from_universe = routing.loc[routing["end_id"].isin(top_sinks)]["trip_count"].sum()
total_into_sinks_from_any = int(top_sub.loc[top_sinks, "n_deposit"].sum())

print("=" * 72)
print(f"TOP CLUSTER SANITY CHECK  ({top_row['cluster_id'][:60]}...)")
print("=" * 72)
print(f"members           : {len(top_ids)}  "
      f"({top_row['n_source']} source + {top_row['n_sink']} sink + "
      f"{len(top_ids) - top_row['n_source'] - top_row['n_sink']} balanced)")
print(f"total events      : {top_row['sum_events']:,}")
print(f"source volume     : {top_row['source_volume']:.1f}/h  (sum lam+mu on sources)")
print(f"sink   volume     : {top_row['sink_volume']:.1f}/h  (sum lam+mu on sinks)")
print(f"balance factor    : {top_row['balance_factor']:.3f}")
print(f"mean |log rho|    : {top_row['mean_abs_log_rho']:.3f}")
print(f"composite score   : {top_row['score']:.1f}")
print()
print(f"Intra-cluster source->sink trips : {int(intra_src_to_snk):,}")
print(f"Total trips into cluster sinks (universe)  : {int(total_into_sinks_from_universe):,}")
print(f"Closure fraction (universe-scoped) : "
      f"{intra_src_to_snk / max(total_into_sinks_from_universe, 1):.3f}")
print(f"(Higher = cluster sinks are mostly fed by cluster sources "
      f"-- the independent-station approximation is less lossy.)")
print()
# Per-sink closure: share of *each* sink's deposits coming from cluster sources.
for sid in top_sinks:
    srow = top_sub.loc[sid]
    from_cluster = int(routing.loc[(routing["end_id"] == sid)
                                    & (routing["start_id"].isin(top_sources))]
                             ["trip_count"].sum())
    total_deposits = int(srow["n_deposit"])
    frac = from_cluster / total_deposits if total_deposits else 0.0
    print(f"  {str(srow['name'])[:40]:40s}  "
          f"deposits from cluster sources: {from_cluster:>4d} / {total_deposits:>4d}  "
          f"({frac:.1%})")
""")


# ---------------------------------------------------------------------------
# Part 2 — Exponential validation
# ---------------------------------------------------------------------------
md(r"""## Part 2 — Validate the exponential inter-arrival assumption

The CTMC model assumes withdrawal and deposit streams at each station are Poisson with constant rate in the 07:00–10:00 window — equivalently, that inter-arrival times are iid Exponential($\lambda$). We test this per (station, event-type) with:

* **Q-Q plot** vs. Exp(MLE rate).
* **Histogram** with the fitted exponential PDF overlaid.
* **Chi-squared** goodness-of-fit on equal-probability bins (ddof=1 because we estimated the rate from the sample).
* **Kolmogorov–Smirnov** one-sample test against Exp.

Inter-arrivals that cross a day boundary (i.e. gap from Monday 09:58 to Tuesday 07:03) are discarded — that gap is not a sample from the morning-rush arrival process we are modelling. This is important rigour: failing to do it biases the interval distribution toward long tails and almost guarantees a rejected fit.
""")

code(r"""# Cluster selection. Default: the top-ranked candidate. Override by editing below.
SELECTED_CLUSTER_RANK = 0          # 0 = best-scoring cluster from Part 1
# Or set SELECTED_STATION_IDS directly to pin a specific set, e.g.
# SELECTED_STATION_IDS = ["6140.05", "6948.10", "6926.01"]
SELECTED_STATION_IDS = None

if SELECTED_STATION_IDS is None:
    SELECTED_STATION_IDS = clusters_summary.iloc[SELECTED_CLUSTER_RANK]["member_station_ids"]

selected = (
    stations.set_index("station_id").loc[SELECTED_STATION_IDS, :].reset_index()
)
print(f"Selected cluster (rank={SELECTED_CLUSTER_RANK}): {len(selected)} stations")
selected[["station_id", "name", "type", "capacity", "lam", "mu", "rho_hat", "n_events"]]
""")

code(r"""def exp_gof(interarrivals_sec):
    '''MLE rate plus chi-squared and KS against Exp(rate).

    Returns dict with n, rate (1/mean, in 1/sec), chi2_stat/p, ks_stat/p.
    Chi-squared uses ~10 equal-probability bins under H0; ddof=1 for the
    estimated rate. We skip the test (return NaNs) when n < 30 because the
    bin count and the chi-squared asymptotics degrade rapidly.
    '''
    x = np.asarray(interarrivals_sec, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    out = {"n": int(len(x)), "rate": np.nan, "chi2_stat": np.nan,
           "chi2_p": np.nan, "ks_stat": np.nan, "ks_p": np.nan}
    if len(x) < 30:
        return out
    mean = x.mean()
    if mean <= 0:
        return out
    rate = 1.0 / mean                 # MLE for Exp
    out["rate"] = rate

    # KS first (no binning choices).
    ks = stats.kstest(x, "expon", args=(0.0, mean))
    out["ks_stat"] = float(ks.statistic)
    out["ks_p"]    = float(ks.pvalue)

    # Chi-squared on equal-probability bins under H0.
    n_bins = max(5, min(10, int(len(x) / 5)))   # Cochran's rule of thumb: >=5 expected/bin
    probs = np.linspace(0, 1, n_bins + 1)
    # Inverse CDF of Exp: -ln(1-u)/rate. Drop 0 and 1 endpoints.
    edges = -np.log(1 - probs[1:-1]) / rate
    edges = np.concatenate(([0.0], edges, [np.inf]))
    obs, _ = np.histogram(x, bins=edges)
    expected = np.full(n_bins, len(x) / n_bins)
    chi2 = stats.chisquare(obs, f_exp=expected, ddof=1)   # ddof=1 (fitted rate)
    out["chi2_stat"] = float(chi2.statistic)
    out["chi2_p"]    = float(chi2.pvalue)
    return out
""")

code(r"""# For each station x event-type: collect inter-arrivals, run GOF, plot.
EVENT_TYPES = ["withdraw", "deposit"]
gof_rows = []
streams = {}  # (station_id, kind) -> inter-arrival array

n_stations = len(selected)
fig, axes = plt.subplots(n_stations, 4, figsize=(16, 3.2 * n_stations),
                          squeeze=False)

for i, srow in selected.iterrows():
    sid = srow["station_id"]
    for j, kind in enumerate(EVENT_TYPES):
        ev = dataio.station_event_times(trips, sid, kind, START_HOUR, END_HOUR)
        ia = dataio.within_day_interarrivals_seconds(ev)
        streams[(sid, kind)] = ia
        res = exp_gof(ia)
        res.update(station_id=sid, name=srow["name"], kind=kind,
                   rate_per_hour=res["rate"] * 3600 if np.isfinite(res["rate"]) else np.nan)
        gof_rows.append(res)

        # --- plots -----
        ax_qq   = axes[i, 2*j + 0]
        ax_hist = axes[i, 2*j + 1]
        if len(ia) >= 30 and np.isfinite(res["rate"]):
            mean = 1.0 / res["rate"]
            stats.probplot(ia, dist="expon", sparams=(0, mean), plot=ax_qq)
            ax_qq.set_title(f"Q-Q  {srow['name'][:25]}  [{kind}]", fontsize=9)

            # Histogram truncated to 99th pct to avoid outlier smearing.
            hi = np.quantile(ia, 0.99)
            ax_hist.hist(ia[ia <= hi], bins=40, density=True, alpha=0.55, color="steelblue")
            xs = np.linspace(0, hi, 200)
            ax_hist.plot(xs, res["rate"] * np.exp(-res["rate"] * xs),
                         color="crimson", lw=2, label=f"Exp({res['rate']*3600:.1f}/h)")
            ax_hist.set_title(
                f"Hist  n={res['n']}  chi2 p={res['chi2_p']:.3f}  KS p={res['ks_p']:.3f}",
                fontsize=9,
            )
            ax_hist.legend(fontsize=8)
            ax_hist.set_xlabel("inter-arrival (sec)")
        else:
            ax_qq.set_title(f"Q-Q  (n={res['n']} insufficient)", fontsize=9)
            ax_hist.set_title(f"Hist (n={res['n']} insufficient)", fontsize=9)

plt.tight_layout()
plt.show()
""")

code(r"""# Per-station diagnostic table + pass/fail verdict.
gof = pd.DataFrame(gof_rows)
gof["pass"] = (gof["chi2_p"] > 0.05) & (gof["ks_p"] > 0.05)
display_cols = ["station_id", "name", "kind", "n", "rate_per_hour",
                "chi2_stat", "chi2_p", "ks_stat", "ks_p", "pass"]
print("Pass-fail verdict: pass iff both chi2_p > 0.05 and KS_p > 0.05.")
gof[display_cols].round(4)
""")

code(r"""fails = gof.loc[~gof["pass"]]
if len(fails):
    print("WARNING: the following (station, event_type) streams FAILED the exponential fit:")
    for _, r in fails.iterrows():
        print(f"  - {r['name']:<40s} [{r['kind']:<8s}]  "
              f"chi2_p={r['chi2_p']:.3f}  KS_p={r['ks_p']:.3f}  n={int(r['n'])}")
    print("\nConsider narrowing the window (e.g. 8:00-9:30 AM) to reduce non-stationarity")
    print("within the morning rush. Not retrying automatically -- this is a modelling choice,")
    print("not a data-cleaning step.")
else:
    print("All selected streams pass the exponential fit at alpha = 0.05.")
""")


# ---------------------------------------------------------------------------
# Part 3 — Base CTMC
# ---------------------------------------------------------------------------
md(r"""## Part 3 — Base CTMC model at current capacities

For each station in the selected cluster, with the current real capacity $c_n$ from GBFS and the estimated $(\hat\lambda_n, \hat\mu_n)$ from Part 2 (events per hour):

* compute the full stationary distribution $\pi_k(n)$;
* report stockout probability $\pi_0(n)$, dockblock probability $\pi_{c_n}(n)$;
* compute the long-run failure rate $F_n(c_n) = \hat\lambda_n \pi_0(n) + \hat\mu_n \pi_{c_n}(n)$ (events/hour).

Total baseline $F_{\text{base}} = \sum_n F_n(c_n)$ is what the reallocation in notebook 02 must beat.
""")

code(r"""# Compute per-station pi_k, pi_0, pi_c, F_n.
base_rows = []
pi_arrays = {}
for _, s in selected.iterrows():
    lam = float(s["lam"])
    mu  = float(s["mu"])
    c   = int(s["capacity"])
    rho = mu / lam
    pi  = ctmc.stationary_distribution(rho, c)
    F_n = ctmc.failure_rate(lam, mu, c)
    pi_arrays[s["station_id"]] = pi
    base_rows.append({
        "station_id": s["station_id"],
        "name": s["name"],
        "lam": lam, "mu": mu, "c": c, "rho": rho,
        "pi_0": float(pi[0]),
        "pi_c": float(pi[-1]),
        "F_n": F_n,
        "stockout_per_hour": lam * float(pi[0]),
        "dockblock_per_hour": mu * float(pi[-1]),
        "n_events": int(s["n_events"]),
    })

base = pd.DataFrame(base_rows)
F_base = float(base["F_n"].sum())

print(f"F_base (total failure rate across the cluster) = {F_base:.3f} events/hour")
print(f"  -> over the 3-hour morning window: {F_base * 3:.1f} failed user events per weekday\n")
base.round(4)
""")

code(r"""# Bar chart of pi_k per station (visualises where each station 'lives').
fig, axes = plt.subplots(1, len(selected), figsize=(4.2 * len(selected), 3.4), squeeze=False)
for ax, (_, s) in zip(axes[0], selected.iterrows()):
    pi = pi_arrays[s["station_id"]]
    ks = np.arange(len(pi))
    colors = ["#d7301f"] + ["#bdbdbd"] * (len(pi) - 2) + ["#2b8cbe"]
    ax.bar(ks, pi, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_title(f"{s['name'][:22]}\nc={int(s['capacity'])} rho={s['mu']/s['lam']:.2f}", fontsize=10)
    ax.set_xlabel("bikes at station $k$")
    ax.set_ylabel(r"$\pi_k$")
plt.tight_layout()
plt.show()
""")

code(r"""# Save the base model table for the next notebook to consume.
OUT = PROCESSED_DIR / "base_model_results.csv"
base.to_csv(OUT, index=False)
print(f"Wrote {OUT.relative_to(ROOT)} ({len(base)} rows).")
print("\nNext notebook (02_capacity_reallocation.ipynb) reads from this file and")
print(f"searches over integer allocations subject to sum(c_n) = {int(base['c'].sum())}.")
""")

md(r"""### Recap

* **Part 1** ranked candidate clusters by mean($|\log\rho|$) × volume and mapped the top 3. The selected cluster for the base model is the top-ranked candidate; override `SELECTED_CLUSTER_RANK` or `SELECTED_STATION_IDS` in the Part 2 config cell to pick a different one.
* **Part 2** produced per-(station, event-type) GOF diagnostics. Any failed stream is flagged above with a suggestion to tighten the window.
* **Part 3** saved `data/processed/base_model_results.csv` — the incumbent benchmark for the reallocation search in notebook 02.

Run `pytest tests/test_ctmc.py` to verify the closed-form helpers are still green.
""")


NB.cells = cells
NB.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}

out = Path(__file__).parent / "01_exploration_and_base_model.ipynb"
nbf.write(NB, out)
print(f"wrote {out}")
