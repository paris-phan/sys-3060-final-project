"""Geographic helpers for station clustering."""
# Citation
# Developed alongside Claude Code (Opus 4.6)

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres. Accepts scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def pairwise_haversine_m(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """N-by-N matrix of pairwise great-circle distances in metres."""
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    lat2 = lats[:, None]
    lon2 = lons[:, None]
    return haversine_m(lats[None, :], lons[None, :], lat2, lon2)


def nearest_neighbors_within(
    df: pd.DataFrame,
    seed_idx: int,
    max_neighbors: int = 3,
    radius_m: float = 500.0,
) -> list[int]:
    """Return indices of the k nearest stations to df.iloc[seed_idx] within radius.

    Excludes the seed itself. df must have 'lat' and 'lng' columns.
    """
    lats = df["lat"].to_numpy()
    lons = df["lng"].to_numpy()
    d = haversine_m(lats[seed_idx], lons[seed_idx], lats, lons)
    order = np.argsort(d)
    picks = []
    for j in order:
        if j == seed_idx:
            continue
        if d[j] > radius_m:
            break
        picks.append(int(j))
        if len(picks) >= max_neighbors:
            break
    return picks
