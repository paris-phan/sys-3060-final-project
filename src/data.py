"""Data loading, filtering, and rate-estimation helpers for Citi Bike trips.

Responsibilities:
    * Download and cache the monthly trip zip from the public S3 bucket
      (handles the nested-zip format used for recent months).
    * Load trips into a single DataFrame with parsed datetimes.
    * Fetch and cache the GBFS station_information feed.
    * Filter trips to a weekday hour window (e.g. 7:00-10:00 AM).
    * Estimate per-station withdrawal / deposit rates per hour.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

TRIPS_URL_TEMPLATE = "https://s3.amazonaws.com/tripdata/{year_month}-citibike-tripdata.zip"
TRIPS_YEARLY_URL_TEMPLATE = "https://s3.amazonaws.com/tripdata/{year}-citibike-tripdata.zip"
GBFS_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"

TRIP_COLUMNS = [
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_station_name",
    "start_station_id",
    "end_station_name",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
]


# ---------------------------------------------------------------------------
# Trip data
# ---------------------------------------------------------------------------


def _stream_download(url: str, dest: Path) -> None:
    log.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".zip.part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp.replace(dest)


def download_trip_zip(year_month: str, raw_dir: Path) -> Path:
    """Download the parent zip for the given YYYYMM into raw_dir. No-op if cached.

    Tries the monthly file first; falls back to the yearly zip if the monthly
    one is missing on S3 (older months -- e.g. 2021 -- are only published as
    a single yearly archive).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    year = year_month[:4]

    monthly = raw_dir / f"{year_month}-citibike-tripdata.zip"
    yearly  = raw_dir / f"{year}-citibike-tripdata.zip"

    # Prefer cached monthly, then cached yearly.
    for path in (monthly, yearly):
        if path.exists() and path.stat().st_size > 0:
            log.info("Cache hit: %s (%d bytes)", path.name, path.stat().st_size)
            return path

    # Neither cached; try monthly download first.
    monthly_url = TRIPS_URL_TEMPLATE.format(year_month=year_month)
    try:
        _stream_download(monthly_url, monthly)
        return monthly
    except requests.HTTPError as exc:
        if exc.response is None or exc.response.status_code != 404:
            raise
        log.info("Monthly zip not found on S3; falling back to yearly archive for %s", year)

    yearly_url = TRIPS_YEARLY_URL_TEMPLATE.format(year=year)
    _stream_download(yearly_url, yearly)
    return yearly


def _iter_csv_members(zip_path: Path) -> Iterable[tuple[zipfile.ZipFile, str]]:
    """Yield (zipfile, member_name) pairs for every CSV inside zip_path,
    including CSVs nested inside inner zips. Skips __MACOSX/ entries.
    """
    outer = zipfile.ZipFile(zip_path)
    try:
        for name in outer.namelist():
            if name.startswith("__MACOSX") or name.endswith("/"):
                continue
            lower = name.lower()
            if lower.endswith(".csv"):
                yield outer, name
            elif lower.endswith(".zip"):
                # Nested zip -- extract to temp bytes, open recursively.
                with outer.open(name) as inner_bytes:
                    inner_zf = zipfile.ZipFile(_BytesIOCompat(inner_bytes.read()))
                    for inner_name in inner_zf.namelist():
                        if inner_name.startswith("__MACOSX") or inner_name.endswith("/"):
                            continue
                        if inner_name.lower().endswith(".csv"):
                            yield inner_zf, inner_name
    finally:
        # We don't close outer here; caller consumes lazily. This is a small
        # leak but the process is short-lived.
        pass


class _BytesIOCompat:
    """Minimal BytesIO shim so zipfile can open in-memory bytes."""

    def __new__(cls, data: bytes):
        import io

        return io.BytesIO(data)


def load_trips(year_month: str, raw_dir: Path) -> pd.DataFrame:
    """Download (if needed) and load trip CSVs for the given YYYYMM.

    Uses pyarrow under the hood (via pandas) and parses the datetime columns.
    When the zip is a yearly archive (e.g. 2021), only inner members whose
    name contains the target `YYYYMM` substring are read, so memory cost
    stays at ~one-month scale. A final timestamp filter on
    `started_at.dt.year/month` guards against inner-naming variations.
    """
    zip_path = download_trip_zip(year_month, raw_dir)
    is_yearly = year_month[:4] + "-citibike-tripdata.zip" == zip_path.name
    target_year, target_month = int(year_month[:4]), int(year_month[4:])

    frames: list[pd.DataFrame] = []
    for zf, member in _iter_csv_members(zip_path):
        # When we're inside a yearly zip, skip CSVs that obviously belong to
        # other months. Use the basename so nested-zip paths don't confuse us.
        if is_yearly and year_month not in Path(member).name:
            continue
        log.info("  reading %s", member)
        with zf.open(member) as fh:
            df = pd.read_csv(
                fh,
                dtype={
                    "ride_id": "string",
                    "rideable_type": "category",
                    "start_station_name": "string",
                    "start_station_id": "string",
                    "end_station_name": "string",
                    "end_station_id": "string",
                    "member_casual": "category",
                },
                parse_dates=["started_at", "ended_at"],
                low_memory=False,
            )
        frames.append(df)
    if not frames:
        raise RuntimeError(
            f"No CSV members matched {year_month} in {zip_path}. "
            "Check the year-month string and the zip's inner layout."
        )
    trips = pd.concat(frames, ignore_index=True, copy=False)

    # Drop rows with missing station IDs or timestamps -- can't use them.
    before = len(trips)
    trips = trips.dropna(
        subset=["start_station_id", "end_station_id", "started_at", "ended_at"]
    )

    # Safety filter on actual timestamps: restrict to the target month.
    # (Protects against inner-file naming oddities in yearly zips.)
    s = trips["started_at"]
    in_month = (s.dt.year == target_year) & (s.dt.month == target_month)
    trips = trips.loc[in_month].reset_index(drop=True)

    log.info("Loaded %d trips for %s (%d dropped for missing/OOB).",
             len(trips), year_month, before - len(trips))
    return trips


# ---------------------------------------------------------------------------
# GBFS station info
# ---------------------------------------------------------------------------


def fetch_station_info(
    raw_dir: Path, url: str = GBFS_URL, force: bool = False
) -> pd.DataFrame:
    """Fetch (and cache) the GBFS station_information feed; return a flat DataFrame.

    Columns: station_id, name, lat, lng, capacity, short_name (if present).
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_dir / "station_information.json"
    if force or not cache.exists():
        log.info("Fetching GBFS %s", url)
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        cache.write_text(r.text)
    payload = json.loads(cache.read_text())
    stations = payload["data"]["stations"]
    df = pd.DataFrame(stations)
    # Lyft GBFS uses `lon`; stick with `lng` internally to match trip schema.
    if "lon" in df.columns and "lng" not in df.columns:
        df = df.rename(columns={"lon": "lng"})
    # The trip CSVs' `start_station_id` / `end_station_id` correspond to
    # GBFS's `short_name` (e.g. "5500.07"), not the opaque long `station_id`.
    # We standardize on the trip-CSV encoding and keep the GBFS id under
    # `gbfs_station_id` for reference.
    required = ["station_id", "short_name", "name", "lat", "lng", "capacity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"GBFS payload missing expected fields: {missing}")
    df = df.rename(columns={"station_id": "gbfs_station_id", "short_name": "station_id"})
    df = df[["station_id", "gbfs_station_id", "name", "lat", "lng", "capacity"]].copy()
    df["station_id"] = df["station_id"].astype("string")
    df["gbfs_station_id"] = df["gbfs_station_id"].astype("string")
    df["name"] = df["name"].astype("string")
    df["capacity"] = df["capacity"].astype("Int64")
    # A handful of stations may share a short_name if the feed has duplicates;
    # keep the first to maintain one_to_one merges downstream.
    df = df.drop_duplicates(subset=["station_id"], keep="first").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Filtering / rate estimation
# ---------------------------------------------------------------------------


def filter_weekday_window(
    trips: pd.DataFrame,
    start_hour: int,
    end_hour: int,
) -> pd.DataFrame:
    """Keep only trips whose started_at falls in [start_hour, end_hour) on a weekday.

    Mon=0 ... Fri=4. We key on started_at only here; deposits are re-filtered
    separately using ended_at in `station_rates`.
    """
    s = trips["started_at"]
    mask = (s.dt.dayofweek < 5) & (s.dt.hour >= start_hour) & (s.dt.hour < end_hour)
    return trips.loc[mask].copy()


def _count_weekdays_in_month(trips: pd.DataFrame) -> int:
    """Distinct weekday (Mon-Fri) dates spanned by this slice of trips, based on started_at."""
    dates = trips["started_at"].dt.floor("D")
    weekday = dates[trips["started_at"].dt.dayofweek < 5]
    return int(weekday.dt.date.nunique())


def station_rates(
    trips_all: pd.DataFrame,
    start_hour: int,
    end_hour: int,
) -> pd.DataFrame:
    """Per-station withdrawal / deposit rates (events per hour) in the weekday window.

    Withdrawal rate lambda_n is estimated from trips whose *started_at* lies in the
    window at station n; deposit rate mu_n is estimated from trips whose *ended_at*
    lies in the window at station n. We normalize by the total exposure time, which
    equals (# distinct weekday-dates in the month) * (window_hours).

    Returns columns: station_id, lam, mu, n_withdraw, n_deposit, n_events, rho_hat.
    """
    hours = float(end_hour - start_hour)
    if hours <= 0:
        raise ValueError("end_hour must be strictly greater than start_hour")

    # Exposure: count distinct weekday dates present in the month. We use the
    # full trip DataFrame (not just the window) to detect all weekdays present,
    # since some days may have no trips in the morning window at a given station.
    s = trips_all["started_at"]
    all_weekday_dates = s[s.dt.dayofweek < 5].dt.floor("D").dt.date.unique()
    n_weekdays = len(all_weekday_dates)
    if n_weekdays == 0:
        raise ValueError("no weekday data found in trips")
    exposure_hours = n_weekdays * hours

    # Withdrawals: started_at in window.
    start_mask = (
        (s.dt.dayofweek < 5) & (s.dt.hour >= start_hour) & (s.dt.hour < end_hour)
    )
    withdrawals = (
        trips_all.loc[start_mask]
        .groupby("start_station_id", observed=True)
        .size()
        .rename("n_withdraw")
    )

    e = trips_all["ended_at"]
    end_mask = (e.dt.dayofweek < 5) & (e.dt.hour >= start_hour) & (e.dt.hour < end_hour)
    deposits = (
        trips_all.loc[end_mask]
        .groupby("end_station_id", observed=True)
        .size()
        .rename("n_deposit")
    )

    rates = (
        pd.concat([withdrawals, deposits], axis=1)
        .fillna(0)
        .astype({"n_withdraw": "int64", "n_deposit": "int64"})
        .rename_axis("station_id")
        .reset_index()
    )
    rates["lam"] = rates["n_withdraw"] / exposure_hours
    rates["mu"] = rates["n_deposit"] / exposure_hours
    rates["n_events"] = rates["n_withdraw"] + rates["n_deposit"]
    # rho_hat only defined when lam > 0; leave NaN otherwise.
    rates["rho_hat"] = np.where(rates["lam"] > 0, rates["mu"] / rates["lam"], np.nan)
    rates.attrs["exposure_hours"] = exposure_hours
    rates.attrs["n_weekdays"] = n_weekdays
    return rates


# ---------------------------------------------------------------------------
# Event streams (for inter-arrival analysis in Part 2)
# ---------------------------------------------------------------------------


def station_event_times(
    trips_all: pd.DataFrame,
    station_id: str,
    event_type: str,
    start_hour: int,
    end_hour: int,
) -> pd.Series:
    """Sorted event timestamps for a station in the weekday morning window.

    event_type: 'withdraw' (uses started_at at start_station_id)
                or 'deposit' (uses ended_at at end_station_id).
    """
    if event_type == "withdraw":
        ts_col, id_col = "started_at", "start_station_id"
    elif event_type == "deposit":
        ts_col, id_col = "ended_at", "end_station_id"
    else:
        raise ValueError(f"event_type must be 'withdraw' or 'deposit', got {event_type!r}")
    ts = trips_all[ts_col]
    mask = (
        (trips_all[id_col] == station_id)
        & (ts.dt.dayofweek < 5)
        & (ts.dt.hour >= start_hour)
        & (ts.dt.hour < end_hour)
    )
    return trips_all.loc[mask, ts_col].sort_values().reset_index(drop=True)


def within_day_interarrivals_seconds(
    event_times: pd.Series,
) -> np.ndarray:
    """Inter-arrival times (in seconds), restricted to consecutive same-day events.

    We drop inter-arrivals that cross a day boundary, since the 3-hour window
    is closed each night -- the gap between 10am Monday and 7am Tuesday is not
    a sample from the stationary arrival process we are modeling.
    """
    if len(event_times) < 2:
        return np.array([], dtype=float)
    t = pd.Series(event_times.to_numpy())
    day = t.dt.floor("D")
    dt = t.diff().dt.total_seconds().to_numpy()
    same_day = (day.diff().dt.total_seconds() == 0).to_numpy()
    return dt[same_day & np.isfinite(dt)]
