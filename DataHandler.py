#!/usr/bin/env python3
"""
data_handler.py
===============

Builds ONE merged DataFrame for TAS1 price forecasting + battery dispatch,
and turns it into leakage-free supervised training tables.

THE CENTRAL IDEA: forecast vintages
-----------------------------------
"The weather" in a row is not one number. For a row with valid time T, we
store several columns, one per *vintage* -- how far ahead the forecast was
issued:

    temperature_2m_fc24h  ->  the value predicted for T, issued 24 h before T
    temperature_2m_fc48h  ->  the value predicted for T, issued 48 h before T
    temperature_2m_obs    ->  what actually happened at T (ERA5 hindsight)

Which one you may use depends on the HORIZON you are predicting at, not on
the row. Standing at origin t, predicting target time T = t + h:

    a `_fcNh` column for T was issued at T - N.
    It is available to you at t  <=>  T - N <= t  <=>  h <= N.

So h = 6 h may use _fc24h. h = 30 h may NOT (that forecast doesn't exist yet
at t) and must use _fc48h. `forecast_column_for_horizon()` picks the band and
`build_supervised_frame()` asserts the inequality, so leakage is a crash
rather than a silent 20% improvement in your results.

`_obs` columns are ERA5 hindsight. They exist for exactly one purpose: the
"perfect weather knowledge" counterfactual that tells you how much the
forecast error costs you. `validate_master_frame()` shouts if they're present
and `build_supervised_frame()` refuses to put them in X unless you pass
allow_hindsight=True.

DAYLIGHT SAVING
---------------
Three clocks, kept separate on purpose:

  time_utc    the index. Unambiguous, no DST, all merging happens here.
  time_aest   market time. Fixed UTC+10, DST NEVER applies. AEMO's
              SETTLEMENTDATE lives here.
  time_local  Australia/Hobart, DST-aware. Human behaviour follows the wall
              clock, so ALL calendar features are derived from this.

`to_utc_series()` handles the two DST pathologies when a source is stored in
local civil time: the repeated hour at DST end (ambiguous) and the missing
hour at DST start (nonexistent). It reports what it found instead of silently
dropping or duplicating rows.

NOTE ON PRICES
--------------
This handler deals in wholesale spot price (RRP, $/MWh) only. No network
adder, no retail margin. That is a deliberate choice -- see the note in
README_NOTES at the bottom of this file for what it does to your result.

Usage
-----
    python data_handler.py --self-test          # offline, run this first

    python data_handler.py \
        --price-csv tas_price_demand.csv \
        --start-date 2024-03-01 --end-date 2026-07-14 \
        --output-dir build

Requirements: pip install pandas numpy requests holidays pyarrow
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("data_handler")

try:
    import holidays as _holidays_lib
    _HAS_HOLIDAYS = True
except ImportError:  # pragma: no cover
    _holidays_lib = None
    _HAS_HOLIDAYS = False
    logger.warning("`holidays` not installed -- holiday flags will be False. "
                   "pip install holidays")


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegionInfo:
    region_id: str
    state_code: str
    city: str
    latitude: float
    longitude: float
    local_tz: str


NEM_REGIONS: dict[str, RegionInfo] = {
    "NSW1": RegionInfo("NSW1", "NSW", "Sydney", -33.8688, 151.2093, "Australia/Sydney"),
    "QLD1": RegionInfo("QLD1", "QLD", "Brisbane", -27.4698, 153.0251, "Australia/Brisbane"),
    "SA1": RegionInfo("SA1", "SA", "Adelaide", -34.9285, 138.6007, "Australia/Adelaide"),
    "TAS1": RegionInfo("TAS1", "TAS", "Hobart", -42.8821, 147.3272, "Australia/Hobart"),
    "VIC1": RegionInfo("VIC1", "VIC", "Melbourne", -37.8136, 144.9631, "Australia/Melbourne"),
}

MARKET_UTC_OFFSET = timedelta(hours=10)
MARKET_TZ = timezone(MARKET_UTC_OFFSET)  # fixed +10:00; DST never applies

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OPEN_ELECTRICITY_BASE_URL = "https://api.openelectricity.org.au/v4"

# Available from the Previous Runs API. Purely astronomical quantities
# (is_day) and ERA5-only derived fields (et0, sunshine_duration) are absent;
# is_day is geometry rather than weather, so it's computed locally -- no leak.
FORECAST_WEATHER_VARS: list[str] = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "dew_point_2m", "precipitation", "rain", "cloud_cover", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "wind_speed_100m",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
]

OBSERVED_WEATHER_VARS: list[str] = FORECAST_WEATHER_VARS + ["sunshine_duration", "is_day"]

OE_INTERVAL_MAX_DAYS = {"5m": 8, "1h": 32, "1d": 366}
_INTERVALS_PER_BUCKET = {"1h": 12, "1d": 288}
GENERATION_METRICS = ["power"]
MARKET_METRICS = ["demand", "flow_imports", "flow_exports", "renewable_proportion"]

# Open-Meteo weights a request by (variables x days). 14 vars x 2 leads = 28
# columns, so keep chunks modest or the free tier starts refusing.
WEATHER_CHUNK_DAYS = 60
_DEFAULT_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "*/*",
}


# ---------------------------------------------------------------------------
# Timezone / DST handling -- the part that silently ruins projects
# ---------------------------------------------------------------------------


def to_utc_series(
    s: pd.Series,
    tz: str = "market",
    fmt: Optional[str] = None,
    label: str = "timestamps",
) -> pd.Series:
    """Convert a timestamp column to tz-aware UTC, handling DST properly.

    tz:
      "market"  AEST, fixed UTC+10, no DST. AEMO's SETTLEMENTDATE. Safe.
      "utc"     already UTC.
      <IANA>    e.g. "Australia/Hobart". Civil local time, WITH DST. This is
                the dangerous one and is why this function exists.

    Two DST pathologies are handled explicitly for the IANA case:

      AMBIGUOUS (DST ends, first Sunday in April in TAS): 02:00-03:00 local
        happens twice. `ambiguous="infer"` uses the monotonic ordering of a
        regular series to tell the first pass from the second. If the series
        is irregular and inference fails, we fall back to assuming the FIRST
        (still-DST) pass and say so loudly -- that costs you at most one hour
        per year, mislabelled, and never silently deletes rows.

      NONEXISTENT (DST starts, first Sunday in October): 02:00-03:00 local
        never happens. Such stamps are shifted forward, which is what any
        sane logger would have produced anyway.
    """
    parsed = pd.to_datetime(s, format=fmt) if fmt else pd.to_datetime(s)

    if getattr(parsed.dt, "tz", None) is not None:
        return parsed.dt.tz_convert("UTC")

    if tz == "market":
        return parsed.dt.tz_localize(MARKET_TZ).dt.tz_convert("UTC")
    if tz == "utc":
        return parsed.dt.tz_localize("UTC")

    # Civil local time with DST.
    try:
        localized = parsed.dt.tz_localize(tz, ambiguous="infer",
                                          nonexistent="shift_forward")
        logger.info("%s: localized %s -> UTC (DST inferred from ordering)", label, tz)
    except Exception as exc:  # pytz.AmbiguousTimeError and friends
        logger.warning("%s: could not infer the DST-end repeated hour (%s). "
                       "Falling back to first (DST) pass. At most one hour per "
                       "year is mislabelled; no rows are dropped.", label, exc)
        localized = parsed.dt.tz_localize(tz, ambiguous=True,
                                          nonexistent="shift_forward")
    return localized.dt.tz_convert("UTC")


def dst_report(s_local_naive: pd.Series, tz: str) -> dict:
    """Diagnostics for a naive local-time column.

    ambiguous_stamps   -- fall inside the hour that repeats when DST ends.
    nonexistent_stamps -- fall inside the hour that is skipped when DST starts.

    A source that is genuinely local civil time will show a nonzero count for
    at least one of these across a multi-year span. If BOTH are zero, the
    column is probably not local time at all (most likely it is already UTC
    or fixed-offset market time) -- worth knowing before you convert it.
    """
    parsed = pd.to_datetime(s_local_naive)
    nonexistent = int(parsed.dt.tz_localize(tz, ambiguous=True, nonexistent="NaT").isna().sum())
    ambiguous = int(parsed.dt.tz_localize(tz, ambiguous="NaT",
                                          nonexistent="shift_forward").isna().sum())
    out = {"n": len(parsed), "ambiguous_stamps": ambiguous, "nonexistent_stamps": nonexistent}
    logger.info("DST report (%s): %s", tz, out)
    if ambiguous == 0 and nonexistent == 0:
        logger.info("  -> no DST artefacts found; this column may already be UTC or "
                    "fixed-offset market time rather than local civil time.")
    return out


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _request_with_retries(method: str, url: str, *, max_retries: int = 5,
                          backoff_base: float = 1.5, **kwargs) -> requests.Response:
    headers = {**_DEFAULT_HEADERS, **(kwargs.pop("headers", None) or {})}
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.request(method, url, timeout=60, headers=headers, **kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            wait = backoff_base ** attempt
            logger.warning("Network error (%s) %d/%d, retry in %.1fs",
                           exc, attempt, max_retries, wait)
            time.sleep(wait)
            continue
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", backoff_base ** attempt))
            logger.warning("Rate limited; retry in %.1fs", wait)
            time.sleep(wait)
            continue
        if resp.status_code >= 500:
            wait = backoff_base ** attempt
            logger.warning("Server error %d; retry in %.1fs", resp.status_code, wait)
            time.sleep(wait)
            continue
        if not resp.ok:
            snip = resp.content[:400].decode("utf-8", errors="replace") if resp.content else "(empty)"
            raise requests.HTTPError(f"{method} {url} -> HTTP {resp.status_code}. Body: {snip!r}",
                                     response=resp)
        return resp
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to fetch {url}")


def _chunk_date_range(start: date, end: date, max_days: int) -> list[tuple[date, date]]:
    if start > end:
        raise ValueError(f"start_date {start} is after end_date {end}")
    chunks, cur, step = [], start, timedelta(days=max_days - 1)
    while cur <= end:
        chunk_end = min(cur + step, end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


# ---------------------------------------------------------------------------
# Weather: FORECAST vintages (Previous Runs API)
# ---------------------------------------------------------------------------


def fetch_weather_forecast(region: RegionInfo, start_date: date, end_date: date,
                           lead_days: Sequence[int] = (1, 2),
                           variables: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Weather as it was forecast N days *before* each valid time.

    Returns hourly rows on a UTC index with columns `<var>_fc24h`,
    `<var>_fc48h`, ... The archive begins January 2024.
    """
    variables = list(variables or FORECAST_WEATHER_VARS)
    lead_days = sorted(set(lead_days))
    if not lead_days or min(lead_days) < 1 or max(lead_days) > 7:
        raise ValueError("lead_days must be a non-empty subset of 1..7")

    hourly_vars = [f"{v}_previous_day{n}" for n in lead_days for v in variables]
    frames = []
    for c0, c1 in _chunk_date_range(start_date, end_date, WEATHER_CHUNK_DAYS):
        logger.info("Forecast weather %s: %s -> %s (leads=%s)", region.region_id, c0, c1, lead_days)
        params = {"latitude": region.latitude, "longitude": region.longitude,
                  "start_date": c0.isoformat(), "end_date": c1.isoformat(),
                  "hourly": ",".join(hourly_vars), "timezone": "UTC"}
        payload = _request_with_retries("GET", OPEN_METEO_PREVIOUS_RUNS_URL, params=params).json()
        if "hourly" not in payload:
            raise ValueError(f"Unexpected Previous-Runs response: {str(payload)[:400]}")
        frames.append(pd.DataFrame(payload["hourly"]))
        time.sleep(0.25)

    df = pd.concat(frames, ignore_index=True).rename(columns={"time": "time_utc"})
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.rename(columns={f"{v}_previous_day{n}": f"{v}_fc{24 * n}h"
                            for n in lead_days for v in variables})

    keep = ["time_utc"] + [c for c in df.columns if "_fc" in c and c.endswith("h")]
    df = df[keep].drop_duplicates(subset="time_utc").sort_values("time_utc")

    empty = [c for c in df.columns if c != "time_utc" and df[c].isna().all()]
    if empty:
        logger.warning("All-NaN forecast columns dropped (model doesn't carry them): %s", empty)
        df = df.drop(columns=empty)
    return df.reset_index(drop=True)


def fetch_weather_observed(region: RegionInfo, start_date: date, end_date: date,
                           variables: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """ERA5 reanalysis, suffixed `_obs`. Hindsight -- counterfactual runs only."""
    variables = list(variables or OBSERVED_WEATHER_VARS)
    frames = []
    for c0, c1 in _chunk_date_range(start_date, end_date, 366):
        logger.info("Observed weather %s: %s -> %s", region.region_id, c0, c1)
        params = {"latitude": region.latitude, "longitude": region.longitude,
                  "start_date": c0.isoformat(), "end_date": c1.isoformat(),
                  "hourly": ",".join(variables), "timezone": "UTC"}
        payload = _request_with_retries("GET", OPEN_METEO_ARCHIVE_URL, params=params).json()
        if "hourly" not in payload:
            raise ValueError(f"Unexpected ERA5 response: {str(payload)[:400]}")
        frames.append(pd.DataFrame(payload["hourly"]))
        time.sleep(0.2)
    df = pd.concat(frames, ignore_index=True).rename(columns={"time": "time_utc"})
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.rename(columns={c: f"{c}_obs" for c in df.columns if c != "time_utc"})
    return df.drop_duplicates(subset="time_utc").sort_values("time_utc").reset_index(drop=True)


def forecast_column_for_horizon(base_var: str, horizon_minutes: float,
                                lead_days: Sequence[int] = (1, 2)) -> str:
    """Smallest vintage N with N >= horizon. See module docstring."""
    for n in sorted(lead_days):
        if horizon_minutes <= n * 24 * 60:
            return f"{base_var}_fc{24 * n}h"
    raise ValueError(
        f"Horizon {horizon_minutes / 60:.1f} h exceeds the longest vintage you fetched "
        f"({max(lead_days) * 24} h). Re-fetch with a larger --lead-days, or shorten "
        f"the horizon. Using a shorter vintage here would be leakage."
    )


# ---------------------------------------------------------------------------
# OpenElectricity (optional)
# ---------------------------------------------------------------------------


def _parse_oe(payload: dict) -> pd.DataFrame:
    if not payload.get("success", True):
        raise ValueError(f"OpenElectricity error: {payload.get('error')}")
    rows = []
    for series in payload.get("data", []) or []:
        metric = series.get("metric")
        for result in series.get("results", []) or []:
            groups = result.get("columns") or {}
            for pt in result.get("data", []) or []:
                rec = {"time_utc": pt.get("timestamp"), "metric": metric, "value": pt.get("value")}
                rec.update(groups)
                rows.append(rec)
    if not rows:
        return pd.DataFrame(columns=["time_utc", "metric", "value"])
    df = pd.DataFrame.from_records(rows)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    return df


def _fetch_oe(endpoint: str, api_key: str, region: RegionInfo, metrics: list[str],
              start_date: date, end_date: date, interval: str,
              secondary_grouping: Optional[str] = None) -> pd.DataFrame:
    frames = []
    for c0, c1 in _chunk_date_range(start_date, end_date, OE_INTERVAL_MAX_DAYS.get(interval, 32)):
        params: dict = {"metrics": metrics, "interval": interval,
                        "date_start": f"{c0.isoformat()}T00:00:00",
                        "date_end": f"{c1.isoformat()}T23:59:59",
                        "network_region": region.region_id}
        if secondary_grouping:
            params["secondary_grouping"] = secondary_grouping
        logger.info("OpenElectricity %s %s -> %s", endpoint, c0, c1)
        resp = _request_with_retries("GET", f"{OPEN_ELECTRICITY_BASE_URL}/{endpoint}/NEM",
                                     headers={"Authorization": f"Bearer {api_key}"}, params=params)
        frames.append(_parse_oe(resp.json()))
        time.sleep(0.3)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_generation_mix(api_key: str, region: RegionInfo, start_date: date,
                         end_date: date, interval: str = "1h") -> pd.DataFrame:
    long_df = _fetch_oe("data/network", api_key, region, GENERATION_METRICS,
                        start_date, end_date, interval, secondary_grouping="fueltech_group")
    if long_df.empty:
        logger.warning("No generation-mix data for %s", region.region_id)
        return pd.DataFrame()
    if "fueltech_group" not in long_df.columns:
        long_df["fueltech_group"] = "unknown"
    long_df["column"] = "gen_" + long_df["fueltech_group"].fillna("unknown").astype(str)
    wide = long_df.pivot_table(index="time_utc", columns="column", values="value", aggfunc="mean")
    wide.columns = [str(c) for c in wide.columns]
    return wide.reset_index()


def fetch_market_fundamentals(api_key: str, region: RegionInfo, start_date: date,
                              end_date: date, interval: str = "1h") -> pd.DataFrame:
    long_df = _fetch_oe("market/network", api_key, region, MARKET_METRICS,
                        start_date, end_date, interval)
    if long_df.empty:
        return pd.DataFrame()
    wide = long_df.pivot_table(index="time_utc", columns="metric", values="value", aggfunc="mean")
    wide.columns = [str(c) for c in wide.columns]
    wide = wide.reset_index()
    n = _INTERVALS_PER_BUCKET.get(interval)  # documented OE bug: MW summed, not averaged
    if n:
        for col in ("demand", "flow_imports", "flow_exports"):
            if col in wide.columns:
                wide[col] = wide[col] / n
    return wide.rename(columns={c: f"oe_{c}" for c in wide.columns if c != "time_utc"})


# ---------------------------------------------------------------------------
# Local tables
# ---------------------------------------------------------------------------


def load_price_5min(
    source: "str | Path | pd.DataFrame",
    time_col: str = "SETTLEMENTDATE",
    price_col: str = "RRP",
    demand_col: Optional[str] = "TOTALDEMAND",
    region_col: Optional[str] = "REGION",
    region_id: Optional[str] = "TAS1",
    tz: str = "market",
    time_format: Optional[str] = "%Y/%m/%d %H:%M:%S",
    interval_ending: bool = True,
    interval_minutes: int = 5,
) -> pd.DataFrame:
    """Load an AEMO 5-minute price/demand table, normalised to UTC interval-START.

    `interval_ending=True`: AEMO stamps a dispatch interval by its END time --
    the 00:05 row describes 00:00-00:05. Everything downstream assumes
    interval-start, so we shift back one interval. Leaving this wrong is a
    silent 5-minute lookahead that will make your model look great.

    `time_format` is pinned because "2024/03/01" is ambiguous to a parser that
    guesses; AEMO writes YYYY/MM/DD. Pass None to let pandas infer.

    `source` may be a CSV path OR an already-loaded DataFrame, so if you
    already concatenate the AEMO monthly files yourself you can hand the
    frame straight in.
    """
    if isinstance(source, pd.DataFrame):
        df, origin = source.copy(), "<DataFrame>"
    else:
        df, origin = pd.read_csv(source), str(source)
    for col in (time_col, price_col):
        if col not in df.columns:
            raise KeyError(f"{col!r} not in {origin}. Found: {list(df.columns)[:25]}")

    n_raw = len(df)
    if region_col and region_col in df.columns:
        regions = sorted(df[region_col].dropna().unique())
        if region_id:
            df = df[df[region_col] == region_id]
            if df.empty:
                raise ValueError(f"No rows for region {region_id!r}. Present: {regions}")
            if len(regions) > 1:
                logger.warning("Input held %d regions %s; kept %s (%d of %d rows)",
                               len(regions), regions, region_id, len(df), n_raw)

    dup = df.duplicated(subset=[time_col]).sum()
    if dup:
        logger.warning("%d duplicate %s values -- keeping the first of each. "
                       "Overlapping monthly downloads are the usual cause.", dup, time_col)
        df = df.drop_duplicates(subset=[time_col], keep="first")

    ts = to_utc_series(df[time_col], tz=tz, fmt=time_format, label="price table")
    if interval_ending:
        ts = ts - pd.Timedelta(minutes=interval_minutes)

    # .values on a tz-aware Series silently strips the timezone. Reset the
    # index instead so the UTC awareness survives into the frame.
    out = pd.DataFrame({
        "time_utc": ts.reset_index(drop=True),
        "price": pd.to_numeric(df[price_col], errors="coerce").reset_index(drop=True),
    })
    if demand_col and demand_col in df.columns:
        out["demand"] = pd.to_numeric(df[demand_col], errors="coerce").reset_index(drop=True)

    out = out.dropna(subset=["time_utc"]).sort_values("time_utc").reset_index(drop=True)

    span_days = (out["time_utc"].max() - out["time_utc"].min()).total_seconds() / 86400
    expected = int(span_days * 24 * 60 / interval_minutes) + 1
    logger.info("Price table: %d rows kept (of %d raw), %s -> %s, %.0f days, "
                "%d expected at %d-min spacing",
                len(out), n_raw, out["time_utc"].min(), out["time_utc"].max(),
                span_days, expected, interval_minutes)
    if abs(len(out) - expected) > 0.01 * expected:
        logger.warning("Row count is %+d vs expected -- gaps or leftover duplicates. "
                       "assemble_master_frame() will reindex onto a clean grid and "
                       "report the holes.", len(out) - expected)
    return out


def load_hourly_table(path: str | Path, time_col: str, tz: str = "market",
                      time_format: Optional[str] = None,
                      suffix: str = "") -> pd.DataFrame:
    """Load your existing hourly feature table (e.g. the weather CSV you
    already have) onto a UTC index. Set tz="Australia/Hobart" if it was
    fetched in local civil time -- then DST is handled by to_utc_series."""
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise KeyError(f"{time_col!r} not in {path}. Found: {list(df.columns)[:25]}")
    if tz not in ("market", "utc"):
        dst_report(df[time_col], tz)
    ts = to_utc_series(df[time_col], tz=tz, fmt=time_format, label=str(path))
    out = df.drop(columns=[time_col]).reset_index(drop=True)
    if suffix:
        out = out.rename(columns={c: f"{c}{suffix}" for c in out.columns})
    out.insert(0, "time_utc", ts.reset_index(drop=True))
    return out.drop_duplicates(subset="time_utc").sort_values("time_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Calendar features (from LOCAL wall clock)
# ---------------------------------------------------------------------------


def add_calendar_features(df: pd.DataFrame, region: RegionInfo) -> pd.DataFrame:
    local = df["time_local"]
    df["hour"] = local.dt.hour
    df["minute_of_day"] = local.dt.hour * 60 + local.dt.minute
    df["day_of_week"] = local.dt.dayofweek
    df["day_of_year"] = local.dt.dayofyear
    df["month"] = local.dt.month
    df["quarter"] = local.dt.quarter
    df["year"] = local.dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if _HAS_HOLIDAYS:
        years = range(int(df["year"].min()), int(df["year"].max()) + 1)
        au = _holidays_lib.country_holidays("AU", subdiv=region.state_code, years=years)
        df["is_public_holiday"] = pd.Series(local.dt.date.values, index=df.index).isin(au).astype(int)
    else:
        df["is_public_holiday"] = 0
    df["is_business_day"] = ((1 - df["is_weekend"]) * (1 - df["is_public_holiday"])).astype(int)

    df["season"] = df["month"].map({12: "summer", 1: "summer", 2: "summer",
                                    3: "autumn", 4: "autumn", 5: "autumn",
                                    6: "winter", 7: "winter", 8: "winter",
                                    9: "spring", 10: "spring", 11: "spring"})

    # Trees can't learn that 23:55 sits next to 00:00 from a raw integer.
    tod = 2 * np.pi * df["minute_of_day"] / 1440.0
    doy = 2 * np.pi * df["day_of_year"] / 365.25
    df["tod_sin"], df["tod_cos"] = np.sin(tod), np.cos(tod)
    df["doy_sin"], df["doy_cos"] = np.sin(doy), np.cos(doy)

    df["is_day"] = ((df["hour"] >= 6) & (df["hour"] < 19)).astype(int)
    # DST flag: 1 when the wall clock runs an hour ahead of market time.
    # Vectorised -- a .map() over hundreds of thousands of tz-aware stamps is slow.
    if "time_aest" in df.columns:
        wall_gap = (df["time_local"].dt.tz_localize(None)
                    - df["time_aest"].dt.tz_localize(None))
        df["is_dst"] = (wall_gap == pd.Timedelta(hours=1)).astype(int)
    else:
        df["is_dst"] = 0
    return df


# ---------------------------------------------------------------------------
# The merge
# ---------------------------------------------------------------------------


def _upsample(hourly: pd.DataFrame, grid: pd.DatetimeIndex,
              interpolate_limit: int) -> pd.DataFrame:
    """Hourly -> grid by time interpolation. `interpolate_limit` is in grid
    steps, so a genuine multi-hour hole stays NaN rather than being invented."""
    h = hourly.copy()
    h["time_utc"] = pd.to_datetime(h["time_utc"], utc=True)
    h = h.drop_duplicates(subset="time_utc").set_index("time_utc").sort_index()
    out = h.reindex(h.index.union(grid))
    num = out.select_dtypes(include=[np.number]).columns
    other = [c for c in out.columns if c not in num]
    if len(num):
        out[num] = out[num].interpolate(method="time", limit=interpolate_limit,
                                        limit_direction="both")
    if other:
        out[other] = out[other].ffill(limit=interpolate_limit)
    return out.reindex(grid)


def assemble_master_frame(price: pd.DataFrame, region: RegionInfo,
                          weather_fc: Optional[pd.DataFrame] = None,
                          weather_obs: Optional[pd.DataFrame] = None,
                          generation: Optional[pd.DataFrame] = None,
                          market: Optional[pd.DataFrame] = None,
                          freq: str = "5min") -> pd.DataFrame:
    """One gap-free frame at `freq`, indexed on UTC interval-start."""
    if price.empty:
        raise ValueError("price table is empty")

    p = price.copy()
    p["time_utc"] = pd.to_datetime(p["time_utc"], utc=True)
    p = p.drop_duplicates(subset="time_utc").set_index("time_utc").sort_index()

    grid = pd.date_range(p.index.min(), p.index.max(), freq=freq, tz="UTC")
    master = p.reindex(grid)
    master.index.name = "time_utc"

    missing = int(master["price"].isna().sum())
    if missing:
        logger.warning("%d of %d grid intervals have no price (source gaps)", missing, len(master))

    step_min = pd.Timedelta(freq).total_seconds() / 60
    limit = int(120 / step_min)  # never interpolate across more than 2 hours

    for frame, name in ((weather_fc, "forecast weather"), (weather_obs, "observed weather"),
                        (generation, "generation mix"), (market, "market fundamentals")):
        if frame is None or frame.empty:
            continue
        logger.info("Merging %s (%d cols)", name, frame.shape[1] - 1)
        up = _upsample(frame, grid, limit)
        clash = [c for c in up.columns if c in master.columns]
        if clash:
            raise ValueError(f"Column collision merging {name}: {clash}")
        master = master.join(up)

    master = master.reset_index()
    # A properly labelled fixed +10:00, not "UTC plus ten hours".
    master["time_aest"] = master["time_utc"].dt.tz_convert(MARKET_TZ)
    master["time_local"] = master["time_utc"].dt.tz_convert(region.local_tz)
    master["region"] = region.region_id
    master = add_calendar_features(master, region)

    gen_cols = [c for c in master.columns if c.startswith("gen_")]
    vre = [c for c in gen_cols if any(k in c for k in ("solar", "wind"))]
    if "demand" in master.columns and vre:
        master["residual_demand"] = master["demand"] - master[vre].sum(axis=1, min_count=1)

    lead = ["time_utc", "time_aest", "time_local", "region", "price"]
    return master[lead + [c for c in master.columns if c not in lead]]


def resample_master(master: pd.DataFrame, freq: str = "30min") -> pd.DataFrame:
    """Aggregate the 5-minute master to a coarser decision grid.

    Prices and demand are time-averaged; flags take the first value. Keep the
    5-minute frame for settlement -- decide on this one, pay on that one.
    """
    df = master.set_index("time_utc").sort_index()
    num = df.select_dtypes(include=[np.number]).columns
    agg = df[num].resample(freq).mean()
    for col in ("region", "season"):
        if col in df.columns:
            agg[col] = df[col].resample(freq).first()
    agg = agg.reset_index()
    region = master["region"].iloc[0]
    info = NEM_REGIONS[region]
    agg["time_aest"] = agg["time_utc"].dt.tz_convert(MARKET_TZ)
    agg["time_local"] = agg["time_utc"].dt.tz_convert(info.local_tz)
    agg = agg.drop(columns=[c for c in ("hour", "minute_of_day", "day_of_week", "day_of_year",
                                        "month", "quarter", "year", "is_weekend",
                                        "is_public_holiday", "is_business_day", "tod_sin",
                                        "tod_cos", "doy_sin", "doy_cos", "is_day", "is_dst")
                            if c in agg.columns])
    return add_calendar_features(agg, info)


# ---------------------------------------------------------------------------
# Supervised frame -- where the vintage rule is enforced
# ---------------------------------------------------------------------------


def build_supervised_frame(
    df: pd.DataFrame,
    horizon_minutes: int,
    lead_days: Sequence[int] = (1, 2),
    target_col: str = "price",
    lag_minutes: Sequence[int] = (30, 60, 120, 180, 360, 1440, 2880, 10080),
    roll_minutes: Sequence[int] = (60, 360, 1440),
    weather_vars: Optional[Sequence[str]] = None,
    calendar_cols: Sequence[str] = ("tod_sin", "tod_cos", "doy_sin", "doy_cos",
                                    "day_of_week", "is_weekend", "is_public_holiday",
                                    "is_business_day", "is_day", "is_dst"),
    allow_hindsight: bool = False,
) -> pd.DataFrame:
    """Build (X, y) for ONE horizon, origin-anchored and leakage-checked.

    Standing at origin t, predicting target time T = t + horizon:

      * lags / rolling stats  -- computed at or before t. Identical for every
        horizon at a given origin. These are what you actually observe.
      * calendar of T         -- deterministic, known years in advance. Safe.
      * weather forecast of T -- taken from the vintage band satisfying
        horizon <= N, asserted below. This is the only genuinely
        forward-looking input, and the only one that can leak.

    Returns one row per origin with `origin_time`, `target_time`,
    `horizon_minutes`, feature columns, and `target`.
    """
    d = df.sort_values("time_utc").reset_index(drop=True)
    step = int(pd.Timedelta(d["time_utc"].diff().dropna().mode().iloc[0]).total_seconds() // 60)
    if horizon_minutes % step:
        raise ValueError(f"horizon {horizon_minutes} min is not a multiple of the "
                         f"{step}-min grid")
    h_steps = horizon_minutes // step

    out = pd.DataFrame({
        "origin_time": d["time_utc"],
        "target_time": d["time_utc"].shift(-h_steps),
        "horizon_minutes": horizon_minutes,
    })

    # --- observed history, anchored at the origin -------------------------
    for base in [c for c in (target_col, "demand", "residual_demand") if c in d.columns]:
        for lag in lag_minutes:
            if lag % step:
                continue
            out[f"{base}_lag{lag}m"] = d[base].shift(lag // step)
        for win in roll_minutes:
            w = max(1, win // step)
            out[f"{base}_rollmean{win}m"] = d[base].rolling(w, min_periods=1).mean()
            out[f"{base}_rollstd{win}m"] = d[base].rolling(w, min_periods=2).std()
        out[f"{base}_now"] = d[base]

    # --- calendar of the TARGET time (deterministic, safe) ----------------
    for col in calendar_cols:
        if col in d.columns:
            out[f"{col}_at_target"] = d[col].shift(-h_steps)

    # --- weather forecast of the TARGET time, correct vintage -------------
    if weather_vars is None:
        weather_vars = sorted({c.rsplit("_fc", 1)[0] for c in d.columns
                               if "_fc" in c and c.endswith("h")})
    for var in weather_vars:
        col = forecast_column_for_horizon(var, horizon_minutes, lead_days)  # asserts h <= N
        if col not in d.columns:
            logger.warning("%s missing for horizon %d min -- skipped", col, horizon_minutes)
            continue
        out[f"{var}_fcast_at_target"] = d[col].shift(-h_steps)

    if allow_hindsight:
        for col in [c for c in d.columns if c.endswith("_obs")]:
            out[f"{col}_at_target"] = d[col].shift(-h_steps)
        logger.warning("HINDSIGHT ENABLED: %d _obs columns included. Counterfactual only.",
                       sum(c.endswith("_obs") for c in d.columns))
    else:
        leaked = [c for c in out.columns if c.endswith("_obs") or "_obs_" in c]
        assert not leaked, f"observed-weather columns leaked into X: {leaked}"

    out["target"] = d[target_col].shift(-h_steps)
    out = out.dropna(subset=["target", "target_time"]).reset_index(drop=True)

    # Cheap end-to-end guard: every target must sit exactly `horizon` after
    # its origin. Catches an off-by-one shift better than reading the code.
    delta = (out["target_time"] - out["origin_time"]).dt.total_seconds() / 60
    assert (delta == horizon_minutes).all(), "origin/target spacing is wrong"

    logger.info("Supervised frame h=%d min: %d rows, %d features",
                horizon_minutes, len(out), out.shape[1] - 4)
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_master_frame(df: pd.DataFrame, freq_minutes: int = 5,
                          max_nan_fraction: float = 0.05) -> pd.DataFrame:
    assert "time_utc" in df.columns, "missing time_utc"
    ts = pd.to_datetime(df["time_utc"], utc=True)
    assert ts.is_monotonic_increasing, "time_utc not sorted"
    assert not ts.duplicated().any(), f"{int(ts.duplicated().sum())} duplicate timestamps"

    gaps = ts.diff().dropna().unique()
    expected = pd.Timedelta(minutes=freq_minutes)
    assert len(gaps) == 1 and gaps[0] == expected, f"irregular grid: {gaps[:5]}"

    assert df["time_local"].dt.tz is not None, "time_local must be tz-aware"
    assert df["time_aest"].dt.tz is not None, "time_aest must be tz-aware"

    obs = [c for c in df.columns if c.endswith("_obs")]
    if obs:
        logger.warning("HINDSIGHT COLUMNS PRESENT (%d, e.g. %s) -- counterfactual only.",
                       len(obs), obs[:3])

    if "is_dst" in df.columns:
        logger.info("DST: %.1f%% of rows are in daylight saving (expect ~45%% for TAS)",
                    100 * df["is_dst"].mean())

    nan = df.isna().mean().sort_values(ascending=False)
    bad = nan[(nan > max_nan_fraction) & (nan.index != "price")]
    if len(bad):
        logger.warning("Columns above %.0f%% NaN:\n%s", max_nan_fraction * 100, bad.head(15))

    logger.info("VALID: %d rows x %d cols | %s -> %s (%.1f days)", len(df), df.shape[1],
                ts.min(), ts.max(), (ts.max() - ts.min()).total_seconds() / 86400)
    return nan.rename("nan_fraction").to_frame()


# ---------------------------------------------------------------------------
# Self-test (offline)
# ---------------------------------------------------------------------------


def _self_test() -> None:
    logger.info("=== SELF TEST (synthetic, no network) ===")
    region = NEM_REGIONS["TAS1"]
    rng = np.random.default_rng(0)

    # --- 1. DST handling -------------------------------------------------
    # Tasmanian DST ends 06 Apr 2025 (02:00 local repeats) and starts
    # 05 Oct 2025 (02:00-03:00 local never happens). Same switch dates as
    # the mainland south-east; only the start date differs in some years.
    # Round-tripping UTC -> local -> naive reproduces the real repeated hour
    # that a local-time logger would have written (02:00 appears twice).
    apr_utc = pd.date_range("2025-04-05 12:00", "2025-04-07 12:00", freq="h", tz="UTC")
    apr = apr_utc.tz_convert("Australia/Hobart").tz_localize(None)
    assert apr.duplicated().any(), "test fixture should contain the repeated hour"
    rep = dst_report(pd.Series(apr), "Australia/Hobart")
    assert rep["ambiguous_stamps"] > 0, "fixture should look ambiguous to the reporter"
    utc_apr = to_utc_series(pd.Series(apr), tz="Australia/Hobart", label="DST-end test")
    assert utc_apr.is_monotonic_increasing, "DST-end conversion broke ordering"
    assert utc_apr.is_unique, "DST-end conversion produced duplicate instants"
    assert len(utc_apr) == len(apr), "rows lost at DST end"

    # A naive range across DST start DOES contain the hour that never happened.
    oct_ = pd.date_range("2025-10-04 12:00", "2025-10-06 12:00", freq="h")
    rep_oct = dst_report(pd.Series(oct_), "Australia/Hobart")
    assert rep_oct["nonexistent_stamps"] > 0, "fixture should contain the skipped hour"
    utc_oct = to_utc_series(pd.Series(oct_), tz="Australia/Hobart", label="DST-start test")
    assert len(utc_oct) == len(oct_) and utc_oct.is_monotonic_increasing
    logger.info("  DST end/start conversion OK (no rows lost, ordering intact)")

    # Market time must be immune to DST.
    m = to_utc_series(pd.Series(pd.date_range("2025-04-05 12:00", periods=48, freq="h")),
                      tz="market")
    assert (m.diff().dropna() == pd.Timedelta(hours=1)).all(), "market tz should have no DST jump"
    logger.info("  market time correctly ignores DST")

    # --- 2. price loading, AEMO-shaped ----------------------------------
    n = 40 * 288
    aest = pd.date_range("2024-03-01 00:05:00", periods=n, freq="5min")
    raw = pd.DataFrame({
        "REGION": "TAS1",
        "SETTLEMENTDATE": aest.strftime("%Y/%m/%d %H:%M:%S"),
        "TOTALDEMAND": 1150 + 200 * np.sin(np.arange(n) * 2 * np.pi / 288),
        "RRP": 90 + 60 * np.sin(np.arange(n) * 2 * np.pi / 288) + rng.normal(0, 8, n),
        "PERIODTYPE": "TRADE",
    })
    tmp = Path("/tmp/_selftest_price.csv")
    raw.to_csv(tmp, index=False)

    price = load_price_5min(tmp)
    assert price["time_utc"].iloc[0] == pd.Timestamp("2024-02-29 14:00:00", tz="UTC"), \
        price["time_utc"].iloc[0]
    logger.info("  YYYY/MM/DD parsing + interval-ending shift OK")

    # --- 3. merge --------------------------------------------------------
    hours = pd.date_range(price["time_utc"].min().floor("h"),
                          price["time_utc"].max().ceil("h"), freq="h", tz="UTC")
    fc = pd.DataFrame({"time_utc": hours})
    for var in ("temperature_2m", "wind_speed_100m", "shortwave_radiation"):
        for lead in (24, 48):
            fc[f"{var}_fc{lead}h"] = rng.normal(20, 3, len(hours))
    obs = pd.DataFrame({"time_utc": hours, "temperature_2m_obs": rng.normal(20, 3, len(hours))})
    gen = pd.DataFrame({"time_utc": hours,
                        "gen_solar": np.abs(rng.normal(0, 500, len(hours))),
                        "gen_wind": np.abs(rng.normal(0, 400, len(hours)))})

    master = assemble_master_frame(price, region, weather_fc=fc, weather_obs=obs, generation=gen)
    validate_master_frame(master)
    assert "residual_demand" in master.columns

    row = master.iloc[0]
    assert row["time_aest"].utcoffset() == MARKET_UTC_OFFSET
    assert row["time_aest"].hour == 0 and row["time_aest"].minute == 0
    assert row["time_local"].utcoffset() == timedelta(hours=11), "expect AEDT on 1 Mar"
    logger.info("  merge OK | market=%s local=%s", row["time_aest"], row["time_local"])

    # --- 4. vintage rule -------------------------------------------------
    assert forecast_column_for_horizon("temperature_2m", 60) == "temperature_2m_fc24h"
    assert forecast_column_for_horizon("temperature_2m", 24 * 60) == "temperature_2m_fc24h"
    assert forecast_column_for_horizon("temperature_2m", 30 * 60) == "temperature_2m_fc48h"
    try:
        forecast_column_for_horizon("temperature_2m", 72 * 60)
    except ValueError:
        logger.info("  vintage rule refuses horizons beyond the fetched leads (correct)")
    else:
        raise AssertionError("should have refused a 72 h horizon with leads (1, 2)")

    # --- 5. supervised frames -------------------------------------------
    m30 = resample_master(master, "30min")
    validate_master_frame(m30, freq_minutes=30)

    s6 = build_supervised_frame(m30, horizon_minutes=360)
    s30 = build_supervised_frame(m30, horizon_minutes=30 * 60)
    assert "temperature_2m_fcast_at_target" in s6.columns
    assert not [c for c in s6.columns if c.endswith("_obs")], "hindsight leaked"

    # The 6 h model must draw on the 24 h vintage, the 30 h model on the 48 h one.
    j = m30[["time_utc", "temperature_2m_fc24h", "temperature_2m_fc48h"]].rename(
        columns={"time_utc": "target_time"})
    chk6 = s6.merge(j, on="target_time")
    assert np.allclose(chk6["temperature_2m_fcast_at_target"], chk6["temperature_2m_fc24h"],
                       equal_nan=True), "6 h model did not use the 24 h vintage"
    chk30 = s30.merge(j, on="target_time")
    assert np.allclose(chk30["temperature_2m_fcast_at_target"], chk30["temperature_2m_fc48h"],
                       equal_nan=True), "30 h model did not use the 48 h vintage"

    # Lag features must be anchored at the origin, not the target.
    price_by_time = m30.set_index("time_utc")["price"]
    probe = s6.iloc[100]
    assert np.isclose(probe["price_now"], price_by_time.loc[probe["origin_time"]]), \
        "price_now is not the value at the origin"
    assert np.isclose(probe["target"], price_by_time.loc[probe["target_time"]]), \
        "target is not the value at the target time"
    logger.info("  vintages + origin anchoring verified against the source series")

    tmp.unlink(missing_ok=True)
    logger.info("=== SELF TEST PASSED (master %d x %d) ===", *master.shape)


README_NOTES = """
Wholesale-only pricing: a note for your slide
---------------------------------------------
This handler carries spot price (RRP, $/MWh) with no network charge, no
retail margin, no export fee. That is a clean, well-defined problem and a
fine scope for the project -- but the direction of the bias is the opposite
of what you might expect, so state it explicitly:

  * Charging at spot and discharging at spot loses only round-trip
    efficiency (~10%) plus degradation. This is the MOST favourable case.
  * A real household on a retail tariff pays spot + ~$200/MWh to import and
    receives about spot to export. That adder is a dead loss on every
    grid-to-grid cycle, and it is usually larger than the daily spread --
    so pure wholesale arbitrage typically stops paying entirely.
  * The adder only becomes a gain when the battery discharges into
    household load, because then it avoids the import charge.

So "if it works on wholesale it will work with charges" is backwards. Say
instead: "we measure the wholesale arbitrage value, which is an UPPER BOUND
on what a retail-tariff household could capture from grid-to-grid trading."
That is defensible, and it is one line on the limitations slide.

TAS1 specifics worth a sentence of their own
--------------------------------------------
Tasmania is not a scaled-down NSW, and the differences all point the same
way -- at the arbitrage spread, which is what the battery earns from:

  * Generation is overwhelmingly hydro, which is itself a storage asset.
    Hydro schedulers already arbitrage the intraday shape, so the residual
    spread left for a battery is usually thinner than on the mainland.
  * Basslink couples TAS1 to VIC1 with a ~500 MW limit. Prices track
    Victoria while the link is unconstrained and decouple sharply when it
    binds or trips, so interconnector flow matters more here than any
    weather variable. If you get an OpenElectricity key, the
    flow_imports / flow_exports columns are the ones to look at first.
  * There is almost no rooftop or utility solar, so the mainland "duck
    curve" midday price trough is largely absent. Features built around
    shortwave_radiation will carry much less signal than they would in
    NSW1 or SA1; wind_speed_100m (large wind fleet) will carry more.
  * Demand is winter-peaking and heavily industrial (the smelters are a
    large, fairly flat block), so the weather-to-demand relationship is
    weaker and skewed towards cold rather than heat.
  * Hydro storage levels are a slow-moving state variable that drives price
    over months. Nothing in this handler captures it; that is a known gap,
    not an oversight.
"""



# ---------------------------------------------------------------------------
# HIGH-LEVEL API -- this is what you call
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Everything the build needs. Edit the defaults or pass overrides:

        df = build_dataset(data_folder="data", region="TAS1")
    """
    # --- your AEMO CSVs ---
    data_folder: Path = Path("data")
    price_glob: str = "*.csv"
    region: str = "TAS1"
    time_col: str = "SETTLEMENTDATE"
    price_col: str = "RRP"
    demand_col: Optional[str] = "TOTALDEMAND"
    region_col: Optional[str] = "REGION"
    price_tz: str = "market"                      # AEMO publishes AEST, no DST
    price_time_format: Optional[str] = "%Y/%m/%d %H:%M:%S"
    price_interval_ending: bool = True            # AEMO stamps interval END

    # --- weather ---
    lead_days: Sequence[int] = (1, 2)             # forecast vintages to fetch
    include_observed: bool = True                 # ERA5, hindsight counterfactual
    start_date: Optional[date] = None             # None -> inferred from prices
    end_date: Optional[date] = None

    # --- optional generation mix ---
    oe_api_key: Optional[str] = None
    oe_interval: str = "1h"

    # --- downstream grids ---
    decision_freq: str = "30min"
    horizons: Sequence[int] = (30, 360, 1440, 2160)   # minutes

    # --- caching: weather is fetched once and reused ---
    cache_dir: Optional[Path] = Path("cache")
    refresh_cache: bool = False


# Previous-Runs archive begins here; asking for earlier just returns NaN.
FORECAST_ARCHIVE_START = date(2024, 1, 1)


def _cache_io(path: Path, fetch, refresh: bool) -> pd.DataFrame:
    """Read a cached frame if present, otherwise fetch and cache it.

    Weather is the slow part of the build and it never changes once archived,
    so this turns a multi-minute run into a two-second one after the first go.
    """
    if path is not None and path.exists() and not refresh:
        logger.info("Cache hit: %s", path)
        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        return df

    df = fetch()

    if path is not None and not df.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(path, index=False)
        except Exception as exc:  # noqa: BLE001 - pyarrow missing, etc.
            logger.warning("Parquet cache failed (%s); using CSV.", exc)
            path = path.with_suffix(".csv")
            df.to_csv(path, index=False)
        logger.info("Cached -> %s", path)
    return df


def load_price_folder(cfg: DatasetConfig) -> pd.DataFrame:
    """Read and concatenate every AEMO CSV in `cfg.data_folder`.

    Replaces the loose concat loop: it also filters to one region, drops the
    duplicate rows that overlapping monthly downloads always produce, and
    normalises timestamps to UTC interval-START.
    """
    folder = Path(cfg.data_folder)
    if not folder.exists():
        raise FileNotFoundError(f"{folder.resolve()} does not exist")

    files = sorted(folder.glob(cfg.price_glob))
    if not files:
        raise FileNotFoundError(f"No files matching {cfg.price_glob!r} in {folder.resolve()}")

    frames = [pd.read_csv(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    logger.info("Read %d file(s) from %s -> %d raw rows", len(files), folder, len(combined))

    return load_price_5min(
        combined,
        time_col=cfg.time_col, price_col=cfg.price_col, demand_col=cfg.demand_col,
        region_col=cfg.region_col, region_id=cfg.region, tz=cfg.price_tz,
        time_format=cfg.price_time_format, interval_ending=cfg.price_interval_ending,
    )


def build_dataset(cfg: Optional[DatasetConfig] = None, **overrides) -> pd.DataFrame:
    """Build and return THE DataFrame: 5-minute grid, prices, demand, weather
    forecast vintages, ERA5 hindsight, calendar features, everything merged.

        from data_handler import build_dataset
        df = build_dataset()                       # uses ./data
        df = build_dataset(data_folder="mydata")   # or override anything

    This is the frame to use for EDA. For CDA and for model training, pass it
    through `resample_master()` and `build_supervised_frame()`.
    """
    cfg = cfg or DatasetConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise TypeError(f"Unknown config option {key!r}. Valid: "
                            f"{sorted(cfg.__dataclass_fields__)}")
        setattr(cfg, key, value)

    region = NEM_REGIONS[cfg.region]
    cache = Path(cfg.cache_dir) if cfg.cache_dir else None

    # 1. prices ----------------------------------------------------------
    price = load_price_folder(cfg)

    # 2. weather window: infer from the price data unless told otherwise --
    start = cfg.start_date or price["time_utc"].min().date()
    end = cfg.end_date or price["time_utc"].max().date()
    if start < FORECAST_ARCHIVE_START:
        logger.warning("Price data starts %s but the forecast archive starts %s. "
                       "Forecast columns will be NaN before then -- consider "
                       "trimming the training period.", start, FORECAST_ARCHIVE_START)
        start = FORECAST_ARCHIVE_START
    logger.info("Weather window: %s -> %s", start, end)

    leads = "-".join(str(n) for n in sorted(cfg.lead_days))
    fc = _cache_io(
        cache / f"wfc_{cfg.region}_{start}_{end}_lead{leads}.parquet" if cache else None,
        lambda: fetch_weather_forecast(region, start, end, lead_days=cfg.lead_days),
        cfg.refresh_cache,
    )

    obs = None
    if cfg.include_observed:
        obs = _cache_io(
            cache / f"wobs_{cfg.region}_{start}_{end}.parquet" if cache else None,
            lambda: fetch_weather_observed(region, start, end),
            cfg.refresh_cache,
        )

    # 3. optional generation mix -----------------------------------------
    gen = mkt = None
    api_key = cfg.oe_api_key or os.environ.get("OPENELECTRICITY_API_KEY")
    if api_key:
        gen = _cache_io(
            cache / f"gen_{cfg.region}_{start}_{end}_{cfg.oe_interval}.parquet" if cache else None,
            lambda: fetch_generation_mix(api_key, region, start, end, interval=cfg.oe_interval),
            cfg.refresh_cache,
        )
        mkt = _cache_io(
            cache / f"mkt_{cfg.region}_{start}_{end}_{cfg.oe_interval}.parquet" if cache else None,
            lambda: fetch_market_fundamentals(api_key, region, start, end, interval=cfg.oe_interval),
            cfg.refresh_cache,
        )
    else:
        logger.warning("No OpenElectricity key -- no generation mix, so no "
                       "residual_demand, and no Basslink flow columns. In TAS1 "
                       "the interconnector is the strongest price predictor "
                       "you're leaving on the table. Free key: "
                       "https://platform.openelectricity.org.au")

    # 4. merge + validate -------------------------------------------------
    master = assemble_master_frame(price, region, weather_fc=fc, weather_obs=obs,
                                   generation=gen, market=mkt)
    validate_master_frame(master)
    return master


def build_all(cfg: Optional[DatasetConfig] = None, **overrides) -> dict:
    """Everything at once, ready for the three analysis modes.

    Returns a dict:
        "master"      5-min frame  -> EDA, and settlement in the battery backtest
        "decision"    30-min frame -> CDA regressions, and the MPC grid
        "supervised"  {horizon_minutes: (X, y) frame} -> PDA model training
    """
    cfg = cfg or DatasetConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)

    master = build_dataset(cfg)
    decision = resample_master(master, cfg.decision_freq)
    step = int(pd.Timedelta(cfg.decision_freq).total_seconds() // 60)
    validate_master_frame(decision, freq_minutes=step)

    supervised: dict[int, pd.DataFrame] = {}
    for h in cfg.horizons:
        try:
            supervised[h] = build_supervised_frame(decision, horizon_minutes=h,
                                                   lead_days=cfg.lead_days)
        except ValueError as exc:
            logger.warning("Skipping horizon %d min: %s", h, exc)

    return {"master": master, "decision": decision, "supervised": supervised, "config": cfg}


def _self_test_build_dataset() -> None:
    """End-to-end check of build_dataset() with the network stubbed out."""
    import tempfile

    logger.info("--- build_dataset() end-to-end (network stubbed) ---")
    rng = np.random.default_rng(1)
    n = 20 * 288
    aest = pd.date_range("2024-03-01 00:05:00", periods=n, freq="5min")

    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir) / "data"
        folder.mkdir()
        # Two overlapping monthly files, exactly like real AEMO downloads.
        for i, sl in enumerate([slice(0, 12 * 288), slice(10 * 288, n)]):
            pd.DataFrame({
                "REGION": "TAS1",
                "SETTLEMENTDATE": aest[sl].strftime("%Y/%m/%d %H:%M:%S"),
                "TOTALDEMAND": 1150 + rng.normal(0, 60, len(aest[sl])),
                "RRP": 90 + rng.normal(0, 20, len(aest[sl])),
                "PERIODTYPE": "TRADE",
            }).to_csv(folder / f"part{i}.csv", index=False)

        def fake_fc(region, s, e, lead_days=(1, 2), variables=None):
            hrs = pd.date_range(f"{s}", f"{e} 23:00", freq="h", tz="UTC")
            out = pd.DataFrame({"time_utc": hrs})
            for v in ("temperature_2m", "wind_speed_100m"):
                for nd in lead_days:
                    out[f"{v}_fc{24 * nd}h"] = rng.normal(20, 3, len(hrs))
            return out

        def fake_obs(region, s, e, variables=None):
            hrs = pd.date_range(f"{s}", f"{e} 23:00", freq="h", tz="UTC")
            return pd.DataFrame({"time_utc": hrs,
                                 "temperature_2m_obs": rng.normal(20, 3, len(hrs))})

        g = globals()
        real_fc, real_obs = g["fetch_weather_forecast"], g["fetch_weather_observed"]
        g["fetch_weather_forecast"], g["fetch_weather_observed"] = fake_fc, fake_obs
        try:
            bundle = build_all(data_folder=folder, cache_dir=Path(tmpdir) / "cache",
                               horizons=(30, 1440, 2160), oe_api_key=None)
        finally:
            g["fetch_weather_forecast"], g["fetch_weather_observed"] = real_fc, real_obs

    master, decision, sup = bundle["master"], bundle["decision"], bundle["supervised"]
    assert len(master) == n, f"overlapping files not deduped: {len(master)} vs {n}"
    assert "temperature_2m_fc24h" in master.columns
    assert master["is_dst"].isin((0, 1)).all()
    assert set(sup) == {30, 1440, 2160}
    assert "temperature_2m_fcast_at_target" in sup[1440].columns
    logger.info("  build_all OK: master %s, decision %s, horizons %s",
                master.shape, decision.shape, sorted(sup))


# ---------------------------------------------------------------------------
# Run directly: no CLI, no arguments. Edit CONFIG and press play.
# ---------------------------------------------------------------------------

CONFIG = DatasetConfig(
    data_folder=Path("data"),   # folder holding your AEMO CSVs
    region="TAS1",
    lead_days=(1, 2),           # forecast vintages: 24 h and 48 h
    decision_freq="30min",
    horizons=(30, 360, 1440, 2160),
    cache_dir=Path("cache"),
)


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        _self_test()
        _self_test_build_dataset()
        logger.info("=== ALL SELF TESTS PASSED ===")
    else:
        bundle = build_all(CONFIG)
        master = bundle["master"]
        print(f"\nmaster: {master.shape[0]:,} rows x {master.shape[1]} cols")
        print(master[["time_aest", "time_local", "price", "demand", "is_dst"]].head())
        print("\nforecast columns:",
              [c for c in master.columns if "_fc" in c][:6], "...")
        for h, frame in bundle["supervised"].items():
            print(f"supervised h={h:>5} min: {frame.shape[0]:,} rows x "
                  f"{frame.shape[1] - 4} features")
        print(README_NOTES)

