#!/usr/bin/env python3
"""
=====================================================================
 EDA for TAS1 wholesale PRICE forecasting  ->  home-battery dispatch
=====================================================================

Companion to `data_handler.py`. The target of this project is the PRICE,
so this EDA is built around the price: demand, weather and interconnector
flows appear as *drivers of the price*, never as ends in themselves.

    from eda_price import apply_eda
    from data_handler import build_dataset

    df = build_dataset()                 # the 5-min master frame
    figs = apply_eda(df)                 # all figures, each one fig.show()

    figs = apply_eda(df, sections=["price_dyn", "corr"])   # a subset
    figs = apply_eda(df, show=False, save_html=True)       # for the report

Conventions used everywhere
---------------------------
* Every axis carries a unit. Price is $/MWh, demand MW, temperature degC,
  wind m/s, radiation W/m2, flows MW, time is the LOCAL Hobart wall clock
  (Australia/Hobart, DST-aware) because human behaviour follows it.
* Because price spans -1000 to +16600 $/MWh, most price axes use the
  signed log  sgn(p)*log10(1+|p|).  Read it as: 2 -> $100, 3 -> $1000,
  -2 -> -$100. Every such axis says so in its title.
* Each figure carries a subtitle explaining how to read it, and prints a
  longer interpretation note to the console.
* One figure = one function returning (figure, note). A failure is caught,
  reported and skipped, never fatal.

DST caveat: the index is local wall-clock time, so one hour repeats each
April and one is missing each October. That is ~12 five-minute rows a year
and is irrelevant for every statistic here, but it is why the ACF section
counts lags in STEPS rather than in wall-clock hours.
"""

from __future__ import annotations

import os
import textwrap
import warnings
from statistics import NormalDist

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = "browser"

warnings.filterwarnings("ignore")

TEMPLATE = "plotly_white"
SEQ, DIV = "Viridis", "RdBu"
pio.templates.default = TEMPLATE

PRICE, DEMAND = "price", "demand"
TIME_CANDIDATES = ["time_local", "time_aest", "time_utc", "time", "datetime"]
REGION_TZ = {"TAS1": "Australia/Hobart", "NSW1": "Australia/Sydney",
             "VIC1": "Australia/Melbourne", "SA1": "Australia/Adelaide",
             "QLD1": "Australia/Brisbane"}

SEASON_ORDER = ["Summer", "Autumn", "Winter", "Spring"]
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

PRICE_LOG_LAB = "signed log10 price  [2 = $100/MWh, 3 = $1000/MWh]"
PRICE_LAB = "spot price  [$/MWh]"
DEMAND_LAB = "operational demand  [MW]"
HOUR_LAB = "hour of day  [local Hobart time]"


# =====================================================================
#  small helpers
# =====================================================================
def _has(df, *cols):
    return all(c in df.columns and df[c].notna().any() for c in cols)


def _first(df, *cands):
    """First column of `cands` that exists and holds data (None otherwise)."""
    for c in cands:
        if _has(df, c):
            return c
    return None


def _num_cols(df):
    out = []
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].notna().sum() > 10 and df[c].nunique(dropna=True) > 1:
            out.append(c)
    return out


def _feat_cols(df):
    """Model-candidate columns: numeric, not internal, not an ID."""
    drop = {"year", "day_of_year", "minute_of_day"}
    return [c for c in _num_cols(df) if not c.startswith("_") and c not in drop]


def _slog(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log10(1.0 + np.abs(x))


def _sample(df, n):
    return df if len(df) <= n else df.sample(n, random_state=0).sort_index()


def _style(fig, title, subtitle, xlab=None, ylab=None, height=500, **kw):
    """Title + wrapped explanatory subtitle + axis labels, consistently."""
    sub = "<br>".join(textwrap.wrap(subtitle, 118))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><sub>{sub}</sub>", x=0.01, xanchor="left"),
        height=height, margin=dict(l=75, r=45, t=40 + 22 * (sub.count("<br>") + 2), b=70),
        **kw)
    if xlab:
        fig.update_xaxes(title_text=xlab)
    if ylab:
        fig.update_yaxes(title_text=ylab)
    return fig


def _acf(x, nlags):
    s = pd.Series(np.asarray(x, float)).interpolate(limit_direction="both")
    v = s.to_numpy()
    v = v[np.isfinite(v)]
    n = len(v)
    nlags = int(min(nlags, n - 2))
    v = v - v.mean()
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(v, nfft)
    acov = np.fft.irfft(f * np.conjugate(f), nfft)[: nlags + 1].real / n
    return (np.zeros(nlags + 1), n) if acov[0] <= 0 else (acov / acov[0], n)


def _pacf(r, nlags):
    r = np.asarray(r, float)
    nlags = int(min(nlags, len(r) - 1))
    pac = np.zeros(nlags + 1)
    pac[0] = 1.0
    if nlags == 0:
        return pac
    phi = np.zeros((nlags + 1, nlags + 1))
    phi[1, 1] = pac[1] = r[1]
    for k in range(2, nlags + 1):
        num = r[k] - np.sum(phi[k - 1, 1:k] * r[1:k][::-1])
        den = 1.0 - np.sum(phi[k - 1, 1:k] * r[1:k])
        phi[k, k] = 0.0 if abs(den) < 1e-12 else num / den
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pac[k] = phi[k, k]
    return pac


def _ccf(x, y, max_lag):
    a = pd.Series(np.asarray(x, float)).interpolate(limit_direction="both")
    b = pd.Series(np.asarray(y, float)).interpolate(limit_direction="both")
    lags = np.arange(-max_lag, max_lag + 1)
    return lags, np.array([a.shift(-L).corr(b) for L in lags], dtype=float)


def _binned(x, y, bins=40, q=(0.25, 0.5, 0.75, 0.9)):
    """Conditional quantiles of y in equal-count bins of x."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 100:
        return pd.DataFrame()
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    g = pd.DataFrame({"b": idx, "x": x, "y": y}).groupby("b")
    out = pd.DataFrame({"x": g["x"].mean(), "mean": g["y"].mean(), "n": g["y"].size()})
    for qq in q:
        out[f"q{int(qq*100)}"] = g["y"].quantile(qq)
    return out.reset_index(drop=True)


def _kmeans(X, k, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        new = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
        if (new == lab).all():
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab, C


def _cluster(X, k):
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        return km.labels_, km.cluster_centers_
    except Exception:
        return _kmeans(X, k)


def _order_corr(corr):
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform
        d = 1.0 - corr.abs().fillna(0.0).to_numpy()
        np.fill_diagonal(d, 0.0)
        d = (d + d.T) / 2.0
        return [corr.columns[i] for i in leaves_list(linkage(squareform(d, checks=False),
                                                             method="average"))]
    except Exception:
        return list(corr.columns)


# =====================================================================
#  preparation
# =====================================================================
def prepare(data, time_col=None, region=None, local_tz=None):
    """Load the master frame and index it on the LOCAL wall clock.

    Accepts a DataFrame (what `build_dataset()` returns), a .csv/.parquet/.pkl
    path. Handles tz-aware columns and mixed-offset strings, which is what a
    DST-aware `time_local` column looks like after a CSV round trip.
    """
    if isinstance(data, str):
        print(f"[load] {data}")
        if data.endswith(".parquet"):
            df = pd.read_parquet(data)
        elif data.endswith((".pkl", ".pickle")):
            df = pd.read_pickle(data)
        else:
            df = pd.read_csv(data)
    else:
        df = data.copy()

    if region is None and "region" in df.columns:
        region = str(df["region"].dropna().iloc[0])
    if region and "region" in df.columns and df["region"].nunique() > 1:
        df = df[df["region"] == region]
        print(f"[prep] filtered to region {region}")
    local_tz = local_tz or REGION_TZ.get(str(region), None)

    if time_col is None:
        time_col = next((c for c in TIME_CANDIDATES if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"no timestamp column; tried {TIME_CANDIDATES}")

    # Which clock the plots run on. Default is the local wall clock, because
    # behaviour (and therefore the daily price shape) follows it. Ask for
    # time_aest or time_utc explicitly and that choice is honoured.
    target_tz = {"time_utc": "UTC", "time_aest": "Etc/GMT-10"}.get(time_col, local_tz)
    raw = df[time_col]
    t = pd.to_datetime(raw, errors="coerce")
    aware = pd.api.types.is_datetime64_any_dtype(t) and getattr(t.dt, "tz", None) is not None
    if aware or not pd.api.types.is_datetime64_any_dtype(t):
        t = pd.to_datetime(raw, errors="coerce", utc=True)      # mixed offsets ok
        if target_tz:
            t = t.dt.tz_convert(target_tz)
        t = t.dt.tz_localize(None)                              # naive wall clock
    df["_t"] = t.to_numpy()

    keep_utc = "time_utc" in df.columns
    if keep_utc:
        df = df.drop_duplicates(subset="time_utc", keep="first")
    df = df[df["_t"].notna()].sort_values("_t")
    df = df.set_index("_t")
    df = df.drop(columns=[c for c in ("time_utc", "time_aest", "time_local")
                          if c in df.columns], errors="ignore")

    dt = pd.Series(df.index).diff().dt.total_seconds().dropna()
    step = float(dt[dt > 0].median()) if len(dt) else 300.0
    df.attrs.update(step_min=step / 60.0, per_day=int(round(86400 / step)),
                    region=region or "?", time_col=time_col,
                    tz=target_tz or "naive (as supplied)")

    idx = df.index
    df["_hour"] = idx.hour
    df["_tod"] = idx.hour + idx.minute / 60.0
    df["_dow"] = idx.dayofweek
    df["_month"] = idx.month
    df["_doy"] = idx.dayofyear
    df["_year"] = idx.year
    df["_date"] = idx.normalize()
    df["_season"] = pd.Categorical(
        np.select([np.isin(df["_month"], [12, 1, 2]), np.isin(df["_month"], [3, 4, 5]),
                   np.isin(df["_month"], [6, 7, 8]), np.isin(df["_month"], [9, 10, 11])],
                  SEASON_ORDER, default="NA"), categories=SEASON_ORDER, ordered=True)
    df["_daytype"] = np.where(df["_dow"] >= 5, "Weekend", "Weekday")
    if "is_public_holiday" in df.columns:
        df["_daytype"] = np.where(df["is_public_holiday"].fillna(0).astype(float) > 0,
                                  "Holiday", df["_daytype"])

    p = df[PRICE]
    df["_slog"] = _slog(p)
    df["_dp"] = p.diff()
    df["_neg"] = (p < 0).astype(int)
    thr = float(np.nanquantile(p, 0.995))
    df.attrs["spike_thr"] = thr
    df["_spike"] = (p > thr).astype(int)
    if _has(df, DEMAND):
        df["_dd"] = df[DEMAND].diff()

    net = None
    if _has(df, "oe_flow_imports", "oe_flow_exports"):
        net = df["oe_flow_imports"] - df["oe_flow_exports"]
    elif _has(df, "flow_imports", "flow_exports"):
        net = df["flow_imports"] - df["flow_exports"]
    if net is not None:
        df["_net_import"] = net                 # >0 importing from VIC over Basslink
    df.attrs["has_flow"] = net is not None

    print(f"[prep] region={df.attrs['region']} clock={time_col} ({df.attrs['tz']})")
    print(f"[prep] {len(df):,} rows | {idx.min()} -> {idx.max()} | "
          f"step {df.attrs['step_min']:.0f} min ({df.attrs['per_day']}/day)")
    print(f"[prep] price: median {p.median():.1f} | mean {p.mean():.1f} | "
          f"neg {100*df['_neg'].mean():.2f}% | spike threshold (P99.5) {thr:.0f} $/MWh")
    return df


# =====================================================================
#  SECTION 1 - data quality  (3 figures)
# =====================================================================
def q_missing(df):
    cols = [c for c in df.columns if not c.startswith("_")]
    m = (df[cols].isna().mean() * 100).sort_values(ascending=False)
    m = m[m > 0]
    if m.empty:
        m = pd.Series({"(nothing missing)": 0.0})
    fig = go.Figure(go.Bar(x=m.values, y=m.index, orientation="h", marker_color="crimson"))
    _style(fig, "1.1 Missing values per column",
           "Share of the 5-minute grid where each column is NaN. Anything above ~5% is a "
           "feature you cannot rely on; the forecast columns should be near zero after "
           "2024-01-01, the ERA5 _obs columns can lag at the end of the sample.",
           "share of rows that are NaN  [%]", "column",
           height=max(420, 15 * len(m)))
    return fig, (
        "Read the top of the bar chart first. Two failure modes matter here. (a) A weather "
        "column that is mostly NaN was not carried by the Open-Meteo model for that vintage - "
        "drop it rather than impute it. (b) The _obs columns typically stop a few days before "
        "the end of the sample because ERA5 is published with a lag; if you plan a "
        "perfect-foresight counterfactual, your evaluation window has to end where ERA5 ends. "
        "Price NaNs are different: assemble_master_frame() reindexes onto a clean grid, so a "
        "NaN price is a genuine hole in the AEMO download, not an interpolation artefact.")


def q_missing_time(df):
    cols = [c for c in df.columns if not c.startswith("_")]
    d = df[cols].isna().astype(int).resample("D").mean() * 100
    d = d.loc[:, d.max() > 0]
    if d.shape[1] == 0:
        idx = df.resample("D").size().index
        d = pd.DataFrame({"(nothing missing)": np.zeros(len(idx))}, index=idx)
    fig = go.Figure(go.Heatmap(z=d.T.values, x=d.index, y=d.columns, colorscale="Reds",
                               colorbar=dict(title="% NaN<br>that day")))
    _style(fig, "1.2 Where in time are the gaps?",
           "Daily share of NaN per column. Vertical stripes = one bad day across many columns "
           "(an outage or a failed API chunk). Horizontal stripes = one column broken "
           "throughout.", "date  [local time]", "column",
           height=max(430, 14 * d.shape[1]))
    return fig, (
        "Vertical stripes are the ones to act on: a whole day missing across many columns must "
        "be excluded from training and from the battery backtest, because both the features "
        "and the settlement price for that day are unreliable. A block of red at the very "
        "start of the sample usually means you are before FORECAST_ARCHIVE_START "
        "(2024-01-01) - trim the training period rather than fight it.")


def q_coverage(df):
    cnt = df.resample("D").size()
    pna = df[PRICE].isna().resample("D").sum()
    exp = df.attrs["per_day"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("rows present per day", "missing PRICE values per day"))
    fig.add_trace(go.Bar(x=cnt.index, y=cnt.values, name="rows", marker_color="#4c78a8"), 1, 1)
    fig.add_hline(y=exp, line_dash="dash", line_color="green", row=1, col=1,
                  annotation_text=f"complete day = {exp} rows")
    fig.add_trace(go.Bar(x=pna.index, y=pna.values, name="NaN price",
                         marker_color="crimson"), 2, 1)
    fig.update_yaxes(title_text="rows per day  [count]", row=1, col=1)
    fig.update_yaxes(title_text="NaN prices  [count]", row=2, col=1)
    fig.update_xaxes(title_text="date  [local time]", row=2, col=1)
    _style(fig, "1.3 Grid completeness and price availability",
           "The DST days show 276 and 300 rows instead of 288 - that is correct behaviour on a "
           "local wall clock, not a bug. Everything else below the line is a real gap.",
           height=620, showlegend=False)
    return fig, (
        "Expect exactly two anomalous days per year: the October DST start loses an hour (276 "
        "rows) and the April DST end gains one (300 rows). Any OTHER short day is a genuine "
        "hole. Cross-check the bottom panel: days with many NaN prices are days your battery "
        "backtest cannot settle, so exclude them from the revenue totals or you will "
        "understate performance without knowing why.")


# =====================================================================
#  SECTION 2 - the price itself  (6 figures)
# =====================================================================
def p_history(df):
    d = df[PRICE].resample("D").agg(["min", "median", "max"])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        row_heights=[0.62, 0.38],
                        subplot_titles=("daily min / median / max price",
                                        "30-day rolling mean and volatility"))
    fig.add_trace(go.Scatter(x=d.index, y=_slog(d["max"]), name="daily max",
                             line=dict(color="#d62728", width=0.8)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=_slog(d["min"]), name="daily min",
                             line=dict(color="#1f77b4", width=0.8)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=_slog(d["median"]), name="daily median",
                             line=dict(color="black", width=1.6)), 1, 1)
    roll = df[PRICE].resample("D").mean()
    fig.add_trace(go.Scatter(x=roll.index, y=roll.rolling(30).mean(), name="30d mean",
                             line=dict(color="#2ca02c", width=2)), 2, 1)
    fig.add_trace(go.Scatter(x=roll.index, y=roll.rolling(30).std(), name="30d std",
                             line=dict(color="#ff7f0e", width=2, dash="dot")), 2, 1)
    fig.update_yaxes(title_text=PRICE_LOG_LAB, row=1, col=1)
    fig.update_yaxes(title_text="price level / spread  [$/MWh]", row=2, col=1)
    fig.update_xaxes(title_text="date  [local time]", row=2, col=1)
    _style(fig, "2.1 Price history: level, envelope and volatility",
           "Top panel on a signed-log axis so the -$1000 floor and the $16,600 cap fit "
           "together. The vertical distance between the red and blue lines is the daily "
           "arbitrage window your battery is trying to capture.",
           height=720)
    return fig, (
        "Three questions to answer from this one. (1) Is the median line drifting? A visible "
        "level shift means old data is a different market - consider training only on the "
        "recent regime, or adding a slow-moving level feature. In TAS1 hydro storage drives "
        "exactly this kind of multi-month drift and nothing in your feature set captures it, "
        "so state it as a known limitation. (2) Do the red and blue lines widen together or "
        "independently? Widening only at the top means spikes; widening at the bottom means "
        "wind-driven negative pricing. (3) Does the orange volatility line track the green "
        "level line? If volatility moves independently, your model needs to predict spread, "
        "not just level - and the battery is paid on spread.")


def p_distribution(df):
    p = df[PRICE].dropna()
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("raw price, clipped to [-100, 400] for legibility",
                                        "signed log10 - the whole range"))
    fig.add_trace(go.Histogram(x=p.clip(-100, 400), nbinsx=140, marker_color="#4c78a8"), 1, 1)
    fig.add_trace(go.Histogram(x=_slog(p), nbinsx=140, marker_color="seagreen"), 1, 2)
    fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=2)
    fig.update_xaxes(title_text=PRICE_LAB, row=1, col=1)
    fig.update_xaxes(title_text=PRICE_LOG_LAB, row=1, col=2)
    fig.update_yaxes(title_text="number of 5-min intervals  [count]")
    _style(fig, "2.2 Price distribution",
           f"Negative in {100*(p<0).mean():.2f}% of intervals, above $300/MWh in "
           f"{100*(p>300).mean():.2f}%, P99.5 = {df.attrs['spike_thr']:.0f} $/MWh, "
           f"median = {p.median():.1f} $/MWh, mean = {p.mean():.1f} $/MWh. Mean above median "
           f"is the tail doing the work.",
           height=470, showlegend=False, bargap=0.02)
    return fig, (
        "This plot decides your loss function, so treat it as a modelling result and not "
        "decoration. The gap between mean and median tells you how much of the average price "
        "is produced by a handful of intervals. Squared error on the raw price will spend "
        "almost all of its gradient on those intervals and your model will underfit the 99% of "
        "the time when the battery actually operates. Three defensible answers, pick one and "
        "justify it: (a) model the signed-log price and accept a bias when back-transforming, "
        "(b) keep the raw target but use a robust or quantile loss, (c) split the problem - "
        "a regression for the normal regime plus a classifier for 'will there be a spike in "
        "this window'. For a battery, (c) maps directly onto the decision you care about.")


def p_ecdf(df):
    p = np.sort(df[PRICE].dropna().to_numpy())
    y = np.arange(1, len(p) + 1) / len(p)
    st = max(1, len(p) // 6000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_slog(p[::st]), y=y[::st], mode="lines",
                             line=dict(color="#333", width=2), name="all hours"))
    for lbl, mask, col in [("18:00-20:00 (peak)", df["_hour"].between(18, 19), "#d62728"),
                           ("01:00-05:00 (night)", df["_hour"].between(1, 4), "#1f77b4")]:
        v = np.sort(df.loc[mask, PRICE].dropna().to_numpy())
        if len(v) < 100:
            continue
        yy = np.arange(1, len(v) + 1) / len(v)
        s2 = max(1, len(v) // 4000)
        fig.add_trace(go.Scatter(x=_slog(v[::s2]), y=yy[::s2], mode="lines",
                                 line=dict(color=col, width=1.6), name=lbl))
    qs = {q: np.quantile(p, q) for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)}
    for q, v in qs.items():
        fig.add_hline(y=q, line_dash="dot", line_color="lightgrey")
    fig.add_annotation(x=_slog(qs[0.1]), y=0.10, text=f"P10 = {qs[0.1]:.0f}",
                       showarrow=True, arrowhead=2, ax=-45, ay=-20)
    fig.add_annotation(x=_slog(qs[0.9]), y=0.90, text=f"P90 = {qs[0.9]:.0f}",
                       showarrow=True, arrowhead=2, ax=40, ay=20)
    _style(fig, "2.3 Empirical CDF of the price, overall and by time of day",
           "Horizontal dotted lines are the 5/10/25/50/75/90/95/99% quantiles. The horizontal "
           "gap between the blue (night) and red (peak) curves at a given probability is the "
           "systematic time-of-day premium a fixed-schedule battery would harvest.",
           PRICE_LOG_LAB, "cumulative probability  [fraction of intervals below x]",
           height=560)
    return fig, (
        f"Read your battery thresholds straight off this curve. Charging in the cheapest decile "
        f"means buying below {qs[0.1]:.0f} $/MWh; discharging in the dearest decile means "
        f"selling above {qs[0.9]:.0f} $/MWh; the raw spread between them is "
        f"{qs[0.9]-qs[0.1]:.0f} $/MWh before efficiency. Multiply by your round-trip efficiency "
        f"and compare against the retail import adder (~$200/MWh, see README_NOTES): if the "
        f"decile spread is smaller than the adder, grid-to-grid arbitrage on a retail tariff "
        f"loses money and your project's honest conclusion is that the value lies in "
        f"self-consumption instead. Do this arithmetic early - it shapes the whole report.")


def p_qq(df):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("raw price vs Normal", "signed-log price vs Normal"))
    nd = NormalDist()
    for i, v in enumerate([df[PRICE].dropna().to_numpy(), _slog(df[PRICE].dropna())], start=1):
        s = np.sort(v)
        s = s[:: max(1, len(s) // 3000)]
        n = len(s)
        theo = np.array([nd.inv_cdf((k + 0.5) / n) for k in range(n)])
        fig.add_trace(go.Scattergl(x=theo, y=s, mode="markers", marker=dict(size=3)), 1, i)
        a, b = np.polyfit(theo, s, 1)
        fig.add_trace(go.Scatter(x=[theo.min(), theo.max()],
                                 y=[a * theo.min() + b, a * theo.max() + b],
                                 mode="lines", line=dict(color="red", dash="dash")), 1, i)
    fig.update_xaxes(title_text="theoretical Normal quantile  [z]")
    fig.update_yaxes(title_text=PRICE_LAB, row=1, col=1)
    fig.update_yaxes(title_text=PRICE_LOG_LAB, row=1, col=2)
    _style(fig, "2.4 Q-Q plots: how far from Normal is the price?",
           "Points on the red line would mean Normally distributed. The right panel shows how "
           "much of the non-normality the log transform actually removes - in an energy market "
           "the answer is usually 'most of it, except the extreme upper tail'.",
           height=470, showlegend=False)
    return fig, (
        "The upward hook at the right of both panels is the spike regime, and it never goes "
        "away entirely. Practical consequences: prediction intervals from an OLS-style model "
        "will be too narrow exactly where being wrong is expensive; and R2 computed on raw "
        "price is close to meaningless because a single spike day can move it. Report MAE and "
        "pinball loss on the raw price, plus R2 on the log scale if you want a familiar "
        "number - and say why.")


def p_violin_hour(df):
    d = _sample(df.dropna(subset=[PRICE]), 150_000)
    fig = px.violin(d, x="_hour", y="_slog", color="_daytype", box=True, points=False,
                    category_orders={"_daytype": ["Weekday", "Weekend", "Holiday"]},
                    color_discrete_sequence=px.colors.qualitative.Set2)
    _style(fig, "2.5 Full price distribution for every hour of the day",
           "Not just the average: the width of each violin is the risk in that hour, and the "
           "long upper tails mark the hours where spikes actually occur. Split by weekday / "
           "weekend / public holiday.",
           HOUR_LAB, PRICE_LOG_LAB, height=580, violinmode="group")
    fig.update_xaxes(dtick=1)
    return fig, (
        "Two separate signals live here and you should quote both. The MEDIAN across hours is "
        "what a fixed-schedule battery exploits. The WIDTH across hours is what a forecasting "
        "battery exploits, because a wide hour is one where the outcome is genuinely uncertain "
        "and therefore one where a good prediction is worth money. If the widest hours are also "
        "the highest-median hours (typically the evening ramp), your model's errors will be "
        "concentrated where the stakes are highest - which is an argument for weighting the "
        "training loss by hour, or at least for reporting error by hour rather than one global "
        "number.")


def p_regime_over_time(df):
    d = pd.DataFrame({
        "negative": df["_neg"].resample("ME").mean() * 100,
        "spike": df["_spike"].resample("ME").mean() * 100,
        "median": df[PRICE].resample("ME").median(),
    })
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=d.index, y=d["negative"], name="% intervals price < 0",
                         marker_color="#1f77b4", opacity=.75), secondary_y=False)
    fig.add_trace(go.Bar(x=d.index, y=d["spike"], name=f"% intervals > P99.5",
                         marker_color="#d62728", opacity=.75), secondary_y=False)
    fig.add_trace(go.Scatter(x=d.index, y=d["median"], name="monthly median price",
                             line=dict(color="black", width=2.5)), secondary_y=True)
    fig.update_yaxes(title_text="share of intervals in the month  [%]", secondary_y=False)
    fig.update_yaxes(title_text="monthly median price  [$/MWh]", secondary_y=True)
    _style(fig, "2.6 Extreme-price regimes month by month",
           "Blue = how often the price went negative, red = how often it exceeded the P99.5 "
           "spike threshold, black line = the ordinary median price for scale.",
           "month", height=520, barmode="group")
    return fig, (
        "This is the plot that tells you whether your train/test split is fair. If negative "
        "prices are concentrated in the last six months (growing wind capacity) and you split "
        "chronologically, your test set contains a regime the training set never saw - your "
        "model will look bad for a reason that has nothing to do with the model. Options: "
        "report it honestly as distribution shift, use a rolling-origin evaluation so every "
        "month gets tested by a model trained on its own past, or shorten the training window. "
        "Rolling-origin is the right answer for a forecasting project and is also what makes "
        "the battery backtest realistic.")


# =====================================================================
#  SECTION 3 - price shape in time  (6 figures)
# =====================================================================
def s_daily_profile(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for dt_, col in [("Weekday", "#d62728"), ("Weekend", "#1f77b4"), ("Holiday", "#2ca02c")]:
        s = df[df["_daytype"] == dt_]
        if len(s) < 500:
            continue
        g = s.groupby("_tod")[PRICE]
        med, lo, hi = g.median(), g.quantile(.25), g.quantile(.75)
        fig.add_trace(go.Scatter(x=med.index, y=med.values, name=f"{dt_} median price",
                                 line=dict(color=col, width=2.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=list(hi.index) + list(lo.index[::-1]),
                                 y=list(hi.values) + list(lo.values[::-1]),
                                 fill="toself", fillcolor=col, opacity=.13,
                                 line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"), secondary_y=False)
    if _has(df, DEMAND):
        dm = df.groupby("_tod")[DEMAND].mean()
        fig.add_trace(go.Scatter(x=dm.index, y=dm.values, name="mean demand (context)",
                                 line=dict(color="grey", width=2, dash="dot")),
                      secondary_y=True)
    fig.update_yaxes(title_text="median price  [$/MWh]  (band = P25-P75)", secondary_y=False)
    fig.update_yaxes(title_text=DEMAND_LAB, secondary_y=True)
    _style(fig, "3.1 Average price shape over the day",
           "Median price by time of day with the inter-quartile band, split by day type. The "
           "grey dotted line is demand, shown only as context - notice whether the price peak "
           "and the demand peak actually coincide.",
           HOUR_LAB, height=560)
    fig.update_xaxes(dtick=1)
    return fig, (
        "The single most important shape in the project: the vertical distance between the "
        "daily low and the daily high on this curve is the gross margin per MWh cycled. In "
        "TAS1 expect a morning and an evening peak and NO deep midday solar trough, because "
        "Tasmania has very little PV - so if you see a pronounced midday dip, check whether "
        "your frame really is TAS1. Also compare the price peak against the grey demand peak: "
        "where they diverge, something other than local load is setting the price, which in "
        "Tasmania means hydro bidding and the Basslink flow. That divergence is your argument "
        "for including interconnector features.")


def s_seasonal_profile(df):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.11,
                        subplot_titles=("median price by season",
                                        "mean demand by season (context)"))
    cols = {"Summer": "#d62728", "Autumn": "#ff7f0e", "Winter": "#1f77b4", "Spring": "#2ca02c"}
    for s in SEASON_ORDER:
        sub = df[df["_season"] == s]
        if len(sub) < 500:
            continue
        m = sub.groupby("_tod")[PRICE].median()
        fig.add_trace(go.Scatter(x=m.index, y=m.values, name=s,
                                 line=dict(color=cols[s], width=2.5)), 1, 1)
        if _has(df, DEMAND):
            d2 = sub.groupby("_tod")[DEMAND].mean()
            fig.add_trace(go.Scatter(x=d2.index, y=d2.values, name=s, showlegend=False,
                                     line=dict(color=cols[s], width=2, dash="dot")), 1, 2)
    fig.update_xaxes(title_text=HOUR_LAB, dtick=3)
    fig.update_yaxes(title_text="median price  [$/MWh]", row=1, col=1)
    fig.update_yaxes(title_text=DEMAND_LAB, row=1, col=2)
    _style(fig, "3.2 Does the daily price shape change with the season?",
           "Southern-hemisphere seasons (Summer = Dec-Feb). For a winter-peaking, "
           "hydro-dominated region expect winter to sit above the others and to have the "
           "sharpest evening ramp.", height=520)
    return fig, (
        "If the four curves differ mainly in LEVEL, a season dummy plus your existing doy_sin / "
        "doy_cos features is enough. If they differ in SHAPE - a peak at a different hour, or a "
        "much steeper evening ramp in winter - then hour and season interact, and a linear "
        "model with additive terms cannot represent it. That is a concrete, testable "
        "justification for gradient boosting, and you can quantify it in the CDA step by "
        "comparing a model with hour+season against one with hour x season.")


def s_calendar_heat(df):
    d = df.copy()
    d["_hh"] = (d["_tod"] * 2).round() / 2
    piv = d.pivot_table(index="_date", columns="_hh", values=PRICE, aggfunc="median")
    fig = go.Figure(go.Heatmap(z=_slog(piv.to_numpy()).T, x=piv.index, y=piv.columns,
                               colorscale="Inferno", zmid=0,
                               colorbar=dict(title="signed log10<br>price [$/MWh]")))
    _style(fig, "3.3 Every half-hour of the sample on one page",
           "x = calendar date, y = time of day, colour = median price in that half-hour "
           "(signed log). Horizontal bands are the daily rhythm; bright vertical stripes are "
           "individual extreme days; dark patches are sustained low or negative prices.",
           "date  [local time]", HOUR_LAB, height=560)
    fig.update_yaxes(dtick=3)
    return fig, (
        "The highest information-per-square-inch plot in the deck, and the one to put on a "
        "slide. Three things to hunt for. (1) Bright vertical stripes: whole-day price events. "
        "Count them - if the year's arbitrage revenue depends on fewer than ten such days, your "
        "battery result is dominated by tail events and needs to be reported with an interval, "
        "not a point estimate. (2) Dark horizontal bands appearing only in part of the year: "
        "seasonal negative pricing, usually windy spring. (3) A change in texture partway "
        "along the x-axis: a regime change, which matters for your train/test split.")


def s_price_surface(df):
    d = df.copy()
    d["_hh"] = (d["_tod"] * 2).round() / 2
    d["_doyb"] = (d["_doy"] // 5) * 5
    piv = d.pivot_table(index="_doyb", columns="_hh", values=PRICE, aggfunc="median")
    piv = piv.interpolate(axis=0).interpolate(axis=1)
    fig = go.Figure(go.Surface(z=piv.to_numpy().T, x=piv.index, y=piv.columns,
                               colorscale="Inferno",
                               colorbar=dict(title="median price<br>[$/MWh]"),
                               contours={"z": {"show": True, "usecolormap": True,
                                               "project": {"z": True}}}))
    _style(fig, "3.4 Median price surface: season x time of day",
           "The same information as the calendar heat map but averaged over years, so the "
           "systematic seasonal-daily structure is separated from individual events. Rotate it "
           "in the browser.", height=720,
           scene=dict(xaxis_title="day of year  [1-365]",
                      yaxis_title="hour of day  [local]",
                      zaxis_title="median price  [$/MWh]"))
    return fig, (
        "This surface IS the deterministic, learnable part of the price - everything a model "
        "with only calendar features could ever get right. Two uses. First, as a baseline: "
        "predicting each interval by its (day-of-year, time-of-day) median is a legitimate "
        "climatological benchmark, and if your ML model cannot beat it, something is wrong. "
        "Second, as a diagnostic: the ridges tell you where the price is systematically high, "
        "and the flat regions tell you where the price is essentially unpredictable from the "
        "calendar alone and must come from weather and market state.")


def s_dow_hour(df):
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14,
                        subplot_titles=("median price  [$/MWh]",
                                        "price volatility - IQR  [$/MWh]"))
    p1 = df.pivot_table(index="_dow", columns="_hour", values=PRICE, aggfunc="median")
    p2 = (df.pivot_table(index="_dow", columns="_hour", values=PRICE, aggfunc=lambda v: v.quantile(.75))
          - df.pivot_table(index="_dow", columns="_hour", values=PRICE, aggfunc=lambda v: v.quantile(.25)))
    fig.add_trace(go.Heatmap(z=p1.values, x=p1.columns, y=[DOW_NAMES[i] for i in p1.index],
                             colorscale="Inferno", colorbar=dict(title="$/MWh", x=0.43)), 1, 1)
    fig.add_trace(go.Heatmap(z=p2.values, x=p2.columns, y=[DOW_NAMES[i] for i in p2.index],
                             colorscale="Cividis", colorbar=dict(title="$/MWh", x=1.02)), 1, 2)
    fig.update_xaxes(title_text=HOUR_LAB, dtick=3)
    fig.update_yaxes(title_text="day of week")
    _style(fig, "3.5 Weekly rhythm of price level and price risk",
           "Left: the typical price in each (weekday, hour) cell. Right: the inter-quartile "
           "range in the same cell, i.e. how uncertain that cell is. The two maps rarely look "
           "the same.", height=460)
    return fig, (
        "Compare the two panels rather than reading them separately. Cells that are dark on "
        "the left but bright on the right are cheap-on-average yet risky - exactly the cells "
        "where a battery scheduled by a rule of thumb gets caught out. Also check whether "
        "Saturday and Sunday differ from each other: if they do, the single is_weekend flag "
        "your handler builds is too coarse and you should use the full day_of_week (one-hot for "
        "linear models, raw integer is fine for trees).")


def s_monthly_box(df):
    d = _sample(df, 180_000)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.6, 0.4],
                        subplot_titles=("price distribution per month",
                                        "mean demand per month (context)"))
    fig.add_trace(go.Box(x=d["_month"], y=d["_slog"], marker_color="#d62728",
                         boxpoints=False), 1, 1)
    if _has(df, DEMAND):
        md = df.groupby("_month")[DEMAND].mean()
        fig.add_trace(go.Bar(x=md.index, y=md.values, marker_color="grey", opacity=.6), 2, 1)
    fig.update_yaxes(title_text=PRICE_LOG_LAB, row=1, col=1)
    fig.update_yaxes(title_text=DEMAND_LAB, row=2, col=1)
    fig.update_xaxes(title_text="month  [1 = Jan]", dtick=1, row=2, col=1)
    _style(fig, "3.6 Seasonality of price and demand side by side",
           "Southern hemisphere: month 7 is mid-winter. For TAS1 the demand bars should peak in "
           "June-August; check whether the price boxes peak in the same months or somewhere "
           "else entirely.", height=680, showlegend=False)
    return fig, (
        "The interesting outcome is a MISMATCH. In a thermal-dominated region the price peak "
        "follows the demand peak closely. In hydro-dominated Tasmania the schedulers move water "
        "between months, so the price can peak in a different month than demand does - and if "
        "you see that here, you have direct evidence that a demand-driven price model is "
        "structurally incomplete. That is a genuinely good finding for the report: it motivates "
        "either interconnector features or an explicit admission that hydro storage state is an "
        "unobserved variable driving your residuals.")


# =====================================================================
#  SECTION 4 - price dynamics  (6 figures)
# =====================================================================
def d_acf_price(df):
    per_day = df.attrs["per_day"]
    step_h = df.attrs["step_min"] / 60.0
    a, n = _acf(df[PRICE], int(8 * per_day))
    p = _pacf(a, min(int(2 * per_day), len(a) - 1))
    ci = 1.96 / np.sqrt(n)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.13,
                        subplot_titles=("ACF of the price - 8 days of lags",
                                        "PACF of the price - 2 days of lags"))
    fig.add_trace(go.Scatter(x=np.arange(len(a)) * step_h, y=a, line=dict(color="#d62728")), 1, 1)
    fig.add_trace(go.Bar(x=np.arange(len(p)) * step_h, y=p, marker_color="#4c78a8"), 2, 1)
    for r in (1, 2):
        fig.add_hline(y=ci, line_dash="dot", line_color="grey", row=r, col=1)
        fig.add_hline(y=-ci, line_dash="dot", line_color="grey", row=r, col=1)
    for k in range(1, 8):
        fig.add_vline(x=24 * k, line_dash="dash", line_color="black", opacity=.3, row=1, col=1)
    fig.update_xaxes(title_text="lag  [hours]", row=1, col=1)
    fig.update_xaxes(title_text="lag  [hours]", row=2, col=1)
    fig.update_yaxes(title_text="autocorrelation  [-1 .. 1]", row=1, col=1)
    fig.update_yaxes(title_text="partial autocorrelation  [-1 .. 1]", row=2, col=1)
    _style(fig, "4.1 How much memory does the price have?",
           "Dashed verticals mark whole days. Dotted horizontals are the 95% significance band "
           "for white noise. The ACF shows total correlation with the past; the PACF shows how "
           "much each lag adds once the shorter lags are known - that is the one that picks "
           "your features.", height=760, showlegend=False)
    return fig, (
        "Expect a fast decay over the first hour or two, then clear bumps at 24, 48 and 168 "
        "hours. Turn this directly into a feature set: take every PACF bar that pokes outside "
        "the dotted band and nothing else. Typically that means the last two or three "
        "intervals, the same interval yesterday, and the same interval last week - which is "
        "exactly the lag list your build_supervised_frame() already offers. Two warnings "
        "specific to price. First, compare the decay length against your forecast horizon: if "
        "the ACF has essentially died by 6 hours, then at a 24-hour horizon your lag features "
        "carry almost nothing and the model is living entirely on calendar and weather - which "
        "means you should not expect a 24-hour price forecast to be anywhere near as accurate "
        "as a 30-minute one, and you should say so before you show the numbers. Second, the "
        "spikes inflate the ACF; if you want a robust picture, rerun it on the log price.")


def d_volatility(df):
    per_day = df.attrs["per_day"]
    step_h = df.attrs["step_min"] / 60.0
    series = [("price level", df[PRICE], "#d62728"),
              ("price change  dP", df["_dp"].dropna(), "#4c78a8"),
              ("absolute change  |dP|", df["_dp"].abs().dropna(), "#2ca02c")]
    fig = go.Figure()
    n = 1
    for lbl, s, c in series:
        a, n = _acf(s, int(6 * per_day))
        fig.add_trace(go.Scatter(x=np.arange(len(a)) * step_h, y=a, name=lbl,
                                 line=dict(color=c, width=2)))
    fig.add_hline(y=1.96 / np.sqrt(n), line_dash="dot", line_color="grey")
    for k in range(1, 7):
        fig.add_vline(x=24 * k, line_dash="dash", line_color="black", opacity=.25)
    _style(fig, "4.2 Volatility clustering: the level forgets, the risk does not",
           "Three ACFs on one axis. If the green |dP| curve stays high while the blue dP curve "
           "collapses to zero, price CHANGES are unpredictable but their SIZE is predictable - "
           "the classic signature of volatility clustering.",
           "lag  [hours]", "autocorrelation  [-1 .. 1]", height=520)
    return fig, (
        "This is the most under-used plot in energy-price EDA and it maps perfectly onto your "
        "battery problem. A battery does not need to know the price exactly; it needs to know "
        "whether the coming evening is worth saving charge for. That is a question about "
        "volatility, not level. If the green curve is persistently positive, you can forecast "
        "'today will be a volatile day' from yesterday - a cheap, high-value feature (rolling "
        "std of the last 24 h, which build_supervised_frame already computes as "
        "price_rollstd1440m). Consider making risk an explicit second output of your model: "
        "predict the conditional spread alongside the conditional mean, and let the dispatch "
        "logic use both.")


def d_lag_scatter(df):
    per_day = df.attrs["per_day"]
    lags = {"previous interval": 1, "same time yesterday": per_day,
            "same time last week": 7 * per_day}
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07,
                        subplot_titles=[f"{k} (r = )" for k in lags])
    # positional, not label-based: the local wall clock repeats an hour each April
    p = df[PRICE].to_numpy(dtype=float)
    hh = df["_hour"].to_numpy()
    n = len(p)
    pos = (np.arange(n) if n <= 40_000
           else np.sort(np.random.default_rng(0).choice(n, 40_000, replace=False)))
    titles = []
    for i, (name, L) in enumerate(lags.items(), start=1):
        keep = pos[pos >= L]
        x, y = _slog(p[keep - L]), _slog(p[keep])
        r = pd.Series(x).corr(pd.Series(y))
        titles.append(f"{name}   r = {r:.2f}")
        fig.add_trace(go.Scattergl(x=x, y=y, mode="markers",
                                   marker=dict(size=2.5, opacity=.25, color=hh[keep],
                                               colorscale="Twilight",
                                               colorbar=dict(title="hour", x=1.02) if i == 3 else None)),
                      1, i)
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(color="red", dash="dash")), 1, i)
        fig.update_xaxes(title_text=f"price at lag  [{PRICE_LOG_LAB}]", row=1, col=i)
    for i, t in enumerate(titles):
        fig.layout.annotations[i].text = t
    fig.update_yaxes(title_text=f"price now  [{PRICE_LOG_LAB}]", row=1, col=1)
    _style(fig, "4.3 Persistence of the price at three lags",
           "Each point is one interval plotted against itself L steps earlier, coloured by hour "
           "of day. A tight cloud on the red 1:1 line means a naive 'same as last time' "
           "forecast is hard to beat at that lag.", height=470, showlegend=False)
    return fig, (
        "These three panels define your baselines, and a forecasting project without baselines "
        "is not evaluable. Panel 1 is the persistence baseline (predict the last observed "
        "price), panel 2 is the daily-naive baseline (predict the same time yesterday), panel 3 "
        "is the weekly-naive one. Compute the MAE of each and put them in the results table "
        "next to your ML model. Note the colour structure too: if the cloud separates by hour, "
        "the naive forecast fails systematically at particular times of day, and that is the "
        "gap your model should be closing.")


def d_spike_event(df):
    thr = df.attrs["spike_thr"]
    per_day = df.attrs["per_day"]
    win = int(round(6 * 60 / df.attrs["step_min"]))          # +/- 6 hours
    p = df[PRICE].to_numpy(dtype=float)
    above = p > thr
    onset = np.where(above[1:] & ~above[:-1])[0] + 1
    onset = onset[(onset > win) & (onset < len(p) - win)]
    if len(onset) < 5:
        raise RuntimeError(f"only {len(onset)} spike onsets above {thr:.0f} $/MWh")
    mat = np.vstack([p[i - win:i + win + 1] for i in onset])
    x = (np.arange(-win, win + 1) * df.attrs["step_min"]) / 60.0
    fig = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.12,
                        subplot_titles=(f"average price around a spike onset (n = {len(onset)})",
                                        "how long does a spike episode last?"))
    fig.add_trace(go.Scatter(x=x, y=np.nanmedian(mat, axis=0), name="median",
                             line=dict(color="#d62728", width=3)), 1, 1)
    fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]],
                             y=np.r_[np.nanquantile(mat, .75, axis=0),
                                     np.nanquantile(mat, .25, axis=0)[::-1]],
                             fill="toself", fillcolor="rgba(214,39,40,.15)",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
    fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=1)
    # episode lengths
    lens, run = [], 0
    for a in above:
        if a:
            run += 1
        elif run:
            lens.append(run)
            run = 0
    lens = np.array(lens) * df.attrs["step_min"]
    fig.add_trace(go.Histogram(x=lens, nbinsx=40, marker_color="#4c78a8"), 1, 2)
    fig.update_xaxes(title_text="time relative to spike onset  [hours]", row=1, col=1)
    fig.update_yaxes(title_text="price  [$/MWh]", row=1, col=1)
    fig.update_xaxes(title_text="episode duration  [minutes]", row=1, col=2)
    fig.update_yaxes(title_text="number of episodes  [count]", row=1, col=2)
    _style(fig, f"4.4 Anatomy of a price spike (threshold = P99.5 = {thr:.0f} $/MWh)",
           "Left: every spike aligned at its onset and averaged, with the inter-quartile band - "
           "does the price ramp up gradually or jump from nowhere? Right: how many minutes a "
           "spike episode survives once it starts.", height=520, showlegend=False)
    return fig, (
        "This decides whether spikes are forecastable at all, so run it before you promise "
        "anything about them. If the median curve rises noticeably in the hours BEFORE zero, "
        "spikes are preceded by an observable build-up (a demand ramp, a wind lull) and a model "
        "with lag features has a chance. If the curve is flat and then jumps vertically, spikes "
        "are effectively unpredictable at 5-minute resolution and you should say so plainly "
        "rather than reporting a model that appears to catch them in-sample. The duration "
        "histogram matters for the battery: if the median episode is shorter than the time your "
        "battery needs to discharge, you cannot actually capture the spike price on the whole "
        "stored energy, and your revenue estimate must be capped by the power rating.")


def d_decomposition(df):
    h = df[PRICE].resample("h").mean().interpolate()
    trend = h.rolling(24 * 30, center=True, min_periods=48).median()
    detr = h - trend
    seas = detr.groupby(detr.index.hour).transform("mean")
    resid = detr - seas
    share = {"trend": trend.var(), "daily cycle": seas.var(), "residual": resid.var()}
    tot = sum(v for v in share.values() if np.isfinite(v))
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
                        subplot_titles=(
                            "observed hourly mean price",
                            f"trend, 30-day centred median  ({100*share['trend']/tot:.0f}% of variance)",
                            f"daily cycle  ({100*share['daily cycle']/tot:.0f}%)",
                            f"residual - what the model must actually predict  ({100*share['residual']/tot:.0f}%)"))
    for i, (y, c) in enumerate([(h, "#333"), (trend, "#2ca02c"), (seas, "#4c78a8"),
                                (resid, "#d62728")], start=1):
        fig.add_trace(go.Scattergl(x=y.index, y=y.values, line=dict(width=.8, color=c)), i, 1)
        fig.update_yaxes(title_text="$/MWh", row=i, col=1)
    fig.update_xaxes(title_text="date  [local time]", row=4, col=1)
    _style(fig, "4.5 Decomposing the price into trend, daily cycle and residual",
           "Additive decomposition on hourly means. The percentages in the panel titles are "
           "each component's share of total variance - that is the honest budget of what is "
           "easy and what is hard in this forecasting problem.", height=820, showlegend=False)
    return fig, (
        "Quote these three percentages in your report; they set expectations for everything "
        "that follows. A large trend share means slow-moving conditions (in TAS1: hydro storage "
        "and Basslink availability) dominate, and none of your weather features will capture "
        "them - consider adding a rolling 30-day mean price as an explicit feature, being "
        "careful that it only uses information available at the origin. A large daily-cycle "
        "share means a calendar-only model is already decent. The residual panel is the real "
        "target: look at whether its amplitude changes over the year, because if it does, your "
        "error metric will be seasonal too and a single MAE hides that.")


def d_ccf(df):
    h = df.resample("h").mean(numeric_only=True)
    fig = go.Figure()
    pairs = [(_first(h, DEMAND), "demand", "#d62728"),
             (_first(h, "wind_speed_100m_obs", "wind_speed_100m_fc24h"), "wind speed 100m", "#2ca02c"),
             (_first(h, "temperature_2m_obs", "temperature_2m_fc24h"), "temperature", "#4c78a8"),
             (_first(h, "_net_import"), "net import over Basslink", "#9467bd"),
             (_first(h, "shortwave_radiation_obs"), "solar radiation", "#ff7f0e")]
    for col, lbl, c in pairs:
        if col is None:
            continue
        lags, cc = _ccf(h[PRICE], h[col], 48)
        fig.add_trace(go.Scatter(x=lags, y=cc, name=lbl, line=dict(color=c, width=2)))
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.add_hline(y=0, line_color="lightgrey")
    _style(fig, "4.6 Cross-correlation of the price with its drivers",
           "corr(price at t+lag, driver at t). A peak at a POSITIVE lag means the driver moves "
           "first and the price follows, which is what makes a variable genuinely predictive "
           "rather than merely contemporaneous.",
           "lag  [hours]   (positive = driver leads price)",
           "correlation with price  [-1 .. 1]", height=520)
    return fig, (
        "Use this to choose lagged features and to sanity-check signs. Wind should be "
        "NEGATIVE against price in TAS1 (more wind, cheaper energy - the merit-order effect) "
        "and that is worth stating explicitly because it is one of the few weather variables "
        "with real signal in Tasmania. Solar should be nearly flat, unlike NSW, because "
        "Tasmania has almost no PV; if solar radiation shows a strong correlation here, it is "
        "probably acting as a proxy for time of day rather than for generation, which is "
        "exactly the kind of confounding the next section is designed to expose. Where the peak "
        "sits at a lag of several hours, add that lagged variable to your feature set.")


# =====================================================================
#  SECTION 5 - correlation structure  (5 figures)
# =====================================================================
def c_matrix(df):
    cols = _feat_cols(df)
    corr = df[cols].corr()
    order = _order_corr(corr)
    corr = corr.loc[order, order]
    fig = go.Figure(go.Heatmap(z=corr.to_numpy(), x=order, y=order, colorscale=DIV,
                               zmid=0, zmin=-1, zmax=1,
                               colorbar=dict(title="Pearson r")))
    _style(fig, "5.1 Correlation between every pair of numeric features",
           "Hierarchically ordered so that mutually redundant features sit next to each other "
           "and form dark blocks on the diagonal. This is a redundancy map, not a causal map.",
           "feature", "feature", height=1000, width=1120)
    return fig, (
        "Look for the blocks, not the individual cells. You will find at least three: the "
        "thermal group (temperature, apparent temperature, dew point, humidity), the radiation "
        "group (shortwave, direct, diffuse, sunshine duration) and the wind group (10m, 100m). "
        "Within a block, features carry almost the same information, so keep one representative "
        "each. This is not about model accuracy - trees handle collinearity fine - it is about "
        "INTERPRETABILITY: with four near-identical temperature columns, permutation importance "
        "splits the credit arbitrarily between them and your feature-importance slide becomes "
        "unreadable. Also check the _fc24h / _fc48h / _obs versions of the same variable: they "
        "should correlate above ~0.95, and if one does not, that vintage has a quality problem.")


def c_nonlinear(df):
    cols = _feat_cols(df)
    pe, sp = df[cols].corr(), df[cols].corr(method="spearman")
    diff = sp.abs() - pe.abs()
    order = _order_corr(sp)
    fig = go.Figure(go.Heatmap(z=diff.loc[order, order].to_numpy(), x=order, y=order,
                               colorscale="PuOr", zmid=0,
                               colorbar=dict(title="|Spearman|<br>- |Pearson|")))
    _style(fig, "5.2 Where is the relationship monotone but NOT linear?",
           "Positive (orange) cells: the rank correlation is much stronger than the linear one, "
           "so a straight line through the scatter would understate a real relationship. "
           "Negative cells usually mean outliers are inflating the Pearson value.",
           "feature", "feature", height=1000, width=1120)
    return fig, (
        "This is the empirical argument for your choice of model class, and it is much more "
        "persuasive than asserting that 'energy prices are non-linear'. Pull out the three or "
        "four strongest orange cells involving price and mention them by name. The price row is "
        "the one that matters: if price-versus-demand is strongly orange, that is the convex "
        "bid stack showing up in a statistic, and it tells you a linear price model is "
        "mis-specified in precisely the region - high demand - where the battery earns its "
        "money. Note the limitation honestly: Spearman only detects MONOTONE non-linearity, so "
        "a genuine U-shape (demand versus temperature) shows up weakly in BOTH measures. That "
        "is why section 6 plots the curves directly.")


def c_with_price(df):
    cols = [c for c in _feat_cols(df) if c != PRICE]
    pe = df[cols].corrwith(df[PRICE])
    sp = df[cols].corrwith(df[PRICE], method="spearman")
    d = pd.DataFrame({"Pearson": pe, "Spearman": sp})
    d = d.reindex(d.abs().max(axis=1).sort_values().index)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=d.index, x=d["Pearson"], name="Pearson (linear)",
                         orientation="h", marker_color="#4c78a8"))
    fig.add_trace(go.Bar(y=d.index, x=d["Spearman"], name="Spearman (rank)",
                         orientation="h", marker_color="#d62728"))
    fig.add_vline(x=0, line_color="black")
    _style(fig, "5.3 Every feature ranked by its correlation with the PRICE",
           "Sorted by the larger of the two absolute values. Where the red bar clearly exceeds "
           "the blue one, the relationship is monotone but curved. This is a marginal ranking - "
           "it says nothing about what survives once the other features are known.",
           "correlation with price  [-1 .. 1]", "feature",
           height=max(650, 17 * len(d)), barmode="group")
    return fig, (
        "Your first feature shortlist - with one large caveat that you should raise before your "
        "supervisor does. Almost every variable here is correlated with the time of day, and "
        "the price is too, so a lot of these bars are measuring the daily cycle over and over "
        "again through different proxies. Solar radiation is the clearest example: it will "
        "appear to predict price in any region simply because it is zero at night. The next two "
        "figures strip that confounding out, and the comparison between this ranking and the "
        "one in 5.5 is worth a slide of its own - it is exactly the kind of thing EDA is "
        "supposed to catch before modelling starts.")


def c_by_hour(df):
    cands = [c for c in [DEMAND, "_net_import",
                         _first(df, "temperature_2m_obs", "temperature_2m_fc24h"),
                         _first(df, "wind_speed_100m_obs", "wind_speed_100m_fc24h"),
                         _first(df, "shortwave_radiation_obs", "shortwave_radiation_fc24h"),
                         _first(df, "cloud_cover_obs", "cloud_cover_fc24h"),
                         _first(df, "relative_humidity_2m_obs")] if c and _has(df, c)]
    m = pd.DataFrame({h: {c: df.loc[df["_hour"] == h, c].corr(df.loc[df["_hour"] == h, PRICE],
                                                              method="spearman")
                          for c in cands} for h in range(24)})
    labels = [c.replace("_obs", "").replace("_fc24h", "").replace("_net_import", "net import")
              for c in m.index]
    fig = go.Figure(go.Heatmap(z=m.to_numpy(), x=m.columns, y=labels, colorscale=DIV,
                               zmid=0, zmin=-1, zmax=1,
                               colorbar=dict(title="Spearman rho<br>with price")))
    _style(fig, "5.4 Correlation with price, computed separately for each hour",
           "Each cell is the rank correlation between that driver and the price, using only "
           "intervals from that hour. Because the hour is held fixed, the daily cycle can no "
           "longer manufacture a correlation.",
           HOUR_LAB, "driver", height=470)
    fig.update_xaxes(dtick=1)
    return fig, (
        "Watch for cells that change SIGN along a row. A variable whose correlation with price "
        "is positive in the morning and negative in the evening cannot be represented by a "
        "single coefficient, and no amount of feature scaling will fix that - it needs an "
        "interaction with the hour, or a model that builds interactions itself. Equally "
        "important is the row that goes pale: solar radiation in Tasmania should largely "
        "vanish here even though it looked respectable in 5.3, which is the confounding being "
        "removed in front of you. Compare the demand row against the wind row - in a hydro "
        "system it is quite possible that wind is the stronger hour-by-hour driver, and that "
        "would be a genuinely interesting finding to lead with.")


def c_residualised(df):
    """Marginal vs residualised correlation: remove the (hour x season) mean first."""
    cols = [c for c in _feat_cols(df) if c != PRICE]
    key = [df["_hour"].to_numpy(), df["_season"].astype(str).to_numpy()]
    y = df[PRICE] - df.groupby(key)[PRICE].transform("mean")
    marg, resi = {}, {}
    for c in cols:
        s = df[c]
        if s.notna().sum() < 500:
            continue
        marg[c] = s.corr(df[PRICE], method="spearman")
        resi[c] = (s - df.groupby(key)[c].transform("mean")).corr(y, method="spearman")
    d = pd.DataFrame({"marginal": marg, "after removing hour x season": resi}).dropna()
    d = d.reindex(d["after removing hour x season"].abs().sort_values().index).tail(25)
    fig = go.Figure()
    fig.add_trace(go.Bar(y=d.index, x=d["marginal"], name="marginal (raw)",
                         orientation="h", marker_color="#bbb"))
    fig.add_trace(go.Bar(y=d.index, x=d["after removing hour x season"],
                         name="after removing hour x season", orientation="h",
                         marker_color="#d62728"))
    fig.add_vline(x=0, line_color="black")
    _style(fig, "5.5 Which correlations survive once the daily and seasonal cycle is removed?",
           "Grey = the raw correlation from 5.3. Red = the same correlation computed on "
           "deviations from the mean of that (hour, season) cell. Top 25 features by the red "
           "value. A grey bar with almost no red bar next to it was pure calendar confounding.",
           "Spearman correlation with price  [-1 .. 1]", "feature",
           height=760, barmode="group")
    return fig, (
        "This is the most decision-relevant plot in the correlation section and it is the one "
        "that separates a careful project from a checkbox one. The red bars are what each "
        "feature adds ON TOP of simply knowing what time of year and time of day it is - and "
        "since your model will always know the calendar, that is the only contribution that "
        "counts. Features where grey is tall and red is flat should be dropped or, better, kept "
        "only as an interaction. Features where red is nearly as tall as grey are carrying "
        "genuine independent information and belong in the model. Be precise about what this is "
        "and is not: removing the cell mean is a coarse control, not a proper partial "
        "correlation, and it cannot tell you about causality. It is a screening tool that "
        "hands you a short, defensible list to take into the CDA stage, where you test those "
        "few relationships properly.")


# =====================================================================
#  SECTION 6 - what actually moves the price  (6 figures)
# =====================================================================
def m_bid_stack(df):
    if not _has(df, DEMAND):
        raise RuntimeError("no demand column")
    b = _binned(df[DEMAND], df[PRICE], bins=45)
    d = _sample(df.dropna(subset=[PRICE, DEMAND]), 50_000)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=d[DEMAND], y=d["_slog"], mode="markers", name="5-min intervals",
                               marker=dict(size=2.5, opacity=.2, color=d["_hour"],
                                           colorscale="Twilight",
                                           colorbar=dict(title="hour"))))
    for q, c, w in [("q90", "#d62728", 2), ("q75", "#ff7f0e", 1.5), ("q50", "black", 3),
                    ("q25", "#4c78a8", 1.5)]:
        fig.add_trace(go.Scatter(x=b["x"], y=_slog(b[q]), name=q.upper(),
                                 line=dict(color=c, width=w)))
    _style(fig, "6.1 The bid stack: price as a function of demand",
           "Conditional quantiles of the price in 45 equal-count demand bins. The vertical "
           "distance between the P90 and P50 lines is the upside risk at that demand level - "
           "it is usually far from constant.",
           DEMAND_LAB, PRICE_LOG_LAB, height=580)
    return fig, (
        "The classic hockey stick: flat while there is spare capacity, then convex once the "
        "expensive plant is needed. Three things to take away. (1) Locate the knee and quote "
        "the demand level in MW - that is a concrete, quotable finding and a natural place for "
        "a threshold feature. (2) Notice that the P90 line bends up long before the median "
        "does: high demand raises the RISK of a high price well before it raises the typical "
        "price, which is precisely the regime where a battery should be holding charge. (3) The "
        "convexity means demand-forecast errors are amplified asymmetrically into price errors, "
        "so a demand model with symmetric loss is the wrong intermediate target. In TAS1 expect "
        "this curve to be flatter than on the mainland, because hydro can respond quickly and "
        "Basslink imports cap the local price whenever the link has headroom - if your knee is "
        "weak, that is the reason, and it is worth a sentence.")


def m_price_surface_demand(df):
    if not _has(df, DEMAND):
        raise RuntimeError("no demand column")
    d = df.dropna(subset=[PRICE, DEMAND]).copy()
    d["_db"] = pd.qcut(d[DEMAND], 22, duplicates="drop")
    piv = d.pivot_table(index="_db", columns="_hour", values=PRICE, aggfunc="median",
                        observed=True).interpolate(axis=0).interpolate(axis=1)
    fig = go.Figure(go.Surface(z=piv.to_numpy(), x=piv.columns, y=[iv.mid for iv in piv.index],
                               colorscale="Inferno",
                               colorbar=dict(title="median price<br>[$/MWh]")))
    _style(fig, "6.2 Median price as a joint function of hour and demand",
           "Same demand level, different hour of day: if the surface is not flat along the hour "
           "axis, then load alone does not set the price and the time of day is carrying "
           "information about supply.", height=720,
           scene=dict(xaxis_title="hour of day  [local]", yaxis_title=DEMAND_LAB,
                      zaxis_title="median price  [$/MWh]"))
    return fig, (
        "Slice this mentally along the hour axis at a fixed demand. On the mainland that slice "
        "is strongly tilted because solar changes the residual supply through the day. In "
        "Tasmania the tilt should be weaker, and what remains is mostly hydro bidding behaviour "
        "and interconnector state. Either result is publishable in your report as long as you "
        "interpret it: a flat surface means demand is a sufficient statistic for price and your "
        "model can be simple, while a tilted one means you need the hour as a first-class "
        "feature interacting with demand rather than a nuisance to be removed.")


def m_wind(df):
    col = _first(df, "wind_speed_100m_obs", "wind_speed_100m_fc24h", "wind_speed_10m_obs")
    if col is None:
        raise RuntimeError("no wind column")
    b = _binned(df[col], df[PRICE], bins=30)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=b["x"], y=b["q50"], name="median price",
                             line=dict(color="black", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=b["x"], y=b["q90"], name="P90 price",
                             line=dict(color="#d62728", width=1.8, dash="dot")),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=b["x"], y=b["q25"], name="P25 price",
                             line=dict(color="#4c78a8", width=1.8, dash="dot")),
                  secondary_y=False)
    neg = df.groupby(pd.cut(df[col], bins=20), observed=True)["_neg"].mean() * 100
    fig.add_trace(go.Bar(x=[iv.mid for iv in neg.index], y=neg.values,
                         name="% intervals with negative price", marker_color="seagreen",
                         opacity=.35), secondary_y=True)
    fig.update_yaxes(title_text="price  [$/MWh]", secondary_y=False)
    fig.update_yaxes(title_text="negative-price frequency  [%]", secondary_y=True)
    _style(fig, "6.3 The merit-order effect: price against wind speed",
           "Conditional price quantiles in wind-speed bins, plus the share of intervals that "
           "went negative. In a region with a large wind fleet and little solar, this is the "
           "weather variable that actually matters.",
           "wind speed at 100 m  [m/s]", height=560)
    return fig, (
        "Expect the median line to slope downward and the green bars to climb steeply at the "
        "right-hand end. Quantify both: the price drop per m/s, and the wind speed above which "
        "negative prices become common. Those two numbers are directly actionable for the "
        "battery - a high-wind forecast is a charging signal that has nothing to do with the "
        "time of day - and they are a much stronger justification for keeping the wind columns "
        "than any correlation coefficient. Also compare the P90 line: if it stays flat while "
        "the median falls, wind lowers typical prices without removing spike risk, which means "
        "you should not let a windy forecast talk the battery out of holding reserve.")


def m_temperature(df):
    tcol = _first(df, "temperature_2m_obs", "temperature_2m_fc24h")
    if tcol is None:
        raise RuntimeError("no temperature column")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("price vs temperature", "demand vs temperature"))
    bp = _binned(df[tcol], df[PRICE], bins=30)
    fig.add_trace(go.Scatter(x=bp["x"], y=bp["q50"], name="median price",
                             line=dict(color="black", width=3)), 1, 1)
    fig.add_trace(go.Scatter(x=np.r_[bp["x"], bp["x"][::-1]],
                             y=np.r_[bp["q75"], bp["q25"][::-1]], fill="toself",
                             fillcolor="rgba(0,0,0,.12)", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"), 1, 1)
    if _has(df, DEMAND):
        bd = _binned(df[tcol], df[DEMAND], bins=30)
        fig.add_trace(go.Scatter(x=bd["x"], y=bd["mean"], name="mean demand",
                                 line=dict(color="#d62728", width=3)), 1, 2)
        fig.add_trace(go.Scatter(x=np.r_[bd["x"], bd["x"][::-1]],
                                 y=np.r_[bd["q75"], bd["q25"][::-1]], fill="toself",
                                 fillcolor="rgba(214,39,40,.12)", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"), 1, 2)
    fig.update_xaxes(title_text="air temperature at 2 m  [degC]")
    fig.update_yaxes(title_text="price  [$/MWh]  (band = P25-P75)", row=1, col=1)
    fig.update_yaxes(title_text=DEMAND_LAB + "  (band = P25-P75)", row=1, col=2)
    _style(fig, "6.4 Temperature: the heating branch that dominates in Tasmania",
           "Binned conditional curves, not a fitted line. For a cold, winter-peaking region the "
           "left (heating) branch should be steep and the right (cooling) branch short and "
           "flat - the opposite emphasis to a mainland region.", height=520)
    return fig, (
        "A DOWNWARD slope on the right-hand panel is the expected TAS1 result, not an error: "
        "colder weather means more electric heating means more demand. Verify the comfort "
        "minimum - the temperature at which demand bottoms out - and read the slope of the "
        "heating branch in MW per degree, because that is the number to quote. Then engineer "
        "the feature accordingly: heating degrees, HDD = max(0, T_base - T), with T_base taken "
        "from where this curve actually turns rather than from a textbook. Adding a cooling "
        "term as well is defensible but check the right-hand branch first; if Hobart barely "
        "reaches the temperatures where cooling load appears, a CDD feature will be almost "
        "always zero and will only add noise. Note also that the price panel is usually much "
        "flatter than the demand panel - temperature reaches price only through demand, and in "
        "a hydro system that transmission is weak.")


def m_flow_or_renewables(df):
    if df.attrs.get("has_flow"):
        b = _binned(df["_net_import"], df[PRICE], bins=30)
        fig = go.Figure()
        for q, c, w in [("q90", "#d62728", 1.8), ("q50", "black", 3), ("q25", "#4c78a8", 1.8)]:
            fig.add_trace(go.Scatter(x=b["x"], y=b[q], name=q.upper(),
                                     line=dict(color=c, width=w,
                                               dash="dot" if q != "q50" else "solid")))
        fig.add_vline(x=0, line_dash="dash", line_color="grey",
                      annotation_text="0 = balanced")
        _style(fig, "6.5 Price against net flow over the interconnector",
               "Positive x = Tasmania is importing from Victoria, negative = exporting. "
               "Conditional price quantiles per flow bin. In an interconnected hydro region "
               "this is often the strongest single predictor there is.",
               "net import over Basslink  [MW]   (positive = importing)",
               "price  [$/MWh]", height=540)
        note = (
            "If this curve is steep, the interconnector is your headline feature and the report "
            "should say so. The mechanism is worth explaining: when Tasmania imports heavily, "
            "the local price is being set by Victorian conditions and by whether the link is "
            "constrained; when the link binds, TAS1 decouples and can price far away from VIC1. "
            "The awkward part is honest and important - flow is CONTEMPORANEOUS, so you cannot "
            "use it at a 24-hour horizon unless you can forecast it, and forecasting Basslink "
            "flow is its own project. Two legitimate uses: as a diagnostic that explains your "
            "residuals, and as a feature at very short horizons (30 minutes) where the current "
            "flow is known. Do not let it into a 24-hour model without a forecast of its own.")
    else:
        col = _first(df, "shortwave_radiation_obs", "shortwave_radiation_fc24h")
        if col is None:
            raise RuntimeError("no flow and no radiation column")
        day = df[df[col] > 20]
        b = _binned(day[col], day[PRICE], bins=25)
        fig = go.Figure()
        for q, c in [("q90", "#d62728"), ("q50", "black"), ("q25", "#4c78a8")]:
            fig.add_trace(go.Scatter(x=b["x"], y=b[q], name=q.upper(),
                                     line=dict(color=c, width=2.5 if q == "q50" else 1.6)))
        _style(fig, "6.5 Price against solar radiation (daytime intervals only)",
               "No interconnector columns were found in the frame, so this shows the solar "
               "merit-order effect instead. In Tasmania it should be weak - there is very "
               "little PV - which is itself the finding.",
               "shortwave radiation  [W/m2]", "price  [$/MWh]", height=540)
        note = (
            "A flat set of curves here is the expected TAS1 result and worth stating: unlike "
            "NSW or SA, Tasmania has almost no solar, so radiation carries little merit-order "
            "signal and mostly acts as a proxy for daylight. Say this explicitly, because a "
            "reader who knows the mainland market will expect a midday trough and will wonder "
            "why it is missing. Then act on it: drop most of the radiation family and spend the "
            "feature budget on wind instead. Separately - you have no interconnector columns in "
            "this frame. Your own handler notes that Basslink flow is likely the strongest "
            "single predictor in TAS1. A free OpenElectricity key would add it, and it is "
            "probably the highest-value hour you could spend on this project.")
    return fig, note


def m_extremes_map(df):
    neg = df.pivot_table(index="_month", columns="_hour", values="_neg", aggfunc="mean") * 100
    spk = df.pivot_table(index="_month", columns="_hour", values="_spike", aggfunc="mean") * 100
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14,
                        subplot_titles=("share of intervals with price < 0  [%]",
                                        f"share above P99.5 = {df.attrs['spike_thr']:.0f} $/MWh  [%]"))
    fig.add_trace(go.Heatmap(z=neg.values, x=neg.columns, y=neg.index, colorscale="Blues",
                             colorbar=dict(title="%", x=0.43)), 1, 1)
    fig.add_trace(go.Heatmap(z=spk.values, x=spk.columns, y=spk.index, colorscale="Reds",
                             colorbar=dict(title="%", x=1.02)), 1, 2)
    fig.update_xaxes(title_text=HOUR_LAB, dtick=3)
    fig.update_yaxes(title_text="month  [1 = Jan]", dtick=1)
    _style(fig, "6.6 When exactly do the extremes happen?",
           "Two conditional frequency maps over the same (month, hour) grid. Blue = when you "
           "would love to be charging, red = when you would love to be discharging.",
           height=480)
    return fig, (
        "Treat these two maps as the null hypothesis for your battery controller: a rule that "
        "charges in the darkest blue cells and discharges in the darkest red cells needs no "
        "machine learning at all. Your ML dispatch has to beat it, and if it does not, that is "
        "still a legitimate result to report as long as you show the comparison. Also check how "
        "CONCENTRATED the red cells are. If spikes cluster in a few (month, hour) cells, a "
        "classifier restricted to those cells is a far easier problem than forecasting the "
        "price everywhere, and it is where a battery earns disproportionately.")


# =====================================================================
#  SECTION 7 - can the forecasts actually be used?  (3 figures)
# =====================================================================
def f_skill(df):
    bases = sorted({c.rsplit("_fc", 1)[0] for c in df.columns if "_fc" in c and c.endswith("h")})
    rows = []
    for b in bases:
        obs = f"{b}_obs"
        if not _has(df, obs):
            continue
        for lead in ("24", "48"):
            fc = f"{b}_fc{lead}h"
            if not _has(df, fc):
                continue
            e = (df[fc] - df[obs]).dropna()
            if len(e) < 100:
                continue
            rows.append({"variable": b, "lead": f"{lead} h", "MAE": e.abs().mean(),
                         "bias": e.mean(), "corr": df[fc].corr(df[obs])})
    if not rows:
        raise RuntimeError("no forecast/observation pairs to compare")
    r = pd.DataFrame(rows)
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.09,
                        subplot_titles=("mean absolute error", "bias (forecast - observed)",
                                        "correlation with the observation"))
    for i, (m, unit) in enumerate([("MAE", "native units"), ("bias", "native units"),
                                   ("corr", "-1 .. 1")], start=1):
        for lead, c in [("24 h", "#4c78a8"), ("48 h", "#d62728")]:
            s = r[r["lead"] == lead]
            fig.add_trace(go.Bar(x=s["variable"], y=s[m], name=lead, marker_color=c,
                                 showlegend=(i == 1)), 1, i)
        fig.update_yaxes(title_text=unit, row=1, col=i)
    fig.add_hline(y=0, line_color="black", row=1, col=2)
    fig.update_xaxes(title_text="weather variable", tickangle=-40)
    _style(fig, "7.1 How good are the forecasts you will actually have at run time?",
           "Each weather variable compared against its ERA5 observation, separately for the "
           "24-hour and the 48-hour vintage. Units differ per variable, so compare the two "
           "bars within a variable rather than across variables.",
           height=520, barmode="group")
    return fig, (
        "This figure protects you from the single most common way a forecasting project "
        "silently cheats. At run time you have only the _fc columns; the _obs columns are "
        "hindsight, and your handler is right to refuse them unless allow_hindsight=True. What "
        "this plot adds is the SIZE of the penalty. Variables where the 48-hour bar is much "
        "worse than the 24-hour bar will degrade your day-ahead model specifically, so if you "
        "report accuracy across several horizons, expect - and explain - a step down as you "
        "cross the 24-hour boundary where build_supervised_frame switches vintage. A nonzero "
        "bias is good news, because a constant offset can simply be subtracted; a large MAE "
        "with near-zero bias cannot. Finally, this gives you the design of a genuinely "
        "interesting experiment: train one model on _obs and one on _fc, and report the gap as "
        "'the cost of imperfect weather knowledge'. That is a real result, and it is the "
        "counterfactual your _obs columns exist for.")


def f_signal_loss(df):
    bases = sorted({c.rsplit("_fc", 1)[0] for c in df.columns if "_fc" in c and c.endswith("h")})
    rows = {}
    for b in bases:
        e = {}
        for lbl, col in [("observed", f"{b}_obs"), ("fc 24h", f"{b}_fc24h"),
                         ("fc 48h", f"{b}_fc48h")]:
            if _has(df, col):
                e[lbl] = df[col].corr(df[PRICE], method="spearman")
        if len(e) >= 2:
            rows[b] = e
    if not rows:
        raise RuntimeError("nothing to compare")
    d = pd.DataFrame(rows).T
    d = d.reindex(d.abs().max(axis=1).sort_values(ascending=False).index)
    fig = go.Figure()
    for c, col in [("observed", "#2ca02c"), ("fc 24h", "#4c78a8"), ("fc 48h", "#d62728")]:
        if c in d.columns:
            fig.add_trace(go.Bar(x=d.index, y=d[c], name=c, marker_color=col))
    fig.add_hline(y=0, line_color="black")
    _style(fig, "7.2 How much price signal survives the forecast error?",
           "Rank correlation of each weather variable with the price, computed once with the "
           "perfect observation and once with each forecast vintage. The drop from green to "
           "blue to red is the signal you lose by having to predict the weather first.",
           "weather variable", "Spearman correlation with price  [-1 .. 1]",
           height=520, barmode="group")
    fig.update_xaxes(tickangle=-40)
    return fig, (
        "A cleaner way to prioritise features than raw correlation, because it ranks variables "
        "by the signal you can actually USE. A variable with a strong green bar and a much "
        "weaker red bar is one whose value evaporates at longer horizons - keep it for the "
        "short-horizon model and drop it from the day-ahead one, which is easy to do since you "
        "build a separate supervised frame per horizon anyway. Where green and red are almost "
        "equal, the forecast is good enough that the variable is safe at any horizon; for "
        "smooth, large-scale fields such as temperature and pressure this is usually the case, "
        "while wind at a point is typically the one that degrades most - and in TAS1 wind is "
        "also the one you most want, which is a tension worth naming in the report.")


def f_error_structure(df):
    pairs = [(b, f"{b}_fc24h", f"{b}_obs") for b in
             ["temperature_2m", "wind_speed_100m"]]
    pairs = [p for p in pairs if _has(df, p[1], p[2])]
    if not pairs:
        raise RuntimeError("no fc24h/obs pair for temperature or wind")
    fig = make_subplots(rows=1, cols=len(pairs) * 2, horizontal_spacing=0.07,
                        subplot_titles=sum([[f"{b}: error distribution",
                                             f"{b}: error by hour"] for b, _, _ in pairs], []))
    for i, (b, fc, obs) in enumerate(pairs):
        e = (df[fc] - df[obs])
        fig.add_trace(go.Histogram(x=e, nbinsx=70, marker_color="#4c78a8"), 1, 2 * i + 1)
        fig.add_vline(x=0, line_dash="dash", line_color="black", row=1, col=2 * i + 1)
        g = pd.DataFrame({"e": e, "h": df["_hour"]}).dropna()
        fig.add_trace(go.Box(x=g["h"], y=g["e"], marker_color="#d62728",
                             boxpoints=False), 1, 2 * i + 2)
        unit = "degC" if "temperature" in b else "m/s"
        fig.update_xaxes(title_text=f"forecast error  [{unit}]", row=1, col=2 * i + 1)
        fig.update_yaxes(title_text="count", row=1, col=2 * i + 1)
        fig.update_xaxes(title_text=HOUR_LAB, dtick=6, row=1, col=2 * i + 2)
        fig.update_yaxes(title_text=f"forecast error  [{unit}]", row=1, col=2 * i + 2)
    _style(fig, "7.3 Structure of the 24-hour forecast error",
           "Left of each pair: is the error centred and symmetric? Right: does it depend on the "
           "time of day? A structured error is a correctable error.",
           height=480, showlegend=False)
    return fig, (
        "Any structure you find here is free accuracy. A constant offset is removed with a "
        "single subtraction. A time-of-day pattern - forecasts too warm at dawn, say - is "
        "removed by de-biasing per hour, which is three lines of pandas and belongs in the "
        "feature pipeline rather than in the model. What you should NOT do is fit that "
        "correction on the whole sample and then evaluate on part of it; estimate the bias on "
        "the training window only, exactly as you would any other learned parameter, or you "
        "have introduced a subtle leak of the same family your handler works so hard to "
        "prevent.")


# =====================================================================
#  SECTION 8 - battery economics  (6 figures)
# =====================================================================
def b_spread(df):
    g = df.groupby("_date")[PRICE]
    sp = (g.max() - g.min()).dropna()
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.13, row_heights=[0.55, 0.45],
                        subplot_titles=("daily max minus daily min",
                                        "distribution of the daily spread (99th pct clipped)"))
    fig.add_trace(go.Scatter(x=sp.index, y=sp.values, mode="lines", name="daily spread",
                             line=dict(width=.7, color="#4c78a8")), 1, 1)
    fig.add_trace(go.Scatter(x=sp.index, y=sp.rolling(30).median(), name="30-day median",
                             line=dict(width=2.5, color="#d62728")), 1, 1)
    fig.add_trace(go.Histogram(x=sp.clip(0, sp.quantile(.99)), nbinsx=70,
                               marker_color="teal"), 2, 1)
    fig.update_yaxes(title_text="price spread  [$/MWh]", type="log", row=1, col=1)
    fig.update_xaxes(title_text="date  [local time]", row=1, col=1)
    fig.update_xaxes(title_text="daily price spread  [$/MWh]", row=2, col=1)
    fig.update_yaxes(title_text="number of days  [count]", row=2, col=1)
    _style(fig, "8.1 The arbitrage opportunity, day by day",
           f"Median daily spread {sp.median():.0f} $/MWh, 90th percentile {sp.quantile(.9):.0f}, "
           f"maximum {sp.max():.0f}. The top panel is on a log axis because the distribution is "
           f"extremely skewed.", height=700, showlegend=True)
    return fig, (
        f"The daily spread is the raw material your battery converts into money, so anchor the "
        f"whole project on the median value of {sp.median():.0f} $/MWh. Multiply it by your "
        f"usable capacity in MWh and by round-trip efficiency to get the best possible revenue "
        f"on a typical day, then compare that against the retail import adder discussed in your "
        f"handler's README_NOTES. Look hard at the histogram's right tail as well: if the mean "
        f"spread sits far above the median, annual revenue is dominated by a handful of days, "
        f"and a controller that misses those days loses most of the value while still looking "
        f"fine on average error. That is the argument for evaluating your dispatch on realised "
        f"dollars rather than on forecast MAE - they are not the same objective, and saying so "
        f"is one of the strongest points you can make in the report.")


def b_timing(df):
    imin = df.groupby("_date")[PRICE].idxmin().dropna()
    imax = df.groupby("_date")[PRICE].idxmax().dropna()
    hmin = pd.DatetimeIndex(imin).hour
    hmax = pd.DatetimeIndex(imax).hour
    seas = df.groupby("_date")["_season"].first().reindex(pd.DatetimeIndex(imin).normalize())
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.12,
                        subplot_titles=("when the daily minimum / maximum occurs",
                                        "hour of the daily maximum, by season"))
    fig.add_trace(go.Histogram(x=hmin, name="daily MIN (charge here)", nbinsx=24,
                               marker_color="#4c78a8", opacity=.75), 1, 1)
    fig.add_trace(go.Histogram(x=hmax, name="daily MAX (discharge here)", nbinsx=24,
                               marker_color="#d62728", opacity=.75), 1, 1)
    ct = pd.crosstab(pd.Series(seas.to_numpy(), name="season"),
                     pd.Series(hmax, name="hour"), normalize="index") * 100
    fig.add_trace(go.Heatmap(z=ct.to_numpy(), x=ct.columns, y=ct.index.astype(str),
                             colorscale=SEQ, colorbar=dict(title="% of days", x=1.02)), 1, 2)
    fig.update_xaxes(title_text=HOUR_LAB, dtick=2)
    fig.update_yaxes(title_text="number of days  [count]", row=1, col=1)
    fig.update_yaxes(title_text="season", row=1, col=2)
    _style(fig, "8.2 Is the timing of the best trade stable enough to schedule blindly?",
           "Left: histograms of the hour containing each day's cheapest and dearest interval. "
           "Right: the same maximum-hour distribution split by season, since the evening peak "
           "moves with sunset and with heating load.", height=500, barmode="overlay")
    return fig, (
        "This measures how much a forecast is worth. Tight, well-separated histograms mean the "
        "timing is nearly deterministic, a fixed schedule captures most of the value, and your "
        "ML controller can only add a few percent - a perfectly respectable finding, but one "
        "you want to know at the start rather than discover in the conclusion. Broad or "
        "overlapping histograms mean the timing genuinely varies day to day, which is where "
        "forecasting pays. The seasonal panel refines this: if the peak hour shifts by two or "
        "three hours between winter and summer, even the fixed-schedule benchmark should be "
        "season-dependent, and comparing against a naive year-round schedule would flatter your "
        "model unfairly.")


def b_duration(df):
    fig = go.Figure()
    for s in SEASON_ORDER:
        v = df.loc[df["_season"] == s, PRICE].dropna().sort_values(ascending=False)
        if len(v) < 500:
            continue
        x = np.linspace(0, 100, len(v))
        st = max(1, len(v) // 3000)
        fig.add_trace(go.Scatter(x=x[::st], y=_slog(v.to_numpy())[::st], name=s,
                                 line=dict(width=2)))
    fig.add_hline(y=0, line_dash="dot", line_color="grey", annotation_text="$0/MWh")
    _style(fig, "8.3 Price duration curves by season",
           "Prices sorted from highest to lowest within each season. A steep left edge means "
           "the money is concentrated in very few intervals; a long tail below zero means "
           "frequent free or paid charging.",
           "share of intervals in the season that exceed this price  [%]",
           PRICE_LOG_LAB, height=520)
    return fig, (
        "Read the two ends separately, because the battery uses them differently. The left edge "
        "is the discharge opportunity: how steep it is tells you how much of the annual revenue "
        "sits in the top 1% of intervals, and a very steep edge means your controller must be "
        "right on a small number of occasions rather than roughly right all year. The right end "
        "is the charging opportunity: wherever the curve dips below the zero line you are being "
        "paid to charge, which is the single most profitable thing a battery can do. Compare "
        "the seasons - in a wind-heavy hydro region the negative tail is usually a spring and "
        "autumn phenomenon, and if so, your battery's economics are seasonal and the annual "
        "average is a misleading summary.")


def b_premium(df):
    piv = df.pivot_table(index="_hour", columns="_month", values=PRICE, aggfunc="median")
    z = piv.to_numpy() - np.nanmedian(piv.to_numpy(), axis=0, keepdims=True)
    fig = go.Figure(go.Heatmap(z=z, x=piv.columns, y=piv.index, colorscale=DIV, zmid=0,
                               colorbar=dict(title="deviation from<br>monthly median<br>[$/MWh]")))
    _style(fig, "8.4 Hour-of-day price premium within each month",
           "Each column is one month with its own median removed, so the colours show the "
           "within-day pattern only. Blue = systematically cheap hours (charge), red = "
           "systematically expensive hours (discharge).",
           "month  [1 = Jan]", HOUR_LAB, height=580)
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(dtick=2)
    return fig, (
        "This table IS a dispatch policy: read off the bluest hour and the reddest hour for each "
        "month and you have a seasonal fixed schedule, derived entirely from history, with no "
        "model at all. Implement it as your benchmark controller - it takes ten lines and it is "
        "a far more honest comparison than 'do nothing'. Then look at how much the pattern "
        "shifts across the months: a stable pattern means the benchmark is strong and your ML "
        "controller must earn its keep on the unusual days, while a pattern that moves around "
        "means the schedule is unreliable and forecasting has obvious value. Either way you now "
        "have a number to beat instead of an open-ended goal.")


def b_arbitrage(df, kwh=13.5, kw=5.0, rte=0.90):
    """Perfect-foresight vs fixed-schedule daily arbitrage revenue (upper bound)."""
    step_h = df.attrs["step_min"] / 60.0
    e_int = kw * step_h                                   # kWh moved per interval at full power
    k = max(1, int(np.ceil(kwh / e_int)))                 # intervals to fill the battery
    piv = df.pivot_table(index="_date", columns="_tod", values=PRICE, aggfunc="mean")
    piv = piv.dropna(thresh=int(0.9 * piv.shape[1]))
    if len(piv) < 30:
        raise RuntimeError("not enough complete days")
    arr = piv.to_numpy()
    srt = np.sort(arr, axis=1)
    buy = np.nanmean(srt[:, :k], axis=1)
    sell = np.nanmean(srt[:, -k:], axis=1)
    perfect = (sell * np.sqrt(rte) - buy / np.sqrt(rte)) * (kwh / 1000.0)   # $/day
    # fixed schedule: historically cheapest and dearest hour, same every day
    prof = df.groupby("_hour")[PRICE].median()
    ch, dh = int(prof.idxmin()), int(prof.idxmax())
    cols = np.array(piv.columns, dtype=float)
    cmask = (cols >= ch) & (cols < ch + 1)
    dmask = (cols >= dh) & (cols < dh + 1)
    fixed = (np.nanmean(arr[:, dmask], axis=1) * np.sqrt(rte)
             - np.nanmean(arr[:, cmask], axis=1) / np.sqrt(rte)) * (kwh / 1000.0)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.14, row_heights=[0.55, 0.45],
                        subplot_titles=("cumulative revenue over the sample",
                                        "distribution of daily revenue"))
    fig.add_trace(go.Scatter(x=piv.index, y=np.nancumsum(perfect), name="perfect foresight",
                             line=dict(color="#2ca02c", width=2.5)), 1, 1)
    fig.add_trace(go.Scatter(x=piv.index, y=np.nancumsum(fixed),
                             name=f"fixed schedule (charge {ch:02d}h, discharge {dh:02d}h)",
                             line=dict(color="#4c78a8", width=2.5, dash="dash")), 1, 1)
    fig.add_trace(go.Histogram(x=perfect, name="perfect foresight", nbinsx=60,
                               marker_color="#2ca02c", opacity=.65), 2, 1)
    fig.add_trace(go.Histogram(x=fixed, name="fixed schedule", nbinsx=60,
                               marker_color="#4c78a8", opacity=.65), 2, 1)
    fig.add_vline(x=0, line_dash="dash", line_color="black", row=2, col=1)
    fig.update_yaxes(title_text="cumulative revenue  [$]", row=1, col=1)
    fig.update_xaxes(title_text="date  [local time]", row=1, col=1)
    fig.update_xaxes(title_text="revenue per day  [$/day]", row=2, col=1)
    fig.update_yaxes(title_text="number of days  [count]", row=2, col=1)
    gap = np.nansum(perfect) - np.nansum(fixed)
    _style(fig, f"8.5 What is the forecast actually worth? ({kwh:.1f} kWh / {kw:.1f} kW, "
                f"round trip {100*rte:.0f}%)",
           f"Perfect foresight (green) charges in the {k} cheapest intervals of each day and "
           f"discharges in the {k} dearest; the fixed schedule (blue) always uses the same "
           f"hours. Over the sample the gap is ${gap:,.0f} - that is the entire prize your "
           f"forecasting model is competing for. Wholesale prices only, no network charges.",
           height=740, barmode="overlay")
    return fig, (
        f"The most important number in your project comes from this figure: perfect foresight "
        f"earns ${np.nansum(perfect):,.0f} over the sample, the fixed schedule "
        f"${np.nansum(fixed):,.0f}, so everything a forecast could possibly buy you is the "
        f"${gap:,.0f} in between. Any realistic model captures a fraction of that gap; quoting "
        f"the fraction is a far more meaningful result than quoting an MAE. Be explicit about "
        f"the assumptions, because they all flatter the battery: this is an upper bound that "
        f"ignores the requirement that charging must precede discharging within the day, allows "
        f"exactly one cycle, ignores degradation, and settles at wholesale prices with no "
        f"network charge or retail margin. Recompute the green line with the ~$200/MWh import "
        f"adder from your README_NOTES and check whether it stays positive at all - if it does "
        f"not, you have found the real answer to your research question, and it is a more "
        f"interesting one than a marginally better RMSE.")


def b_predictability(df):
    g = df.groupby("_date")[PRICE]
    sp = (g.max() - g.min()).dropna()
    a, n = _acf(sp, min(40, len(sp) // 3))
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], horizontal_spacing=0.12,
                        subplot_titles=("autocorrelation of the daily spread",
                                        "today's spread vs yesterday's"))
    fig.add_trace(go.Bar(x=np.arange(len(a)), y=a, marker_color="#4c78a8"), 1, 1)
    fig.add_hline(y=1.96 / np.sqrt(n), line_dash="dot", line_color="grey", row=1, col=1)
    prev, cur = pd.Series(_slog(sp.shift(1))), pd.Series(_slog(sp))
    fig.add_trace(go.Scattergl(x=prev, y=cur, mode="markers",
                               marker=dict(size=4, opacity=.45, color="#d62728")), 1, 2)
    lo, hi = float(np.nanmin(cur)), float(np.nanmax(cur))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="black", dash="dash")), 1, 2)
    fig.update_xaxes(title_text="lag  [days]", row=1, col=1)
    fig.update_yaxes(title_text="autocorrelation  [-1 .. 1]", row=1, col=1)
    fig.update_xaxes(title_text="yesterday's spread  [signed log10 $/MWh]", row=1, col=2)
    fig.update_yaxes(title_text="today's spread  [signed log10 $/MWh]", row=1, col=2)
    r = prev.corr(cur)
    _style(fig, f"8.6 Is a profitable day predictable from the day before? (r = {r:.2f})",
           "If the daily spread is autocorrelated, then yesterday already tells you something "
           "about whether today is worth trading hard - a cheap feature that needs no weather "
           "model at all.", height=500, showlegend=False)
    return fig, (
        "A significant bar at lag 1 means the profitable days cluster, and clustering is "
        "exploitable: a battery can raise its reserve target after a high-spread day without "
        "forecasting anything. If the bars die immediately, spread is essentially "
        "unpredictable from its own past and all the value has to come from weather and load "
        "features - which sharpens the case for the model but also lowers your expectations. "
        "Either way, put yesterday's spread and yesterday's maximum price into the feature set "
        "and let the model decide; they cost nothing, they are strictly backward-looking, and "
        "they are already available at the origin so there is no leakage risk.")


# =====================================================================
#  SECTION 9 - price regimes  (3 figures)
# =====================================================================
def _daily_shape(df, col, per_hour=1):
    piv = df.pivot_table(index="_date", columns="_hour", values=col, aggfunc="median")
    piv = piv.dropna(axis=0, thresh=20).interpolate(axis=1, limit_direction="both")
    return piv.dropna()


def r_clusters(df, k=4):
    piv = _daily_shape(df, "_slog")
    X = piv.to_numpy()
    Xn = (X - X.mean(1, keepdims=True))
    lab, cen = _cluster(Xn, k)
    order = np.argsort([-c.max() for c in cen])
    fig = make_subplots(rows=1, cols=2, column_widths=[0.56, 0.44], horizontal_spacing=0.12,
                        subplot_titles=("cluster centroids: typical daily price SHAPES",
                                        "share of each month's days in each cluster"))
    for rank, j in enumerate(order):
        fig.add_trace(go.Scatter(x=np.arange(24), y=cen[j],
                                 name=f"C{rank} - {int((lab == j).sum())} days",
                                 line=dict(width=3)), 1, 1)
    remap = {j: r for r, j in enumerate(order)}
    lab2 = np.array([remap[l] for l in lab])
    share = pd.crosstab(piv.index.month, lab2, normalize="index") * 100
    fig.add_trace(go.Heatmap(z=share.to_numpy(), x=[f"C{c}" for c in share.columns],
                             y=share.index, colorscale=SEQ,
                             colorbar=dict(title="% of days<br>in that month", x=1.02)), 1, 2)
    fig.update_xaxes(title_text=HOUR_LAB, dtick=3, row=1, col=1)
    fig.update_yaxes(title_text="price shape  [signed log10, daily mean removed]", row=1, col=1)
    fig.update_xaxes(title_text="cluster", row=1, col=2)
    fig.update_yaxes(title_text="month  [1 = Jan]", dtick=1, row=1, col=2)
    _style(fig, f"9.1 Typical daily price shapes ({k}-means on de-meaned daily profiles)",
           "The daily mean is removed before clustering, so these are SHAPES, not levels - two "
           "days with the same pattern but different price levels land in the same cluster. "
           "That is what a battery cares about.", height=540)
    return fig, (
        "Name the clusters in your report; it makes the rest of the analysis much easier to "
        "discuss. Typically you will find a flat day with almost no spread (nothing to trade), "
        "an evening-peak day (the standard case), a day with a deep negative midday or "
        "overnight trough (charge cheaply), and a spiky day (all the revenue in one or two "
        "intervals). Then ask the question that matters: can you predict tomorrow's cluster "
        "from information available today? Classifying four regimes is a much easier learning "
        "problem than regressing 48 half-hourly prices, and for dispatch it may be nearly as "
        "useful - a strong, tractable secondary experiment if you have time.")


def r_cluster_calendar(df, k=4):
    piv = _daily_shape(df, "_slog")
    X = piv.to_numpy()
    lab, _ = _cluster(X - X.mean(1, keepdims=True), k)
    s = pd.DataFrame({"date": piv.index, "c": lab})
    s["dow"] = s["date"].dt.dayofweek
    s["season"] = np.select(
        [s["date"].dt.month.isin([12, 1, 2]), s["date"].dt.month.isin([3, 4, 5]),
         s["date"].dt.month.isin([6, 7, 8]), s["date"].dt.month.isin([9, 10, 11])],
        SEASON_ORDER, default="NA")
    ct1 = pd.crosstab(s["dow"], s["c"], normalize="index") * 100
    ct2 = pd.crosstab(s["season"], s["c"], normalize="index") * 100
    ct2 = ct2.reindex([x for x in SEASON_ORDER if x in ct2.index])
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14,
                        subplot_titles=("by day of week", "by season"))
    fig.add_trace(go.Heatmap(z=ct1.to_numpy(), x=[f"C{c}" for c in ct1.columns],
                             y=[DOW_NAMES[i] for i in ct1.index], colorscale=SEQ,
                             colorbar=dict(title="%", x=0.43)), 1, 1)
    fig.add_trace(go.Heatmap(z=ct2.to_numpy(), x=[f"C{c}" for c in ct2.columns],
                             y=list(ct2.index), colorscale=SEQ,
                             colorbar=dict(title="%", x=1.02)), 1, 2)
    fig.update_xaxes(title_text="cluster")
    _style(fig, "9.2 Are the price regimes explained by the calendar?",
           "Row-normalised: each row sums to 100%. If a cluster were purely a weekend effect it "
           "would light up only in the Sat/Sun rows. Clusters spread evenly across the calendar "
           "are driven by something else - weather, or market state.", height=440)
    return fig, (
        "The useful outcome here is a NEGATIVE one. If the clusters map cleanly onto weekday "
        "and season, they add nothing your calendar features do not already encode. If they cut "
        "across the calendar, then a large part of the daily price shape is driven by variables "
        "your model has to learn from weather and market state - which is both the "
        "justification for the whole modelling exercise and a warning that calendar-only "
        "baselines will have a ceiling. Quote the strongest and weakest association you find.")


def r_spike_days(df):
    thr = df.attrs["spike_thr"]
    daily = df.groupby("_date")["_spike"].max()
    spike_days = set(daily[daily > 0].index)
    if len(spike_days) < 5:
        raise RuntimeError("too few spike days")
    is_sp = df["_date"].isin(spike_days)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for lbl, mask, c in [(f"days containing a spike (n={len(spike_days)})", is_sp, "#d62728"),
                         (f"all other days (n={df['_date'].nunique()-len(spike_days)})",
                          ~is_sp, "#4c78a8")]:
        m = df[mask].groupby("_tod")[PRICE].median()
        fig.add_trace(go.Scatter(x=m.index, y=m.values, name=lbl,
                                 line=dict(color=c, width=2.5)), secondary_y=False)
        if _has(df, DEMAND):
            dm = df[mask].groupby("_tod")[DEMAND].mean()
            fig.add_trace(go.Scatter(x=dm.index, y=dm.values, name=lbl + " - demand",
                                     line=dict(color=c, width=1.5, dash="dot"),
                                     showlegend=False), secondary_y=True)
    fig.update_yaxes(title_text="median price  [$/MWh]", secondary_y=False)
    fig.update_yaxes(title_text=DEMAND_LAB + "  (dotted)", secondary_y=True)
    _style(fig, f"9.3 What does a spike day look like BEFORE the spike?",
           f"Median price (solid) and mean demand (dotted) through the day, comparing days that "
           f"contained at least one interval above {thr:.0f} $/MWh against all other days.",
           HOUR_LAB, height=540)
    fig.update_xaxes(dtick=1)
    return fig, (
        "Look at the morning, well before the spike happens. If the red curves sit above the "
        "blue ones from early in the day, then a spike day is identifiable hours in advance and "
        "a day-ahead classifier is worth building - the morning price level and the demand "
        "forecast are your features. If the two sets of curves are indistinguishable until the "
        "spike itself, spikes arrive without warning from the data you have, and the honest "
        "conclusion is that your battery should hold a precautionary reserve rather than try to "
        "time them. That conclusion is worth stating explicitly; a project that establishes "
        "what is NOT predictable, with evidence, is stronger than one that quietly avoids the "
        "question.")


# =====================================================================
#  registry
# =====================================================================
REGISTRY = [
    ("quality",   "1.1 missing values",              q_missing),
    ("quality",   "1.2 gaps over time",              q_missing_time),
    ("quality",   "1.3 grid completeness",           q_coverage),

    ("price",     "2.1 price history",               p_history),
    ("price",     "2.2 price distribution",          p_distribution),
    ("price",     "2.3 price ECDF",                  p_ecdf),
    ("price",     "2.4 Q-Q plots",                   p_qq),
    ("price",     "2.5 price by hour (violins)",     p_violin_hour),
    ("price",     "2.6 extreme regimes by month",    p_regime_over_time),

    ("shape",     "3.1 daily price profile",         s_daily_profile),
    ("shape",     "3.2 seasonal price profiles",     s_seasonal_profile),
    ("shape",     "3.3 price calendar heat map",     s_calendar_heat),
    ("shape",     "3.4 price surface",               s_price_surface),
    ("shape",     "3.5 weekly price rhythm",         s_dow_hour),
    ("shape",     "3.6 monthly seasonality",         s_monthly_box),

    ("price_dyn", "4.1 ACF / PACF of price",         d_acf_price),
    ("price_dyn", "4.2 volatility clustering",       d_volatility),
    ("price_dyn", "4.3 lag scatter of price",        d_lag_scatter),
    ("price_dyn", "4.4 anatomy of a spike",          d_spike_event),
    ("price_dyn", "4.5 price decomposition",         d_decomposition),
    ("price_dyn", "4.6 cross-correlation drivers",   d_ccf),

    ("corr",      "5.1 correlation matrix",          c_matrix),
    ("corr",      "5.2 non-linearity map",           c_nonlinear),
    ("corr",      "5.3 correlation with price",      c_with_price),
    ("corr",      "5.4 correlation by hour",         c_by_hour),
    ("corr",      "5.5 residualised correlation",    c_residualised),

    ("drivers",   "6.1 bid stack",                   m_bid_stack),
    ("drivers",   "6.2 price surface hour x demand", m_price_surface_demand),
    ("drivers",   "6.3 wind and merit order",        m_wind),
    ("drivers",   "6.4 temperature branches",        m_temperature),
    ("drivers",   "6.5 interconnector / renewables", m_flow_or_renewables),
    ("drivers",   "6.6 map of extremes",             m_extremes_map),

    ("forecast",  "7.1 forecast skill",              f_skill),
    ("forecast",  "7.2 usable signal by vintage",    f_signal_loss),
    ("forecast",  "7.3 forecast error structure",    f_error_structure),

    ("battery",   "8.1 daily spread",                b_spread),
    ("battery",   "8.2 timing of min and max",       b_timing),
    ("battery",   "8.3 price duration curves",       b_duration),
    ("battery",   "8.4 hourly premium table",        b_premium),
    ("battery",   "8.5 value of a perfect forecast", b_arbitrage),
    ("battery",   "8.6 spread predictability",       b_predictability),

    ("regimes",   "9.1 daily price shapes",          r_clusters),
    ("regimes",   "9.2 regimes vs calendar",         r_cluster_calendar),
    ("regimes",   "9.3 anatomy of a spike day",      r_spike_days),
]

SECTIONS = list(dict.fromkeys(s for s, _, _ in REGISTRY))


# =====================================================================
#  ENTRY POINT
# =====================================================================
def apply_eda(data,
              sections=None,
              show=True,
              save_html=False,
              out_dir="eda_plots",
              region=None,
              time_col=None,
              renderer=None,
              battery_kwh=13.5,
              battery_kw=5.0,
              round_trip_efficiency=0.90,
              verbose=True):
    """Run the price-focused EDA.

    data        DataFrame from build_dataset(), or a path to csv/parquet/pkl
    sections    subset of: quality, price, shape, price_dyn, corr, drivers,
                forecast, battery, regimes    (None = all)
    show        call fig.show() on every figure
    save_html   also write standalone .html files into out_dir
    renderer    force a plotly renderer, e.g. "browser" for a plain script
    battery_*   sizing used by figure 8.5 (defaults: a Powerwall-ish 13.5 kWh / 5 kW)
    """
    if renderer:
        pio.renderers.default = renderer

    df = prepare(data, time_col=time_col, region=region)

    todo = [x for x in REGISTRY if sections is None or x[0] in sections]
    if save_html:
        os.makedirs(out_dir, exist_ok=True)

    figs, notes, failed = {}, [], []
    print(f"\n[eda] building {len(todo)} figures\n" + "=" * 72)
    for sec, name, fn in todo:
        try:
            if fn is b_arbitrage:
                fig, note = fn(df, kwh=battery_kwh, kw=battery_kw,
                               rte=round_trip_efficiency)
            else:
                fig, note = fn(df)
            fig.update_layout(template=TEMPLATE)
            figs[name] = fig
            notes.append((sec, name, note))
            if verbose:
                print(f"  [ok]   {name}")
            if save_html:
                fig.write_html(os.path.join(out_dir, name.replace(" ", "_") + ".html"),
                               include_plotlyjs="cdn")
            if show:
                fig.show()
        except Exception as exc:                                    # noqa: BLE001
            failed.append((name, repr(exc)))
            print(f"  [skip] {name}  ->  {exc}")

    print("\n" + "=" * 72 + "\nHOW TO READ EACH FIGURE\n" + "=" * 72)
    cur = None
    for sec, name, note in notes:
        if sec != cur:
            cur = sec
            print(f"\n{'=' * 72}\n  SECTION: {sec.upper()}\n{'=' * 72}")
        print(f"\n>>> {name}")
        for line in textwrap.wrap(note, 100):
            print(f"    {line}")
    if failed:
        print("\n" + "=" * 72 + "\nSKIPPED\n" + "=" * 72)
        for n, e in failed:
            print(f"  - {n}: {e}")
    print(f"\n[done] {len(figs)} figures"
          + (f" written to ./{out_dir}" if save_html else ""))
    return figs


