#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8_H8_transit_network_vs_economic_geographic_factors.py — Streamlit "H8" page
-----------------------------------------------------------------------------

Hypothesis (H8):
  Global transit network connectivity patterns reflect economic and geographic factors.

Live-data approach
  • Connectivity proxy: Day-of-week ridership index (0–100) derived from live open datasets
    - New York (NYC): Hourly -> weekday average
    - Singapore (SG): Daily/Monthly -> weekday average
    - London (LON): Weekly -> daily approx -> weekday average
  • Economic factor: GDP per capita (World Bank live API by country)
  • Geographic factor: region (by country)

UI
  • Data Load Summary (status, rows)
  • Sidebar: Region filter, GDP range (if available), log toggle, Top-N
  • Visuals: Heatmap (DoW patterns), Scatter GDP vs ridership density (R²), Top-N bar, Correlation heatmap, Table
  • Narrative pills: About / Discovery / Hypothesis / Audience

Note
  • This page uses live HTTP calls with caching; network access must be permitted.
"""

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------- Streamlit page config & styles ----------------------
st.set_page_config(page_title="H8 — Transit Connectivity vs Economic & Geographic (Live)", layout="wide")
st.title("H8 — Global Transit Connectivity vs Economic & Geographic Factors (Live Data)")

st.markdown(
    """
    <style>
      .card {
        border: 1px solid #e5e7eb; border-radius: 12px; padding: 18px 20px;
        background: #ffffff; margin: 8px 0 18px 0;
      }
      .section-title { color:#2563eb; font-weight:700; font-size:1.05rem; margin:0 0 6px 0; }
      .pill-group > div[role="radiogroup"] { display:flex; flex-wrap:wrap; gap:10px; }
      .pill-group [role="radio"] {
        border:1px solid #2563eb; color:#2563eb; border-radius:9999px; padding:6px 14px; background:#fff;
        transition:all .12s ease-in-out; box-shadow:0 1px 0 rgba(0,0,0,0.02); font-weight:600;
      }
      .pill-group [role="radio"][aria-checked="true"] {
        background: linear-gradient(90deg, #60a5fa, #2563eb); border-color:#2563eb; color:#fff;
      }
      .pill-divider { height:1px; background:#e5e7eb; margin:6px 0 16px 0; }
      .status-ok { color:#059669; font-weight:600; }
      .status-fail { color:#dc2626; font-weight:600; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- Constants ----------------------
DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

CITY_META = {
    # city: (country_code_iso3, country_name, region)
    "New York": ("USA", "United States", "North America"),
    "Singapore": ("SGP", "Singapore", "Asia"),
    "London": ("GBR", "United Kingdom", "Europe"),
}

# Live sources
NYC_URL = "https://data.ny.gov/resource/wujg-7c2s.csv?$limit=50000"
SG_URL  = "https://data.gov.sg/api/action/datastore_search?resource_id=75248cf2-fbf3-40de-6a74-6dc91ec9223c&limit=5000"
LON_URL = "https://data.london.gov.uk/download/public-transport-journeys-type-transport/b9c48f37-journeys-by-type.xls"
# World Bank GDP per capita (current USD): NY.GDP.PCAP.CD
WB_GDP_TPL = "https://api.worldbank.org/v2/country/{iso3}/indicator/NY.GDP.PCAP.CD?format=json"

# ---------------------- Helpers ----------------------
def enforce_day_order(s):
    return pd.Categorical(s, categories=DAYS, ordered=True)

def normalize_to_index(series):
    mx = series.max()
    return (series / mx * 100.0) if pd.notna(mx) and np.isfinite(mx) and mx != 0 else series*0

@st.cache_data(show_spinner=False)
def fetch_csv(url, **kwargs):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), **kwargs)

@st.cache_data(show_spinner=False)
def fetch_json(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def fetch_excel(url, sheet_name=0):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content), sheet_name=sheet_name)

@st.cache_data(show_spinner=False)
def wb_latest_gdp_per_capita(iso3: str) -> float | None:
    """Return the latest non-null GDP per capita (current USD) for a country."""
    try:
        js = fetch_json(WB_GDP_TPL.format(iso3=iso3))
        if not isinstance(js, list) or len(js) < 2:
            return None
        data = pd.DataFrame(js[1])
        data = data.sort_values("date", ascending=False)
        val = pd.to_numeric(data["value"], errors="coerce").dropna()
        return float(val.iloc[0]) if not val.empty else None
    except Exception:
        return None

# ---------------------- City-specific loaders (build weekday averages) ----------------------
@st.cache_data(show_spinner=False)
def load_nyc_hourly_to_dow() -> tuple[pd.DataFrame, dict]:
    status = {"source": NYC_URL, "rows": 0, "ok": False, "note": ""}
    try:
        df = fetch_csv(NYC_URL)
        status["rows"] = len(df)
        # best-effort column picks
        time_col_candidates = ["date_hour_begin","datetime","timestamp","date_time","hour_begin"]
        ridership_col_candidates = ["ridership","estimated_ridership","entries","count"]
        def pick(col_list):
            for c in col_list:
                if c in df.columns: return c
            return None
        tcol = pick(time_col_candidates)
        rcol = pick(ridership_col_candidates)
        if not tcol or not rcol:
            status["note"] = "Could not find time/ridership columns."
            return pd.DataFrame(), status
        ts = pd.to_datetime(df[tcol], errors="coerce")
        df = df.loc[ts.notna()].copy()
        df["day_of_week"] = ts.dt.day_name()
        df["ridership"]   = pd.to_numeric(df[rcol], errors="coerce")
        out = (
            df.groupby("day_of_week", as_index=False)["ridership"]
              .mean()
              .assign(city="New York")
        )
        out["day_of_week"] = enforce_day_order(out["day_of_week"])
        status["ok"] = True
        return out, status
    except Exception as e:
        status["note"] = f"Error: {e}"
        return pd.DataFrame(), status

@st.cache_data(show_spinner=False)
def load_sg_daily_to_dow() -> tuple[pd.DataFrame, dict]:
    status = {"source": SG_URL, "rows": 0, "ok": False, "note": ""}
    try:
        js = fetch_json(SG_URL)
        recs = js.get("result", {}).get("records", [])
        df = pd.DataFrame(recs)
        status["rows"] = len(df)
        # Try common fields
        date_col_candidates = ["date","day","dt","obs_date"]
        value_col_candidates = ["average_daily_ridership","ridership","value","count"]
        def pick(col_list):
            for c in col_list:
                if c in df.columns: return c
            return None
        dcol = pick(date_col_candidates)
        vcol = pick(value_col_candidates)
        # Case A: daily date available
        if dcol and vcol:
            tmp = df[[dcol, vcol]].dropna()
            tmp["date"] = pd.to_datetime(tmp[dcol], errors="coerce")
            tmp = tmp.loc[tmp["date"].notna()].copy()
            tmp["day_of_week"] = tmp["date"].dt.day_name()
            tmp["ridership"] = pd.to_numeric(tmp[vcol], errors="coerce")
            out = (
                tmp.groupby("day_of_week", as_index=False)["ridership"]
                   .mean()
                   .assign(city="Singapore")
            )
            out["day_of_week"] = enforce_day_order(out["day_of_week"])
            status["ok"] = True
            return out, status
        # Case B: monthly average daily (no actual day column)
        if {"year","month"}.issubset(df.columns):
            # look for a value column
            mv = pick(value_col_candidates)
            if mv:
                avg_daily = pd.to_numeric(df[mv], errors="coerce").dropna().mean()
                if pd.notna(avg_daily):
                    out = pd.DataFrame({"day_of_week": DAYS})
                    out["ridership"] = avg_daily
                    out["city"] = "Singapore"
                    out["day_of_week"] = enforce_day_order(out["day_of_week"])
                    status["ok"] = True
                    status["note"] = "Used monthly average as flat weekday profile."
                    return out, status
        status["note"] = "Could not infer Singapore schema."
        return pd.DataFrame(), status
    except Exception as e:
        status["note"] = f"Error: {e}"
        return pd.DataFrame(), status

@st.cache_data(show_spinner=False)
def load_london_weekly_to_dow() -> tuple[pd.DataFrame, dict]:
    status = {"source": LON_URL, "rows": 0, "ok": False, "note": ""}
    try:
        df = fetch_excel(LON_URL, sheet_name=0)
        status["rows"] = len(df)
        # Column picks
        week_col_candidates = ["week_start_date","Week starting","week","week_start","week_commencing"]
        val_col_candidates  = ["journeys_millions","Journeys (m)","journeys","total_journeys"]
        def pick(col_list):
            for c in col_list:
                if c in df.columns: return c
            return None
        wcol = pick(week_col_candidates) or next((c for c in df.columns if "week" in c.lower() or "date" in c.lower()), None)
        vcol = pick(val_col_candidates) or (df.select_dtypes(include=[np.number]).columns.tolist()[0] if len(df.select_dtypes(include=[np.number]).columns) else None)
        if not wcol or not vcol:
            status["note"] = "Could not find week/value columns."
            return pd.DataFrame(), status
        tmp = df[[wcol, vcol]].copy()
        tmp[vcol] = pd.to_numeric(tmp[vcol], errors="coerce")
        tmp = tmp.loc[tmp[vcol].notna()]
        tmp["daily_avg"] = tmp[vcol] / 7.0
        out = pd.DataFrame({"day_of_week": DAYS})
        out["ridership"] = tmp["daily_avg"].mean()
        out["city"] = "London"
        out["day_of_week"] = enforce_day_order(out["day_of_week"])
        status["ok"] = True
        return out, status
    except Exception as e:
        status["note"] = f"Error: {e}"
        return pd.DataFrame(), status

# ---------------------- Build combined DoW panel & GDP ----------------------
@st.cache_data(show_spinner=False)
def build_panel_with_gdp():
    nyc, s1 = load_nyc_hourly_to_dow()
    sg,  s2 = load_sg_daily_to_dow()
    lon, s3 = load_london_weekly_to_dow()

    parts = [d for d in [nyc, sg, lon] if not d.empty]
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["city","day_of_week","ridership"])

    if not combined.empty:
        combined["ridership_index"] = combined.groupby("city")["ridership"].transform(normalize_to_index)
        # Attach country & region
        meta = []
        for c in combined["city"].unique():
            iso3, country, region = CITY_META.get(c, (None, None, None))
            meta.append({"city": c, "country_iso3": iso3, "country": country, "region": region})
        meta = pd.DataFrame(meta)
        combined = combined.merge(meta, on="city", how="left")
        # Fetch GDP for available iso3
        gdp_rows = []
        for iso3 in combined["country_iso3"].dropna().unique():
            g = wb_latest_gdp_per_capita(iso3)
            gdp_rows.append({"country_iso3": iso3, "gdp_per_capita": g})
        gdp = pd.DataFrame(gdp_rows)
        combined = combined.merge(gdp, on="country_iso3", how="left")
    return combined, (s1, s2, s3)

# ---------------------- Data Load Summary ----------------------
panel, statuses = build_panel_with_gdp()
s1, s2, s3 = statuses

st.markdown("### Data Load Summary")
cols = st.columns(3)
for col, (label, stt) in zip(cols, [("New York", s1), ("Singapore", s2), ("London", s3)]):
    ok = stt.get("ok", False)
    rows = stt.get("rows", 0)
    src  = stt.get("source", "")
    note = stt.get("note", "")
    with col:
        st.markdown(f"**{label}**")
        st.markdown(f"- Source: <span class='mono'>{src}</span>", unsafe_allow_html=True)
        st.markdown(f"- Status: {'<span class=\"status-ok\">OK</span>' if ok else '<span class=\"status-fail\">FAILED</span>'}", unsafe_allow_html=True)
        st.markdown(f"- Rows fetched: **{rows}**")
        if note:
            st.caption(note)

if panel.empty:
    st.error("No live data available right now. Please try again later or check network access.")
    st.stop()

# ---------------------- Sidebar controls ----------------------
with st.sidebar:
    st.markdown("### Filters")
    regions = sorted(panel["region"].dropna().unique().tolist())
    sel_regions = st.multiselect("Region(s)", ["All"] + regions, default=["All"])
    if "All" in sel_regions or not sel_regions:
        sel_regions = regions

    has_gdp = panel["gdp_per_capita"].notna().any()
    if has_gdp:
        gmin, gmax = float(panel["gdp_per_capita"].min()), float(panel["gdp_per_capita"].max())
        gdp_range = st.slider("GDP per Capita range", min_value=int(gmin), max_value=int(gmax),
                              value=(int(gmin), int(gmax)), step=1000)
        use_logx = st.checkbox("Log-scale GDP axis", value=False)
    else:
        gdp_range, use_logx = None, False

    st.markdown("---")
    st.markdown("### Display")
    y_metric = st.selectbox(
        "Y-axis (density proxy)",
        options=["ridership_index","ridership"],
        index=0,
        format_func=lambda s: "Ridership index (0–100)" if s=="ridership_index" else "Ridership (avg)"
    )
    topn = st.slider("Top N (table & bar)", 3, 7, 3, step=1)

# Apply filters
f = panel["region"].isin(sel_regions)
if has_gdp and gdp_range is not None:
    f &= panel["gdp_per_capita"].between(gdp_range[0], gdp_range[1])
fpanel = panel.loc[f].copy()

# Aggregate for city-level (mean over DoW)
city_agg = (
    fpanel.groupby(["city","country","region","gdp_per_capita"], dropna=False)[["ridership","ridership_index"]]
          .mean()
          .reset_index()
)

# ---------------------- Row 1: Heatmap of DoW patterns ----------------------
st.markdown("### Weekly Public Transport Pattern (Ridership Index by Day of Week)")
pivot = fpanel.pivot_table(index="city", columns="day_of_week", values="ridership_index", aggfunc="mean")
fig_heatmap = px.imshow(
    pivot,
    text_auto=True,
    aspect="auto",
    labels=dict(x="Day of week", y="City", color="Ridership Index"),
    title="Ridership Index (0–100) by Day of Week"
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# ---------------------- Row 2: GDP vs density proxy (R²) ----------------------
if has_gdp and city_agg["gdp_per_capita"].notna().any():
    st.markdown("### GDP per Capita vs Connectivity Proxy")
    x = city_agg["gdp_per_capita"].to_numpy(dtype=float)
    y = city_agg[y_metric].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x2, y2 = x[mask], y[mask]

    fig_scatter = px.scatter(
        city_agg, x="gdp_per_capita", y=y_metric,
        color="region", hover_name="city",
        labels={"gdp_per_capita": "GDP per Capita (USD)"},
        title=("Ridership Index vs GDP per Capita" if y_metric=="ridership_index" else "Ridership vs GDP per Capita")
    )
    r2_txt = ""
    if len(x2) >= 2 and np.ptp(x2) > 0:
        A = np.vstack([x2, np.ones_like(x2)]).T
        m, c = np.linalg.lstsq(A, y2, rcond=None)[0]
        xs = np.linspace(x2.min(), x2.max(), 200)
        ys = m*xs + c
        ss_res = np.sum((y2 - (m*x2 + c))**2)
        ss_tot = np.sum((y2 - y2.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
        r2_txt = f"Trendline: y = {m:.4f}x + {c:.2f}  •  R² = {r2:.2f}"
        fig_scatter.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="OLS trend"))

    if use_logx:
        fig_scatter.update_xaxes(type="log")
    st.plotly_chart(fig_scatter, use_container_width=True)
    if r2_txt:
        st.caption(r2_txt)
else:
    st.info("GDP per capita not available. Using connectivity visuals only.")

# ---------------------- Row 3: Top-N cities & Table ----------------------
st.subheader("Leaders by Connectivity Proxy (Ridership Index)")
# pick DoW day with highest index per city to rank
peak = fpanel.loc[fpanel.groupby("city")["ridership_index"].idxmax()].copy()
peak = peak.sort_values("ridership_index", ascending=False)

c1, c2 = st.columns([1,1])
with c1:
    st.dataframe(
        peak[["city","country","region","day_of_week","ridership_index","gdp_per_capita"]]
        .head(int(topn))
        .round(2),
        use_container_width=True
    )
with c2:
    fig_bar = px.bar(
        peak.head(int(topn)).sort_values("ridership_index", ascending=True),
        x="ridership_index", y="city", orientation="h", color="region",
        labels={"ridership_index":"Ridership index","city":"City"},
        title="Top-N cities by peak ridership index"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------- Row 4: Correlation (aggregated) ----------------------
st.subheader("Correlation (city-level aggregates)")
num_cols = [c for c in ["gdp_per_capita","ridership","ridership_index"] if c in city_agg.columns]
corr = city_agg[num_cols].corr().round(3) if len(city_agg) and len(num_cols) >= 2 else pd.DataFrame()
if not corr.empty:
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", labels=dict(color="corr"))
    st.plotly_chart(fig_corr, use_container_width=True)
    st.dataframe(corr)
else:
    st.info("Not enough numeric columns to compute correlations.")

# ---------------------- Narrative pills ----------------------
SECTION_CONTENT = {
    "About this dataset": """
<div class="card">
  <div class="section-title">About this dataset</div>
  <p>We derive a <b>connectivity proxy</b> — day-of-week ridership index (0–100) — from live city datasets:
  NYC hourly, SG daily/monthly, LON weekly journeys. We attach live <b>GDP per capita</b> from the World Bank for each country's city.</p>
</div>
""",
    "Exploration & discovery": """
<div class="card">
  <div class="section-title">Exploration & discovery</div>
  <ul>
    <li>Heatmap reveals the <b>weekday/weekend</b> ridership shape by city.</li>
    <li>Scatter plot examines <b>GDP vs connectivity proxy</b> with an OLS trendline and <b>R²</b>.</li>
    <li>Top-N ranks cities by <b>peak</b> ridership index; correlation summarizes numeric relationships.</li>
  </ul>
</div>
""",
    "Working hypothesis (H8)": """
<div class="card">
  <div class="section-title">Working hypothesis (H8)</div>
  <p>Global transit connectivity patterns (approximated by ridership index) reflect <b>economic</b> (GDP per capita)
  and <b>geographic</b> (region) factors. Wealthier regions often show stronger or steadier ridership patterns.</p>
</div>
""",
    "Target audience": """
<div class="card">
  <div class="section-title">Target audience</div>
  <ul>
    <li><b>Transit planners</b> — benchmark connectivity pattern shapes across cities.</li>
    <li><b>Policy teams</b> — link fiscal capacity (GDP) to network usage patterns.</li>
    <li><b>Researchers</b> — extend with station/line/km live sources as they become available.</li>
  </ul>
</div>
"""
}

st.markdown('<div class="pill-divider"></div>', unsafe_allow_html=True)
st.markdown("#### Details")

if "h8_section" not in st.session_state:
    st.session_state["h8_section"] = "About this dataset"

with st.container():
    st.markdown('<div class="pill-group">', unsafe_allow_html=True)
    choice = st.radio(
        "h8_section_pills",
        options=list(SECTION_CONTENT.keys()),
        index=list(SECTION_CONTENT.keys()).index(st.session_state["h8_section"]),
        horizontal=True,
        label_visibility="collapsed",
        key="h8_section_radio",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state["h8_section"] = choice

st.markdown(SECTION_CONTENT[st.session_state["h8_section"]], unsafe_allow_html=True)
