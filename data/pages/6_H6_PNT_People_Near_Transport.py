#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6_H6 - PNT (People Near Transport).py — Streamlit "H6" page
-----------------------------------------------------------
Focus: Additional hypothesis from Analysis 1
"Cities with longer networks tend to show higher PNT (People Near Transport)".

This app loads:
  - mrt_access.csv  (PNT shares within 500m/1000m/1500m)
  - metro_systems.csv (city, country, region, length, stations, ridership)

It merges on City + Country (case-insensitive) and presents:
  - Scatter: System length (km) vs chosen PNT (500m/1000m/1500m)
  - Correlation bullets and heatmap across merged numeric fields
  - Narrative details in pill-style tabs at the bottom

Run:
  streamlit run "6_H6 - PNT (People Near Transport).py"
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------- Style tweaks (match H3) ----------------------
st.set_page_config(page_title="H6 — PNT (People Near Transport)", layout="wide")

st.markdown(
    """
    <style>
      .main > div:first-child h1 {
        font-size: 2.0rem;
        letter-spacing: 0.2px;
        margin-bottom: 0.25rem;
      }
      .card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px 20px;
        background: #ffffff;
        margin: 8px 0 18px 0;
      }
      .section-title {
        color: #b91c1c; /* red-700 */
        margin: 0 0 6px 0;
        font-weight: 700;
        font-size: 1.15rem;
      }
      .pill-group > div[role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }
      .pill-group [role="radio"] {
        border: 1px solid #e11d48;
        color: #be123c;
        border-radius: 9999px;
        padding: 6px 14px;
        background: #fff;
        transition: all .12s ease-in-out;
        box-shadow: 0 1px 0 rgba(0,0,0,0.02);
        font-weight: 600;
      }
      .pill-group [role="radio"][aria-checked="true"] {
        background: linear-gradient(90deg, #fb7185, #e11d48);
        border-color: #e11d48;
        color: #fff;
      }
      .pill-divider {
        height: 1px;
        background: #e5e7eb;
        margin: 6px 0 16px 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- Helpers ----------------------
ACCESS_CANDIDATES = ["mrt_access.csv"]
SYSTEMS_CANDIDATES = ["metro_systems (1).csv", "metro_systems.csv"]

def find_local_file(candidates):
    here = Path(__file__).resolve().parent
    for name in candidates:
        p = here / name
        if p.exists():
            return str(p)
    return None

def normalize_cols(cols):
    return [re.sub(r"[^a-z0-9]+", "", str(c).strip().lower()) for c in cols]

def pick_col(cols_norm, original_cols, *key_parts):
    for i, cn in enumerate(cols_norm):
        if all(part in cn for part in key_parts):
            return original_cols[i]
    return None

def prepare_systems(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    cols_norm = normalize_cols(d.columns)
    city_col = pick_col(cols_norm, d.columns, "city") or "City"
    country_col = pick_col(cols_norm, d.columns, "country") or "Country"
    length_col = (pick_col(cols_norm, d.columns, "system", "length") or
                  pick_col(cols_norm, d.columns, "length"))
    ridership_col = pick_col(cols_norm, d.columns, "annual", "ridership")
    stations_col = pick_col(cols_norm, d.columns, "stations")
    region_col = pick_col(cols_norm, d.columns, "region")

    for c in [length_col, ridership_col, stations_col]:
        if c and c in d.columns:
            d[c] = pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce")
    if length_col and ridership_col:
        d["ridership_per_km"] = d[ridership_col] / d[length_col]

    ren = {}
    if city_col in d.columns: ren[city_col] = "City"
    if country_col in d.columns: ren[country_col] = "Country"
    if length_col in d.columns: ren[length_col] = "System length (km)"
    if ridership_col in d.columns: ren[ridership_col] = "Annual ridership (millions)"
    if stations_col and stations_col in d.columns: ren[stations_col] = "Stations"
    if region_col and region_col in d.columns: ren[region_col] = "Region"
    d = d.rename(columns=ren)
    return d

def prepare_access(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip() for c in d.columns]
    rename_candidates = {}
    for c in d.columns:
        cn = c.strip().lower()
        if cn.replace(" ", "") in ["500m", "access500m"]:
            rename_candidates[c] = "500m"
        if cn.replace(" ", "") in ["1000m", "1km", "access1000m"]:
            rename_candidates[c] = "1000m"
        if cn.replace(" ", "") in ["1500m", "1_5km", "access1500m"]:
            rename_candidates[c] = "1500m"
    d = d.rename(columns=rename_candidates)
    return d

def compute_merge(access_df: pd.DataFrame, systems_df: pd.DataFrame):
    if access_df is None or systems_df is None:
        return None
    a = prepare_access(access_df)
    s = prepare_systems(systems_df)

    for frame in (a, s):
        for k in ("City", "Country"):
            if k not in frame.columns:
                lower = {c.lower(): c for c in frame.columns}
                if k.lower() in lower:
                    frame.rename(columns={lower[k.lower()]: k}, inplace=True)

    need = ["City", "Country", "System length (km)", "Annual ridership (millions)", "ridership_per_km"]
    if not all(c in s.columns for c in need):
        return None

    a["City_key"] = a["City"].astype(str).str.strip().str.lower()
    a["Country_key"] = a["Country"].astype(str).str.strip().str.lower()
    s["City_key"] = s["City"].astype(str).str.strip().str.lower()
    s["Country_key"] = s["Country"].astype(str).str.strip().str.lower()

    keep_cols = [c for c in ["500m", "1000m", "1500m", "Population"] if c in a.columns]
    base_cols = ["City", "Country", "Region"] if "Region" in s.columns else ["City", "Country"]
    m = pd.merge(
        s[base_cols + ["City_key", "Country_key", "System length (km)", "Annual ridership (millions)", "ridership_per_km", "Stations"]],
        a[["City_key", "Country_key"] + keep_cols],
        on=["City_key", "Country_key"],
        how="inner",
    )
    for c in ["500m", "1000m", "1500m", "System length (km)", "Annual ridership (millions)", "ridership_per_km", "Stations"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    return m

def compute_access_hypothesis(merged: pd.DataFrame):
    if merged is None or merged.empty:
        return None
    corr_targets = [c for c in ["500m", "1000m", "1500m"] if c in merged.columns]
    if not corr_targets:
        return None
    res = {}
    valid = merged.dropna(subset=corr_targets + ["ridership_per_km"])
    if valid.empty:
        return None
    for c in corr_targets:
        res[c] = round(valid[c].corr(valid["ridership_per_km"]), 3)
    valid2 = merged.dropna(subset=["System length (km)", "Annual ridership (millions)", "ridership_per_km"])
    if not valid2.empty:
        res["len_vs_total"] = round(valid2["System length (km)"].corr(valid2["Annual ridership (millions)"]), 3)
        res["len_vs_rpk"] = round(valid2["System length (km)"].corr(valid2["ridership_per_km"]), 3)
    return res

# ---------------------- UI ----------------------
st.title("H6 — Cities with longer networks tend to show higher PNT (People Near Transport)")

with st.sidebar:
    st.markdown("### Data Sources")
    st.caption("Upload to override local files (if present).")
    up_access = st.file_uploader("Upload city access file (mrt_access.csv)", type=["csv", "xlsx"], key="up_access")
    up_systems = st.file_uploader("Upload metro systems file (metro_systems*.csv)", type=["csv", "xlsx"], key="up_systems")
    st.divider()
    st.caption("If not uploaded, the app will try to load files next to the script.")

# Load helpers
def read_any(file):
    if file is None:
        return None
    try:
        return pd.read_csv(file)
    except Exception:
        file.seek(0)
        return pd.read_excel(file)

def read_optional_local(candidates):
    p = find_local_file(candidates)
    if p:
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None

# Load data
status_msgs = []
access_df = read_any(up_access) if up_access is not None else read_optional_local(ACCESS_CANDIDATES)
systems_df = read_any(up_systems) if up_systems is not None else read_optional_local(SYSTEMS_CANDIDATES)
merged = compute_merge(access_df, systems_df) if (access_df is not None and systems_df is not None) else None

# ---------------------- Charts ----------------------
st.markdown("### Charts & Tables")

cols_top = st.columns([1,1,1])
with cols_top[0]:
    metric_pnt = st.radio("Access metric (PNT)", ["500m","1000m","1500m"], horizontal=True)
with cols_top[1]:
    if merged is not None and "Region" in merged.columns:
        regions = sorted([r for r in merged["Region"].dropna().unique()])
        selected_regions = st.multiselect("Region filter", regions, default=[])
    else:
        selected_regions = []

if merged is None:
    st.info("Upload or place both 'mrt_access.csv' and 'metro_systems.csv' next to this script. Merge key: City + Country (case-insensitive).")
else:
    m = merged.copy()
    if selected_regions and "Region" in m.columns:
        m = m[m["Region"].isin(selected_regions)]

    # Correlation bullets
    res = compute_access_hypothesis(m) or {}
    bullets = []
    if "500m" in res: bullets.append(f"- 500 m access vs ridership per km: **r = {res['500m']}**")
    if "1000m" in res: bullets.append(f"- 1000 m access vs ridership per km: **r = {res['1000m']}**")
    if "1500m" in res: bullets.append(f"- 1500 m access vs ridership per km: **r = {res['1500m']}**")
    len_total = f"{res.get('len_vs_total','n/a')}"
    len_rpk = f"{res.get('len_vs_rpk','n/a')}"

    st.markdown(
        "Using your city-access and metro-systems files, we compute simple Pearson correlations on the merged city-level table.\n\n"
        + ("\n".join(bullets) if bullets else "_No access columns found._") + "\n\n"
        + f"Also: length vs total ridership: **r = {len_total}**; length vs ridership per km: **r = {len_rpk}**.\n\n"
        "Interpretation: higher **station accessibility (coverage)** associates with better **per-km efficiency**, while sheer network length relates more to total demand than to efficiency."
    )

    # Scatter: Length vs selected PNT
    xcol, ycol = "System length (km)", metric_pnt
    if ycol in m.columns and xcol in m.columns:
        dfp = m.dropna(subset=[xcol, ycol]).copy()
        if not dfp.empty:
            fig = px.scatter(
                dfp, x=xcol, y=ycol, hover_name="City",
                color="Region" if "Region" in dfp.columns else None,
                labels={xcol:"Network length (km)", ycol:f"{ycol} access (share)"}
            )
            fig.update_layout(title=f"Network length vs {ycol} access (PNT)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No valid rows for the selected columns.")
    else:
        st.warning("Columns missing for the scatter plot.")

    # Correlation heatmap for merged numeric fields
    num_cols_m = [c for c in ["500m","1000m","1500m","System length (km)","Annual ridership (millions)","ridership_per_km","Stations"] if c in m.columns]
    if num_cols_m:
        corr_m = m[num_cols_m].corr().round(3)
        fig_m = px.imshow(corr_m, text_auto=True, aspect="auto", labels=dict(color="corr"))
        st.plotly_chart(fig_m, use_container_width=True)
        st.dataframe(corr_m)
    else:
        st.info("No numeric columns available for merged heatmap.")

# ---------------------- Narrative pills (BOTTOM) ----------------------
SECTION_CONTENT = {
    "About this dataset": """
<div class="card">
  <div class="section-title">About this dataset</div>
  <p>This page joins two city-level sources to understand how metro <b>network design</b> relates to population <b>accessibility</b> (PNT):</p>
  <ul>
    <li><code>metro_systems.csv</code>: City, Country, Region, <b>System length (km)</b>, <b>Stations</b>, <b>Annual ridership (millions)</b>, derived <b>ridership per km</b>.</li>
    <li><code>mrt_access.csv</code>: Share of population within <b>500 m / 1000 m / 1500 m</b> of a station (PNT thresholds).</li>
  </ul>
  <p>We merge by City + Country (case-insensitive) to explore how <b>network scale</b> and <b>coverage</b> move together.</p>
</div>
""",
    "Exploration & discovery": """
<div class="card">
  <div class="section-title">Exploration & discovery</div>
  <ol>
    <li><b>Clean & merge</b> the two files; coerce numerics; create <em>ridership per km</em> for context.</li>
    <li><b>Chart 1</b> — Network length vs PNT (pick 500/1000/1500 m): shows how longer networks relate to population coverage; region colouring reveals geographic patterns.</li>
    <li><b>Chart 2</b> — Correlation heatmap (merged numeric fields): quantifies linear relations among length, stations, total ridership, per‑km ridership, and PNT thresholds.</li>
  </ol>
</div>
""",
    "Working hypotheses": """
<div class="card">
  <div class="section-title">Working hypotheses</div>
  <ol>
    <li><b>Coverage scales with size</b>: Cities with longer networks tend to show higher PNT.</li>
    <li><b>Diminishing returns</b>: Marginal PNT gains may flatten at larger lengths—especially at 500 m.</li>
    <li><b>Stations as mediator</b>: For similar lengths, more stations (denser spacing) lift PNT.</li>
    <li><b>Regional design effects</b>: Some regions achieve high PNT with moderate length (dense, infill‑oriented networks).</li>
  </ol>
</div>
""",
    "Target audience": """
<div class="card">
  <div class="section-title">Target audience</div>
  <ul>
    <li><b>Public transport operators & regulators</b> — link coverage to network expansion plans; monitor accessibility KPIs.</li>
    <li><b>Urban planners & infrastructure agencies</b> — test TOD/access assumptions; balance <em>length vs station density</em>.</li>
    <li><b>Government policy teams & ministries</b> — support funding and service standards with measurable coverage metrics.</li>
    <li><b>Development banks & IFIs</b> — compare cities/regions for investment screening and outcome tracking.</li>
    <li><b>Transport & urban consulting firms</b> — baseline opportunities; communicate network design trade‑offs.</li>
    <li><b>Researchers, academics & students</b> — cross‑city hypothesis testing and teaching demos on PNT vs design.</li>
  </ul>
</div>
"""
}

st.markdown('<div class="pill-divider"></div>', unsafe_allow_html=True)
st.markdown("#### Details")

if "section" not in st.session_state:
    st.session_state["section"] = "About this dataset"

with st.container():
    st.markdown('<div class="pill-group">', unsafe_allow_html=True)
    choice = st.radio(
        "section_pills",
        options=list(SECTION_CONTENT.keys()),
        index=list(SECTION_CONTENT.keys()).index(st.session_state["section"]),
        horizontal=True,
        label_visibility="collapsed",
        key="section_radio",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state["section"] = choice

st.markdown(SECTION_CONTENT[st.session_state["section"]], unsafe_allow_html=True)

st.divider()
# status message
msgs = []
if access_df is not None: msgs.append("Loaded city access data")
if systems_df is not None: msgs.append("Loaded metro systems data")
st.caption(" | ".join(msgs) if msgs else "Awaiting files...")
