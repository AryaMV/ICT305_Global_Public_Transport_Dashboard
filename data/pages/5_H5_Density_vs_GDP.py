#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5_H5_Density_vs_GDP.py — Streamlit "H5" page
--------------------------------------------
Hypothesis: Cities with higher GDP per capita tend to support denser metro
networks (more stations per million population), reflecting stronger
economic capacity for transit investments.

Features
- CSV upload override (falls back to data/city_economics.csv or synthetic)
- Region multi-select, GDP range slider, log-axis toggle
- Two scatters: Stations vs GDP and Stations-per-million vs GDP with OLS line + R²
- Top-N table & bar, Correlation heatmap
- Narrative "pill" sections (About / Discovery / Hypothesis / Audience)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from utils import load_csv_or_synth, upload_override, badge, synth_city_econ

# ---------------------- Page & style ----------------------
st.set_page_config(page_title="H5 — Density vs GDP", layout="wide")
st.title("H5 — Network Density vs GDP per Capita")

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
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- Data loading ----------------------
expected = ["city","country","gdp_per_capita","population_millions","stations","region"]
up_df, up_kind = upload_override("Upload city_economics.csv (optional)", expected, synth_city_econ)
if up_df is not None:
    df, kind = up_df, up_kind
else:
    df, kind = load_csv_or_synth(Path("data/city_economics.csv"), synth_city_econ)

badge(kind)

# Guard: ensure expected columns present
missing = [c for c in expected if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}. Expected: {expected}")
    st.stop()

# Basic engineering
df = df.copy()
df["stations_per_million_pop"] = df["stations"] / df["population_millions"]

# ---------------------- Controls ----------------------
with st.sidebar:
    st.markdown("### Filters")
    regions = sorted(df["region"].dropna().unique().tolist())
    sel_regions = st.multiselect("Region(s)", ["All"] + regions, default=["All"])
    if "All" in sel_regions or not sel_regions:
        sel_regions = regions

    g_min, g_max = float(df["gdp_per_capita"].min()), float(df["gdp_per_capita"].max())
    gdp_range = st.slider("GDP per Capita range", min_value=int(g_min), max_value=int(g_max),
                          value=(int(g_min), int(g_max)), step=1000)
    use_logx = st.checkbox("Log-scale GDP axis", value=False)

    st.markdown("---")
    st.markdown("### Display")
    y_metric = st.selectbox(
        "Y-axis for 2nd chart",
        options=["stations_per_million_pop", "stations"],
        index=0,
        format_func=lambda s: "Stations per million population" if s=="stations_per_million_pop" else "Stations (raw)",
    )
    topn = st.slider("Top N (table & bar)", 5, 20, 10, step=1)

# Apply filters
fdf = df[df["region"].isin(sel_regions)].query("@gdp_range[0] <= gdp_per_capita <= @gdp_range[1]").copy()

# ---------------------- Row 1: Stations vs GDP ----------------------
c1, c2 = st.columns([1.2, 1])
with c1:
    fig1 = px.scatter(
        fdf, x="gdp_per_capita", y="stations",
        size="population_millions", color="region", hover_name="city",
        labels={"gdp_per_capita":"GDP per Capita (USD)","stations":"Stations","population_millions":"Population (millions)"},
        title="Stations vs GDP per Capita (bubble=size population)"
    )
    if use_logx:
        fig1.update_xaxes(type="log")
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.markdown("#### Insight")
    st.write(
        "This view shows **network scale** (stations) as economic capacity rises. "
        "Use the filters to compare regions or focus on specific GDP ranges."
    )
    st.metric("Cities shown", len(fdf))

# ---------------------- Row 2: Density vs GDP with OLS ----------------------
st.markdown("### Density vs GDP (with trendline)")

# Prepare OLS for selected y_metric (default stations_per_million_pop)
x = fdf["gdp_per_capita"].to_numpy()
y = fdf[y_metric].to_numpy()
mask = np.isfinite(x) & np.isfinite(y)
x2, y2 = x[mask], y[mask]

fig2 = px.scatter(
    fdf, x="gdp_per_capita", y=y_metric, color="region", hover_name="city",
    labels={"gdp_per_capita":"GDP per Capita (USD)", y_metric: ("Stations per million population" if y_metric=="stations_per_million_pop" else "Stations (raw)")},
    title=("Stations per million vs GDP per Capita" if y_metric=="stations_per_million_pop" else "Stations vs GDP per Capita (again)")
)
r2_txt = ""
if len(x2) >= 3 and np.ptp(x2) > 0:
    A = np.vstack([x2, np.ones_like(x2)]).T
    m, c = np.linalg.lstsq(A, y2, rcond=None)[0]
    xs = np.linspace(x2.min(), x2.max(), 200)
    ys = m*xs + c
    ss_res = np.sum((y2 - (m*x2 + c))**2)
    ss_tot = np.sum((y2 - y2.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    r2_txt = f"Trendline: y = {m:.4f}x + {c:.2f}  •  R² = {r2:.2f}"
    fig2.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="OLS trend"))

if use_logx:
    fig2.update_xaxes(type="log")
st.plotly_chart(fig2, use_container_width=True)
if r2_txt:
    st.caption(r2_txt)

# ---------------------- Row 3: Top-N & Table ----------------------
st.subheader("Leaders by density / scale")
rank_df = fdf.assign(stations_per_million_pop=lambda d: d["stations"]/d["population_millions"])
rank_df = rank_df.sort_values(by=("stations_per_million_pop" if y_metric=="stations_per_million_pop" else "stations"), ascending=False)

b1, b2 = st.columns([1,1])
with b1:
    disp = rank_df.head(int(topn))[["city","country","region","gdp_per_capita","population_millions","stations","stations_per_million_pop"]]
    st.dataframe(disp.round(3), use_container_width=True)

with b2:
    fig_bar = px.bar(
        disp.sort_values(by=("stations_per_million_pop" if y_metric=="stations_per_million_pop" else "stations"), ascending=True),
        x=("stations_per_million_pop" if y_metric=="stations_per_million_pop" else "stations"),
        y="city", orientation="h", color="region",
        labels={"stations_per_million_pop":"Stations per million","stations":"Stations","city":"City"},
        title=("Top-N cities by stations per million population" if y_metric=="stations_per_million_pop" else "Top-N cities by stations")
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------- Row 4: Correlation heatmap ----------------------
st.subheader("Correlation (numeric fields)")
num_cols = ["gdp_per_capita","population_millions","stations","stations_per_million_pop"]
corr = fdf[num_cols].corr().round(3) if len(fdf) else pd.DataFrame()
if not corr.empty:
    fig_heat = px.imshow(corr, text_auto=True, aspect="auto", labels=dict(color="corr"))
    st.plotly_chart(fig_heat, use_container_width=True)
    st.dataframe(corr)
else:
    st.info("Not enough data after filters to compute correlations.")

# ---------------------- Narrative pills ----------------------
SECTION_CONTENT = {
    "About this dataset": """
<div class="card">
  <div class="section-title">About this dataset</div>
  <p>City-level metro attributes: <code>gdp_per_capita</code>, <code>population_millions</code>, <code>stations</code>, and <code>region</code>.
  We derive <code>stations_per_million_pop = stations / population_millions</code> as a density proxy.</p>
</div>
""",
    "Exploration & discovery": """
<div class="card">
  <div class="section-title">Exploration & discovery</div>
  <ul>
    <li>Scatter 1 highlights scale growth with GDP per capita; bubbles encode population.</li>
    <li>Scatter 2 focuses on <b>density</b> (stations per million) vs GDP, with OLS trend & R².</li>
    <li>Top-N view quickly surfaces outliers or leaders by density or raw station count.</li>
    <li>Correlation matrix summarizes linear relationships among key numeric fields.</li>
  </ul>
</div>
""",
    "Working hypothesis": """
<div class="card">
  <div class="section-title">Working hypothesis (H5)</div>
  <p>Higher GDP per capita is associated with <b>denser metro networks</b> (more stations per million population),
  reflecting greater fiscal capacity and sustained investment in transit accessibility.</p>
</div>
""",
    "Target audience": """
<div class="card">
  <div class="section-title">Target audience</div>
  <ul>
    <li><b>Transport planners & regulators</b> — benchmark city density vs GDP, spot funding gaps.</li>
    <li><b>Policy teams</b> — motivate investment cases with density & accessibility indicators.</li>
    <li><b>Researchers</b> — test robustness with different groupings (region filters) and scales (log GDP).</li>
  </ul>
</div>
"""
}

st.markdown('<div class="pill-divider"></div>', unsafe_allow_html=True)
st.markdown("#### Details")

if "h5_section" not in st.session_state:
    st.session_state["h5_section"] = "About this dataset"

with st.container():
    st.markdown('<div class="pill-group">', unsafe_allow_html=True)
    choice = st.radio(
        "h5_section_pills",
        options=list(SECTION_CONTENT.keys()),
        index=list(SECTION_CONTENT.keys()).index(st.session_state["h5_section"]),
        horizontal=True,
        label_visibility="collapsed",
        key="h5_section_radio",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state["h5_section"] = choice

st.markdown(SECTION_CONTENT[st.session_state["h5_section"]], unsafe_allow_html=True)
