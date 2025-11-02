#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3_H3_City_Ridership_Patterns.py — Streamlit "H3" page (Analysis 1 content)
---------------------------------------------------------------------------
Single-page view with country-level charts, then a pill-style section selector
at the bottom for: About this dataset, Exploration & discovery,
Working hypotheses, and Target audience.

Run:
  streamlit run 3_H3_City_Ridership_Patterns.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------- Style tweaks ----------------------
st.set_page_config(page_title="H3 — City Ridership Patterns", layout="wide")

# Custom CSS for "pill" tabs, spacing, and colors
st.markdown(
    """
    <style>
      /* Title styling */
      .main > div:first-child h1, h1#h3--city-ridership-patterns {
        font-size: 2.0rem;
        letter-spacing: 0.2px;
        margin-bottom: 0.25rem;
      }
      /* Card-like container */
      .card {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px 20px;
        background: #ffffff;
        margin: 8px 0 18px 0;
      }
      /* Section headings (red) */
      .section-title {
        color: #b91c1c; /* red-700 */
        margin: 0 0 6px 0;
        font-weight: 700;
        font-size: 1.15rem;
      }
      /* Pill radio group */
      .pill-group > div[role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }
      .pill-group label {
        border: 1px solid #e11d48;       /* rose-600 */
        color: #be123c;                   /* rose-700 */
        border-radius: 9999px;
        padding: 6px 14px;
        background: #fff;
        transition: all .12s ease-in-out;
        box-shadow: 0 1px 0 rgba(0,0,0,0.02);
        font-weight: 600;
      }
      /* selected state */
      .pill-group input:checked + div, .pill-group label[data-baseweb="radio"] div[dir="auto"] {
        /* some Streamlit themes use nested div; we rely on :has support limitedly */
      }
      .pill-group input:checked + div > p {
        color: #fff !important;
      }
      /* highlight the selected pill by targeting aria-checked true label */
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
MAIN_CANDIDATES = ["metro_countries_total (1).csv", "metro_countries_total.csv"]

def find_local_file(candidates):
    here = Path(__file__).resolve().parent
    for name in candidates:
        p = here / name
        if p.exists():
            return str(p)
    return None

def clean_and_engineer_country(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Flexible column name handling
    cols_norm = [re.sub(r"[^a-z0-9]+", "", str(c).lower()) for c in df.columns]
    def pick(*parts):
        for i, cn in enumerate(cols_norm):
            if all(p in cn for p in parts):
                return df.columns[i]
        return None

    length = pick("length")
    ridership = pick("annual", "ridership")
    stations = pick("stations")
    country = pick("country")
    region = pick("region")
    systems = pick("systems")  # optional

    req = [c for c in [country, region, length, ridership, stations] if c]
    if len(req) < 5:
        raise ValueError("Missing required column(s): need country, region, length, annual ridership, stations")

    for c in [length, ridership, stations, systems]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")

    df = df.dropna(subset=[length, ridership, stations])
    df = df[df[length] > 0]

    df["ridership_per_km"] = df[ridership] / df[length]

    # Unify canonical names
    ren = {
        country: "country",
        region: "region",
        length: "length",
        ridership: "annual_ridership_mill",
        stations: "stations",
    }
    if systems and systems in df.columns:
        ren[systems] = "systems"
    df = df.rename(columns=ren)
    return df

# ---------------------- UI ----------------------
st.title("H3 — City Ridership Patterns")

with st.sidebar:
    st.markdown("### Data Source")
    st.caption("Upload to override local files (if present).")
    up_main = st.file_uploader("Upload country-level file (metro_countries_total*.csv)", type=["csv", "xlsx"], key="up_main")
    st.divider()
    st.caption("If not uploaded, the app will try to load files next to the script.")

# Load country-level
def read_any(file):
    if file is None:
        return None
    try:
        return pd.read_csv(file)
    except Exception:
        file.seek(0)
        return pd.read_excel(file)

status_msgs = []

if up_main is not None:
    try:
        df_country = clean_and_engineer_country(read_any(up_main))
        status_msgs.append("Loaded country-level data from upload.")
    except Exception as e:
        df_country = None
        status_msgs.append(f"Country-level upload failed — {e}")
else:
    main_path = find_local_file(MAIN_CANDIDATES)
    if main_path:
        try:
            df_country = clean_and_engineer_country(pd.read_csv(main_path))
            status_msgs.append(f"Loaded country-level data from {Path(main_path).name}")
        except Exception as e:
            df_country = None
            status_msgs.append(f"Found {Path(main_path).name} but failed to load — {e}")
    else:
        df_country = None
        status_msgs.append("No local country-level CSV found — please upload.")

# ---------------------- Charts ----------------------
st.markdown("### Charts & Tables")

metric = st.selectbox(
    "Y-axis metric",
    ["annual_ridership_mill", "ridership_per_km"],
    format_func=lambda s: "Annual ridership (millions)" if s=="annual_ridership_mill" else "Ridership per km (millions/km)"
)

if df_country is None:
    st.info("Please upload or place a valid country-level CSV to render charts.")
else:
    fig_scatter = px.scatter(
        df_country, x="length", y=metric, size="stations", color="region", hover_name="country",
        labels={"length":"Network length (km)", metric:("Annual ridership (millions)" if metric=="annual_ridership_mill" else "Ridership per km (millions/km)"),
                "stations":"Stations","region":"Region"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Top-N by selected metric")
    topn = st.slider("Top N", min_value=5, max_value=20, value=10, step=1)
    d_sorted = df_country.sort_values(by=metric, ascending=False).head(int(topn)).copy()
    fig_bar = px.bar(d_sorted, x="country", y=metric, color="region",
                     labels={"country":"Country", metric:metric.replace("_"," ").title()})
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.dataframe(d_sorted[[c for c in ["country","region",metric,"length","stations","systems"] if c in d_sorted.columns]].round(3))

    st.subheader("Correlation heatmap (numeric columns)")
    num_cols = [c for c in ["length","stations","systems","annual_ridership_mill","ridership_per_km"] if c in df_country.columns]
    if num_cols:
        corr = df_country[num_cols].corr().round(3)
        fig_heat = px.imshow(corr, text_auto=True, aspect="auto", labels=dict(color="corr"))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.dataframe(corr)
    else:
        st.warning("No numeric columns found for correlation.")

# ---------------------- Narrative pills (BOTTOM) ----------------------
SECTION_CONTENT = {
    "About this dataset": """
<div class="card">
  <div class="section-title">About this dataset</div>
  <p>Country-level metro characteristics and usage. Each row is a country with at least one metro system, including network scale (length, lines, stations, systems), demand (<code>annual_ridership_mill</code>), and context (region, earliest inauguration).</p>
</div>
""",
    "Exploration & discovery": """
<div class="card">
  <div class="section-title">Exploration & discovery</div>
  <ul>
    <li>Cleaning & typing; removed invalid/zero lengths; engineered <code>ridership_per_km = annual_ridership_mill / length</code>.</li>
    <li>Scatter: length vs metric (annual ridership / ridership per km); bubbles encode stations; colour encodes region.</li>
    <li>Bar: Top-N countries by chosen metric to highlight leaders.</li>
    <li>Heatmap: correlations among numeric fields to reveal linear relationships.</li>
  </ul>
</div>
""",
    "Working hypotheses": """
<div class="card">
  <div class="section-title">Working hypotheses</div>
  <ol>
    <li>Longer networks and more stations correlate with higher annual ridership (scale–demand link).</li>
    <li>Ridership per km may flatten or fall at very large sizes (efficiency vs scale).</li>
    <li>Greater station density (more stations for a given length) boosts ridership via accessibility.</li>
    <li>More separate systems can raise total ridership but not necessarily per-km efficiency.</li>
    <li>Regional patterns: dense Asian cities tend to achieve higher ridership and per-km utilisation.</li>
    <li>Earlier inauguration dates may signal maturity and integration, correlating with higher demand.</li>
  </ol>
</div>
""",
    "Target audience": """
<div class="card">
  <div class="section-title">Target audience</div>
  <ul>
    <li><b>Public transport operators & regulators</b> — benchmark ridership vs. network size, spot efficiency gaps, prioritise upgrades, monitor recovery trends.</li>
    <li><b>Urban planners & infrastructure agencies</b> — test TOD assumptions, relate station density/length to demand, compare design choices across cities.</li>
    <li><b>Government policy teams & ministries</b> — build evidence for funding, fare policy, service standards, and decarbonisation targets.</li>
    <li><b>Development banks & IFIs (e.g., ADB, World Bank)</b> — screen/justify investments, compare regional performance, track KPIs for lending programs.</li>
    <li><b>Transport & urban consulting firms</b> — rapid baselining, opportunity sizing, and strategy/comms visuals for client reports.</li>
    <li><b>Researchers, academics & students</b> — cross-city comparative studies, hypothesis testing (e.g., ridership-per-km vs. station density), teaching demos.</li>
  </ul>
</div>
"""
}

st.markdown('<div class="pill-divider"></div>', unsafe_allow_html=True)
st.markdown("#### Details")

# Initialize session state
if "section" not in st.session_state:
    st.session_state["section"] = "About this dataset"

# Radio as pills
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

# Render selected content
st.markdown(SECTION_CONTENT[st.session_state["section"]], unsafe_allow_html=True)

st.divider()
st.caption(" | ".join(status_msgs))
