
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from utils import load_csv_or_synth, upload_override, badge, synth_metro_transit

st.title("H1 — Network Integration & Efficiency (Ridership per km)")

expected = ["city","continent","stations","lines","system_km","ridership_millions"]

st.sidebar.header("Data")
up_df, up_kind = upload_override("Upload metro_transit_data.csv (optional)", expected, synth_metro_transit)
if up_df is not None:
    df, kind = up_df, up_kind
else:
    df, kind = load_csv_or_synth(Path("data/metro_transit_data.csv"), synth_metro_transit)

badge(kind)

df["efficiency_passengers_per_km"] = (df["ridership_millions"]*1_000_000) / df["system_km"]

left, right = st.columns([1,2])
with left:
    conts = ["All"] + sorted(df["continent"].dropna().unique().tolist())
    sel_cont = st.selectbox("Filter by continent", conts)
    if sel_cont != "All":
        fdf = df[df["continent"]==sel_cont]
    else:
        fdf = df.copy()
    st.metric("Cities in view", len(fdf))
with right:
    st.dataframe(fdf, use_container_width=True)

st.subheader("1) Scatter — System size vs Ridership (log axes)")
fig1 = px.scatter(
    fdf, x="system_km", y="ridership_millions", color="continent", hover_name="city",
    size="stations", trendline="ols", log_x=True, log_y=True
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("2) Boxplot — Efficiency by Continent")
fig2 = px.box(fdf, x="continent", y="efficiency_passengers_per_km", points="all")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3) Correlation Heatmap")
corr = fdf[["stations","lines","system_km","ridership_millions","efficiency_passengers_per_km"]].corr()
fig3 = px.imshow(corr, text_auto=True, aspect="auto")
st.plotly_chart(fig3, use_container_width=True)
