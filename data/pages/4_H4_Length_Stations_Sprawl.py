
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from utils import load_csv_or_synth, upload_override, badge, synth_worldwide

st.title("H4 — Length & Stations vs Urban Sprawl & Population Served")

expected = ["city","stations","lines","system_km","population_millions","sprawl_index"]

up_df, up_kind = upload_override("Upload metro_systems_worldwide.csv (optional)", expected, synth_worldwide)
if up_df is not None:
    df, kind = up_df, up_kind
else:
    df, kind = load_csv_or_synth(Path("data/metro_systems_worldwide.csv"), synth_worldwide)

badge(kind)

df["people_per_km"] = (df["population_millions"]*1_000_000) / df["system_km"]

st.subheader("Scatter — Stations/km vs Sprawl (lower sprawl = denser)")
df["stations_per_km"] = df["stations"]/df["system_km"]
fig1 = px.scatter(df, x="stations_per_km", y="sprawl_index", size="population_millions", hover_name="city",
                  title="Stations per km vs Sprawl Index")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Bar — People served per km (approx)")
fig2 = px.bar(df.sort_values("people_per_km", ascending=False), x="city", y="people_per_km")
st.plotly_chart(fig2, use_container_width=True)
