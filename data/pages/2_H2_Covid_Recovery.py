import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from utils import load_csv_or_synth, upload_override, badge, synth_mobility, synth_lta_sg
st.title("H2 — COVID-19 Impact & Recovery")

st.markdown("""
This page implements **five sub-hypotheses (H2.1–H2.5)** using Google Mobility and local ridership data.
You may **upload** your real CSVs below or place them in `./data`:
- `Global_Mobility_Report.csv` (Google Mobility)
- `full_data_clean.csv` (Ridership — e.g., UK `tfl_tube`, `tfl_bus`, etc.)
""")

# =====================================================================================
# Helpers
# =====================================================================================
def normalize_mobility_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to: date, country, transit_index (≈100 = baseline)."""
    cols = {c.lower(): c for c in df.columns}

    # date
    date_col = None
    for cand in ["date"]:
        if cand in cols:
            date_col = cols[cand]; break
    if date_col is None:
        raise ValueError("No 'date' column found in mobility CSV.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()
    df = df.rename(columns={date_col: "date"})

    # country
    country_col = None
    for cand in ["country","country_region","location","region"]:
        if cand in cols:
            country_col = cols[cand]; break
    if country_col is None:
        df["country"] = df.get("country_region_code", "Unknown")
    else:
        df = df.rename(columns={country_col: "country"})

    # mobility metric
    transit_index_col = None
    for cand in ["transit_index","transit_station_index"]:
        if cand in cols:
            transit_index_col = cols[cand]; break
    if transit_index_col is not None:
        df = df.rename(columns={transit_index_col: "transit_index"})
        return df[["date","country","transit_index"]].copy()

    # Google % change from baseline → convert to index around 100
    perc_col = None
    for cand in [
        "transit_stations_percent_change_from_baseline",
        "transitstation_percent_change_from_baseline",
        "transit_percent_change_from_baseline",
    ]:
        if cand in cols:
            perc_col = cols[cand]; break
    if perc_col is None:
        # best-effort: any numeric column as proxy
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            df["transit_index"] = df[num_cols[0]]
        else:
            df["transit_index"] = 100.0
    else:
        df["transit_index"] = 100.0 + pd.to_numeric(df[perc_col], errors="coerce")

    return df[["date","country","transit_index"]].copy()

def guess_region(c: str) -> str:
    if not isinstance(c, str): return "Other"
    c = c.lower()
    asia = ["singapore","japan","china","india","taiwan","hong kong","korea","indonesia","malaysia","thailand","philippines","vietnam"]
    europe = ["united kingdom","france","germany","italy","spain","netherlands","sweden","norway","denmark","poland"]
    na = ["united states","canada","mexico"]
    oce = ["australia","new zealand"]
    sa = ["brazil","argentina","chile","peru","colombia"]
    if c in asia: return "Asia"
    if c in europe: return "Europe"
    if c in na: return "North America"
    if c in oce: return "Oceania"
    if c in sa: return "South America"
    return "Other"

def normalize_ridership(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes to ['date','avg_daily_ridership'] or ['date','transport_type','avg_daily_ridership'].
    Supports layout: date (day-first), transport_type, value (index ~1.0).
    """
    if df is None or df.empty:
        years = pd.date_range("2019-01-01","2024-12-31", freq="Y")
        vals = np.array([5.2e6, 3.0e6, 3.5e6, 4.2e6, 4.7e6, 5.0e6])
        return pd.DataFrame({"date": years, "avg_daily_ridership": vals})

    cols = {c.lower(): c for c in df.columns}

    # Your file: date, transport_type, value
    if {"date","transport_type","value"}.issubset(set(cols)):
        dcol = cols["date"]; tcol = cols["transport_type"]; vcol = cols["value"]
        out = df[[dcol, tcol, vcol]].copy()
        out[dcol] = pd.to_datetime(out[dcol], dayfirst=True, errors="coerce")
        out = out.dropna(subset=[dcol])
        out.rename(columns={dcol:"date", tcol:"transport_type", vcol:"value_index"}, inplace=True)
        out["avg_daily_ridership"] = pd.to_numeric(out["value_index"], errors="coerce") * 1_000_000.0
        return out[["date","transport_type","avg_daily_ridership"]]

    # Generic fallbacks
    dcol = cols.get("date")
    for rid_guess in ["avg_daily_ridership","ridership","value","avg_daily","passengers","ridership_avg_daily"]:
        if rid_guess in cols and dcol:
            out = df[[dcol, cols[rid_guess]]].copy()
            out.columns = ["date","avg_daily_ridership"]
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
            return out.dropna(subset=["date"])

    for ycol in ["year","yr","fiscal_year"]:
        if ycol in cols:
            rid = next((cols[r] for r in ["avg_daily_ridership","ridership","value","avg_daily","passengers","ridership_avg_daily"] if r in cols), None)
            if rid:
                out = df[[cols[ycol], rid]].copy()
                out.columns = ["year","avg_daily_ridership"]
                out["date"] = pd.to_datetime(out["year"].astype(str) + "-12-31")
                return out[["date","avg_daily_ridership"]]

    years = pd.date_range("2019-01-01","2024-12-31", freq="Y")
    vals = np.array([5.2e6, 3.0e6, 3.5e6, 4.2e6, 4.7e6, 5.0e6])
    return pd.DataFrame({"date": years, "avg_daily_ridership": vals})

# =====================================================================================
# DATA INPUTS
# =====================================================================================
st.sidebar.header("Data Inputs")

# Mobility (Google)
up_mob_df, up_mob_kind = upload_override(
    "Upload Global_Mobility_Report.csv (optional)", [], synth_mobility
)
if up_mob_df is not None:
    raw_mob, kind_mob = up_mob_df, "upload"
else:
    raw_mob, kind_mob = load_csv_or_synth(
        Path("data/Global_Mobility_Report.csv"), synth_mobility, parse_dates=["date"]
    )

# UK/local ridership
up_sg_df, up_sg_kind = upload_override(
    "Upload full_data_clean.csv (optional)", [], synth_lta_sg
)
if up_sg_df is not None:
    raw_sg, kind_sg = up_sg_df, "upload"
else:
    raw_sg, kind_sg = load_csv_or_synth(Path("data/full_data_clean.csv"), synth_lta_sg)

badge(kind_mob if isinstance(kind_mob, str) else "file")
badge(kind_sg if isinstance(kind_sg, str) else "file")

# Normalize
try:
    mob = normalize_mobility_columns(raw_mob)
except Exception as e:
    st.error(f"Mobility file issue: {e}. Using synthetic mobility.")
    mob = synth_mobility()
mob["country"] = mob["country"].astype(str)
mob["region_guess"] = mob["country"].map(guess_region)

try:
    rid = normalize_ridership(raw_sg)
except Exception as e:
    st.error(f"Ridership file issue: {e}. Using synthetic ridership.")
    rid = normalize_ridership(pd.DataFrame())

# =====================================================================================
# H2.1 — Mobility Drop Magnitude by Region (Mar–May 2020)
# =====================================================================================
st.subheader("H2.1 — Mobility Drop Magnitude by Region (Mar–May 2020)")
c1, c2 = st.columns([1,3])
with c1:
    start_2020 = st.date_input("Start (2020)", pd.to_datetime("2020-03-01"))
    end_2020 = st.date_input("End (2020)", pd.to_datetime("2020-05-31"))
m20 = mob[(mob["date"]>=pd.to_datetime(start_2020)) & (mob["date"]<=pd.to_datetime(end_2020))].copy()
m20["delta_from_baseline"] = m20["transit_index"] - 100.0
drop_by_region = (m20.groupby("region_guess")["delta_from_baseline"]
                     .mean(numeric_only=True).reset_index()
                     .sort_values("delta_from_baseline"))
fig_h21 = px.bar(drop_by_region, x="region_guess", y="delta_from_baseline",
                 labels={"region_guess":"Region","delta_from_baseline":"Avg % change vs baseline"},
                 title="Average transit mobility change (negative = bigger drop)")
st.plotly_chart(fig_h21, use_container_width=True)

# =====================================================================================
# H2.2 — Recovery Rate & Timeline by Country
# =====================================================================================
st.subheader("H2.2 — Recovery Rate & Timeline by Country")
countries_all = sorted(mob["country"].dropna().unique().tolist())
default_sel = [c for c in ["Singapore","United Kingdom","United States","Japan"] if c in countries_all][:3] or countries_all[:3]
sel_countries = st.multiselect("Select countries", countries_all, default=default_sel)
date_min = st.date_input("From date", pd.to_datetime("2020-01-01"))
date_max = st.date_input("To date", pd.to_datetime("2024-12-31"))
mrec = mob[(mob["country"].isin(sel_countries)) &
           (mob["date"]>=pd.to_datetime(date_min)) &
           (mob["date"]<=pd.to_datetime(date_max))].copy()
mrec = mrec.sort_values("date").copy()
mrec["transit_index_ra"] = (
    mrec.groupby("country")["transit_index"]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
)

fig_h22 = px.line(mrec, x="date", y="transit_index_ra", color="country",
                  labels={"transit_index_ra":"Transit index (7-day avg)"})
st.plotly_chart(fig_h22, use_container_width=True)

# =====================================================================================
# H2.3 — Mobility vs Local Ridership (Correlation)  [UK + transport_type selector]
# =====================================================================================
st.subheader("H2.3 — Mobility vs Local Ridership (Correlation)")

# Mobility country selector (default UK if present)
mob_countries = sorted(mob["country"].dropna().astype(str).unique().tolist())
default_country = "United Kingdom" if "United Kingdom" in mob_countries else (mob_countries[0] if mob_countries else "Unknown")
default_idx = mob_countries.index(default_country) if default_country in mob_countries else 0
sel_country = st.selectbox("Mobility country", mob_countries, index=default_idx)

# Ridership transport_type selector (if available)
if "transport_type" in rid.columns:
    rid_types = sorted(rid["transport_type"].dropna().astype(str).unique().tolist())
    default_type = "tfl_tube" if "tfl_tube" in rid_types else rid_types[0]
    sel_rid_type = st.selectbox("Ridership series (transport_type)", rid_types, index=rid_types.index(default_type))
    rid_series = rid[rid["transport_type"] == sel_rid_type][["date","avg_daily_ridership"]].copy()
else:
    sel_rid_type = None
    rid_series = rid[["date","avg_daily_ridership"]].copy()

# Slice mobility
cty_mob = mob[mob["country"].astype(str) == sel_country][["date","transit_index"]].copy()

if cty_mob.empty or rid_series.empty:
    st.warning("Selected data series is empty. Check country or transport_type.")
else:
    # MONTHLY merge
    cty_mob["month"] = cty_mob["date"].dt.to_period("M").dt.to_timestamp()
    cty_mob_m = cty_mob.groupby("month", as_index=False)["transit_index"].mean()

    rid_series["month"] = rid_series["date"].dt.to_period("M").dt.to_timestamp()
    rid_m = rid_series.groupby("month", as_index=False)["avg_daily_ridership"].mean()

    merged = pd.merge(cty_mob_m, rid_m, on="month", how="inner")

    label_right = f"Avg daily ridership{' — ' + sel_rid_type if sel_rid_type else ''}"

    if len(merged) >= 3:
        corr = merged[["transit_index","avg_daily_ridership"]].corr().iloc[0,1]
        st.caption(f"Pearson correlation (monthly) = **{corr:.3f}**")

        fig_h23 = go.Figure()
        fig_h23.add_trace(go.Scatter(x=merged["month"], y=merged["transit_index"],
                                     mode="lines+markers", name="Mobility (index, left)", yaxis="y1"))
        fig_h23.add_trace(go.Scatter(x=merged["month"], y=merged["avg_daily_ridership"],
                                     mode="lines+markers", name=label_right, yaxis="y2"))
        fig_h23.update_layout(
            title=f"{sel_country} Mobility vs {sel_rid_type or 'ridership'} (monthly)",
            xaxis=dict(title="Month"),
            yaxis=dict(title="Mobility index (≈100 baseline)"),
            yaxis2=dict(title=label_right, overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        st.plotly_chart(fig_h23, use_container_width=True)
    else:
        # YEARLY fallback (works if ridership is annual or sparse)
        cty_mob["year"] = cty_mob["date"].dt.year
        cty_mob_y = cty_mob.groupby("year", as_index=False)["transit_index"].mean()

        rid_series["year"] = rid_series["date"].dt.year
        rid_y = rid_series.groupby("year", as_index=False)["avg_daily_ridership"].mean()

        merged_y = pd.merge(cty_mob_y, rid_y, on="year", how="inner")
        if len(merged_y) == 0:
            st.warning("No overlapping years found between mobility and ridership.")
        else:
            corr_y = merged_y[["transit_index","avg_daily_ridership"]].corr().iloc[0,1]
            st.caption(f"Pearson correlation (yearly) = **{corr_y:.3f}**")

            fig_h23y = go.Figure()
            fig_h23y.add_trace(go.Bar(x=merged_y["year"], y=merged_y["transit_index"],
                                      name="Mobility index (left)", yaxis="y1"))
            fig_h23y.add_trace(go.Scatter(x=merged_y["year"], y=merged_y["avg_daily_ridership"],
                                          mode="lines+markers", name=label_right, yaxis="y2"))
            fig_h23y.update_layout(
                title=f"{sel_country} Mobility vs {sel_rid_type or 'ridership'} (yearly)",
                xaxis=dict(title="Year"),
                yaxis=dict(title="Mobility index (≈100 baseline)"),
                yaxis2=dict(title=label_right, overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            st.plotly_chart(fig_h23y, use_container_width=True)

# =====================================================================================
# H2.4 — Pre vs Post COVID Distribution
# =====================================================================================
st.subheader("H2.4 — Distribution Shift: Pre-COVID vs Post-COVID")
p1, p2, p3 = st.columns(3)
with p1: pre_start = st.date_input("Pre start", pd.to_datetime("2019-10-01"))
with p2: pre_end   = st.date_input("Pre end",   pd.to_datetime("2019-12-31"))
with p3: post_year = st.selectbox("Post period (year)", [2021,2022,2023,2024], index=2)

pre  = mob[(mob["date"]>=pd.to_datetime(pre_start)) & (mob["date"]<=pd.to_datetime(pre_end))].assign(period="Pre-COVID")
post = mob[(mob["date"]>=pd.to_datetime(f"{post_year}-01-01")) & (mob["date"]<=pd.to_datetime(f"{post_year}-12-31"))].assign(period=f"Post-COVID {post_year}")
dist = pd.concat([pre, post], ignore_index=True)
fig_h24 = px.violin(dist, x="period", y="transit_index", color="period", box=True, points="all",
                    labels={"transit_index":"Mobility index"})
st.plotly_chart(fig_h24, use_container_width=True)

# =====================================================================================
# H2.5 — 2019 vs 2024 Regional Averages
# =====================================================================================
st.subheader("H2.5 — 2019 vs 2024 Regional Averages")

c1, c2 = st.columns(2)
with c1:
    y1 = st.selectbox("Baseline year", [2019, 2020, 2021, 2022, 2023, 2024], index=0)
with c2:
    y2 = st.selectbox("Recovery year", [2024, 2023, 2022, 2021, 2020, 2019], index=0)

# Ensure helpers/columns
if "region_guess" not in mob.columns:
    mob["region_guess"] = mob["country"].map(guess_region)
mob["year"] = mob["date"].dt.year.astype("int64")

# Use isin (or == comparisons) instead of "in @y1"
reg_year = (
    mob[mob["year"].isin([y1, y2])]
      .groupby(["region_guess", "year"], as_index=False)["transit_index"]
      .mean()
)

# Keep only regions that have BOTH years
have_both = reg_year.groupby("region_guess")["year"].nunique().reset_index(name="n_years")
both_regions = have_both.loc[have_both["n_years"] == 2, "region_guess"]
reg_year = reg_year[reg_year["region_guess"].isin(both_regions)]

# Labels / ordering
reg_year["year_label"] = reg_year["year"].map({y1: str(y1), y2: str(y2)})
region_order = ["Asia", "Europe", "North America", "South America", "Oceania", "Africa", "Other"]
reg_year["region_guess"] = pd.Categorical(reg_year["region_guess"], region_order)
reg_year = reg_year.sort_values(["region_guess", "year"])

# Plot
title_txt = f"H2.5 — Stabilization vs Over-Recovery ({y1} vs {y2})"
fig_h25_reg = px.bar(
    reg_year,
    x="region_guess",
    y="transit_index",
    color="year_label",
    barmode="group",
    category_orders={"region_guess": region_order, "year_label": [str(y1), str(y2)]},
    labels={"region_guess": "", "transit_index": "Avg Mobility Index", "year_label": "year"},
    title=title_txt,
)
fig_h25_reg.update_layout(margin=dict(l=10, r=10, t=50, b=10), bargap=0.25)
st.plotly_chart(fig_h25_reg, use_container_width=True)

st.caption("Annual average of weekly transit mobility by region. Values > 100 indicate above pre-COVID baseline.")
# =====================================================================================
# Target Audience, Recommendations and Remedies
# =====================================================================================

st.divider()
st.header("🎯 Target Audience")

st.markdown("""
The findings in this COVID-19 Impact & Recovery module are designed primarily for:

- **Urban Transport Authorities** – such as LTA Singapore, Transport for London, and city councils analysing post-pandemic recovery trends.  
- **Policy Makers and Planners** – developing adaptive mobility strategies to strengthen transport resilience.  
- **Researchers and Data Analysts** – exploring behavioural shifts and modelling recovery trajectories.  
- **Public Transport Operators** – optimising fleet operations, scheduling, and passenger load management in the post-COVID landscape.  
- **Commuters and the General Public** – understanding how mobility patterns changed during and after the pandemic.  
""")

st.divider()
st.header("💡 Recommendations")

st.markdown("""
**1. Enhance Public Confidence:**  
Clear safety communication, real-time crowd monitoring, and transparent sanitation updates encourage commuters to return to public transport.

**2. Strengthen Data-Driven Planning:**  
Authorities should integrate mobility indices, ridership, and health metrics into decision dashboards to predict and mitigate future disruptions.

**3. Diversify Transport Options:**  
Expand first- and last-mile solutions (e-scooters, bike-sharing, feeder buses) to reduce crowding and dependence on single modes.

**4. Support Digital Transformation:**  
Use contactless ticketing, mobile journey planning, and predictive analytics to improve commuter experience and operational flexibility.

**5. Promote Sustainable Mobility:**  
Encourage eco-friendly modes and integrate carbon-reduction targets into post-pandemic transport strategies.
""")

st.divider()
st.header("🩺 Remedies / Policy Actions")

st.markdown("""
- **Short-Term Remedy:** Maintain hybrid work and staggered commute policies to manage peak-hour loads.  
- **Medium-Term Remedy:** Build adaptive scheduling systems using AI/ML to dynamically adjust service frequency by demand.  
- **Long-Term Remedy:** Invest in resilient transport infrastructure and continuous open-data sharing frameworks for pandemic preparedness.

Together, these strategies support **transport system recovery, commuter confidence, and long-term sustainability** — aligning with global post-COVID urban mobility goals.
""")





