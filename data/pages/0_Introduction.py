import streamlit as st

# ============================================================
# ICT305 – Global Public Transport Dashboard Introduction
# ============================================================

st.set_page_config(page_title="ICT305 — Global Public Transport", page_icon="🌍", layout="wide")

st.title("🌍 ICT305 — Global Public Transport Dashboard")

st.markdown("""
This project presents a comprehensive **interactive data analytics dashboard** designed to examine 
global public transport systems through five interrelated hypotheses.  
It integrates data science, visualization, and system performance concepts to evaluate how 
**mobility efficiency, recovery trends, ridership patterns, network scale, and socio-economic factors** 
influence public transport outcomes worldwide.

The dashboard combines **real-world datasets** (e.g., Google Mobility, LTA Singapore, TfL, World Bank indicators) 
with **synthetic data generators** to ensure robustness and reproducibility.  
Users can upload CSV files or rely on preloaded datasets to explore key dimensions of 
transport resilience, sustainability, and efficiency.
""")

st.divider()

st.header("🎯 Project Objectives")
st.markdown("""
- Evaluate **public transport performance** before, during, and after major disruptions such as COVID-19  
- Explore how **network size, density, and station length** relate to ridership and urban sprawl  
- Analyse the **correlation between economic growth (GDP)** and public transport intensity  
- Visualise and compare **mobility recovery trajectories** across regions and transport types  
- Provide a **replicable analytical framework** for evidence-based policy and transport planning  
""")

st.divider()

st.header("🧩 Dashboard Structure")
st.markdown("""
Each hypothesis corresponds to an individual interactive page powered by **Plotly and Streamlit**, 
featuring dynamic selectors, comparative charts, and summary metrics:

1. **H1 — Network Efficiency:** Relationship between system scale, coverage, and ridership efficiency  
2. **H2 — COVID-19 Impact & Recovery:** Global transit recovery trajectories from 2020 to 2024  
3. **H3 — City Ridership Patterns:** Cross-modal ridership trends across major cities and transport types  
4. **H4 — Length, Stations & Sprawl:** Association between urban sprawl, line length, and ridership density  
5. **H5 — Density vs GDP:** Impact of population and economic density on transport usage levels  
""")

st.divider()

st.header("⚙️ Technical Overview")
st.markdown("""
- **Framework:** Streamlit (Python-based web application)  
- **Visualization:** Plotly Express and Graph Objects  
- **Data Handling:** Pandas and NumPy for preprocessing and aggregation  
- **Architecture:** Modular structure with reusable utilities (`utils.py`), cached data loading, and synthetic fallbacks  
""")

st.divider()

st.header("📈 Key Outcomes")
st.markdown("""
The analysis highlights the **multi-dimensional nature of transport performance** — revealing that 
**network efficiency, recovery pace, and socio-economic factors** jointly determine the stability 
and adaptability of public transport systems.  

This project demonstrates the power of **data-driven visualization** for interpreting complex 
mobility phenomena and serves as a flexible, educational model for **urban transport analytics** 
and policy-based planning.
""")

st.divider()

