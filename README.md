# ICT305 — Global Public Transport Dashboard (Arya's Part)

This Streamlit app implements **5 hypotheses** and **5 interactive Python graphs** for the group project.
It runs even without external datasets by generating synthetic samples, and supports **user interaction** via filters.
You can also **upload your own CSVs** to override the generated data.

## Quick start
```bash
# 1) (Recommended) Create venv
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run
streamlit run app.py
```

## Project structure
```
ICT305_Arya_Dashboard/
├── app.py
├── pages/
│   ├── 1_H1_Network_Efficiency.py
│   ├── 2_H2_Covid_Recovery.py
│   ├── 3_H3_City_Ridership_Patterns.py
│   ├── 4_H4_Length_Stations_Sprawl.py
│   └── 5_H5_Density_vs_GDP.py
├── data/               # place your CSVs here (optional)
│   ├── metro_transit_data.csv
│   ├── mobility_global.csv
│   ├── lta_sg_ridership.csv
│   ├── metro_countries_total.csv
│   ├── metro_systems_worldwide.csv
│   └── city_economics.csv
├── requirements.txt
└── README.md
```

## Expected (optional) CSV schema
- `metro_transit_data.csv`: columns like `city,continent,stations,lines,system_km,ridership_millions`
- `mobility_global.csv`: columns like `date,country,transit_index` (baseline 100 ≈ Feb-2020)
- `lta_sg_ridership.csv`: columns like `year,avg_daily_ridership`
- `metro_countries_total.csv`: columns like `country,region,stations,length_km,ridership_millions`
- `metro_systems_worldwide.csv`: columns like `city,stations,lines,system_km,population_millions,sprawl_index`
- `city_economics.csv`: columns like `city,country,gdp_per_capita,population_millions,stations,region`

If a file is missing, the app will generate a small synthetic dataframe so you can still demo the graphs and interactions.

## Notes
- All charts are **interactive** (Plotly). Filters, selectors, and sliders allow **user intervention**.
- Pages and titles follow the assignment's tone and your team's proposal document.
- You can change the color theme in Streamlit settings if needed.

— Prepared for **Manikantan Pillai Vijayakumary Arya (35502575)**
