import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
#  MAIN CACHED LOADER
# ============================================================
@st.cache_data(show_spinner=False)
def load_csv_or_synth(path: Path, _synthesizer, **read_csv_kwargs):
    """
    Load a CSV from disk if present; otherwise return a synthetic DataFrame.
    - Adds sensible defaults for dtype and date parsing to avoid mixed-type warnings.
    - Limits very large files by reading the first ~750k rows (3x 250k chunks).
    - The leading underscore in _synthesizer avoids Streamlit hashing issues.
    Returns: (DataFrame, source_kind) where source_kind ∈ {"file","synthetic"}.
    """
    path = Path(path)

    if path.exists():
        try:
            # --- Default parsing options; caller can override any of these via **read_csv_kwargs
            defaults = {
                "low_memory": False,
                "dtype": {
                    "region": "string",
                    "country": "string",
                    "subway_ridership": "float64",
                    "date": "string",
                },
                "parse_dates": ["date"],
            }

            # Merge caller kwargs, with caller taking precedence
            merged = {**defaults, **read_csv_kwargs}
            if "dtype" in read_csv_kwargs and isinstance(read_csv_kwargs["dtype"], dict):
                # Deep-merge dtype dicts so caller wins per-column
                merged["dtype"] = {**defaults.get("dtype", {}), **read_csv_kwargs["dtype"]}

            file_size = path.stat().st_size
            too_big = file_size > 500 * 1024 * 1024  # 500 MB

            if too_big:
                # Read a limited number of rows for responsiveness
                chunks = []
                for i, chunk in enumerate(pd.read_csv(path, chunksize=250_000, **merged)):
                    chunks.append(chunk)
                    if i >= 2:  # ~750k rows max
                        break
                df = pd.concat(chunks, ignore_index=True)
                st.info(
                    f"Large file detected: loaded the first {len(df):,} rows from {path.name} for speed."
                )
            else:
                df = pd.read_csv(path, **merged)

            if df.empty:
                raise ValueError("CSV is empty")

            return df, "file"

        except Exception as e:
            st.warning(f"Failed to read {path.name}: {e}. Using synthetic data instead.")
            return _synthesizer(), "synthetic"

    # No file found -> synthetic
    return _synthesizer(), "synthetic"


# ============================================================
#  FILE UPLOAD OVERRIDE
# ============================================================
def upload_override(label: str, expected_cols: list[str], synthesizer):
    """
    Optional CSV uploader. If provided, validates that expected columns exist.
    Returns (DataFrame, kind) where kind ∈ {'upload','synthetic'} or (None, None) if no upload.
    """
    up = st.file_uploader(label, type=["csv"], key=f"u_{label}")
    if up is None:
        return None, None

    try:
        df = pd.read_csv(up, low_memory=False)
        missing = [c for c in expected_cols if c and c not in df.columns]
        if missing:
            st.error(f"Uploaded CSV is missing columns: {missing}. Using synthetic data instead.")
            return synthesizer(), "synthetic"
        return df, "upload"
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}. Using synthetic data instead.")
        return synthesizer(), "synthetic"


# ============================================================
#  SMALL BADGE DISPLAY
# ============================================================
def badge(source_kind: str):
    if source_kind == "file":
        st.caption("✅ Using CSV from /data")
    elif source_kind == "upload":
        st.caption("✅ Using your uploaded CSV")
    else:
        st.caption("🧪 Using synthetic sample (replace with your real CSV)")


# ============================================================
#  SYNTHETIC SAMPLE GENERATORS (shared across pages)
# ============================================================
def synth_mobility():
    """Synthetic mobility data showing realistic recovery from 2020 to 2023."""
    rng = pd.date_range("2019-01-01", "2023-12-31", freq="W")
    countries = ["Singapore", "United Kingdom", "United States", "Japan", "Australia"]
    rows = []
    for c in countries:
        base = 100
        for d in rng:
            # Start around baseline
            val = base + np.random.normal(0, 3)

            # 2020 — COVID crash
            if d.year == 2020 and 3 <= d.month <= 6:
                val -= np.random.uniform(35, 65)

            # Gradual recovery 2021–2023
            elif d.year == 2021:
                val += np.random.uniform(-5, 10)
            elif d.year == 2022:
                val += np.random.uniform(5, 15)
            elif d.year == 2023:
                val += np.random.uniform(10, 25)  # recovery beyond baseline

            rows.append((d, c, max(val, 10)))
    return pd.DataFrame(rows, columns=["date", "country", "transit_index"])

def synth_lta_sg():
    # Simple annual series approximating MRT recovery trend
    years = np.arange(2015, 2025)
    ridership = np.array([5.5, 5.7, 5.9, 6.1, 6.2, 4.0, 4.6, 5.1, 5.4, 5.8])  # billions (illustrative)
    return pd.DataFrame(
        {"year": years, "avg_daily_ridership": (ridership * 1e9) / 365}
    )


def synth_metro_transit():
    np.random.seed(7)
    cities = ["Singapore", "Tokyo", "Paris", "Seoul", "New York", "London", "Shanghai", "Delhi"]
    continents = ["Asia", "Asia", "Europe", "Asia", "North America", "Europe", "Asia", "Asia"]
    stations = np.random.randint(50, 350, size=len(cities))
    lines = np.random.randint(3, 16, size=len(cities))
    system_km = np.random.randint(80, 800, size=len(cities))
    ridership = (system_km * np.random.uniform(0.6, 1.4, size=len(cities))) + np.random.randint(50, 250, size=len(cities))
    ridership = np.round(ridership / 10, 2)
    return pd.DataFrame(
        {
            "city": cities,
            "continent": continents,
            "stations": stations,
            "lines": lines,
            "system_km": system_km,
            "ridership_millions": ridership,
        }
    )


def synth_countries_total():
    countries = ["China", "Japan", "Singapore", "USA", "UK", "India", "France"]
    region = ["Asia", "Asia", "Asia", "North America", "Europe", "Asia", "Europe"]
    stations = [9000, 3500, 220, 1800, 1200, 900, 1000]
    length_km = [9000, 3800, 300, 2100, 1600, 900, 1200]
    ridership = [27000, 14000, 3500, 3200, 2400, 4200, 3000]  # millions
    return pd.DataFrame(
        {
            "country": countries,
            "region": region,
            "stations": stations,
            "length_km": length_km,
            "ridership_millions": ridership,
        }
    )


def synth_worldwide():
    cities = ["Singapore", "Los Angeles", "Tokyo", "Hong Kong", "Taipei", "Delhi", "Paris"]
    stations = [220, 93, 682, 93, 131, 250, 303]
    lines = [5, 2, 13, 11, 5, 10, 16]
    system_km = [300, 170, 850, 175, 170, 420, 350]
    pop_m = [5.9, 13.2, 37.0, 7.4, 2.6, 32.9, 11.2]
    sprawl = [0.2, 0.7, 0.3, 0.2, 0.35, 0.55, 0.4]
    return pd.DataFrame(
        {
            "city": cities,
            "stations": stations,
            "lines": lines,
            "system_km": system_km,
            "population_millions": pop_m,
            "sprawl_index": sprawl,
        }
    )


def synth_city_econ():
    cities = ["Singapore", "Tokyo", "Paris", "Seoul", "New York", "London", "Shanghai", "Delhi", "Hong Kong", "Taipei"]
    country = ["Singapore", "Japan", "France", "Korea, Rep.", "United States", "United Kingdom", "China", "India", "China (Hong Kong)", "Taiwan"]
    gdp = [88000, 42000, 48000, 36000, 76000, 56000, 17000, 7800, 51000, 33000]
    pop = [5.9, 37, 11, 10, 19, 9.6, 24, 33, 7.4, 2.6]
    stations = [220, 682, 303, 308, 424, 270, 420, 250, 93, 131]
    region = ["Asia", "Asia", "Europe", "Asia", "North America", "Europe", "Asia", "Asia", "Asia", "Asia"]
    return pd.DataFrame(
        {
            "city": cities,
            "country": country,
            "gdp_per_capita": gdp,
            "population_millions": pop,
            "stations": stations,
            "region": region,
        }
    )
