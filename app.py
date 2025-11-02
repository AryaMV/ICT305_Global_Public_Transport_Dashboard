import streamlit as st

st.set_page_config(
    page_title="ICT305 – Global Public Transport Dashboard",
    page_icon="🌍",
    layout="wide"
)

# --- Try to open the Introduction page automatically ---
try:
    st.switch_page("pages/0_Introduction.py")
except Exception:
    st.write("### Open the **Introduction** page manually from the sidebar 👈")
