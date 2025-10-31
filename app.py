import streamlit as st

st.set_page_config(page_title="ICT305 – Global Public Transport",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Immediately open the Intro page (Streamlit ≥ 1.38)
st.switch_page("pages/0_Introduction.py")


