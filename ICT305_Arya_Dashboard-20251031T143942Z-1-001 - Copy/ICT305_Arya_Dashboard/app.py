
import streamlit as st
st.set_page_config(page_title="ICT305 — Global Public Transport", page_icon="🌍", layout="wide")

# Jump straight to the Introduction page
try:
    st.switch_page("pages/0_Introduction.py")
except Exception:
    # Older Streamlit: show a link instead
    st.write("Open the **Introduction** page from the sidebar.")

