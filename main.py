import streamlit as st
import pandas as pd
import base64

st.set_page_config(page_title="IPAL Helpdesk", layout="wide")

# Logo laden
with open("logo.png", "rb") as image_file:
    encoded_logo = base64.b64encode(image_file.read()).decode()

# CSS voor layout
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
    }
    .logo-container {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .card:hover {
        background-color: #f0f0f0;
    }
    .emoji {
        font-size: 40px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Header met logo
st.markdown(f"""
    <div class='logo-container'>
        <img src='data:image/png;base64,{encoded_logo}' style='height: 80px;'>
        <h1>IPAL Helpdesk</h1>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("🔍 Hoe kunnen we u helpen?")
st.text_input("Zoek helpartikelen...", placeholder="Typ hier uw vraag of trefwoord")

st.markdown("### 📂 Kies een systeem:")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔷 Exact", use_container_width=True):
        st.session_state["systeem"] = "Exact"
        st.switch_page("exact.py")

with col2:
    if st.button("🟣 DocBase", use_container_width=True):
        st.session_state["systeem"] = "DocBase"
        st.switch_page("docbase.py")

st.markdown("---")

st.markdown("""
    <div style='text-align: center; color: gray;'>
        Gemaakt met ❤️ door IPAL · Versie 1.0
    </div>
""", unsafe_allow_html=True)