import pandas as pd
import streamlit as st
import traceback
import io
import os
import base64

st.set_page_config(page_title="Helpdesk Zoekfunctie", layout="wide")

# 📷 Logo's in gekleurde balk
ipal_logo = base64.b64encode(open("logo.png", "rb").read()).decode()
docbase_logo = base64.b64encode(open("logo-docbase-icon.png", "rb").read()).decode()
exact_logo = base64.b64encode(open("Exact.png", "rb").read()).decode()

st.markdown(
    f"""
    <div style='background-color: rgb(42, 68, 173); padding: 10px 20px; display: flex; align-items: center; border-radius: 0 0 10px 10px;'>
        <img src='data:image/png;base64,{ipal_logo}' style='height: 80px; margin-right: 20px;'>
        <img src='data:image/png;base64,{docbase_logo}' style='height: 60px; margin-right: 20px;'>
        <img src='data:image/png;base64,{exact_logo}' style='height: 40px; margin-right: 20px;'>
        <h1 style='color: white; font-size: 2.5rem;'>🔍 Helpdesk Zoekfunctie</h1>
    </div>
    <style>
        html, body, .stApp {{
            background-color: #FFD3AC;
        }}
        .stSelectbox > div {{
            border-radius: 10px !important;
            box-shadow: 1px 1px 6px rgba(0,0,0,0.15);
            border: 1px solid #2A44AD;
            padding: 4px;
        }}
        .stSelectbox > div:hover {{
            border: 1px solid rgb(42, 68, 173);
        }}
        .stSelectbox label, .stTextInput label, .stRadio label {{
            color: rgb(42, 68, 173) !important;
            font-weight: bold;
        }}
        .stDownloadButton button, .stButton button {{
            background-color: rgb(42, 68, 173);
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }}
        .stDownloadButton button:hover, .stButton button:hover {{
            background-color: rgb(30, 50, 130);
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# 🔐 Wachtwoord
if "wachtwoord_ok" not in st.session_state:
    wachtwoord = st.text_input("Wachtwoord:", type="password")
    if wachtwoord != "ipal2024":
        st.warning("Voer het juiste wachtwoord in om toegang te krijgen.")
        st.stop()
    else:
        st.session_state.wachtwoord_ok = True
        st.rerun()

# Rest van de code blijft ongewijzigd...
