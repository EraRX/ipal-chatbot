import os
from datetime import datetime
import streamlit as st
import openai
import pandas as pd
import re
from dotenv import load_dotenv
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---------------------------------------------
# IPAL Directe Interactieve Chatbox
# ---------------------------------------------
# Vereisten: streamlit, openai, pandas, pillow, python-dotenv, tenacity, openpyxl

# Laad API-sleutel en model uit .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Valideer API-sleutel
def validate_api_key():
    if not openai.api_key:
        st.error("⚠️ Stel je OPENAI_API_KEY in in een .env-bestand of Streamlit Cloud Secrets.")
        st.stop()
    try:
        openai.models.list()
    except openai.AuthenticationError:
        st.error("⚠️ Ongeldige OPENAI_API_KEY. Controleer je .env-bestand of Streamlit Cloud Secrets.")
        st.stop()
    except openai.RateLimitError as e:
        st.error("⚠️ API-limiet bereikt bij validatie. Controleer je account op https://platform.openai.com/usage.")
        print(f"RateLimitError bij validatie: {str(e)}")
        st.stop()

validate_api_key()

# Paginaconfiguratie
st.set_page_config(page_title="IPAL Chatbox", layout="centered")

# Laad logo in sidebar
logo_path = "logo.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=160)

# Laad en schaal avatar.png
assistant_avatar = None
avatar_path = 'avatar.png'
if os.path.exists(avatar_path):
    try:
        img = Image.open(avatar_path)
        assistant_avatar = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
    except Exception:
        assistant_avatar = None

# Laad FAQ-data
@st.cache_data
def load_faq(path: str = 'faq.xlsx') -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"⚠️ FAQ-bestand '{path}' niet gevonden in {os.getcwd()}. Plaats faq.xlsx in de juiste map.")
        return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])
    try:
        df = pd.read_excel(path)
        required_columns = ['Systeem', 'Subthema', 'Omschrijving melding', 'Toelichting melding', 'Antwoord of oplossing']
        if not all(col in df.columns for col in required_columns):
            st.error(f"⚠️ FAQ-bestand mist vereiste kolommen. Verwachte kolommen: {required_columns}")
            return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])
        # Converteer
