# Dit is een aangepaste versie van main.py die speciaal is ontworpen voor oudere vrijwilligers
# met beperkte computervaardigheden. Het bevat grote knoppen, groter lettertype, eenvoudige taal
# en duidelijke visuele structuur. Afbeeldingsondersteuning wordt ook toegevoegd via een Excel-kolom.

import os
from datetime import datetime
import streamlit as st
import openai
import pandas as pd
import re
from dotenv import load_dotenv
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import base64
import requests
from bs4 import BeautifulSoup
import pytz

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

st.set_page_config(page_title="IPAL Chatbox", layout="centered")

# Grotere tekst en klikbare elementen voor leesbaarheid
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-size: 20px;
        }
        button[kind="primary"] {
            font-size: 22px !important;
            padding: 0.75em 1.5em;
        }
    </style>
""", unsafe_allow_html=True)

def validate_api_key():
    if not openai.api_key:
        st.error("⚠️ Stel uw toegangssleutel in via een .env-bestand.")
        st.stop()
    try:
        openai.models.list()
    except openai.AuthenticationError:
        st.error("⚠️ Ongeldige sleutel. Controleer uw instellingen.")
        st.stop()
    except openai.RateLimitError as e:
        st.error("⚠️ API-limiet bereikt. Probeer het later opnieuw.")
        st.stop()

validate_api_key()

@st.cache_data
def load_faq(path: str = 'faq.xlsx') -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])
    try:
        df = pd.read_excel(path)
        required_columns = ['Systeem', 'Subthema', 'Omschrijving melding', 'Toelichting melding', 'Antwoord of oplossing']
        optional_columns = ['Afbeelding']

        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            st.error(f"FAQ-bestand mist vereiste kolommen: {missing_required}")
            return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])

        for col in optional_columns:
            if col not in df.columns:
                df[col] = None

        def convert_hyperlink(text):
            if isinstance(text, str) and text.startswith('=HYPERLINK'):
                match = re.match(r'=HYPERLINK\("([^"]+)","([^"]+)"\)', text)
                if match:
                    url, display_text = match.groups()
                    return f"[{display_text}]({url})"
            return text

        df['Antwoord of oplossing'] = df['Antwoord of oplossing'].apply(convert_hyperlink)
        df['combined'] = df[required_columns].fillna('').agg(' '.join, axis=1)
        return df

    except Exception as e:
        st.error(f"Fout bij laden FAQ: {str(e)}")
        return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])

faq_df = load_faq()
producten = ["Exact", "DocBase"]

subthema_dict = {}
if not faq_df.empty:
    for product in producten:
        subthema_dict[product] = sorted(
            [s for s in faq_df[faq_df['Systeem'] == product]['Subthema'].dropna().unique() if isinstance(s, str) and s.strip()]
        )

if 'history' not in st.session_state:
    st.session_state.history = []
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'selected_module' not in st.session_state:
    st.session_state.selected_module = None
if 'reset_triggered' not in st.session_state:
    st.session_state.reset_triggered = False

timezone = pytz.timezone("Europe/Amsterdam")

def add_message(role: str, content: str):
    current_time = datetime.now(timezone).strftime('%d-%m-%Y %H:%M')
    st.session_state.history.append({'role': role, 'content': content, 'time': current_time})
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[-100:]

def render_chat():
    for msg in st.session_state.history:
        if msg['role'] == 'assistant' and os.path.exists("aichatbox.jpg"):
            avatar_img = Image.open("aichatbox.jpg")
            avatar = avatar_img.resize((64, 64))
        else:
            avatar = '🙂'
        st.chat_message(msg['role'], avatar=avatar).markdown(f"{msg['content']}\n\n_{msg['time']}_")

def on_reset():
    st.session_state.reset_triggered = True
    st.session_state.selected_product = None
    st.session_state.selected_module = None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), retry=retry_if_exception_type(openai.RateLimitError))
def get_ai_answer(user_text: str) -> str:
    system_prompt = "You are IPAL Chatbox, a helpful Dutch helpdesk assistant. Answer clearly and concisely."
    messages = [{'role': 'system', 'content': system_prompt}]
    for m in st.session_state.history[-10:]:
        messages.append({'role': m['role'], 'content': m['content']})
    full_question = f"[{st.session_state.selected_module}] {user_text}"
    messages.append({'role': 'user', 'content': full_question})
    try:
        resp = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=400
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None

def get_answer(user_text: str) -> str:
    if not faq_df.empty:
        df_filtered = faq_df[faq_df['Subthema'] == st.session_state.selected_module]
        pattern = re.escape(user_text)
        matches = df_filtered[df_filtered['combined'].str.contains(pattern, case=False, na=False, regex=True)]
        if not matches.empty:
            antwoord = matches.iloc[0]['Antwoord of oplossing']
            afbeelding = matches.iloc[0].get('Afbeelding', None)
            try:
                uitleg_resp = openai.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "Herschrijf dit antwoord eenvoudig, duidelijk en vriendelijk voor oudere gebruikers."},
                        {"role": "user", "content": antwoord}
                    ],
                    temperature=0.2,
                    max_tokens=300
                )
                uitleg = uitleg_resp.choices[0].message.content.strip()
                if afbeelding and os.path.exists(afbeelding):
                    st.image(afbeelding, caption="Voorbeeld", use_column_width=True)
                return uitleg
            except Exception as e:
                return f"{antwoord}\n\n(Kon niet verduidelijken: {str(e)})"
    ai = get_ai_answer(user_text)
    if ai:
        return f"IPAL-Helpdesk antwoord:
{ai}"
    return "Er is geen antwoord gevonden. Formuleer uw vraag anders of klik op 'Nieuw gesprek'."

def main():
    if st.session_state.reset_triggered:
        st.session_state.history = []
        st.session_state.selected_product = None
        st.session_state.selected_module = None
        st.session_state.reset_triggered = False

    st.sidebar.button('🔄 Nieuw gesprek', on_click=on_reset)

    if not st.session_state.selected_product:
        st.header("Welkom bij de IPAL Helpdesk")
        st.write("Klik hieronder op het systeem waarover u een vraag heeft.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("DocBase", use_container_width=True):
                st.session_state.selected_product = "DocBase"
                st.rerun()
        with col2:
            if st.button("Exact", use_container_width=True):
                st.session_state.selected_product = "Exact"
                st.rerun()
        render_chat()
        return

    if st.session_state.selected_product and not st.session_state.selected_module:
        opties = subthema_dict.get(st.session_state.selected_product, [])
        keuze = st.selectbox(
            "Kies een onderwerp waarover u een vraag heeft:",
            ["(Kies een onderwerp)"] + opties
        )
        if keuze != "(Kies een onderwerp)":
            st.session_state.selected_module = keuze
            st.session_state.history = []
            add_message('assistant', f"U heeft gekozen voor '{keuze}'. Wat wilt u hierover weten?")
            st.rerun()
        render_chat()
        return

    render_chat()
    vraag = st.chat_input("Stel hier uw vraag in uw eigen woorden:")
    if vraag:
        add_message('user', vraag)
        with st.spinner("Even zoeken naar het antwoord..."):
            antwoord = get_answer(vraag)
            add_message('assistant', antwoord)
        st.rerun()

if __name__ == '__main__':
    main()
