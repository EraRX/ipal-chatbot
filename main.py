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

# ---------------------------------------------
# IPAL Directe Interactieve Chatbox
# ---------------------------------------------
# Vereisten: streamlit, openai, pandas, pillow, python-dotenv, tenacity, openpyxl, requests, beautifulsoup4, pytz

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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
        st.stop()

validate_api_key()

st.set_page_config(page_title="IPAL Chatbox", layout="centered")

logo_path = "logo.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=160)

assistant_avatar = None
avatar_path = 'aichatbox.jpg'
if os.path.exists(avatar_path):
    try:
        img = Image.open(avatar_path)
        assistant_avatar = img.resize((img.width * 4, img.height * 4), Image.Resampling.LANCZOS)
    except Exception:
        assistant_avatar = None

@st.cache_data
def load_faq(path: str = 'faq.xlsx') -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"⚠️ FAQ-bestand '{path}' niet gevonden in {os.getcwd()}.")
        return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])
    try:
        df = pd.read_excel(path)
        required_columns = ['Systeem', 'Subthema', 'Omschrijving melding', 'Toelichting melding', 'Antwoord of oplossing']
        if not all(col in df.columns for col in required_columns):
            st.error(f"⚠️ FAQ-bestand mist vereiste kolommen. Verwachte kolommen: {required_columns}")
            return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])
        def convert_hyperlink(text):
            if isinstance(text, str) and text.startswith('=HYPERLINK'):
                match = re.match(r'=HYPERLINK\("([^"]+)","([^"]+)"\)', text)
                if match:
                    url, display_text = match.groups()
                    return f"[{display_text}]({url})"
            return text
        df['Antwoord of oplossing'] = df['Antwoord of oplossing'].apply(convert_hyperlink)
        search_columns = [col for col in required_columns if col != 'Antwoord of oplossing']
        df['combined'] = df[search_columns].fillna('').agg(' '.join, axis=1)
        return df
    except Exception as e:
        st.error(f"⚠️ Fout bij laden FAQ: {str(e)}")
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
    current_time = datetime.now(timezone).strftime('%Y-%m-%d %H:%M')
    st.session_state.history.append({'role': role, 'content': content, 'time': current_time})
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[-100:]

def render_chat():
    for msg in st.session_state.history:
        avatar = assistant_avatar if msg['role'] == 'assistant' and assistant_avatar else ('🤖' if msg['role'] == 'assistant' else '🙂')
        st.chat_message(msg['role'], avatar=avatar).markdown(f"{msg['content']}\n*{msg['time']}*")

def on_reset():
    st.session_state.reset_triggered = True
    st.session_state.selected_product = None
    st.session_state.selected_module = None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type(openai.RateLimitError))
def get_ai_answer(user_text: str) -> str:
    system_prompt = "You are IPAL Chatbox, a helpful Dutch helpdesk assistant. Answer clearly and concisely."
    history_limit = 10
    messages = [{'role': 'system', 'content': system_prompt}]
    for m in st.session_state.history[-history_limit:]:
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
    except Exception as e:
        return None

def get_faq_answer(user_text: str) -> str:
    if not faq_df.empty:
        try:
            df_filtered = faq_df[faq_df['Subthema'] == st.session_state.selected_module]
            pattern = re.escape(user_text)
            matches = df_filtered[df_filtered['combined'].str.contains(pattern, case=False, na=False, regex=True)]
            if not matches.empty:
                antwoorden = matches.head(3)['Antwoord of oplossing'].tolist()
                verduidelijkte_antwoorden = []

                for ans in antwoorden:
                    try:
                        uitleg_resp = openai.chat.completions.create(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": "Je bent een Nederlandse helpdeskassistent. Herschrijf een technisch antwoord uit een FAQ op een duidelijke, vriendelijke en goed leesbare manier, geschikt voor niet-technische gebruikers."},
                                {"role": "user", "content": f"Herschrijf dit begrijpelijk:\n{ans}"}
                            ],
                            temperature=0.3,
                            max_tokens=300
                        )
                        uitleg = uitleg_resp.choices[0].message.content.strip()
                        verduidelijkte_antwoorden.append(f"- {ans}\n  ↳ {uitleg}")
                    except Exception as e:
                        verduidelijkte_antwoorden.append(f"- {ans}\n  ↳ (Kon niet verduidelijken: {str(e)})")

                return "\n\n".join(verduidelijkte_antwoorden)
        except Exception as e:
            return f"FAQ-zoekfout: {str(e)}"
    return None

def get_answer(user_text: str) -> str:
    ai = get_ai_answer(user_text)
    faq = get_faq_answer(user_text)
    if ai and faq:
        return f"IPAL-Helpdesk antwoord:\n{ai}\n\nVeelgestelde vragen:\n{faq}"
    elif faq:
        return f"IPAL-Helpdesk antwoord uit FAQ:\n{faq}"
    elif ai:
        return f"IPAL-Helpdesk antwoord:\n{ai}"
    else:
        return "⚠️ Geen antwoord gevonden. Probeer je vraag anders te formuleren."

def main():
    if st.session_state.reset_triggered:
        st.session_state.history = []
        st.session_state.selected_product = None
        st.session_state.selected_module = None
        st.session_state.reset_triggered = False

    st.sidebar.button('🔄 Nieuw gesprek', on_click=on_reset)

    if not st.session_state.selected_product:
        if not st.session_state.history:
            add_message('assistant', 'Hallo, ik ben de IPAL AI-assistent, waarmee kan ik u helpen?')
        st.markdown("### Welkom bij de IPAL-Helpdesk:")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("logo-docbase-icon.png"):
                st.image("logo-docbase-icon.png", use_container_width=False, width=120)
                if st.button("Klik hier", key="docbase_button"):
                    st.session_state.selected_product = "DocBase"
                    st.rerun()
        with col2:
            if os.path.exists("Exact.png"):
                st.image("Exact.png", use_container_width=False, width=120)
                if st.button("Klik hier", key="exact_button"):
                    st.session_state.selected_product = "Exact"
                    st.rerun()
        render_chat()
        return

    if st.session_state.selected_product and not st.session_state.selected_module:
        opties = subthema_dict.get(st.session_state.selected_product, [])
        if not opties:
            st.error("⚠️ Geen subthema's beschikbaar voor dit product.")
            return
        keuze = st.selectbox(
            f"📁 Kies een module voor {st.session_state.selected_product}:",
            ["(Kies een module)"] + opties,
            key="module_select"
        )
        if keuze == "(Kies een module)":
            st.warning("Kies een module voordat je een vraag stelt.")
            render_chat()
            return
        else:
            st.session_state.selected_module = keuze
            st.session_state.history = []
            add_message('assistant', f"Hallo, ik ben de IPAL AI-assistent. Je hebt {keuze} gekozen. Welke vraag heb je?")
            st.rerun()

    render_chat()
    vraag = st.chat_input(f"Stel je vraag over {st.session_state.selected_module}:", key="chat_input_" + str(len(st.session_state.history)))
    if vraag:
        add_message('user', vraag)
        with st.spinner("Even nadenken..."):
            antwoord = get_answer(vraag)
            add_message('assistant', antwoord)
        st.rerun()

if __name__ == '__main__':
    main()
