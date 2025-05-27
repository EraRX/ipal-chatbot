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
import pytz

# Laad omgevingsvariabelen
dotenv_path = '.env'
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Pagina-configuratie
st.set_page_config(page_title="IPAL Chatbox", layout="centered")

# Stijlen voor grotere tekst/knoppen
st.markdown("""
<style>
  html, body, [class*="css"] { font-size: 20px; }
  button[kind="primary"] { font-size: 22px !important; padding: 0.75em 1.5em; }
</style>
""", unsafe_allow_html=True)

# Valideer API-sleutel
def validate_api_key():
    if not openai.api_key:
        st.error("⚠️ Stel uw OPENAI_API_KEY in via .env.")
        st.stop()
    try:
        openai.models.list()
    except openai.AuthenticationError:
        st.error("⚠️ Ongeldige API-sleutel.")
        st.stop()
    except openai.RateLimitError:
        st.error("⚠️ API-limiet bereikt, probeer later opnieuw.")
        st.stop()

validate_api_key()

# Laad FAQ
def load_faq(path: str = 'faq.xlsx') -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])
    try:
        df = pd.read_excel(path)
        required = ['Systeem', 'Subthema', 'Omschrijving melding', 'Toelichting melding', 'Antwoord of oplossing']
        if any(col not in df.columns for col in required):
            miss = [col for col in required if col not in df.columns]
            st.error(f"FAQ-bestand mist kolommen: {miss}")
            return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])
        if 'Afbeelding' not in df.columns:
            df['Afbeelding'] = None
        def convert_link(x):
            if isinstance(x, str) and x.startswith('=HYPERLINK'):
                m = re.match(r'=HYPERLINK\("([^"]+)","([^"]+)"\)', x)
                if m:
                    return f"[{m.group(2)}]({m.group(1)})"
            return x
        df['Antwoord of oplossing'] = df['Antwoord of oplossing'].apply(convert_link)
        df['combined'] = df[required].fillna('').agg(' '.join, axis=1)
        return df
    except Exception as e:
        st.error(f"Fout bij laden FAQ: {e}")
        return pd.DataFrame(columns=['combined', 'Antwoord of oplossing'])

faq_df = load_faq()
producten = ["Exact", "DocBase"]

# Bouw subthema-dict
subthema_dict = {}
if not faq_df.empty:
    for prod in producten:
        subthema_dict[prod] = sorted(
            faq_df.loc[faq_df['Systeem'] == prod, 'Subthema'].dropna().unique().tolist()
        )

# Session state defaults
if 'history' not in st.session_state:
    st.session_state.history = []
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'selected_module' not in st.session_state:
    st.session_state.selected_module = None
if 'reset_triggered' not in st.session_state:
    st.session_state.reset_triggered = False

timezone = pytz.timezone('Europe/Amsterdam')

def add_message(role: str, text: str):
    ts = datetime.now(timezone).strftime('%d-%m-%Y %H:%M')
    st.session_state.history.append({'role': role, 'content': text, 'time': ts})
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[-100:]

def render_chat():
    for msg in st.session_state.history:
        if msg['role'] == 'assistant' and os.path.exists('aichatbox.jpg'):
            avatar_img = Image.open('aichatbox.jpg').resize((64, 64))
            avatar = avatar_img
        else:
            avatar = '🙂'
        st.chat_message(msg['role'], avatar=avatar).markdown(f"{msg['content']}\n\n_{msg['time']}_")

def on_reset():
    st.session_state.history = []
    st.session_state.selected_product = None
    st.session_state.selected_module = None
    st.session_state.reset_triggered = False

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), retry=retry_if_exception_type(openai.RateLimitError))
def get_ai_answer(text: str) -> str:
    system_prompt = 'You are IPAL Chatbox, a helpful Dutch helpdesk assistant. Answer clearly and concisely.'
    messages = [{'role': 'system', 'content': system_prompt}] + [
        {'role': m['role'], 'content': m['content']} for m in st.session_state.history[-10:]
    ]
    messages.append({'role': 'user', 'content': f"[{st.session_state.selected_module}] {text}"})
    try:
        resp = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=400
        )
        return resp.choices[0].message.content.strip()
    except:
        return None

def get_answer(text: str) -> str:
    if faq_df is not None and st.session_state.selected_module:
        sub = faq_df[faq_df['Subthema'] == st.session_state.selected_module]
        pat = re.escape(text)
        matches = sub[sub['combined'].str.contains(pat, case=False, na=False)]
        if not matches.empty:
            row = matches.iloc[0]
            ans = row['Antwoord of oplossing']
            img = row.get('Afbeelding')
            try:
                resp = openai.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {'role': 'system', 'content': 'Herschrijf dit antwoord eenvoudig en vriendelijk.'},
                        {'role': 'user', 'content': ans}
                    ],
                    temperature=0.2,
                    max_tokens=300
                )
                ans2 = resp.choices[0].message.content.strip()
            except:
                ans2 = ans
            if img and os.path.exists(img):
                st.image(img, caption='Voorbeeld', use_column_width=True)
            return ans2
    ai_resp = get_ai_answer(text)
    if ai_resp:
        return f"IPAL-Helpdesk antwoord:\n{ai_resp}"
    return "Er is geen antwoord gevonden. Formuleer uw vraag anders of klik op 'Nieuw gesprek'."

# Hoofdapplicatie
def main():
    if st.session_state.reset_triggered:
        on_reset()
    st.sidebar.button('🔄 Nieuw gesprek', on_click=on_reset)
    if not st.session_state.selected_product:
        st.header('Welkom bij de IPAL Helpdesk')
        st.write('Klik hieronder op het systeem waarover u een vraag heeft.')
        c1, c2 = st.columns(2)
        if c1.button('DocBase', use_container_width=True):
            st.session_state.selected_product = 'DocBase'
            st.rerun()
        if c2.button('Exact', use_container_width=True):
            st.session_state.selected_product = 'Exact'
            st.rerun()
        add_message('assistant', 'Hallo, waarmee kan ik u helpen?')
        render_chat()
        return
    if not st.session_state.selected_module:
        opts = subthema_dict.get(st.session_state.selected_product, [])
        sel = st.selectbox('Kies een onderwerp:', ['(Kies)'] + opts)
        if sel != '(Kies)':
            st.session_state.selected_module = sel
            add_message('assistant', f"U heeft gekozen voor '{sel}'. Wat wilt u hierover weten?")
            st.rerun()
        render_chat()
        return
    render_chat()
    q = st.chat_input('Stel hier uw vraag:')
    if q:
        add_message('user', q)
        with st.spinner('Even zoeken...'):
            ans = get_answer(q)
            add_message('assistant', ans)
        st.rerun()

if __name__ == '__main__':
    main()
