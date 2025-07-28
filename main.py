# main.py
"""
IPAL Chatbox voor oudere vrijwilligers
- Python 3, Streamlit
- Groot lettertype, eenvoudige bediening
- Antwoorden uit FAQ aangevuld met AI voor specifieke modules
- Topicfiltering (blacklist + herstelde fallback op geselecteerde module)
- Logging en foutafhandeling
- Antwoorden downloaden als PDF

Geschatte lengte: ~300+ lijnen
"""

import os
import re
import sys
import logging
from datetime import datetime
import streamlit as st
import openai
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pytz
import io
from fpdf import FPDF

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

BLACKLIST_CATEGORIES = [
    "persoonlijke gegevens", "medische gegevens", "gezondheid", "strafrechtelijk verleden",
    "financiële gegevens", "biometrische gegevens", "geboortedatum", "adresgegevens",
    "identiteitsbewijs", "burgerservicenummer", "persoonlijke overtuiging",
    "seksuele geaardheid", "etniciteit", "nationaliteit",
    "discriminatie", "racisme", "haatzaaiende taal", "xenofobie", "seksisme",
    "homofobie", "transfobie", "antisemitisme", "islamofobie", "vooroordelen",
    "stereotypering", "religie", "geloofsovertuiging", "godsdienstige leer", "religieuze extremisme",
    "sekten", "godslastering", "politiek", "politieke extremisme", "radicalisering", "terrorisme", "propaganda",
    "seksuele inhoud", "adult content", "pornografie", "seks", "sex", "seksueel",
    "seksualiteit", "erotiek", "prostitutie", "geweld", "fysiek geweld", "psychologisch geweld", "huiselijk geweld",
    "oorlog", "mishandeling", "misdaad", "illegale activiteiten", "drugs", "wapens", "smokkel",
    "desinformatie", "nepnieuws", "complottheorie", "misleiding", "fake news", "hoax",
    "gokken", "kansspelen", "verslaving", "online gokken", "casino",
    "zelfbeschadiging", "zelfmoord", "eetstoornissen", "kindermisbruik",
    "dierenmishandeling", "milieuschade", "exploitatie", "mensenhandel",
    "phishing", "malware", "hacking", "cybercriminaliteit", "doxing",
    "identiteitsdiefstal", "obsceniteit", "aanstootgevende inhoud", "schokkende inhoud",
    "gruwelijke inhoud", "sensatiezucht", "privacy schending"
]

MAX_HISTORY = 20

# Load environment and API key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

# Streamlit page config
st.set_page_config(page_title='IPAL Chatbox', layout='centered')
st.markdown(
    '''<style>
      html, body, [class*="css"] { font-size: 20px; }
      button[kind="primary"] { font-size: 22px !important; padding: 0.75em 1.5em; }
    </style>''', unsafe_allow_html=True)

# Input API key if missing
if not openai.api_key:
    api_key_input = st.sidebar.text_input('🔑 Voer uw OpenAI API-sleutel in:', type='password')
    if api_key_input:
        openai.api_key = api_key_input
    else:
        st.sidebar.warning('Geen API-sleutel gevonden. Vul uw sleutel in om verder te gaan.')
        st.stop()

@st.cache_data
def load_faq(path: str = 'faq.xlsx') -> pd.DataFrame:
    if not os.path.exists(path):
        logging.error(f"FAQ niet gevonden: {path}")
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=['combined', 'Antwoord'])
    try:
        df = pd.read_excel(path)
    except Exception as e:
        logging.error(f"Fout bij laden FAQ: {e}")
        st.error('⚠️ Kan FAQ niet laden')
        return pd.DataFrame(columns=['combined', 'Antwoord'])
    required = ['Systeem', 'Subthema', 'Omschrijving melding', 'Toelichting melding', 'Antwoord of oplossing']
    if 'Afbeelding' not in df.columns:
        df['Afbeelding'] = None
    df['Antwoord'] = df['Antwoord of oplossing']
    df['combined'] = df[required].fillna('').agg(' '.join, axis=1)
    return df

faq_df = load_faq()
producten = ['Exact', 'DocBase']
subthema_dict = {p: sorted(faq_df[faq_df['Systeem'] == p]['Subthema'].dropna().unique()) for p in producten}

# Blacklist & topic filter
def check_blacklist(text: str) -> list[str]:
    return [term for term in BLACKLIST_CATEGORIES if term in text.lower()]

def generate_warning(found_terms: list[str]) -> str:
    return ("Je bericht bevat inhoud die niet voldoet aan onze richtlijnen. Vermijd gevoelige onderwerpen en probeer het opnieuw." if found_terms else "")

def filter_chatbot_topics(message: str) -> tuple[bool, str]:
    found = check_blacklist(message)
    return (False, generate_warning(found)) if found else (True, '')

# Session init
def init_session():
    defaults = {'history': [], 'selected_product': None, 'selected_module': None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
init_session()
timezone = pytz.timezone('Europe/Amsterdam')

def add_message(role: str, content: str):
    ts = datetime.now(timezone).strftime('%d-%m-%Y %H:%M')
    st.session_state.history.append({'role': role, 'content': content, 'time': ts})
    st.session_state.history = st.session_state.history[-MAX_HISTORY:]

def render_chat():
    for msg in st.session_state.history:
        avatar = '🙂'
        if msg['role'] == 'assistant' and os.path.exists('aichatbox.jpg'):
            avatar = Image.open('aichatbox.jpg').resize((64,64))
        elif msg['role'] == 'user' and os.path.exists('parochie.jpg'):
            avatar = Image.open('parochie.jpg').resize((64,64))
        st.chat_message(msg['role'], avatar=avatar).markdown(f"{msg['content']}\n\n_{msg['time']}_")

def on_reset():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), retry=retry_if_exception_type(openai.RateLimitError))
def rewrite_answer(text: str) -> str:
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=[{'role':'system','content':'Herschrijf dit antwoord eenvoudig en vriendelijk.'},
                  {'role':'user','content':text}],
        temperature=0.2, max_tokens=800
    )
    return resp.choices[0].message.content.strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), retry_if_exception_type(openai.RateLimitError))
def get_ai_answer(text: str) -> str:
    messages = [{'role':'system','content':'Je bent de IPAL Chatbox, een behulpzame Nederlandse helpdeskassistent.'}]
    messages += [{'role':m['role'],'content':m['content']} for m in st.session_state.history[-10:]]
    messages.append({'role':'user','content':f"[{st.session_state.selected_module}] {text}"})
    resp = openai.chat.completions.create(model=MODEL, messages=messages, temperature=0.3, max_tokens=800)
    return resp.choices[0].message.content.strip()

def get_answer(text: str) -> str:
    mod_sel = st.session_state.get('selected_module')
    if mod_sel and not faq_df.empty:
        dfm = faq_df[faq_df['Subthema'].str.lower()==mod_sel.lower()]
        matches = dfm[dfm['combined'].str.contains(re.escape(text),case=False,na=False)]
        if not matches.empty:
            row = matches.iloc[0]
            ans = row['Antwoord']
            img = row.get('Afbeelding')
            try: ans = rewrite_answer(ans)
            except: pass
            if img and isinstance(img,str) and os.path.exists(img):
                st.image(img,caption='Voorbeeld',use_column_width=True)
            return ans
    try:
        return f"IPAL-Helpdesk antwoord:\n{get_ai_answer(text)}"
    except Exception as e:
        logging.error(f"AI-call mislukt: {e}")
        return "⚠️ Fout tijdens AI-fallback"

def genereer_pdf(tekst: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for lijn in tekst.split('\n'):
        pdf.multi_cell(0, 10, lijn)
    return pdf.output(dest='S').encode('latin-1', 'ignore')

def main():
    if st.sidebar.button('🔄 Nieuw gesprek'):
        on_reset()
        st.rerun()

    # Download last assistant message as PDF
    if st.session_state.history and st.session_state.history[-1]['role'] == 'assistant':
        laatste = st.session_state.history[-1]['content']
        st.download_button('📄 Download antwoord als PDF', data=genereer_pdf(laatste), file_name='antwoord.pdf', mime='application/pdf')

    if not st.session_state.selected_product:
        st.header('Welkom bij IPAL Chatbox')
        c1, c2, c3 = st.columns(3)
        if c1.button('DocBase', use_container_width=True):
            add_message('assistant', 'Gekozen: DocBase')
            st.session_state.selected_product = 'DocBase'
            st.rerun()
        if c2.button('Exact', use_container_width=True):
            add_message('assistant', 'Gekozen: Exact')
            st.session_state.selected_product = 'Exact'
            st.rerun()
        if c3.button('Algemeen', use_container_width=True):
            add_message('assistant', 'Gekozen: Algemeen')
            st.session_state.selected_product = 'Algemeen'
            st.session_state.selected_module = 'alles'
            st.rerun()
        render_chat()
        return

    if st.session_state.selected_product != 'Algemeen' and not st.session_state.selected_module:
        opts = subthema_dict.get(st.session_state.selected_product, [])
        sel = st.selectbox('Kies onderwerp:', ['(Kies)'] + opts)
        if sel != '(Kies)':
            st.session_state.selected_module = sel
            add_message('assistant', f"Gekozen: {sel}")
            st.rerun()
        render_chat()
        return

    render_chat()
    vraag = st.chat_input('Stel uw vraag:')
    if vraag:
        add_message('user', vraag)
        allowed, reason = filter_chatbot_topics(vraag)
        if not allowed:
            add_message('assistant', reason)
            st.rerun()
        with st.spinner('Even zoeken...'):
            if st.session_state.selected_product == 'Algemeen':
                matches = faq_df[faq_df['combined'].str.contains(re.escape(vraag), case=False, na=False)]
                if not matches.empty:
                    row = matches.iloc[0]
                    ans = row['Antwoord']
                    img = row.get('Afbeelding')
                    try:
                        ans = rewrite_answer(ans)
                    except:
                        pass
                    if img and isinstance(img, str) and os.path.exists(img):
                        st.image(img, caption='Voorbeeld', use_column_width=True)
                    add_message('assistant', ans)
                else:
                    try:
                        antwoord = get_ai_answer(vraag)
                        add_message('assistant', antwoord)
                    except Exception as e:
                        logging.error(f"AI-call mislukt: {e}")
                        add_message('assistant', '⚠️ Fout tijdens AI-fallback')
            else:
                antwoord = get_answer(vraag)
                add_message('assistant', antwoord)
        st.rerun()

if __name__ == '__main__':
    main()
