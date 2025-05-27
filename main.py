# main.py
"""
IPAL Chatbox voor oudere vrijwilligers
- Python 3, Streamlit
- Groot lettertype, eenvoudige bediening
- Antwoorden uit FAQ aangevuld met AI voor whitelisted modules
- Topicfiltering (whitelist/blacklist)
- Logging en foutafhandeling

Geschatte lengte: ~230 regels
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# -------------------- Topic Filtering --------------------
WHITELIST_TOPICS = {
    "ledenadministratie": [
        "parochiaan", "lidmaatschap", "inschrijving", "uitschrijving", "doopregister",
        "sila", "parochie", "postcode", "adreswijziging", "verhuizing", "mutatie",
        "kerkledenadministratie", "doopdatum", "ledenbeheer", "registratie"
    ],
    "exact_online_boekhouding_financien": [
        "boekhouding", "financiën", "kerkbijdrage", "factuur", "betaling", "budget",
        "financieel beheer", "exact online", "administratie", "rekening", "kosten",
        "inkomsten", "uitgaven", "begraafplaatsadministratie", "donatie"
    ],
    "rooms_katholieke_kerk_administratief": [
        "parochiebeheer", "bisdom", "kerkprovincie", "kerkelijke administratie",
        "parochieblad", "vrijwilligers", "functionaris gegevensbescherming",
        "verwerkersovereenkomst", "avg-compliance", "statuten"
    ]
}
BLACKLIST_CATEGORIES = [
    "politiek", "religie", "persoonlijke gegevens", "gezondheid", "gokken",
    "adult content", "geweld", "haatzaaiende taal", "desinformatie",
    "geloofsovertuiging", "medische gegevens", "strafrechtelijk verleden",
    "seksuele inhoud", "complottheorie", "nepnieuws", "discriminatie",
    "racisme", "extremisme", "godsdienstige leer", "persoonlijke overtuiging"
]

def filter_chatbot_topics(message: str) -> (bool, str):
    """
    Controleer op blacklist en whitelist.
    """
    text = message.lower()
    # Blacklist check
    for blocked in BLACKLIST_CATEGORIES:
        if re.search(rf"\b{re.escape(blocked)}\b", text):
            return False, f"Geblokkeerd: bevat verboden onderwerp '{blocked}'"
    # Whitelist check in module context
    mod = (st.session_state.get('selected_module') or '').lower().replace(' ', '_')
    if mod in WHITELIST_TOPICS:
        for kw in WHITELIST_TOPICS[mod]:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return True, ''
        return False, '⚠️ Geen geldig onderwerp voor AI-ondersteuning'
    return False, '⚠️ AI-fallback niet toegestaan voor dit onderwerp'

# -------------------- Configuratie --------------------
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

st.set_page_config(page_title='IPAL Chatbox', layout='centered')
# Styling
st.markdown('''
<style>
  html, body, [class*="css"] { font-size: 20px; }
  button[kind="primary"] { font-size: 22px !important; padding: 0.75em 1.5em; }
</style>
''', unsafe_allow_html=True)

# -------------------- Validatie --------------------
def validate_api_key():
    if not openai.api_key:
        logging.error("Geen API-sleutel gevonden")
        st.error('⚠️ Stel uw OPENAI_API_KEY in via .env-bestand')
        st.stop()
    try:
        openai.models.list()
    except openai.AuthenticationError:
        logging.error("Ongeldige API-sleutel")
        st.error('⚠️ Ongeldige API-sleutel')
        st.stop()
    except Exception as e:
        logging.error(f"API-validatie fout: {e}")
        st.error('⚠️ Fout bij API-validatie')
        st.stop()
validate_api_key()

# -------------------- FAQ Laden --------------------
@st.cache_data
def load_faq(path: str = 'faq.xlsx') -> pd.DataFrame:
    if not os.path.exists(path):
        logging.error(f"FAQ niet gevonden: {path}")
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=['combined', 'Antwoord'])
    try:
        df = pd.read_excel(path)
    except Exception as e:
        logging.error(f"Fout bij lezen FAQ: {e}")
        st.error('⚠️ Kan FAQ niet laden')
        return pd.DataFrame(columns=['combined', 'Antwoord'])
    required = ['Systeem', 'Subthema', 'Omschrijving melding', 'Toelichting melding', 'Antwoord of oplossing']
    missing = [c for c in required if c not in df.columns]
    if missing:
        logging.error(f"FAQ mist kolommen: {missing}")
        st.error(f"FAQ mist kolommen: {missing}")
        return pd.DataFrame(columns=['combined', 'Antwoord'])
    if 'Afbeelding' not in df.columns:
        df['Afbeelding'] = None
    df['Antwoord'] = df['Antwoord of oplossing']
    df['combined'] = df[required].fillna('').agg(' '.join, axis=1)
    return df
faq_df = load_faq()
producten = ['Exact', 'DocBase']
subthema_dict = {
    p: sorted(faq_df[faq_df['Systeem'] == p]['Subthema'].dropna().unique().tolist())
    for p in producten
}

# -------------------- Sessiestatus --------------------
def init_session():
    defaults = {
        'history': [],
        'selected_product': None,
        'selected_module': None,
        'reset_triggered': False
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
init_session()
timezone = pytz.timezone('Europe/Amsterdam')

# -------------------- Chat Helpers --------------------
def add_message(role: str, content: str):
    ts = datetime.now(timezone).strftime('%d-%m-%Y %H:%M')
    st.session_state.history.append({'role': role, 'content': content, 'time': ts})
    if len(st.session_state.history) > 100:
        st.session_state.history = st.session_state.history[-100:]


def render_chat():
    for msg in st.session_state.history:
        avatar = Image.open('aichatbox.jpg').resize((64,64)) if msg['role'] == 'assistant' and os.path.exists('aichatbox.jpg') else '🙂'
        st.chat_message(msg['role'], avatar=avatar).markdown(
            f"{msg['content']}\n\n_{msg['time']}_"
        )


def on_reset():
    init_session()

# -------------------- AI Interaction --------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), retry=retry_if_exception_type(openai.RateLimitError))
def rewrite_answer(text: str) -> str:
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': 'Herschrijf dit antwoord eenvoudig en vriendelijk.'},
            {'role': 'user', 'content': text}
        ],
        temperature=0.2, max_tokens=300
    )
    return resp.choices[0].message.content.strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max
