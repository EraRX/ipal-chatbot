import os
import re
import logging
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import pytz
import requests
from bs4 import BeautifulSoup
from PIL import Image as PILImage
from dotenv import load_dotenv
from openai import OpenAI

try:
    from openai.error import RateLimitError
except ImportError:
    RateLimitError = Exception

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

st.set_page_config(page_title='IPAL Chatbox', layout='centered')
st.markdown(
    '<style>html, body, [class*="css"] { font-size:20px; } button[kind="primary"] { font-size:22px !important; padding:.75em 1.5em; }</style>',
    unsafe_allow_html=True
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.sidebar.error("🔑 Voeg je OpenAI API-key toe in .env of Secrets.")
    st.stop()
client = OpenAI(api_key=OPENAI_KEY)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(1,10), retry=retry_if_exception_type(RateLimitError))
@st.cache_data
def chatgpt_cached(messages, temperature=0.3, max_tokens=300):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

if os.path.exists("Calibri.ttf"):
    pdfmetrics.registerFont(TTFont("Calibri", "Calibri.ttf"))
else:
    logging.info("Calibri.ttf niet gevonden, gebruik ingebouwde Helvetica")

def find_answer_by_codeword(df, codeword="[UNIEKECODE123]"):
    match = df[df['Antwoord of oplossing'].str.contains(codeword, case=False, na=False)]
    if not match.empty:
        return match.iloc[0]['Antwoord of oplossing']
    return None

AI_INFO = """
AI-Antwoord Info:  
1. Dit is het AI-antwoord vanuit de IPAL chatbox van het Interdiocesaan Platform Automatisering & Ledenadministratie. Het is altijd een goed idee om de meest recente informatie te controleren via officiële bronnen.  
2. Heeft u hulp nodig met DocBase of Exact? Dan kunt u eenvoudig een melding maken door een ticket aan te maken in DocBase. Maar voordat u een ticket invult, hebben we een handige tip: controleer eerst onze FAQ (het document met veelgestelde vragen en antwoorden). Dit document vindt u op onze site.
"""

# (je make_pdf functie etc. staat hier; niet herhaald ivm lengte)

@st.cache_data
def load_faq(path="faq.csv"):
    # laad faq.csv zoals in jouw code
    pass  # hier jouw originele functie

faq_df = load_faq()
producten = ['Exact', 'DocBase']
subthema_dict = {p: sorted(faq_df.index.get_level_values('Subthema').dropna().unique()) for p in producten}
BLACKLIST = ["persoonlijke gegevens", "medische gegevens", "gezondheid", "privacy schending"]

def filter_topics(msg: str):
    found = [t for t in BLACKLIST if re.search(rf"\b{re.escape(t)}\b", msg.lower())]
    return (False, f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}.") if found else (True, "")

# Preload images
aichatbox_img = PILImage.open("aichatbox.png").resize((256, 256)) if os.path.exists("aichatbox.png") else None
logo_img = PILImage.open("logo.png") if os.path.exists("logo.png") else None

AVATARS = {"assistant": "aichatbox.png", "user": "parochie.png"}
TIMEZONE = pytz.timezone("Europe/Amsterdam")
MAX_HISTORY = 10

def get_avatar(role: str):
    return aichatbox_img if role == "assistant" and aichatbox_img else "parochie.png"

def add_msg(role: str, content: str):
    ts = datetime.now(TIMEZONE).strftime('%d-%m-%Y %H:%M')
    st.session_state.history = (st.session_state.history + [{'role': role, 'content': content, 'time': ts}])[-MAX_HISTORY:]

def render_chat():
    for m in st.session_state.history:
        st.chat_message(m['role'], avatar=get_avatar(m['role'])).markdown(f"{m['content']}\n\n_{m['time']}_")

if 'history' not in st.session_state:
    st.session_state.history = []
    st.session_state.selected_product = None
    st.session_state.selected_module = None
    st.session_state.last_question = ''

def main():
    # Sidebar Nieuw gesprek button altijd tonen
    if st.sidebar.button('🔄 Nieuw gesprek'):
        st.session_state.clear()
        st.rerun()

    # Speel video af als helpdesk.mp4 bestaat en er nog geen product gekozen is
    video_file = "helpdesk.mp4"
    if not st.session_state.selected_product:
        if os.path.exists(video_file):
            video_html = f"""
            <video width="640" height="360" autoplay muted loop playsinline>
                <source src="{video_file}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            """
            st.markdown(video_html, unsafe_allow_html=True)
        elif logo_img:
            st.image(logo_img, width=244)
        st.header('Welkom bij IPAL Chatbox')

        # Toon de productkeuzeknoppen
        c1, c2, c3 = st.columns(3)
        if c1.button('Exact', use_container_width=True):
            st.session_state.selected_product = 'Exact'
            add_msg('assistant', 'Gekozen: Exact')
            st.rerun()
        if c2.button('DocBase', use_container_width=True):
            st.session_state.selected_product = 'DocBase'
            add_msg('assistant', 'Gekozen: DocBase')
            st.rerun()
        if c3.button('Algemeen', use_container_width=True):
            st.session_state.selected_product = 'Algemeen'
            st.session_state.selected_module = 'alles'
            add_msg('assistant', 'Gekozen: Algemeen')
            st.rerun()
        render_chat()
        return

    # Kies module als product Exact of DocBase
    if st.session_state.selected_product in ['Exact', 'DocBase'] and not st.session_state.selected_module:
        opts = subthema_dict.get(st.session_state.selected_product, [])
        sel = st.selectbox('Kies onderwerp:', ['(Kies)'] + opts)
        if sel != '(Kies)':
            st.session_state.selected_module = sel
            add_msg('assistant', f'Gekozen: {sel}')
            st.rerun()
        render_chat()
        return

    render_chat()

    vraag = st.chat_input('Stel uw vraag:')
    if not vraag:
        return

    # Controle op uniek codewoord
    if vraag.strip().upper() == "UNIEKECODE123":
        antwoord = find_answer_by_codeword(faq_df, codeword="[UNIEKECODE123]")
        if antwoord:
            add_msg('user', vraag)
            add_msg('assistant', antwoord + f"\n\n{AI_INFO}")
            st.rerun()

    # Exacte match op 'Omschrijving melding'
    vraag_normalized = vraag.strip().lower()
    faq_df["normalized"] = faq_df["Omschrijving melding"].str.strip().str.lower()
    exact_match = faq_df[faq_df["normalized"] == vraag_normalized]

    if not exact_match.empty:
        antwoord = exact_match.iloc[0]["Antwoord of oplossing"]
        add_msg('user', vraag)
        add_msg('assistant', antwoord + f"\n\n{AI_INFO}")
        st.rerun()

    st.session_state.last_question = vraag
    add_msg('user', vraag)

    ok, warn = filter_topics(vraag)
    if not ok:
        add_msg('assistant', warn)
        st.rerun()

    antwoord = vind_best_passend_antwoord(vraag, st.session_state.selected_product, st.session_state.selected_module)

    if antwoord:
        try:
            antwoord = chatgpt_cached([
                {'role': 'system', 'content': 'Herschrijf eenvoudig en vriendelijk.'},
                {'role': 'user', 'content': antwoord}
            ], temperature=0.2)
        except:
            pass
        add_msg('assistant', antwoord + f"\n\n{AI_INFO}")
        st.rerun()

    with st.spinner('de IPAL Helpdesk zoekt het juiste antwoord…'):
        try:
            web_info = fetch_web_info_cached(vraag)
            if web_info:
                ai = chatgpt_cached([
                    {'role': 'system', 'content': 'Je bent een behulpzame Nederlandse assistent. Gebruik de volgende informatie om de vraag te beantwoorden:\n' + web_info},
                    {'role': 'user', 'content': vraag}
                ])
            else:
                ai = chatgpt_cached([
                    {'role': 'system', 'content': 'Je bent een behulpzame Nederlandse assistent.'},
                    {'role': 'user', 'content': vraag}
                ])
            ai = re.sub(r'\*\*([^\*]+)\*\*', r'\1', ai)
            ai = re.sub(r'###\s*([^\n]+)', r'\1', ai)
            add_msg('assistant', ai + f"\n\n{AI_INFO}")
        except Exception as e:
            logging.exception('AI-fallback mislukt')
            add_msg('assistant', f'⚠️ AI-fallback mislukt: {e}')
        st.rerun()

    # PDF download button **onder** het laatste antwoord, in het chat-gedeelte
    if st.session_state.history and st.session_state.history[-1]['role'] == 'assistant':
        pdf_data = make_pdf(
            question=st.session_state.last_question,
            answer=st.session_state.history[-1]['content']
        )
        st.download_button('📄 Download PDF', data=pdf_data, file_name='antwoord.pdf', mime='application/pdf')

if __name__ == '__main__':
    main()
