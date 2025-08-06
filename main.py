"""
IPAL Chatbox voor oudere vrijwilligers
- Python 3, Streamlit
- Groot lettertype, eenvoudige bediening
- Antwoorden uit FAQ, Exact Online, DocBase, rkkerk.nl, rkk-online.nl, en AI
- Topicfiltering (blacklist + herstelde fallback op geselecteerde module)
- Logging en foutafhandeling
- Antwoorden downloaden als PDF
"""

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
def chatgpt_cached(messages, temperature=0.3, max_tokens=1300):
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

# AI-Antwoord Info
AI_INFO = """
AI-Antwoord Info:  
1. Dit is het AI-antwoord vanuit de IPAL chatbox van het Interdiocesaan Platform Automatisering & Ledenadministratie. Het is altijd een goed idee om de meest recente informatie te controleren via officiële bronnen.  
2. Heeft u hulp nodig met DocBase of Exact? Dan kunt u eenvoudig een melding maken door een ticket aan te maken in DocBase. Maar voordat u een ticket invult, hebben we een handige tip: controleer eerst onze FAQ (het document met veelgestelde vragen en antwoorden). Dit document vindt u op onze site.
"""

# PDF generation with chat-style layout and logo top-left
def make_pdf(question: str, answer: str) -> bytes:
    answer = re.sub(r'\*\*([^\*]+)\*\*', r'\1', answer)  # Remove bold
    answer = re.sub(r'###\s*([^\n]+)', r'\1', answer)  # Remove headings
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=16, spaceAfter=12, alignment=TA_LEFT)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#333333"), spaceBefore=12, spaceAfter=6)
    bullet_style = ParagraphStyle("Bullet", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leftIndent=12, bulletIndent=0, leading=16)

    story = []
    if os.path.exists("logopdf.png"):
        logo = Image("logopdf.png", width=124, height=52)
        logo_table = Table([[logo]], colWidths=[124])
        logo_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
        ]))
        story.append(logo_table)

    story.append(Paragraph(f"Vraag: {question}", heading_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Antwoord:", heading_style))
    avatar_path = "aichatbox.png"
    if os.path.exists(avatar_path):
        avatar = Image(avatar_path, width=30, height=30)
        intro_text = Paragraph(answer.split("\n")[0], body_style)
        story.append(Table([[avatar, intro_text]], colWidths=[30, 440], style=TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')])))
        story.append(Spacer(1, 12))
        for line in answer.split("\n")[1:]:
            line = line.strip()
            if line.startswith("•") or line.startswith("-"):
                bullets = ListFlowable([ListItem(Paragraph(line[1:].strip(), bullet_style))], bulletType="bullet")
                story.append(bullets)
            elif line:
                story.append(Paragraph(line, body_style))
    else:
        for line in answer.split("\n"):
            line = line.strip()
            if line.startswith("•") or line.startswith("-"):
                bullets = ListFlowable([ListItem(Paragraph(line[1].strip(), bullet_style))], bulletType="bullet")
                story.append(bullets)
            elif line:
                story.append(Paragraph(line, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

@st.cache_data
def load_faq(path="faq.csv"):
    if not os.path.exists(path):
        logging.error(f"FAQ niet gevonden: {path}")
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=['Systeem','Subthema','Omschrijving melding','Toelichting melding','Antwoord of oplossing','Afbeelding'])
    try:
        df = pd.read_csv(path, encoding="utf-8", sep=";")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="windows-1252", sep=";")
    if 'Afbeelding' not in df.columns:
        df['Afbeelding'] = None
    df['Antwoord'] = df['Antwoord of oplossing']
    df['combined'] = df[['Systeem','Subthema','Omschrijving melding','Toelichting melding']].fillna('').agg(' '.join, axis=1)
    return df.set_index(['Systeem', 'Subthema'])  # Toegevoegd voor snellere lookups

faq_df = load_faq()
producten = ['Exact', 'DocBase']
subthema_dict = {p: sorted(faq_df.index.get_level_values('Subthema').dropna().unique()) for p in producten}
BLACKLIST = ["persoonlijke gegevens", "medische gegevens", "gezondheid", "privacy schending"]

def filter_topics(msg: str):
    found = [t for t in BLACKLIST if re.search(rf"\b{re.escape(t)}\b", msg.lower())]
    return (False, f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}.") if found else (True, "")

@st.cache_data
def fetch_web_info_cached(query: str):
    result = []
    dfm = faq_df[faq_df['combined'].str.contains(re.escape(query), case=False, na=False)]
    if not dfm.empty:
        row = dfm.iloc[0]
        result.append(f"Vanuit FAQ ({row['Systeem']} - {row['Subthema']}): {row['Antwoord']}")
    try:
        r = requests.get("https://docbase.nl", timeout=5)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = ' '.join([p.get_text(strip=True) for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])
        if text and query.lower() in text.lower():
            result.append(f"Vanuit docbase.nl: {text[:200]}... (verkort)")
    except Exception as e:
        logging.info(f"Kon docbase.nl niet ophalen: {e}")
    try:
        r = requests.get("https://support.exactonline.com/community/s/knowledge-base", timeout=5)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = ' '.join([p.get_text(strip=True) for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])
        if text and query.lower() in text.lower():
            result.append(f"Vanuit Exact Online Knowledge Base: {text[:200]}... (verkort)")
    except Exception as e:
        logging.info(f"Kon Exact Online Knowledge Base niet ophalen: {e}")
    return '\n'.join(result) if result else None

def vind_best_passend_antwoord(vraag, systeem, subthema):
    try:
        resultaten = faq_df.loc[(systeem.lower(), subthema.lower())]
        if not resultaten.empty:
            vraag_lower = vraag.lower()
            def score(tekst):
                return sum(1 for woord in vraag_lower.split() if woord in str(tekst).lower())
            resultaten = resultaten.assign(score=resultaten['combined'].apply(score)).sort_values('score', ascending=False)
            beste = resultaten.iloc[0]
            return beste['Antwoord of oplossing'] if beste['score'] > 0 else None
    except KeyError:
        return None

# Preload afbeeldingen
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
    for i, m in enumerate(st.session_state.history):
        st.chat_message(m['role'], avatar=get_avatar(m['role'])).markdown(f"{m['content']}\n\n_{m['time']}_")
        # Toon PDF-downloadknop direct na laatste assistant-bericht
        if m['role'] == 'assistant' and i == len(st.session_state.history) - 1:
            pdf_data = make_pdf(
                question=st.session_state.last_question,
                answer=m['content']
            )
            st.download_button('📄 Download PDF', data=pdf_data, file_name='antwoord.pdf', mime='application/pdf')

if 'history' not in st.session_state:
    st.session_state.history = []
    st.session_state.selected_product = None
    st.session_state.selected_module = None
    st.session_state.last_question = ''

def main():
    if st.sidebar.button('🔄 Nieuw gesprek'):
        st.session_state.clear()
        st.rerun()

    # Verwijder downloadknop sidebar (niet meer hier)
    # if st.session_state.history and st.session_state.history[-1]['role'] == 'assistant':
    #     pdf_data = make_pdf(
    #         question=st.session_state.last_question,
    #         answer=st.session_state.history[-1]['content']
    #     )
    #     st.sidebar.download_button('📄 Download PDF', data=pdf_data, file_name='antwoord.pdf', mime='application/pdf')

    if not st.session_state.selected_product:
        if logo_img:
            st.image(logo_img, width=244)
        st.header('Welkom bij de IPAL Chatbox')
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

    # Geen exacte match → reguliere verwerking
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

if __name__ == '__main__':
    main()





