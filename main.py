# main.py

import os
import re
import logging
import io
import textwrap
from datetime import datetime

import streamlit as st
import pandas as pd
import pytz
import requests
from bs4 import BeautifulSoup
from PIL import Image as PILImage
from dotenv import load_dotenv
import openai

# Fallback import voor RateLimitError
try:
    from openai.error import RateLimitError
except ImportError:
    RateLimitError = Exception

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ReportLab imports voor Platypus PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# — Streamlit config & styling —
st.set_page_config(page_title="IPAL Chatbox", layout="centered")
st.markdown("""
  <style>
    html, body, [class*="css"] { font-size:20px; }
    button[kind="primary"] { font-size:22px !important; padding:.75em 1.5em; }
  </style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# — OpenAI setup —
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai.api_key:
    st.sidebar.error("🔑 Voeg je OpenAI API-key toe in .env of Secrets.")
    st.stop()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(1,10),
       retry=retry_if_exception_type(RateLimitError))
def chatgpt(messages, temperature=0.3, max_tokens=800):
    resp = openai.chat.completions.create(
        model=MODEL, messages=messages,
        temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# — Register Calibri als beschikbaar —
if os.path.exists("Calibri.ttf"):
    pdfmetrics.registerFont(TTFont("Calibri", "Calibri.ttf"))
else:
    logging.info("Calibri.ttf niet gevonden, gebruik ingebouwde Helvetica")

# — PDF-functie met exacte opmaak —
def make_pdf(question: str, answer: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    font = "Calibri" if "Calibri" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    normal.fontName = font
    normal.fontSize = 11
    normal.alignment = TA_JUSTIFY

    h_bold = styles["Heading4"]
    h_bold.fontName = font
    h_bold.fontSize = 11
    h_bold.leading = 14

    # Vaste punten
    para1 = (
        "1. Dit is het AI-antwoord vanuit de IPAL chatbox van het Interdiocesaan Platform "
        "Automatisering & Ledenadministratie. Het is altijd een goed idee om de meest recente "
        "informatie te controleren via officiële bronnen."
    )
    para2 = (
        "2. Heeft u hulp nodig met DocBase of Exact? Dan kunt u eenvoudig een melding maken door "
        "een ticket aan te maken in DocBase. Maar voordat u een ticket invult, hebben we een "
        "handige tip: controleer eerst onze FAQ (het document met veelgestelde vragen en antwoorden). "
        "Dit document vindt u op onze site."
    )
    faq_tip = (
        "<b>Waarom de FAQ gebruiken?</b><br/>"
        "In het document met veelgestelde vragen vindt u snel en eenvoudig antwoorden op "
        "veelvoorkomende vragen, zonder dat u hoeft te wachten op hulp.<br/><br/>"
        "Klik hieronder om de FAQ te openen en te kijken of uw vraag al beantwoord is:<br/>"
        "– Veel gestelde vragen Docbase nieuw 2024<br/>"
        "– Veel gestelde vragen Exact Online"
    )
    instructie = (
        "<b>Instructie: Ticket aanmaken in DocBase</b><br/>"
        "Geen probleem! Zorg ervoor dat uw melding duidelijk is:<br/>"
        "• Beschrijf het probleem zo gedetailleerd mogelijk.<br/>"
        "• Voegt u geen document toe, zet dan het documentformaat in het ticket op “geen bijlage”.<br/>"
        "• Geef uw telefoonnummer op waarop wij u kunnen bereiken, zodat de helpdesk contact met u kan opnemen."
    )

    story = []
    # Logo linksboven
    if os.path.exists("logo.png"):
        story.append(Image("logo.png", width=50, height=50))
        story.append(Spacer(1, 12))

    # Vraag (vet) + inhoud
    story.append(Paragraph("<b>Vraag:</b>", h_bold))
    story.append(Spacer(1, 4))
    story.append(Paragraph(question, normal))
    story.append(Spacer(1, 12))

    # Antwoord (vet) + inhoud
    story.append(Paragraph("<b>Antwoord:</b>", h_bold))
    story.append(Spacer(1, 4))
    story.append(Paragraph(answer, normal))
    story.append(Spacer(1, 12))

    # AI-Antwoord Info (vet) + vaste tekst
    story.append(Paragraph("<b>AI-Antwoord Info:</b>", h_bold))
    story.append(Spacer(1, 4))
    story.append(Paragraph(para1, normal))
    story.append(Spacer(1, 6))
    story.append(Paragraph(para2, normal))
    story.append(Spacer(1, 6))
    story.append(Paragraph(faq_tip, normal))
    story.append(Spacer(1, 6))
    story.append(Paragraph(instructie, normal))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# — FAQ loader —  
@st.cache_data
def load_faq(path="faq.xlsx"):
    if not os.path.exists(path):
        st.error(f"⚠️ FAQ '{path}' niet gevonden")
        return pd.DataFrame(columns=["combined","Antwoord","Afbeelding"])
    df = pd.read_excel(path, engine="openpyxl")
    if "Afbeelding" not in df.columns:
        df["Afbeelding"] = None
    keys = ["Systeem","Subthema","Omschrijving melding","Toelichting melding"]
    df["combined"] = df[keys].fillna("").agg(" ".join, axis=1)
    df["Antwoord"] = df["Antwoord of oplossing"]
    return df[["combined","Antwoord","Afbeelding"]]

faq_df = load_faq()

# — Blacklist —  
BLACKLIST = ["persoonlijke gegevens","medische gegevens","gezondheid","privacy schending"]
def filter_topics(msg: str):
    found = [t for t in BLACKLIST if re.search(rf"\b{re.escape(t)}\b", msg.lower())]
    return (False, f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}.") if found else (True, "")

# — RKK scraping (zoals eerder) —  
def fetch_bishop_from_rkkerk(loc: str): ...
def fetch_bishop_from_rkk_online(loc: str): ...
def fetch_all_bishops_nl(): ...

# — Avatars & helpers —  
AVATARS = {"assistant":"aichatbox.jpg","user":"parochie.jpg"}
def get_avatar(role: str): ...
def add_msg(role: str, content: str): ...
def render_chat(): ...

# — Session init —  
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.selected_product = None
    st.session_state.last_question = ""

# — Main app —  
def main():
    if st.sidebar.button("🔄 Nieuw gesprek"):
        st.session_state.clear()
        st.rerun()

    # PDF-downloadknop
    if st.session_state.history and st.session_state.history[-1]["role"]=="assistant":
        pdf_data = make_pdf(
            question=st.session_state.last_question,
            answer=st.session_state.history[-1]["content"]
        )
        st.sidebar.download_button("📄 Download PDF", data=pdf_data,
            file_name="antwoord.pdf", mime="application/pdf")

    # Chatflow (productselectie, vraag input, FAQ lookup, scraping, fallback…)
    # Onthoud na elke input:
    #   st.session_state.last_question = vraag

if __name__=="__main__":
    main()
