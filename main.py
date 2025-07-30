# main.py


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
def chatgpt(messages, temperature=0.3, max_tokens=800):
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

def make_pdf(question: str, answer: str, ai_info: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=16, spaceAfter=12, alignment=TA_LEFT)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#333333"), spaceBefore=12, spaceAfter=6)
    bullet_style = ParagraphStyle("Bullet", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leftIndent=12, bulletIndent=0, leading=16)

    story = []

    if os.path.exists("logo.png"):
        story.append(Image("logo.png", width=124, height=52))
        story.append(Spacer(1, 12))

    avatar_path = "aichatbox.jpg"
    if os.path.exists(avatar_path):
        avatar = Image(avatar_path, width=30, height=30)
        intro_text = Paragraph(answer.split("\n")[0], body_style)
        story.append(Table([[avatar, intro_text]], colWidths=[30, 440], style=TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')])))
        story.append(Spacer(1, 12))

    story.append(Paragraph("Antwoord:", heading_style))
    for line in answer.split("\n")[1:]:
        line = line.strip()
        if line.startswith("•") or line.startswith("-"):
            bullets = ListFlowable([ListItem(Paragraph(line[1:].strip(), bullet_style))], bulletType="bullet")
            story.append(bullets)
        elif line:
            story.append(Paragraph(line, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

