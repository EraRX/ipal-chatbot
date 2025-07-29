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
import openai

# Safe import for RateLimitError
try:
    from openai.error import RateLimitError
except ImportError:
    RateLimitError = Exception

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ReportLab imports for PDF generation
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- Streamlit config & styling ---
st.set_page_config(page_title="IPAL Chatbox", layout="centered")
st.markdown("""
  <style>
    html, body, [class*="css"] { font-size:20px; }
    button[kind="primary"] { font-size:22px !important; padding:.75em 1.5em; }
  </style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- OpenAI setup ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai.api_key:
    st.sidebar.error("🔑 Voeg je OpenAI API-key toe in .env of Secrets.")
    st.stop()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(1, 10),
       retry=retry_if_exception_type(RateLimitError))
def chatgpt(messages, temperature=0.3, max_tokens=800):
    resp = openai.chat.completions.create(
        model=MODEL, messages=messages,
        temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# --- Register Calibri if available ---
if os.path.exists("Calibri.ttf"):
    pdfmetrics.registerFont(TTFont("Calibri", "Calibri.ttf"))
else:
    logging.info("Calibri.ttf niet gevonden, gebruik ingebouwde Helvetica")

# === PDF Generation (points 1 & 2 changed only) ===
def make_pdf(question: str, answer: str, ai_info: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    normal = ParagraphStyle(
        "normal", parent=styles["BodyText"],
        fontName="Calibri" if "Calibri" in pdfmetrics.getRegisteredFontNames() else "Helvetica",
        fontSize=11, leading=14, alignment=TA_JUSTIFY
    )
    h_bold = ParagraphStyle(
        "h_bold", parent=styles["Heading4"],
        fontName=normal.fontName, fontSize=11, leading=14, spaceAfter=6
    )

    # Static AI-info paragraphs
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
    faq_heading = "Waarom de FAQ gebruiken?"
    faq_text = (
        "In het document met veelgestelde vragen vindt u snel en eenvoudig antwoorden op "
        "veelvoorkomende vragen, zonder dat u hoeft te wachten op hulp.\n"
        "Klik hieronder om de FAQ te openen en te kijken of uw vraag al beantwoord is:\n"
        "– Veel gestelde vragen Docbase nieuw 2024\n"
        "– Veel gestelde vragen Exact Online"
    )
    instr_heading = "Instructie: Ticket aanmaken in DocBase"
    instr_text = (
        "Geen probleem! Zorg ervoor dat uw melding duidelijk is:\n"
        "• Beschrijf het probleem zo gedetailleerd mogelijk.\n"
        "• Voegt u geen document toe, zet dan het documentformaat in het ticket op “geen bijlage”.\n"
        "• Geef uw telefoonnummer op waarop wij u kunnen bereiken, zodat de helpdesk contact met u kan opnemen."
    )

    story = []

    # Logo top-left
    if os.path.exists("logo.png"):
        story.append(Image("logo.png", width=124, height=52))
        story.append(Spacer(1, 12))

    # Vraag
    story.append(Paragraph("<b>Vraag:</b>", h_bold))
    story.append(Paragraph(question, normal))
    story.append(Spacer(1, 12))

    # Antwoord as numbered bullet list, left-aligned
    story.append(Paragraph("<b>Antwoord:</b>", h_bold))
    story.append(Spacer(1, 4))

    items = []
    for num, name, desc in re.findall(
        r"(\d+)\.\s\*\*(.*?)\*\*\s*-\s*(.*?)(?=\n\d+\.|\Z)", answer, re.S
    ):
        text = f"<b>{num}. {name}</b> - {desc.strip().replace(chr(10), ' ')}"
        items.append(ListItem(Paragraph(text, normal), leftIndent=0, bulletIndent=0))
    if items:
        story.append(ListFlowable(
            items,
            bulletType="bullet",
            start=None,
            leftIndent=0,
            bulletIndent=0
        ))
    else:
        story.append(Paragraph(answer, normal))
    story.append(Spacer(1, 12))

    # AI-Antwoord Info
    story.append(Paragraph("<b>AI-Antwoord Info:</b>", h_bold))
    story.append(Paragraph(para1, normal))
    story.append(Spacer(1, 6))
    story.append(Paragraph(para2, normal))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>{faq_heading}</b>", normal))
    for line in faq_text.split("\n"):
        story.append(Paragraph(line, normal))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>{instr_heading}</b>", normal))
    for line in instr_text.split("\n"):
        story.append(Paragraph(line, normal))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
# === end make_pdf ===

# --- FAQ loader ---
@st.cache_data
def load_faq(path="faq.xlsx"):
    if not os.path.exists(path):
        st.error(f"⚠️ FAQ '{path}' niet gevonden")
        return pd.DataFrame(columns=["combined","Antwoord","Afbeelding","Systeem","Subthema"])
    df = pd.read_excel(path, engine="openpyxl")
    if "Afbeelding" not in df.columns:
        df["Afbeelding"] = None
    df["Antwoord"] = df["Antwoord of oplossing"]
    df["combined"] = df[["Systeem","Subthema","Omschrijving melding","Toelichting melding"]].fillna("").agg(" ".join, axis=1)
    return df
faq_df = load_faq()

# --- Build subthema_dict ---
producten = ["Exact","DocBase"]
subthema_dict = {
    p: sorted(faq_df[faq_df["Systeem"] == p]["Subthema"].dropna().unique())
    for p in producten
}

# --- Blacklist & topic filter ---
BLACKLIST = ["persoonlijke gegevens","medische gegevens","gezondheid","privacy schending"]
def filter_topics(msg: str):
    found = [t for t in BLACKLIST if re.search(rf"\b{re.escape(t)}\b", msg.lower())]
    return (False, f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}.") if found else (True, "")

# --- RKK scraping ---
def fetch_bishop_from_rkkerk(loc: str):
    slug = loc.lower().replace(" ", "-")
    url = f"https://www.rkkerk.nl/bisdom-{slug}/"
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        h1 = soup.find("h1")
        if h1 and "bisschop" in h1.text.lower():
            return h1.text.split("—")[0].strip()
    except:
        pass
    return None

def fetch_bishop_from_rkk_online(loc: str):
    query = loc.replace(" ", "+")
    url = f"https://www.rkk-online.nl/?s={query}"
    try:
        r = requests.get(url, timeout=10); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in ("h1","h2","h3"):
            h = soup.find(tag, string=re.compile(r"bisschop", re.I))
            if h:
                return h.text.split("–")[0].strip()
    except:
        pass
    return None

def fetch_all_bishops_nl():
    dioceses = ["Utrecht","Haarlem-Amsterdam","Rotterdam","Groningen-Leeuwarden",
                "’s-Hertogenbosch","Roermond","Breda"]
    result = {}
    for d in dioceses:
        name = fetch_bishop_from_rkkerk(d) or fetch_bishop_from_rkk_online(d)
        if name:
            result[d] = name
    return result

# --- Avatars & helpers ---
AVATARS = {"assistant":"aichatbox.jpg","user":"parochie.jpg"}
def get_avatar(role: str):
    path = AVATARS.get(role)
    return PILImage.open(path).resize((64,64)) if path and os.path.exists(path) else "🙂"

TIMEZONE = pytz.timezone("Europe/Amsterdam")
MAX_HISTORY = 20
def add_msg(role: str, content: str):
    ts = datetime.now(TIMEZONE).strftime("%d-%m-%Y %H:%M")
    st.session_state.history = (st.session_state.history + [{"role":role,"content":content,"time":ts}])[-MAX_HISTORY:]

def render_chat():
    for m in st.session_state.history:
        st.chat_message(m["role"], avatar=get_avatar(m["role"])).markdown(
            f"{m['content']}\n\n_{m['time']}_"
        )

# --- Session init ---
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.selected_product = None
    st.session_state.selected_module = None
    st.session_state.last_question = ""

# --- Main app ---
def main():
    if st.sidebar.button("🔄 Nieuw gesprek"):
        st.session_state.clear(); st.rerun()

    # PDF download
    if st.session_state.history and st.session_state.history[-1]["role"] == "assistant":
        pdf = make_pdf(
            question=st.session_state.last_question,
            answer=st.session_state.history[-1]["content"],
            ai_info=st.session_state.history[-1]["content"]
        )
        st.sidebar.download_button("📄 Download PDF", data=pdf,
                                   file_name="antwoord.pdf", mime="application/pdf")

    # 1) kies product
    if not st.session_state.selected_product:
        st.header("Welkom bij IPAL Chatbox")
        c1,c2,c3 = st.columns(3)
        if c1.button("Exact",use_container_width=True):
            st.session_state.selected_product="Exact"; add_msg("assistant","Gekozen: Exact"); st.rerun()
        if c2.button("DocBase",use_container_width=True):
            st.session_state.selected_product="DocBase"; add_msg("assistant","Gekozen: DocBase"); st.rerun()
        if c3.button("Algemeen",use_container_width=True):
            st.session_state.selected_product="Algemeen"; st.session_state.selected_module="alles"; add_msg("assistant","Gekozen: Algemeen"); st.rerun()
        render_chat(); return

    # 2) kies module (Exact/DocBase)
    if st.session_state.selected_product in ["Exact","DocBase"] and not st.session_state.selected_module:
        opts = subthema_dict.get(st.session_state.selected_product, [])
        sel = st.selectbox("Kies onderwerp:", ["(Kies)"] + opts)
        if sel != "(Kies)":
            st.session_state.selected_module = sel
            add_msg("assistant", f"Gekozen: {sel}")
            st.rerun()
        render_chat(); return

    # 3) chatflow
    render_chat()
    vraag = st.chat_input("Stel uw vraag:")
    if not vraag: return

    st.session_state.last_question = vraag
    add_msg("user", vraag)

    ok,warn = filter_topics(vraag)
    if not ok:
        add_msg("assistant", warn); st.rerun()

    # specific bishop?
    m = re.match(r'(?i)wie is bisschop(?: van)?\s+(.+)', vraag)
    if m:
        loc = m.group(1).strip()
        bishop = fetch_bishop_from_rkkerk(loc) or fetch_bishop_from_rkk_online(loc)
        if bishop:
            add_msg("assistant", f"De huidige bisschop van {loc} is {bishop}."); st.rerun()

    # all NL bishops?
    if re.search(r'(?i)bisschoppen nederland', vraag):
        allb = fetch_all_bishops_nl()
        if allb:
            lines = [f"Mgr. {n} – Bisschop van {d}" for d,n in allb.items()]
            add_msg("assistant", "Huidige Nederlandse bisschoppen:\n" + "\n".join(lines)); st.rerun()

    # FAQ lookup
    dfm = faq_df[faq_df["combined"].str.contains(re.escape(vraag), case=False, na=False)]
    if not dfm.empty:
        row = dfm.iloc[0]
        ans = row["Antwoord"]
        try:
            ans = chatgpt([
                {"role":"system","content":"Herschrijf eenvoudig en vriendelijk."},
                {"role":"user","content":ans}
            ], temperature=0.2)
        except:
            pass
        if img := row["Afbeelding"]:
            st.image(img, caption="Voorbeeld", use_column_width=True)
        add_msg("assistant", ans); st.rerun()

    # AI fallback
    with st.spinner("ChatGPT even aan het werk…"):
        try:
            ai = chatgpt([
                {"role":"system","content":"Je bent een behulpzame Nederlandse assistent."},
                {"role":"user","content":vraag}
            ])
            add_msg("assistant", ai)
        except Exception as e:
            logging.exception("AI-fallback mislukt")
            add_msg("assistant", f"⚠️ AI-fallback mislukt: {e}")
    st.rerun()

if __name__ == "__main__":
    main()
