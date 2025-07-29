# main.py

import os
import re
import logging
import io
from datetime import datetime
import textwrap

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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
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
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# --- Register Calibri if available ---
if os.path.exists("Calibri.ttf"):
    pdfmetrics.registerFont(TTFont("Calibri", "Calibri.ttf"))
else:
    logging.info("Calibri.ttf niet gevonden, gebruik ingebouwde Helvetica")

# === 1) PDF Generation: aangepast voor logo + vetkopjes + nummering ===
def make_pdf(question: str, answer: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    font_name = "Calibri" if "Calibri" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    normal.fontName = font_name
    normal.fontSize = 11
    normal.alignment = TA_JUSTIFY

    h_bold = styles["Heading4"]
    h_bold.fontName = font_name
    h_bold.fontSize = 11
    h_bold.leading = 14

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
    # Logo
    if os.path.exists("logo.png"):
        story.append(Image("logo.png", width=124, height=52))
        story.append(Spacer(1, 12))

    # Vraag
    story.append(Paragraph("<b>Vraag:</b>", h_bold))
    story.append(Spacer(1, 4))
    story.append(Paragraph(question, normal))
    story.append(Spacer(1, 12))

    # Antwoord
    story.append(Paragraph("<b>Antwoord:</b>", h_bold))
    story.append(Spacer(1, 4))
    story.append(Paragraph(answer, normal))
    story.append(Spacer(1, 12))

    # AI-Antwoord Info
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
# === einde make_pdf ===

# --- FAQ loader ---
@st.cache_data
def load_faq(path="faq.xlsx"):
    if not os.path.exists(path):
        logging.error(f"FAQ niet gevonden: {path}")
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=["combined", "Antwoord"])
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        logging.error(f"Fout bij laden FAQ: {e}")
        st.error("⚠️ Kan FAQ niet laden")
        return pd.DataFrame(columns=["combined", "Antwoord"])
    if "Afbeelding" not in df.columns:
        df["Afbeelding"] = None
    required = ["Systeem", "Subthema", "Omschrijving melding", "Toelichting melding"]
    df["Antwoord"] = df["Antwoord of oplossing"]
    df["combined"] = df[required].fillna("").agg(" ".join, axis=1)
    return df[["combined", "Antwoord", "Afbeelding"]]

faq_df = load_faq()
producten = ["Exact", "DocBase"]
subthema_dict = {
    p: sorted(faq_df[faq_df["Systeem"] == p]["Subthema"].dropna().unique())
    for p in producten
}

# --- Blacklist & topic filter ---
BLACKLIST = [
    "persoonlijke gegevens", "medische gegevens", "gezondheid", "strafrechtelijk verleden",
    # ... etc ...
    "privacy schending"
]

def check_blacklist(text: str) -> list[str]:
    return [term for term in BLACKLIST if term in text.lower()]

def generate_warning(found_terms: list[str]) -> str:
    return ("Je bericht bevat inhoud die niet voldoet aan onze richtlijnen. "
            "Vermijd gevoelige onderwerpen en probeer het opnieuw.") if found_terms else ""

def filter_chatbot_topics(message: str) -> tuple[bool, str]:
    found = check_blacklist(message)
    return (False, generate_warning(found)) if found else (True, "")

# --- Session state init ---
def init_session():
    defaults = {
        "history": [], "selected_product": None,
        "selected_module": None, "last_question": ""
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_session()
timezone = pytz.timezone("Europe/Amsterdam")
MAX_HISTORY = 20

# --- Chat helpers ---
def add_msg(role: str, content: str):
    ts = datetime.now(timezone).strftime("%d-%m-%Y %H:%M")
    st.session_state.history.append({"role": role, "content": content, "time": ts})
    st.session_state.history = st.session_state.history[-MAX_HISTORY:]

def render_chat():
    for msg in st.session_state.history:
        avatar = "🙂"
        if msg["role"] == "assistant" and os.path.exists("aichatbox.jpg"):
            avatar = PILImage.open("aichatbox.jpg").resize((64, 64))
        elif msg["role"] == "user" and os.path.exists("parochie.jpg"):
            avatar = PILImage.open("parochie.jpg").resize((64, 64))
        st.chat_message(msg["role"], avatar=avatar).markdown(f"{msg['content']}\n\n_{msg['time']}_")

def on_reset():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# --- Rewrite & AI-answer wrappers ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(1,10),
       retry=retry_if_exception_type(RateLimitError))
def rewrite_answer(text: str) -> str:
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role":"system","content":"Herschrijf dit antwoord eenvoudig en vriendelijk."},
            {"role":"user","content":text}
        ],
        temperature=0.2, max_tokens=800
    )
    return resp.choices[0].message.content.strip()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(1,10),
       retry=retry_if_exception_type(RateLimitError))
def get_ai_answer(text: str) -> str:
    messages = [{"role":"system","content":"Je bent de IPAL Chatbox, een behulpzame Nederlandse helpdeskassistent."}]
    messages += [{"role":m["role"],"content":m["content"]} for m in st.session_state.history[-10:]]
    messages.append({"role":"user","content":f"[{st.session_state.selected_module}] {text}"})
    resp = openai.chat.completions.create(model=MODEL, messages=messages, temperature=0.3, max_tokens=800)
    return resp.choices[0].message.content.strip()

# --- Main logic ---
def main():
    if st.sidebar.button("🔄 Nieuw gesprek"):
        on_reset(); st.rerun()

    # Download button
    if st.session_state.history and st.session_state.history[-1]["role"] == "assistant":
        pdf_bytes = make_pdf(
            question=st.session_state.last_question,
            answer=st.session_state.history[-1]["content"]
        )
        st.sidebar.download_button("📄 Download PDF", data=pdf_bytes,
                                   file_name="antwoord.pdf", mime="application/pdf")

    # Product selection
    if not st.session_state.selected_product:
        st.header("Welkom bij IPAL Chatbox")
        c1, c2, c3 = st.columns(3)
        if c1.button("Exact", use_container_width=True):
            add_msg("assistant","Gekozen: Exact"); st.session_state.selected_product="Exact"; st.rerun()
        if c2.button("DocBase", use_container_width=True):
            add_msg("assistant","Gekozen: DocBase"); st.session_state.selected_product="DocBase"; st.rerun()
        if c3.button("Algemeen", use_container_width=True):
            add_msg("assistant","Gekozen: Algemeen"); st.session_state.selected_product="Algemeen"; st.rerun()
        render_chat(); return

    # Module selection
    if st.session_state.selected_product != "Algemeen" and not st.session_state.selected_module:
        opts = subthema_dict.get(st.session_state.selected_product, [])
        sel = st.selectbox("Kies onderwerp:", ["(Kies)"] + opts)
        if sel != "(Kies)":
            add_msg("assistant", f"Gekozen: {sel}")
            st.session_state.selected_module = sel
            st.rerun()
        render_chat(); return

    # Chat input & processing
    render_chat()
    vraag = st.chat_input("Stel uw vraag:")
    if not vraag:
        return

    st.session_state.last_question = vraag
    add_msg("user", vraag)
    allowed, reason = filter_chatbot_topics(vraag)
    if not allowed:
        add_msg("assistant", reason); st.rerun()

    with st.spinner("Even zoeken..."):
        # Check bishop
        m = re.match(r'(?i)wie is bisschop(?: van)?\s+(.+)', vraag)
        if m:
            loc = m.group(1).strip()
            bishop = fetch_bishop_from_rkkerk(loc) or fetch_bishop_from_rkk_online(loc)
            if bishop:
                add_msg("assistant", f"De huidige bisschop van {loc} is {bishop}."); st.rerun()
        # Check all bishops NL
        if re.search(r'(?i)bisschoppen nederland', vraag):
            allb = fetch_all_bishops_nl()
            if allb:
                lines=[f"Mgr. {n} – Bisschop van {d}" for d,n in allb.items()]
                add_msg("assistant","Huidige Nederlandse bisschoppen:\n"+ "\n".join(lines)); st.rerun()
        # FAQ lookup
        dfm = faq_df[faq_df["combined"].str.contains(re.escape(vraag),case=False,na=False)]
        if not dfm.empty:
            row=dfm.iloc[0]
            ans=row["Antwoord"]
            try: ans=rewrite_answer(ans)
            except: pass
            if img:=row.get("Afbeelding"):
                st.image(img, caption="Voorbeeld", use_column_width=True)
            add_msg("assistant",ans); st.rerun()
        # AI fallback
        try:
            ant = get_ai_answer(vraag)
            add_msg("assistant",f"IPAL-Helpdesk antwoord:\n{ant}")
        except Exception as e:
            logging.error(f"AI-fallback mislukt: {e}")
            add_msg("assistant","⚠️ Fout tijdens AI-fallback")
    st.rerun()

if __name__ == "__main__":
    main()
