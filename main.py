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

# Safe import for RateLimitError
try:
    from openai.error import RateLimitError
except ImportError:
    RateLimitError = Exception

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- Page config & styling ---
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
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(1,10),
       retry=retry_if_exception_type(RateLimitError))
def chatgpt(messages, temperature=0.3, max_tokens=800):
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# --- FAQ Loader ---
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

# --- Blacklist ---
BLACKLIST = ["persoonlijke gegevens","medische gegevens","gezondheid","privacy schending"]
def filter_topics(msg):
    found = [t for t in BLACKLIST if re.search(rf"\b{re.escape(t)}\b", msg.lower())]
    return (False, f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}.") if found else (True, "")

# --- PDF Export with Calibri 11 Prepend ---
# Register Calibri if available
if os.path.exists("Calibri.ttf"):
    pdfmetrics.registerFont(TTFont("Calibri", "Calibri.ttf"))
else:
    logging.info("Calibri.ttf niet gevonden, gebruik ingebouwde Helvetica")

def make_pdf(answer_text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 40
    usable_w = width - 2 * margin

    # Logo
    y = height - margin
    if os.path.exists("logo.png"):
        img = PILImage.open("logo.png")
        ar = img.width / img.height
        logo_h = 50
        c.drawImage("logo.png", margin, y - logo_h, width=logo_h*ar, height=logo_h, mask="auto")
        y -= (logo_h + 10)
    else:
        y -= 10

    # Set font
    font_name = "Calibri" if "Calibri" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
    c.setFont(font_name, 11)

    # Prepend text
    prepend = (
        "1. Dit is het AI antwoord vanuit de IPAL chatbox van het Interdiocesaan Platform "
        "Automatisering & Ledenadministratie. Het is altijd een goed idee om de meest recente "
        "informatie te controleren via officiële bronnen.\n\n"
        "2. Heeft u hulp nodig met DocBase of Exact? Dan kunt u eenvoudig een melding maken door "
        "een ticket aan te maken in DocBase. Maar voordat u een ticket invult, hebben we een "
        "handige tip: controleer eerst onze FAQ (het document met veelgestelde vragen en antwoorden). "
        "Dit document vindt u op onze site.\n\n"
        "Waarom de FAQ gebruiken?\nIn het document met veelgestelde vragen vindt u snel en eenvoudig "
        "antwoorden op veelvoorkomende vragen, zonder dat u hoeft te wachten op hulp.\n\n"
        "Klik hieronder om de FAQ te openen en te kijken of uw vraag al beantwoord is:\n"
        "– Veel gestelde vragen Docbase nieuw 2024\n"
        "– Veel gestelde vragen Exact Online\n\n"
        "Kan het FAQ document geen hulp bieden, maakt u dan een ticket aan door onderaan op JA te klikken.\n\n"
        "Instructie: Ticket aanmaken in DocBase\n"
        "Geen probleem! Zorg ervoor dat uw melding duidelijk is:\n\n"
        "• Beschrijf het probleem zo gedetailleerd mogelijk.\n"
        "• Voegt u geen document toe, zet dan het documentformaat in het ticket op “geen bijlage”.\n"
        "• Geef uw telefoonnummer op waarop wij u kunnen bereiken, zodat de helpdesk contact met u kan opnemen.\n\n"
    )

    full_text = prepend + answer_text

    text_obj = c.beginText(margin, y)
    text_obj.setFont(font_name, 11)
    max_chars = int(usable_w / (11 * 0.6))
    for para in full_text.split("\n"):
        for line in textwrap.wrap(para, width=max_chars):
            text_obj.textLine(line)

    c.drawText(text_obj)
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()

# --- RKK Scraping ---
def fetch_bishop_from_rkkerk(loc):
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

def fetch_bishop_from_rkk_online(loc):
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

# --- Avatars & Chat Helpers ---
AVATARS = {"assistant":"aichatbox.jpg","user":"parochie.jpg"}
def get_avatar(role):
    path = AVATARS.get(role)
    if path and os.path.exists(path):
        return PILImage.open(path).resize((64,64))
    return "🙂"

TIMEZONE = pytz.timezone("Europe/Amsterdam")
MAX_HISTORY = 20
def add_msg(role, content):
    ts = datetime.now(TIMEZONE).strftime("%d-%m-%Y %H:%M")
    st.session_state.history = (st.session_state.history + [{"role":role,"content":content,"time":ts}])[-MAX_HISTORY:]

def render_chat():
    for msg in st.session_state.history:
        st.chat_message(msg["role"], avatar=get_avatar(msg["role"])).markdown(
            f"{msg['content']}\n\n_{msg['time']}_"
        )

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.selected_product = None

# --- Main app ---
def main():
    if st.sidebar.button("🔄 Nieuw gesprek"):
        st.session_state.clear()
        st.rerun()

    if st.session_state.history and st.session_state.history[-1]["role"] == "assistant":
        st.sidebar.download_button(
            "📄 Download PDF",
            data=make_pdf(st.session_state.history[-1]["content"]),
            file_name="antwoord.pdf",
            mime="application/pdf"
        )

    if not st.session_state.selected_product:
        st.header("Welkom bij IPAL Chatbox")
        c1, c2, c3 = st.columns(3)
        if c1.button("Exact", use_container_width=True):
            st.session_state.selected_product = "Exact"
            add_msg("assistant", "Gekozen: Exact")
            st.rerun()
        if c2.button("DocBase", use_container_width=True):
            st.session_state.selected_product = "DocBase"
            add_msg("assistant", "Gekozen: DocBase")
            st.rerun()
        if c3.button("Algemeen", use_container_width=True):
            st.session_state.selected_product = "Algemeen"
            add_msg("assistant", "Gekozen: Algemeen")
            st.rerun()
        render_chat()
        return

    render_chat()
    vraag = st.chat_input("Stel uw vraag:")
    if not vraag:
        return

    add_msg("user", vraag)
    ok, warn = filter_topics(vraag)
    if not ok:
        add_msg("assistant", warn)
        st.rerun()

    # 1) Specific bishop question
    m = re.match(r'(?i)wie is bisschop(?: van)?\s+(.+)\?*', vraag)
    if m:
        loc = m.group(1).strip()
        bishop = fetch_bishop_from_rkkerk(loc) or fetch_bishop_from_rkk_online(loc)
        if bishop:
            add_msg("assistant", f"De huidige bisschop van {loc} is {bishop}.")
            st.rerun()

    # 2) All Dutch bishops
    if re.search(r'(?i)bisschoppen nederland', vraag):
        allb = fetch_all_bishops_nl()
        if allb:
            lines = [f"Mgr. {n} – Bisschop van {d}" for d,n in allb.items()]
            add_msg("assistant", "Huidige Nederlandse bisschoppen:\n" + "\n".join(lines))
            st.rerun()

    # 3) FAQ lookup
    dfm = faq_df[faq_df["combined"].str.contains(re.escape(vraag), case=False, na=False)]
    if not dfm.empty:
        row = dfm.iloc[0]
        ans = row["Antwoord"]
        try:
            ans = chatgpt([
                {"role":"system","content":"Herschrijf dit eenvoudig en vriendelijk."},
                {"role":"user","content":ans}
            ], temperature=0.2)
        except:
            pass
        if img := row["Afbeelding"]:
            st.image(img, caption="Voorbeeld", use_column_width=True)
        add_msg("assistant", ans)
        st.rerun()

    # 4) AI fallback
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
