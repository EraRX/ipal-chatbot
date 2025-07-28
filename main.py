# main.py

import os
import re
import sys
import logging
from datetime import datetime
import io
import textwrap

import streamlit as st
import pandas as pd
import pytz
from PIL import Image as PILImage
from dotenv import load_dotenv

from openai import OpenAI
try:
    from openai.error import RateLimitError
except ImportError:
    RateLimitError = Exception

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# — Page config & styling —
st.set_page_config(page_title="IPAL Chatbox", layout="centered")
st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-size: 20px; }
      button[kind="primary"] { font-size: 22px !important; padding: 0.75em 1.5em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# — Logging setup —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# — Load OpenAI key & model —
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.sidebar.error("🔑 Voeg je OpenAI API key toe in .env of Streamlit Secrets.")
    st.stop()
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
client = OpenAI(api_key=OPENAI_KEY)

# — Retry wrapper for RateLimitError —
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(RateLimitError),
)
def openai_chat(messages: list[dict], temperature: float = 0.3, max_tokens: int = 800) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def rewrite_answer(text: str) -> str:
    return openai_chat(
        [
            {"role": "system", "content": "Herschrijf dit antwoord eenvoudig en vriendelijk."},
            {"role": "user",   "content": text},
        ],
        temperature=0.2,
        max_tokens=800,
    )

def get_ai_answer(prompt: str) -> str:
    system = "Je bent de IPAL Chatbox, een behulpzame Nederlandse helpdeskassistent."
    history_msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.history[-10:]
    ]
    messages = [{"role": "system", "content": system}] + history_msgs
    messages.append({"role": "user", "content": prompt})
    return openai_chat(messages)

# — Blacklist & filtering —
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

def check_blacklist(message: str) -> list[str]:
    found = []
    text = message.lower()
    for term in BLACKLIST_CATEGORIES:
        if re.search(rf"\b{re.escape(term.lower())}\b", text):
            found.append(term)
    return found

def filter_chatbot_topics(message: str) -> tuple[bool, str]:
    found = check_blacklist(message)
    if not found:
        return True, ""
    logging.info(f"Blacklist terms flagged: {found} in message: {message!r}")
    warning = (
        f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}. "
        "Vermijd deze onderwerpen en probeer het opnieuw."
    )
    return False, warning

# — Load FAQ —
@st.cache_data(show_spinner=False)
def load_faq(path: str = "faq.xlsx") -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=["Systeem","Subthema","combined","Antwoord","Afbeelding"])
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        logging.error(f"Fout bij laden FAQ: {e}")
        st.error("⚠️ Kan FAQ niet laden.")
        return pd.DataFrame(columns=["Systeem","Subthema","combined","Antwoord","Afbeelding"])
    if "Afbeelding" not in df.columns:
        df["Afbeelding"] = None
    df["Antwoord"] = df["Antwoord of oplossing"]
    required = ["Systeem","Subthema","Omschrijving melding","Toelichting melding"]
    df["combined"] = df[required].fillna("").agg(" ".join, axis=1)
    return df

faq_df = load_faq()

# — PDF export with logo & wrapping —
def genereer_pdf(tekst: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    left_margin = 40
    right_margin = 40
    usable_width = width - left_margin - right_margin

    # Draw logo if available
    logo_path = "logo.png"
    logo_height = 50
    if os.path.exists(logo_path):
        img = PILImage.open(logo_path)
        aspect = img.width / img.height
        logo_width = logo_height * aspect
        c.drawImage(
            logo_path,
            left_margin,
            height - logo_height - 10,
            width=logo_width,
            height=logo_height,
            mask="auto"
        )
        text_start_y = height - logo_height - 30
    else:
        text_start_y = height - 50

    text_obj = c.beginText(left_margin, text_start_y)
    text_obj.setFont("Helvetica", 12)

    max_chars = int(usable_width / (12 * 0.6))
    for paragraph in tekst.split("\n"):
        lines = textwrap.wrap(paragraph, width=max_chars)
        if not lines:
            text_obj.textLine("")
        else:
            for line in lines:
                text_obj.textLine(line)

    c.drawText(text_obj)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# — UI & session helpers —
TIMEZONE = pytz.timezone("Europe/Amsterdam")
MAX_HISTORY = 20
AVATARS = {"assistant": "aichatbox.jpg", "user": "parochie.jpg"}

def get_avatar(role: str):
    path = AVATARS.get(role)
    if path and os.path.exists(path):
        return PILImage.open(path).resize((64, 64))
    return "🙂"

def add_message(role: str, content: str):
    ts = datetime.now(TIMEZONE).strftime("%d-%m-%Y %H:%M")
    st.session_state.history = (
        st.session_state.history + [{"role": role, "content": content, "time": ts}]
    )[-MAX_HISTORY:]

def render_chat():
    for msg in st.session_state.history:
        st.chat_message(msg["role"], avatar=get_avatar(msg["role"])).markdown(
            f"{msg['content']}\n\n_{msg['time']}_"
        )

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.selected_product = None
    st.session_state.selected_module = None

# — Main app —
def main():
    # Nieuw gesprek
    if st.sidebar.button("🔄 Nieuw gesprek"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    # PDF-download for last assistant reply
    if st.session_state.history and st.session_state.history[-1]["role"] == "assistant":
        laatste = st.session_state.history[-1]["content"]
        st.sidebar.download_button(
            "📄 Download antwoord als PDF",
            data=genereer_pdf(laatste),
            file_name="antwoord.pdf",
            mime="application/pdf"
        )

    # Product-selectie
    if not st.session_state.selected_product:
        st.header("Welkom bij IPAL Chatbox")
        c1, c2, c3 = st.columns(3)
        if c1.button("Exact", use_container_width=True):
            st.session_state.selected_product = "Exact"
            add_message("assistant", "Gekozen: Exact")
            st.rerun()
        if c2.button("DocBase", use_container_width=True):
            st.session_state.selected_product = "DocBase"
            add_message("assistant", "Gekozen: DocBase")
            st.rerun()
        if c3.button("Algemeen", use_container_width=True):
            st.session_state.selected_product = "Algemeen"
            st.session_state.selected_module = "alles"
            add_message("assistant", "Gekozen: Algemeen")
            st.rerun()
        render_chat()
        return

    # Module-selectie voor Exact/DocBase
    if st.session_state.selected_product != "Algemeen" and not st.session_state.selected_module:
        opts = sorted(
            faq_df[faq_df["Systeem"] == st.session_state.selected_product]["Subthema"]
            .dropna().unique()
        )
        sel = st.selectbox("Kies onderwerp:", ["(Kies)"] + opts)
        if sel != "(Kies)":
            st.session_state.selected_module = sel
            add_message("assistant", f"Gekozen: {sel}")
            st.rerun()
        render_chat()
        return

    # Chat interface
    render_chat()
    vraag = st.chat_input("Stel uw vraag:")
    if not vraag:
        return

    add_message("user", vraag)
    allowed, warning = filter_chatbot_topics(vraag)
    if not allowed:
        add_message("assistant", warning)
        st.rerun()

    with st.spinner("Even zoeken..."):
        if st.session_state.selected_product == "Algemeen":
            dfm = faq_df[faq_df["combined"].str.contains(vraag, case=False, na=False)]
        else:
            dfm = faq_df[
                (faq_df["Systeem"] == st.session_state.selected_product) &
                (faq_df["Subthema"].str.lower() == st.session_state.selected_module.lower())
            ]

        if not dfm.empty:
            row = dfm.iloc[0]
            ans = row["Antwoord"]
            try:
                ans = rewrite_answer(ans)
            except:
                pass

            img = row.get("Afbeelding")
            if isinstance(img, str) and img and os.path.exists(img):
                st.image(img, caption="Voorbeeld", use_column_width=True)

            add_message("assistant", ans)
        else:
            prompt = (
                vraag
                if st.session_state.selected_product == "Algemeen"
                else f"[{st.session_state.selected_module}] {vraag}"
            )
            try:
                ai_ans = get_ai_answer(prompt)
                add_message("assistant", f"IPAL-Helpdesk antwoord:\n{ai_ans}")
            except Exception as e:
                logging.exception("AI-fallback mislukt:")
                add_message("assistant", f"⚠️ AI-fallback mislukt: {type(e).__name__}: {e}")

    st.rerun()

if __name__ == "__main__":
    main()
