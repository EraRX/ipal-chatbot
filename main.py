# main.py

"""
IPAL Chatbox voor oudere vrijwilligers
- Python 3, Streamlit
- Groot lettertype, eenvoudige bediening
- Keuze tussen Exact, DocBase en Algemeen (algemene FAQ)
- Antwoorden uit FAQ, aangevuld met AI voor niet-FAQ vragen
- Topicfiltering via complete-word blacklist
- Retry-logica voor OpenAI-calls bij rate-limits
- Real-time context uit Wikipedia (NL→EN) vóór AI request
- OpenAI Python v1 client interface
- PDF-export met logo (logo.png) en tekst-wrapping
- Avatar-ondersteuning, logging en foutafhandeling
"""

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
import wikipedia

from openai import OpenAI
try:
    from openai.error import RateLimitError
except ImportError:
    RateLimitError = Exception

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# — Streamlit pagina-configuratie & styling —
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

# — OpenAI key & model —
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.sidebar.error("🔑 Voeg je OpenAI API key toe in .env of Streamlit Secrets.")
    st.stop()
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
client = OpenAI(api_key=OPENAI_KEY)

# — Retry-wrapper voor RateLimitError —
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

def fetch_wikipedia_summary(query: str) -> str | None:
    """
    Zoek eerst in de NL-Wikipedia, en val terug op EN-Wikipedia.
    Retourneert maximaal 2 zinnen samenvatting.
    """
    for lang in ("nl", "en"):
        try:
            wikipedia.set_lang(lang)
            results = wikipedia.search(query, results=1, auto_suggest=False)
            if not results:
                continue
            return wikipedia.summary(
                results[0],
                sentences=2,
                auto_suggest=False,
                redirect=True
            )
        except Exception:
            continue
    return None

def get_ai_answer(prompt: str) -> str:
    """
    Genereer AI-antwoord met voorafgaande Wikipedia-context (NL→EN).
    """
    summary = fetch_wikipedia_summary(prompt)
    if summary:
        context = {
            "role": "system",
            "content": f"Volgens Wikipedia (laatst bijgewerkt juli 2025):\n{summary}"
        }
    else:
        context = {"role": "system", "content": ""}
    history_msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.history[-10:]
    ]
    messages = [context] + history_msgs + [{"role": "user", "content": prompt}]
    return openai_chat(messages)

# — Whole-word blacklist & filtering —
BLACKLIST_CATEGORIES = [
    # … je volledige lijst …
    "persoonlijke gegevens", "medische gegevens", "gezondheid", # etc.
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
    if found:
        logging.info(f"Blacklist terms flagged: {found}")
        return False, (
            f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}. "
            "Vermijd deze onderwerpen en probeer het opnieuw."
        )
    return True, ""

# — Load FAQ uit Excel —
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

# — PDF-export met logo & wrapping —
def genereer_pdf(tekst: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    left_margin, right_margin = 40, 40
    usable_width = width - left_margin - right_margin

    # Logo tekenen
    logo_path, logo_h = "logo.png", 50
    if os.path.exists(logo_path):
        img = PILImage.open(logo_path)
        aspect = img.width / img.height
        c.drawImage(
            logo_path,
            left_margin,
            height - logo_h - 10,
            width=logo_h * aspect,
            height=logo_h,
            mask="auto"
        )
        start_y = height - logo_h - 30
    else:
        start_y = height - 50

    text_obj = c.beginText(left_margin, start_y)
    text_obj.setFont("Helvetica", 12)
    max_chars = int(usable_width / (12 * 0.6))
    for para in tekst.split("\n"):
        for line in textwrap.wrap(para, width=max_chars):
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
    return (PILImage.open(path).resize((64,64)) if path and os.path.exists(path)
            else "🙂")

def add_message(role: str, content: str):
    ts = datetime.now(TIMEZONE).strftime("%d-%m-%Y %H:%M")
    st.session_state.history = (
        st.session_state.history + [{"role": role, "content": content, "time": ts}]
    )[-MAX_HISTORY:]

def render_chat():
    for m in st.session_state.history:
        st.chat_message(m["role"], avatar=get_avatar(m["role"])).markdown(
            f"{m['content']}\n\n_{m['time']}_"
        )

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.selected_product = None
    st.session_state.selected_module = None

# — Main app —
def main():
    if st.sidebar.button("🔄 Nieuw gesprek"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    if st.session_state.history and st.session_state.history[-1]["role"] == "assistant":
        laatste = st.session_state.history[-1]["content"]
        st.sidebar.download_button(
            "📄 Download antwoord als PDF",
            data=genereer_pdf(laatste),
            file_name="antwoord.pdf",
            mime="application/pdf"
        )

    if not st.session_state.selected_product:
        st.header("Welkom bij IPAL Chatbox")
        c1, c2, c3 = st.columns(3)
        if c1.button("Exact", use_container_width=True):
            st.session_state.selected_product="Exact"; add_message("assistant","Gekozen: Exact"); st.rerun()
        if c2.button("DocBase", use_container_width=True):
            st.session_state.selected_product="DocBase"; add_message("assistant","Gekozen: DocBase"); st.rerun()
        if c3.button("Algemeen", use_container_width=True):
            st.session_state.selected_product="Algemeen"; st.session_state.selected_module="alles"; add_message("assistant","Gekozen: Algemeen"); st.rerun()
        render_chat(); return

    if st.session_state.selected_product!="Algemeen" and not st.session_state.selected_module:
        opts = sorted(faq_df[faq_df["Systeem"]==st.session_state.selected_product]["Subthema"].dropna().unique())
        sel = st.selectbox("Kies onderwerp:", ["(Kies)"]+opts)
        if sel!="(Kies)":
            st.session_state.selected_module=sel; add_message("assistant",f"Gekozen: {sel}"); st.rerun()
        render_chat(); return

    render_chat()
    vraag = st.chat_input("Stel uw vraag:")
    if not vraag:
        return

    add_message("user", vraag)
    ok, warn = filter_chatbot_topics(vraag)
    if not ok:
        add_message("assistant", warn); st.rerun()

    with st.spinner("Even zoeken..."):
        if st.session_state.selected_product=="Algemeen":
            dfm = faq_df[faq_df["combined"].str.contains(vraag, case=False, na=False)]
        else:
            dfm = faq_df[
                (faq_df["Systeem"]==st.session_state.selected_product)&
                (faq_df["Subthema"].str.lower()==st.session_state.selected_module.lower())
            ]

        if not dfm.empty:
            row = dfm.iloc[0]
            ans = row["Antwoord"]
            try: ans = rewrite_answer(ans)
            except: pass

            img = row.get("Afbeelding")
            if isinstance(img,str) and img and os.path.exists(img):
                st.image(img, caption="Voorbeeld", use_column_width=True)

            add_message("assistant", ans)
        else:
            prompt = vraag if st.session_state.selected_product=="Algemeen" else f"[{st.session_state.selected_module}] {vraag}"
            try:
                ai_ans = get_ai_answer(prompt)
                add_message("assistant", f"IPAL-Helpdesk antwoord:\n{ai_ans}")
            except Exception as e:
                logging.exception("AI-fallback mislukt:")
                add_message("assistant", f"⚠️ AI-fallback mislukt: {type(e).__name__}: {e}")

    st.rerun()

if __name__ == "__main__":
    main()
