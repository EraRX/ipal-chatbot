# main.py

"""
IPAL Chatbox voor oudere vrijwilligers
- Python 3, Streamlit
- Groot lettertype, eenvoudige bediening
- Gebruik van FAQ uit faq.xlsx
- Fallback naar ChatGPT (gpt-4o of wat je in OPENAI_MODEL zet)
- Topicfiltering via blacklist
- Retry-logica OpenAI bij rate-limits
- Download PDF met logo en tekst‐wrapping
"""

import os
import re
import logging
from datetime import datetime
import io
import textwrap

import streamlit as st
import pandas as pd
import pytz
from dotenv import load_dotenv
from PIL import Image as PILImage

import openai
# Veilig RateLimitError importeren, of fallback naar Exception
try:
    from openai.error import RateLimitError
except ImportError:
    RateLimitError = Exception

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# — Config & styling —
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# — OpenAI setup —
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai.api_key:
    st.sidebar.error("🔑 Voeg je OpenAI API-key toe in .env of Streamlit Secrets.")
    st.stop()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# — Retry-wrapper voor ChatGPT calls —
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(RateLimitError),
)
def chatgpt(messages: list[dict], temperature=0.3, max_tokens=800) -> str:
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def rewrite(text: str) -> str:
    return chatgpt([
        {"role":"system","content":"Herschrijf dit antwoord eenvoudig en vriendelijk."},
        {"role":"user","content":text},
    ], temperature=0.2, max_tokens=800)

# — FAQ loader —
@st.cache_data
def load_faq(path="faq.xlsx") -> pd.DataFrame:
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
def filter_topics(msg: str) -> tuple[bool,str]:
    found = [t for t in BLACKLIST if re.search(rf"\b{re.escape(t)}\b", msg.lower())]
    if found:
        return False, f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}."
    return True, ""

# — PDF-export met logo & wrapping —
def make_pdf(text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w,h = A4
    margin = 40
    usable_w = w - 2*margin

    logo = "logo.png"
    if os.path.exists(logo):
        img = PILImage.open(logo)
        ar = img.width/img.height
        height_logo = 50
        c.drawImage(logo, margin, h-height_logo-10, width=height_logo*ar, height=height_logo, mask="auto")
        y = h-height_logo-30
    else:
        y = h-50

    text_obj = c.beginText(margin, y)
    text_obj.setFont("Helvetica", 12)
    max_chars = int(usable_w/(12*0.6))
    for para in text.split("\n"):
        for line in textwrap.wrap(para, width=max_chars):
            text_obj.textLine(line)
    c.drawText(text_obj)
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()

# — UI helpers —
TIMEZONE = pytz.timezone("Europe/Amsterdam")
MAX_HISTORY = 20

def add_msg(role,content):
    ts = datetime.now(TIMEZONE).strftime("%d-%m-%Y %H:%M")
    st.session_state.history = (st.session_state.history + [{"role":role,"content":content,"time":ts}])[-MAX_HISTORY:]

def render():
    for m in st.session_state.history:
        st.chat_message(m["role"]).markdown(f"{m['content']}\n\n_{m['time']}_")

# Initialize session
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.selected_product = None

# — Main app —
def main():
    if st.sidebar.button("🔄 Nieuw gesprek"):
        st.session_state.clear()
        st.experimental_rerun()

    if st.session_state.history and st.session_state.history[-1]["role"]=="assistant":
        st.sidebar.download_button(
            "📄 Download PDF",
            data=make_pdf(st.session_state.history[-1]["content"]),
            file_name="antwoord.pdf", mime="application/pdf"
        )

    if not st.session_state.selected_product:
        st.header("Welkom bij IPAL Chatbox")
        c1,c2,c3 = st.columns(3)
        if c1.button("Exact"):
            st.session_state.selected_product="Exact"; add_msg("assistant","Gekozen: Exact"); st.experimental_rerun()
        if c2.button("DocBase"):
            st.session_state.selected_product="DocBase"; add_msg("assistant","Gekozen: DocBase"); st.experimental_rerun()
        if c3.button("Algemeen"):
            st.session_state.selected_product="Algemeen"; add_msg("assistant","Gekozen: Algemeen"); st.experimental_rerun()
        render(); return

    render()
    vraag = st.chat_input("Stel uw vraag:")
    if not vraag:
        return

    add_msg("user", vraag)
    ok, warn = filter_topics(vraag)
    if not ok:
        add_msg("assistant", warn)
        st.experimental_rerun()

    # FAQ lookup
    dfm = faq_df[faq_df["combined"].str.contains(re.escape(vraag), case=False, na=False)]
    if not dfm.empty:
        row = dfm.iloc[0]
        ans = row["Antwoord"]
        try:
            ans = rewrite(ans)
        except:
            pass
        if img := row["Afbeelding"]:
            st.image(img, caption="Voorbeeld", use_column_width=True)
        add_msg("assistant", ans)
        st.experimental_rerun()

    # ChatGPT fallback
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

    st.experimental_rerun()

if __name__=="__main__":
    main()
