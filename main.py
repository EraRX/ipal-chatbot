# main.py

"""
IPAL Chatbox voor oudere vrijwilligers
- Python 3, Streamlit
- Groot lettertype, eenvoudige bediening
- Keuze tussen Exact, DocBase en Algemeen
- Antwoorden uit FAQ, aangevuld met AI voor niet-FAQ vragen
- Topicfiltering via complete‐word blacklist
- Retry‐logica bij rate‐limits via tenacity
- Real‐time context uit Wikipedia REST‐API voor ambtvragen (president, paus)
- Scraping officiële RKK-website voor bisschoppen in Nederland
- Fallback naar Wikipedia‐samenvatting voor overige queries
- Directe OpenAI Chat Completions API via requests
- PDF‐export met logo en automatische tekst‐wrapping
- Avatar‐ondersteuning, logging en foutafhandeling
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
import requests
from bs4 import BeautifulSoup
from PIL import Image as PILImage
from dotenv import load_dotenv
import wikipedia  # pip install wikipedia==1.4.0

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# — Streamlit pagina‐configuratie & styling —
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
    handlers=[logging.StreamHandler()]
)

# — Laad OpenAI‐sleutel & model uit env/secrets —
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.sidebar.error("🔑 Voeg je OpenAI API‐key toe in .env of Streamlit Secrets.")
    st.stop()
MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# — OpenAI REST endpoint & headers —
API_URL = "https://api.openai.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENAI_KEY}",
    "Content-Type": "application/json",
}

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(requests.exceptions.HTTPError),
)
def openai_chat(messages: list[dict], temperature: float = 0.3, max_tokens: int = 800) -> str:
    """Roep OpenAI Chat Completions REST API aan met retry bij HTTP errors."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

def rewrite_answer(text: str) -> str:
    return openai_chat([
        {"role": "system", "content": "Herschrijf dit antwoord eenvoudig en vriendelijk."},
        {"role": "user",   "content": text},
    ], temperature=0.2, max_tokens=800)

# — Load FAQ from Excel —
@st.cache_data(show_spinner=False)
def load_faq(path: str = "faq.xlsx") -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"FAQ‐bestand '{path}' niet gevonden.")
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
    cols = ["Systeem","Subthema","Omschrijving melding","Toelichting melding"]
    df["combined"] = df[cols].fillna("").agg(" ".join, axis=1)
    return df

faq_df = load_faq()
subthema_dict = {
    p: sorted(faq_df[faq_df["Systeem"] == p]["Subthema"].dropna().unique())
    for p in ["Exact", "DocBase"]
}

# — Blacklist & filtering —
BLACKLIST_CATEGORIES = [
    "persoonlijke gegevens","medische gegevens","gezondheid","strafrechtelijk verleden",
    # … vul verder aan …
    "privacy schending"
]

def check_blacklist(msg: str) -> list[str]:
    found = []
    low = msg.lower()
    for term in BLACKLIST_CATEGORIES:
        if re.search(rf"\b{re.escape(term.lower())}\b", low):
            found.append(term)
    return found

def filter_chatbot_topics(msg: str) -> tuple[bool, str]:
    found = check_blacklist(msg)
    if found:
        warning = (
            f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}. "
            "Vermijd deze onderwerpen en probeer het opnieuw."
        )
        logging.info(f"Blacklist flagged: {found}")
        return False, warning
    return True, ""

# — PDF‐export met logo & wrapping —
def genereer_pdf(text: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    lm, rm = 40, 40
    usable_w = w - lm - rm

    # Logo tekenen als aanwezig
    logo_path, logo_h = "logo.png", 50
    if os.path.exists(logo_path):
        img = PILImage.open(logo_path)
        ar = img.width / img.height
        c.drawImage(logo_path, lm, h-logo_h-10,
                    width=logo_h*ar, height=logo_h, mask="auto")
        y0 = h - logo_h - 30
    else:
        y0 = h - 50

    text_obj = c.beginText(lm, y0)
    text_obj.setFont("Helvetica", 12)
    max_chars = int(usable_w / (12 * 0.6))
    for para in text.split("\n"):
        for line in textwrap.wrap(para, width=max_chars):
            text_obj.textLine(line)

    c.drawText(text_obj)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# — Officiële RKK‐website scraping voor bisschoppen —
def fetch_bishop_from_rkkerk(loc: str) -> str | None:
    """
    Scrape rkkerk.nl voor bisschop van een diocese
    loc: 'Roermond', 'Utrecht', etc.
    """
    slug = loc.lower().replace(" ", "-")
    url = f"https://www.rkkerk.nl/bisdom-{slug}/"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # veronderstel dat bisschopsnaam in <h1> of eerste <p> staat
        h1 = soup.find("h1")
        if h1 and "bisschop" in h1.text.lower():
            # tekst als 'Mgr. dr. C.F.M. van den Hout — Bisschop van Roermond'
            return h1.text.strip().split("—")[0].strip()
        # fallback: zoek eerste paragraaf met 'Mgr'
        p = soup.find("p", string=re.compile(r"Mgr\."))
        if p:
            return p.text.strip().split(",")[0]
    except Exception:
        pass
    return None

# — Wikipedia REST‐API voor ambtvragen (president, paus) —
def fetch_incumbent(office_page: str, lang: str = "en") -> str | None:
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{office_page}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        extract = r.json().get("extract", "")
        m = re.search(r"(?i)(?:current|huidige)\s+\S+\s+is\s+([A-Z][\w\s\-']+)", extract)
        return m.group(1).strip() if m else None
    except Exception:
        return None

def extract_wiki_topic(prompt: str) -> str:
    m = re.match(r'(?i)wie (?:is|zijn) (?:de |het |een )?(.+)\?*$', prompt.strip())
    return m.group(1) if m else prompt

def fetch_wikipedia_summary(topic: str) -> str | None:
    for lang in ("nl", "en"):
        try:
            wikipedia.set_lang(lang)
            results = wikipedia.search(topic, results=1, auto_suggest=False)
            if not results:
                continue
            return wikipedia.summary(results[0], sentences=2,
                                     auto_suggest=False, redirect=True)
        except Exception:
            continue
    return None

def get_ai_answer(prompt: str) -> str:
    low = prompt.lower()

    # President VS
    if low.startswith("wie is") and "president" in low:
        holder = fetch_incumbent("President_of_the_United_States", lang="en")
        if holder:
            return f"De huidige president van de Verenigde Staten is {holder}."

    # Paus
    if low.startswith("wie is") and ("paus" in low or "pope" in low):
        holder = fetch_incumbent("Pope", lang="en")
        if holder:
            return f"De huidige paus is {holder}."

    # Bisschop van X
    if low.startswith("wie is") and "bisschop" in low:
        loc_match = re.search(r'bisschop(?: van)?\s+([\w\s\-]+)', prompt, re.I)
        if loc_match:
            loc = loc_match.group(1).strip()
            bishop = fetch_bishop_from_rkkerk(loc)
            if bishop:
                return f"De huidige bisschop van {loc} is {bishop}."
    # Fallback: Wikipedia‐samenvatting + AI
    topic = extract_wiki_topic(prompt)
    summary = fetch_wikipedia_summary(topic)
    context = (
        {"role": "system",
         "content": f"Volgens Wikipedia (jul 2025) over '{topic}':\n{summary}"}
        if summary else {"role": "system", "content": ""}
    )
    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.history[-10:]]
    messages = [context] + history + [{"role": "user", "content": prompt}]
    return openai_chat(messages)

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
    for msg in st.session_state.history:
        st.chat_message(msg["role"], avatar=get_avatar(msg["role"])).markdown(
            f"{msg['content']}\n\n_{msg['time']}_"
        )

# Init sessiestate
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

    # Module-selectie
    if st.session_state.selected_product != "Algemeen" and not st.session_state.selected_module:
        opts = subthema_dict.get(st.session_state.selected_product, [])
        sel = st.selectbox("Kies onderwerp:", ["(Kies)"] + opts)
        if sel != "(Kies)":
            st.session_state.selected_module = sel
            add_message("assistant", f"Gekozen: {sel}")
            st.rerun()
        render_chat()
        return

    # Chat-interface
    render_chat()
    vraag = st.chat_input("Stel uw vraag:")
    if not vraag:
        return

    add_message("user", vraag)
    ok, warn = filter_chatbot_topics(vraag)
    if not ok:
        add_message("assistant", warn)
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
            try:
                ai_ans = get_ai_answer(vraag)
                add_message("assistant", f"IPAL-Helpdesk antwoord:\n{ai_ans}")
            except Exception as e:
                logging.exception("AI-fallback mislukt:")
                add_message("assistant", f"⚠️ AI-fallback mislukt: {type(e).__name__}: {e}")

    st.rerun()

if __name__ == "__main__":
    main()
