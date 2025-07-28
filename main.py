# main.py

import os
import re
import sys
import logging
from datetime import datetime
import io

import streamlit as st
import pandas as pd
import pytz
from PIL import Image
from dotenv import load_dotenv
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# — Logging config —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# — Load API key —
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai.api_key:
    st.sidebar.error("🔑 Voeg je OpenAI API key toe in .env of Streamlit Secrets.")
    st.stop()

# — RateLimitError import fallback —
try:
    from openai.error import RateLimitError
except ImportError:
    try:
        RateLimitError = openai.RateLimitError
    except AttributeError:
        RateLimitError = Exception

# — OpenAI helper met retry —
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    retry=retry_if_exception_type(RateLimitError)
)
def openai_chat(messages: list[dict], temperature: float = 0.3, max_tokens: int = 800) -> str:
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def rewrite_answer(text: str) -> str:
    system = "Herschrijf dit antwoord eenvoudig en vriendelijk."
    return openai_chat(
        [
            {"role": "system", "content": system},
            {"role": "user",   "content": text}
        ],
        temperature=0.2,
        max_tokens=800,
    )

def get_ai_answer(prompt: str, history: list[dict]) -> str:
    system = "Je bent de IPAL Chatbox, een behulpzame Nederlandse helpdeskassistent."
    messages = [{"role": "system", "content": system}] + history
    messages.append({"role": "user", "content": prompt})
    return openai_chat(messages)

# — Blacklist & filtering —
BLACKLIST_CATEGORIES = [
    "persoonlijke gegevens", "medische gegevens", "gezondheid", "strafrechtelijk verleden",
    # … rest van je categorieën …
    "schokkende inhoud", "gruwelijke inhoud", "privacy schending"
]
BLACKLIST_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, BLACKLIST_CATEGORIES)) + r")\b",
    flags=re.IGNORECASE
)

def check_blacklist(text: str) -> list[str]:
    return list({m.group(0).lower() for m in BLACKLIST_PATTERN.finditer(text)})

def filter_chatbot_topics(message: str) -> tuple[bool, str]:
    found = check_blacklist(message)
    if not found:
        return True, ""
    return False, (
        "Je bericht bevat inhoud die niet voldoet aan onze richtlijnen. "
        "Vermijd gevoelige onderwerpen en probeer het opnieuw."
    )

# — FAQ loader —
@st.cache_data(show_spinner=False)
def load_faq(path: str = "faq.xlsx") -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=["Systeem","Subthema","combined","Antwoord","Afbeelding"])
    try:
        df = pd.read_excel(path)
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

# — PDF-generator —
def genereer_pdf(tekst: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    text_obj = c.beginText(40, height - 50)
    text_obj.setFont("Helvetica", 12)
    for line in tekst.split("\n"):
        text_obj.textLine(line)
    c.drawText(text_obj)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# — Session & UI helpers —
TIMEZONE = pytz.timezone("Europe/Amsterdam")
MAX_HISTORY = 20
AVATARS = {
    "assistant": "aichatbox.jpg",
    "user": "parochie.jpg"
}

def get_avatar(role: str):
    path = AVATARS.get(role)
    if path and os.path.exists(path):
        return Image.open(path).resize((64,64))
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

# — Initialize session state —
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

    # Download PDF-knop
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
        if c1.button("DocBase", use_container_width=True):
            st.session_state.selected_product = "DocBase"
            add_message("assistant", "Gekozen: DocBase")
            st.rerun()
        if c2.button("Exact", use_container_width=True):
            st.session_state.selected_product = "Exact"
            add_message("assistant", "Gekozen: Exact")
            st.rerun()
        if c3.button("Algemeen", use_container_width=True):
            st.session_state.selected_product = "Algemeen"
            st.session_state.selected_module = "alles"
            add_message("assistant", "Gekozen: Algemeen")
            st.rerun()
        render_chat()
        return

    # Module-selectie voor DocBase/Exact
    if st.session_state.selected_product != "Algemeen" and not st.session_state.selected_module:
        opties = sorted(
            faq_df[faq_df["Systeem"] == st.session_state.selected_product]["Subthema"]
            .dropna().unique()
        )
        sel = st.selectbox("Kies onderwerp:", ["(Kies)"] + opties)
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

    allowed, warning = filter_chatbot_topics(vraag)
    if not allowed:
        add_message("assistant", warning)
        st.rerun()

    with st.spinner("Even zoeken..."):
        # Kies juiste subset FAQ
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
            # Veilig checken of img een pad-string is
            if isinstance(img, str) and img and os.path.exists(img):
                st.image(img, caption="Voorbeeld", use_column_width=True)

            add_message("assistant", ans)
        else:
            try:
                prompt = (
                    vraag
                    if st.session_state.selected_product == "Algemeen"
                    else f"[{st.session_state.selected_module}] {vraag}"
                )
                ai_ans = get_ai_answer(prompt, st.session_state.history[-10:])
                add_message("assistant", f"IPAL-Helpdesk antwoord:\n{ai_ans}")
            except Exception as e:
                logging.error(f"AI-fallback mislukt: {e}")
                add_message("assistant", "⚠️ Fout tijdens AI-fallback")

    st.rerun()

if __name__ == "__main__":
    main()
