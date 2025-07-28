# main.py

import os
import sys
import logging
from datetime import datetime

import streamlit as st
import pandas as pd
import pytz
from PIL import Image
from dotenv import load_dotenv
import openai

from faq import load_faq
from ai_client import get_ai_answer, rewrite_answer
from filters import filter_chatbot_topics
from pdf_utils import genereer_pdf

# — Config logging —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# — Laad API key —
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai.api_key:
    st.sidebar.error("🔑 Voeg je OpenAI API key toe in .env of Streamlit Secrets.")
    st.stop()

# — Cache en laad FAQ —
@st.cache_data(show_spinner=False)
def get_faq_df(path: str = "faq.xlsx") -> pd.DataFrame:
    return load_faq(path)

faq_df = get_faq_df()

# — Session defaults —
MAX_HISTORY = 20
TIMEZONE = pytz.timezone("Europe/Amsterdam")
AVATARS = {
    "assistant": "aichatbox.jpg",
    "user": "parochie.jpg"
}

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.selected_product = None
    st.session_state.selected_module = None

# — Helpers —
def get_avatar(role: str):
    path = AVATARS.get(role)
    if path and os.path.exists(path):
        return Image.open(path).resize((64, 64))
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

# — Main app —
def main():
    # Nieuw gesprek
    if st.sidebar.button("🔄 Nieuw gesprek"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

    # PDF-downloadknop voor laatste antwoord
    if st.session_state.history and st.session_state.history[-1]["role"] == "assistant":
        laatste = st.session_state.history[-1]["content"]
        st.sidebar.download_button(
            "📄 Download antwoord als PDF",
            data=genereer_pdf(laatste),
            file_name="antwoord.pdf",
            mime="application/pdf"
        )

    # Product-keuze
    if not st.session_state.selected_product:
        st.header("Welkom bij IPAL Chatbox")
        c1, c2, c3 = st.columns(3)
        if c1.button("DocBase", use_container_width=True):
            st.session_state.selected_product = "DocBase"
            add_message("assistant", "Gekozen: DocBase")
            st.experimental_rerun()
        if c2.button("Exact", use_container_width=True):
            st.session_state.selected_product = "Exact"
            add_message("assistant", "Gekozen: Exact")
            st.experimental_rerun()
        if c3.button("Algemeen", use_container_width=True):
            st.session_state.selected_product = "Algemeen"
            st.session_state.selected_module = "alles"
            add_message("assistant", "Gekozen: Algemeen")
            st.experimental_rerun()
        render_chat()
        return

    # Module-keuze voor DocBase/Exact
    if st.session_state.selected_product != "Algemeen" and not st.session_state.selected_module:
        opties = sorted(
            faq_df[faq_df["Systeem"] == st.session_state.selected_product]["Subthema"]
            .dropna().unique()
        )
        sel = st.selectbox("Kies onderwerp:", ["(Kies)"] + opties)
        if sel != "(Kies)":
            st.session_state.selected_module = sel
            add_message("assistant", f"Gekozen: {sel}")
            st.experimental_rerun()
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
        st.experimental_rerun()

    with st.spinner("Even zoeken..."):
        # Bepaal subset FAQ
        if st.session_state.selected_product == "Algemeen":
            dfm = faq_df[faq_df["combined"].str.contains(vraag, case=False, na=False)]
        else:
            dfm = faq_df[
                (faq_df["Systeem"] == st.session_state.selected_product) &
                (faq_df["Subthema"].str.lower() == st.session_state.selected_module.lower())
            ]
        # FAQ-match
        if not dfm.empty:
            row = dfm.iloc[0]
            antwoord = row["Antwoord"]
            try:
                antwoord = rewrite_answer(antwoord)
            except:
                pass
            img = row.get("Afbeelding")
            if img and os.path.exists(img):
                st.image(img, caption="Voorbeeld", use_column_width=True)
            add_message("assistant", antwoord)
        else:
            # AI-fallback
            try:
                prompt = (
                    vraag
                    if st.session_state.selected_product == "Algemeen"
                    else f"[{st.session_state.selected_module}] {vraag}"
                )
                ai_ans = get_ai_answer(prompt)
                add_message("assistant", f"IPAL-Helpdesk antwoord:\n{ai_ans}")
            except Exception as e:
                logging.error(f"AI-fallback mislukt: {e}")
                add_message("assistant", "⚠️ Fout tijdens AI-fallback")

    st.experimental_rerun()

if __name__ == "__main__":
    main()
