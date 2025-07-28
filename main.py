# main.py

import os
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

from faq import load_faq
from ai_client import get_ai_answer, rewrite_answer
from filters import filter_chatbot_topics
from pdf_utils import genereer_pdf

# — Constants & Config —
MAX_HISTORY = 20
TIMEZONE = pytz.timezone("Europe/Amsterdam")
AVATARS = {
    "assistant": "aichatbox.jpg",
    "user": "parochie.jpg"
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Load env & API key (fallback to Streamlit secrets)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.sidebar.error("🔑 OpenAI API key niet gevonden. Voeg toe aan .env of Streamlit Secrets.")
    st.stop()
openai.api_key = openai_api_key

# — Load & cache FAQ —
@st.cache_data(show_spinner=False)
def get_faq_data(path: str = "faq.xlsx") -> pd.DataFrame:
    return load_faq(path)

faq_df = get_faq_data()

# Build product/module lookup
PRODUCTS = ["DocBase", "Exact", "Algemeen"]
subthema_dict = {
    p: sorted(faq_df[faq_df["Systeem"] == p]["Subthema"].dropna().unique())
    for p in PRODUCTS
}

# — Session state initialization —
def init_session():
    defaults = {
        "history": [],
        "selected_product": None,
        "selected_module": None
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

init_session()

# — Helper functions — 
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
        avatar = get_avatar(msg["role"])
        st.chat_message(msg["role"], avatar=avatar).markdown(
            f"{msg['content']}\n\n_{msg['time']}_"
        )

def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# — Main app —
def main():
    # Sidebar controls
    if st.sidebar.button("🔄 Nieuw gesprek"):
        reset_session()

    # Download last assistant reply as PDF
    if st.session_state.history and st.session_state.history[-1]["role"] == "assistant":
        laatste = st.session_state.history[-1]["content"]
        pdf_bytes = genereer_pdf(laatste)
        st.sidebar.download_button(
            "📄 Download laatste antwoord als PDF",
            data=pdf_bytes,
            file_name="antwoord.pdf",
            mime="application/pdf"
        )

    # Product selection screen
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

    # Module selection for non-Algemeen products
    if st.session_state.selected_product != "Algemeen" and not st.session_state.selected_module:
        opts = subthema_dict.get(st.session_state.selected_product, [])
        choice = st.selectbox("Kies onderwerp:", ["(Kies)"] + opts)
        if choice != "(Kies)":
            st.session_state.selected_module = choice
            add_message("assistant", f"Gekozen: {choice}")
            st.experimental_rerun()
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
        st.experimental_rerun()

    with st.spinner("Even zoeken..."):
        # Algemene FAQ lookup
        if st.session_state.selected_product == "Algemeen":
            dfm = faq_df[faq_df["combined"].str.contains(vraag, case=False, na=False)]
            if not dfm.empty:
                row = dfm.iloc[0]
                ans = row["Antwoord"]
                try:
                    ans = rewrite_answer(ans)
                except:
                    pass
                img = row.get("Afbeelding")
                if img and os.path.exists(img):
                    st.image(img, caption="Voorbeeld", use_column_width=True)
                add_message("assistant", ans)
            else:
                try:
                    ans = get_ai_answer(vraag)
                    add_message("assistant", ans)
                except Exception as e:
                    logging.error(f"AI-fallback mislukt: {e}")
                    add_message("assistant", "⚠️ Fout tijdens AI-fallback")
        # Product-specific FAQ & AI
        else:
            dfm = faq_df[faq_df["Subthema"].str.lower() == st.session_state.selected_module.lower()]
            matches = dfm[dfm["combined"].str.contains(vraag, case=False, na=False)]
            if not matches.empty:
                row = matches.iloc[0]
                ans = row["Antwoord"]
                try:
                    ans = rewrite_answer(ans)
                except:
                    pass
                img = row.get("Afbeelding")
                if img and os.path.exists(img):
                    st.image(img, caption="Voorbeeld", use_column_width=True)
                add_message("assistant", ans)
            else:
                try:
                    ai_ans = get_ai_answer(f"[{st.session_state.selected_module}] {vraag}")
                    add_message("assistant", f"IPAL-Helpdesk antwoord:\n{ai_ans}")
                except Exception as e:
                    logging.error(f"AI-fallback mislukt: {e}")
                    add_message("assistant", "⚠️ Fout tijdens AI-fallback")
    st.experimental_rerun()

if __name__ == "__main__":
    main()
