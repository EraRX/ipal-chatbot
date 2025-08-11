"""
IPAL Chatbox voor oudere vrijwilligers
- Python 3, Streamlit
- Groot lettertype, eenvoudige bediening
- Strikte cascade uit CSV: Systeem → Subthema → Categorie
- Daarna kies je één record (antwoord) en kun je daarover doorvragen
- AI‑QA optioneel (toggle). Zonder AI: deterministische zinnen‑match
- Logging, foutafhandeling, PDF‑export
"""

import os
import re
import io
import logging
from datetime import datetime
from collections import defaultdict
from typing import Optional

import streamlit as st
import pandas as pd
import pytz
import requests
from bs4 import BeautifulSoup
from PIL import Image as PILImage
from dotenv import load_dotenv
from openai import OpenAI

try:
    from openai.error import RateLimitError
except ImportError:
    RateLimitError = Exception

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem, Table, TableStyle
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ──────────────────────────────────────────────────────────────────────────────
# UI-config (moet als eerste Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="IPAL Chatbox", layout="centered")
st.markdown(
    """
    <style>
    html, body, [class*=\"css\"] { font-size:20px; }
    button[kind=\"primary\"] { font-size:22px !important; padding:.75em 1.5em; }
    video { width: 600px !important; height: auto !important; max-width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI setup (alleen nodig als AI‑QA aan staat)
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(1, 10), retry=retry_if_exception_type(RateLimitError))
@st.cache_data(show_spinner=False)
def chatgpt_cached(messages, temperature=0.2, max_tokens=900):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# ──────────────────────────────────────────────────────────────────────────────
# PDF helpers
# ──────────────────────────────────────────────────────────────────────────────
if os.path.exists("Calibri.ttf"):
    pdfmetrics.registerFont(TTFont("Calibri", "Calibri.ttf"))
else:
    logging.info("Calibri.ttf niet gevonden, gebruik ingebouwde Helvetica")

def _strip_md(s: str) -> str:
    # Gebruik lambda i.p.v. backrefs (voorkomt canvas/regex issues)
    s = re.sub(r"\*\*([^*]+)\*\*", lambda m: m.group(1), s)
    s = re.sub(r"#+\s*([^\n]+)", lambda m: m.group(1), s)
    return s

def make_pdf(question: str, answer: str) -> bytes:
    answer = _strip_md(answer)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=16, spaceAfter=12, alignment=TA_LEFT)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#333333"), spaceBefore=12, spaceAfter=6)
    bullet_style = ParagraphStyle("Bullet", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leftIndent=12, bulletIndent=0, leading=16)

    story = []
    if os.path.exists("logopdf.png"):
        logo = Image("logopdf.png", width=124, height=52)
        logo_table = Table([[logo]], colWidths=[124])
        logo_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(logo_table)

    story.append(Paragraph(f"Vraag: {question}", heading_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Antwoord:", heading_style))

    avatar_path = "aichatbox.png"
    if os.path.exists(avatar_path):
        avatar = Image(avatar_path, width=30, height=30)
        first_line, *rest = answer.split("\n")
        intro_text = Paragraph(first_line, body_style)
        story.append(Table([[avatar, intro_text]], colWidths=[30, 440], style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")])))
        story.append(Spacer(1, 12))
        for line in rest:
            line = line.strip()
            if not line:
                continue
            if line.startswith(("•", "-")):
                bullets = ListFlowable([ListItem(Paragraph(line[1:].strip(), bullet_style))], bulletType="bullet")
                story.append(bullets)
            else:
                story.append(Paragraph(line, body_style))
    else:
        for line in answer.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith(("•", "-")):
                bullets = ListFlowable([ListItem(Paragraph(line[1:].strip(), bullet_style))], bulletType="bullet")
                story.append(bullets)
            else:
                story.append(Paragraph(line, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ──────────────────────────────────────────────────────────────────────────────
# CSV laden → 3‑laags MultiIndex: (Systeem, Subthema, Categorie)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_faq(path: str = "faq.csv") -> pd.DataFrame:
    cols = [
        "ID","Systeem","Subthema","Categorie","Omschrijving melding","Toelichting melding","Soort melding","Antwoord of oplossing","Afbeelding",
    ]
    if not os.path.exists(path):
        logging.error(f"FAQ niet gevonden: {path}")
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=cols).set_index(["Systeem","Subthema","Categorie"])  
    try:
        df = pd.read_csv(path, encoding="utf-8", sep=";")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="windows-1252", sep=";")

    # kolommen aanvullen indien ontbreken
    required = {"Systeem","Subthema","Categorie","Omschrijving melding","Antwoord of oplossing"}
    missing = required - set(df.columns)
    for col in missing:
        df[col] = ""

    if "Afbeelding" not in df.columns:
        df["Afbeelding"] = None
    if "Toelichting melding" not in df.columns:
        df["Toelichting melding"] = ""

    # combined voor simpele ranking/preview
    keep_cols = ["Systeem","Subthema","Categorie","Omschrijving melding","Toelichting melding"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = ""
    df["combined"] = df[keep_cols].fillna("").agg(" ".join, axis=1)

    return df.set_index(["Systeem","Subthema","Categorie"], drop=True)

faq_df = load_faq()
PRODUCTEN = ["Exact","DocBase"]

# ──────────────────────────────────────────────────────────────────────────────
# Keuzelijsten per laag
# ──────────────────────────────────────────────────────────────────────────────
def build_subthema_dict(df: pd.DataFrame) -> dict:
    sub = defaultdict(list)
    for systeem in PRODUCTEN:
        try:
            subset = df.xs(systeem, level="Systeem", drop_level=False)
            sub[systeem] = sorted(subset.index.get_level_values("Subthema").dropna().unique())
        except KeyError:
            logging.warning(f"Geen subthema's gevonden voor: {systeem}")
            sub[systeem] = []
    return sub

subthema_dict = build_subthema_dict(faq_df)

@st.cache_data(show_spinner=False)
def list_categorieen(systeem: str, subthema: str) -> list:
    try:
        subset = faq_df.xs((systeem, subthema), level=["Systeem","Subthema"], drop_level=False)
        return sorted(subset.index.get_level_values("Categorie").dropna().unique())
    except KeyError:
        return []

# ──────────────────────────────────────────────────────────────────────────────
# Veiligheidsfilters
# ──────────────────────────────────────────────────────────────────────────────
BLACKLIST = ["persoonlijke gegevens","medische gegevens","gezondheid","privacy schending"]

def filter_topics(msg: str):
    found = [t for t in BLACKLIST if re.search(r"\b" + re.escape(t) + r"\b", msg.lower())]
    return (False, f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}.") if found else (True, "")

# ──────────────────────────────────────────────────────────────────────────────
# Eenvoudige QA zonder AI: kies zinnen die de vraagwoorden bevatten
# ──────────────────────────────────────────────────────────────────────────────
def _token_score(q: str, text: str) -> int:
    qs = [w for w in re.findall(r"\w+", q.lower()) if len(w) > 2]
    ts = set(re.findall(r"\w+", str(text).lower()))
    return sum(1 for w in qs if w in ts)

def antwoord_qna(zinnen_bron: str, vraag: str, max_zinnen: int = 3) -> str:
    zinnen = re.split(r"(?<=[.!?])\s+", str(zinnen_bron))
    scores = [(_token_score(vraag, z), z) for z in zinnen]
    scores.sort(key=lambda x: x[0], reverse=True)
    top = [z for s, z in scores if s > 0][:max_zinnen]
    return "\n".join(top) if top else "Ik kan zonder AI geen betere toelichting uit het gekozen antwoord halen. Zet eventueel AI‑QA aan in de sidebar."

# ──────────────────────────────────────────────────────────────────────────────
# UI helpers & state
# ──────────────────────────────────────────────────────────────────────────────
AI_INFO = (
    "AI-Antwoord Info:\n"
    "1. Dit antwoord is afkomstig uit de IPAL chatbox. Controleer bij twijfel altijd de officiële documentatie.\n"
    "2. Hulp nodig met DocBase of Exact? Maak een ticket aan (bekijk eerst de FAQ)."
)

ASSISTANT_AVATAR = "aichatbox.png" if os.path.exists("aichatbox.png") else None
USER_AVATAR = "parochie.png" if os.path.exists("parochie.png") else None
TIMEZONE = pytz.timezone("Europe/Amsterdam")
MAX_HISTORY = 12


def get_avatar(role: str):
    return ASSISTANT_AVATAR if role == "assistant" and ASSISTANT_AVATAR else USER_AVATAR


def add_msg(role: str, content: str):
    ts = datetime.now(TIMEZONE).strftime("%d-%m-%Y %H:%M")
    st.session_state.history = (
        st.session_state.history + [{"role": role, "content": content, "time": ts}]
    )[-MAX_HISTORY:]


def render_chat():
    for i, m in enumerate(st.session_state.history):
        st.chat_message(m["role"], avatar=get_avatar(m["role"]))\
            .markdown(f"{m['content']}\n\n_{m['time']}_")
        if m["role"] == "assistant" and i == len(st.session_state.history) - 1 and st.session_state.last_question:
            pdf_data = make_pdf(st.session_state.last_question, m["content"])
            st.download_button("📄 Download PDF", data=pdf_data, file_name="antwoord.pdf", mime="application/pdf")

# init session
DEFAULT_STATE = {
    "history": [],
    "selected_product": None,
    "selected_module": None,
    "selected_category": None,
    "selected_answer_id": None,
    "selected_answer_text": None,
    "last_question": "",
    "debug": False,
    "allow_ai": False,
}
for _k, _v in DEFAULT_STATE.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

def main():
    with st.sidebar:
        if st.button("🔄 Nieuw gesprek", use_container_width=True):
            st.session_state.clear(); st.rerun()
        st.session_state.debug = st.toggle("Debug info", value=st.session_state.get("debug", False))
        st.session_state.allow_ai = st.toggle("AI‑QA aan", value=st.session_state.get("allow_ai", False))
        if st.session_state.allow_ai and not OPENAI_KEY:
            st.warning("⚠️ AI‑QA staat aan maar er is geen OPENAI_API_KEY ingesteld.")
        if st.session_state.debug:
            try:
                c_exact = len(faq_df.xs("Exact", level="Systeem", drop_level=False))
            except Exception:
                c_exact = 0
            try:
                c_doc = len(faq_df.xs("DocBase", level="Systeem", drop_level=False))
            except Exception:
                c_doc = 0
            st.caption(f"CSV records: {len(faq_df)} | Exact: {c_exact} | DocBase: {c_doc}")

    # Startscherm (systeemkeuze)
    if not st.session_state.get("selected_product"):
        if os.path.exists("logo.png"):
            st.image("logo.png", width=244)
        st.header("Welkom bij IPAL Chatbox")
        c1, c2, c3 = st.columns(3)
        if c1.button("Exact", use_container_width=True):
            st.session_state.selected_product = "Exact"; add_msg("assistant", "Gekozen: Exact"); st.rerun()
        if c2.button("DocBase", use_container_width=True):
            st.session_state.selected_product = "DocBase"; add_msg("assistant", "Gekozen: DocBase"); st.rerun()
        if c3.button("Algemeen", use_container_width=True):
            st.session_state.selected_product = "Algemeen"; st.session_state.selected_module = "alles"; st.session_state.selected_category = "alles"; add_msg("assistant", "Gekozen: Algemeen"); st.rerun()
        render_chat(); return

    # 1) Subthema
    if st.session_state.selected_product in PRODUCTEN and not st.session_state.selected_module:
        opts = subthema_dict.get(st.session_state.selected_product, [])
        sel = st.selectbox("Kies subthema:", ["(Kies)"] + list(opts))
        if sel != "(Kies)":
            st.session_state.selected_module = sel
            add_msg("assistant", f"Gekozen subthema: {sel}")
            st.rerun()
        render_chat(); return

    # 2) Categorie
    if st.session_state.get("selected_product") in PRODUCTEN and st.session_state.get("selected_module") and not st.session_state.get("selected_category"): 
        cats = list_categorieen(st.session_state.selected_product, st.session_state.selected_module)
        selc = st.selectbox("Kies categorie:", ["(Kies)"] + list(cats))
        if selc != "(Kies)":
            st.session_state.selected_category = selc
            add_msg("assistant", f"Gekozen categorie: {selc}")
            st.rerun()
        render_chat(); return

    # 3) Record (antwoord) kiezen binnen scope
    df_scope = faq_df
    syst = st.session_state.selected_product
    sub = st.session_state.selected_module
    cat = st.session_state.selected_category
    try:
        if syst in PRODUCTEN:
            df_scope = df_scope.xs(syst, level="Systeem", drop_level=False)
        if sub and sub != "alles":
            df_scope = df_scope.xs(sub, level="Subthema", drop_level=False)
        if cat and cat != "alles":
            df_scope = df_scope.xs(cat, level="Categorie", drop_level=False)
    except KeyError:
        df_scope = pd.DataFrame(columns=faq_df.reset_index().columns)

    render_chat()

    if not df_scope.empty:
        df_reset = df_scope.reset_index()
        def mk_label(i, row):
            oms = str(row.get("Omschrijving melding", "")).strip()
            toel = str(row.get("Toelichting melding", "")).strip()
            preview = (oms or toel or str(row.get("Antwoord of oplossing", "")).strip())
            preview = re.sub(r"\s+", " ", preview)[:140]
            return f"{i+1:02d}. {preview}"
        opties = [mk_label(i, r) for i, r in df_reset.iterrows()]
        keuze = st.selectbox("Kies een item:", ["(Kies)"] + opties)
        if keuze != "(Kies)":
            i = int(keuze.split(".")[0]) - 1
            row = df_reset.iloc[i]
            st.session_state.selected_answer_id = row.get("ID", i)
            st.session_state.selected_answer_text = row.get("Antwoord of oplossing", "")
            add_msg("assistant", st.session_state.selected_answer_text + "\n\n" + AI_INFO)
            st.rerun()
    else:
        st.info("Geen records gevonden binnen de gekozen Systeem/Subthema/Categorie.")

    # 4) Doorvraag over het gekozen antwoord
    vraag = st.chat_input("Stel uw vraag (over het gekozen antwoord):")
    if not vraag:
        return

    st.session_state.last_question = vraag
    add_msg("user", vraag)

    if not st.session_state.get("selected_answer_text"): 
        add_msg("assistant", "Kies eerst een item in de lijst hierboven.")
        st.rerun(); return

    if st.session_state.allow_ai and client is not None:
        prompt = [
            {"role": "system", "content": "Beantwoord uitsluitend op basis van de meegegeven bron. Geen aannames buiten de bron. Schrijf kort en duidelijk in het Nederlands."},
            {"role": "user", "content": f"Bron (antwoord uit CSV):\n{st.session_state.selected_answer_text}\n\nVraag: {vraag}"},
        ]
        try:
            reactie = chatgpt_cached(prompt, temperature=0.1, max_tokens=600)
        except Exception as e:
            logging.error(f"AI QA fout: {e}")
            reactie = antwoord_qna(st.session_state.selected_answer_text, vraag)
    else:
        reactie = antwoord_qna(st.session_state.selected_answer_text, vraag)

    add_msg("assistant", reactie + "\n\n" + AI_INFO)
    st.rerun()


if __name__ == "__main__":
    main()
