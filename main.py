"""
IPAL Chatbox voor oudere vrijwilligers
- Python 3, Streamlit
- Groot lettertype, eenvoudige bediening
- Antwoorden uitsluitend uit CSV per gekozen systeem (Exact of DocBase); geen vermenging
- Optionele fallback: web/AI (uit te zetten in sidebar; standaard UIT)
- Topicfiltering (blacklist)
- Logging en foutafhandeling
- Antwoorden downloaden als PDF
"""

import os
import re
import io
import logging
from datetime import datetime
from collections import defaultdict

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
    html, body, [class*="css"] { font-size:20px; }
    button[kind="primary"] { font-size:22px !important; padding:.75em 1.5em; }
    video { width: 600px !important; height: auto !important; max-width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI setup
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not OPENAI_KEY:
    st.sidebar.error("🔑 Voeg je OpenAI API-key toe in .env of Secrets.")
    st.stop()
client = OpenAI(api_key=OPENAI_KEY)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(1, 10),
    retry=retry_if_exception_type(RateLimitError),
)
@st.cache_data(show_spinner=False)
def chatgpt_cached(messages, temperature=0.2, max_tokens=1100):
    """Kleine wrapper met cache om identieke prompts niet dubbel te laten lopen."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# ──────────────────────────────────────────────────────────────────────────────
# PDF
# ──────────────────────────────────────────────────────────────────────────────
if os.path.exists("Calibri.ttf"):
    pdfmetrics.registerFont(TTFont("Calibri", "Calibri.ttf"))
else:
    logging.info("Calibri.ttf niet gevonden, gebruik ingebouwde Helvetica")

def _strip_md(s: str) -> str:
    s = re.sub(r"\*\*([^\*]+)\*\*", r"\1", s)   # **bold** → plain
    s = re.sub(r"#+\s*([^\n]+)", r"\1", s)        # # heading → plain
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
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=16, spaceAfter=12, alignment=TA_LEFT
    )
    heading_style = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#333333"),
        spaceBefore=12,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "Bullet", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leftIndent=12, bulletIndent=0, leading=16
    )

    story = []
    if os.path.exists("logopdf.png"):
        logo = Image("logopdf.png", width=124, height=52)
        logo_table = Table([[logo]], colWidths=[124])
        logo_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
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
# Data laden
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_faq(path: str = "faq.csv") -> pd.DataFrame:
    cols = [
        "ID",
        "Systeem",
        "Subthema",
        "Categorie",
        "Omschrijving melding",
        "Toelichting melding",
        "Soort melding",
        "Antwoord of oplossing",
        "Afbeelding",
    ]
    if not os.path.exists(path):
        logging.error(f"FAQ niet gevonden: {path}")
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        # Altijd een MultiIndex teruggeven om downstream fouten te voorkomen
        return pd.DataFrame(columns=cols).set_index(["Systeem", "Subthema"])  
    try:
        df = pd.read_csv(path, encoding="utf-8", sep=";")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="windows-1252", sep=";")

    # Verplichte kolommen aanvullen indien ontbrekend
    required = {"Systeem", "Subthema", "Omschrijving melding", "Antwoord of oplossing"}
    missing = required - set(df.columns)
    for col in missing:
        df[col] = ""

    if "Afbeelding" not in df.columns:
        df["Afbeelding"] = None
    if "Toelichting melding" not in df.columns:
        df["Toelichting melding"] = ""

    # Combined-veld voor simpele ranking
    for c in ["Systeem", "Subthema", "Omschrijving melding", "Toelichting melding"]:
        if c not in df.columns:
            df[c] = ""
    df["combined"] = df[["Systeem", "Subthema", "Omschrijving melding", "Toelichting melding"]].fillna("").agg(" ".join, axis=1)

    # MultiIndex op Systeem/Subthema
    df = df.set_index(["Systeem", "Subthema"], drop=True)
    return df

faq_df = load_faq()
PRODUCTEN = ["Exact", "DocBase"]

# ──────────────────────────────────────────────────────────────────────────────
# Subthema's strikt per systeem (alleen tonen wat beschikbaar is)
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

# ──────────────────────────────────────────────────────────────────────────────
# Veiligheidsfilters
# ──────────────────────────────────────────────────────────────────────────────
BLACKLIST = [
    "persoonlijke gegevens",
    "medische gegevens",
    "gezondheid",
    "privacy schending",
]

def filter_topics(msg: str):
    found = [t for t in BLACKLIST if re.search(r"\b" + re.escape(t) + r"\b", msg.lower())]
    return (False, f"Je bericht bevat gevoelige onderwerpen: {', '.join(found)}.") if found else (True, "")

# ──────────────────────────────────────────────────────────────────────────────
# Externe bronnen (optioneel, via toggles)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_web_info_cached(query: str):
    result = []
    try:
        r = requests.get("https://docbase.nl", timeout=5)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = " ".join([p.get_text(strip=True) for p in soup.find_all(["p", "h1", "h2", "h3"])])
        if text and query.lower() in text.lower():
            result.append(f"Vanuit docbase.nl: {text[:200]}... (verkort)")
    except Exception as e:
        logging.info(f"Kon docbase.nl niet ophalen: {e}")
    try:
        r = requests.get("https://support.exactonline.com/community/s/knowledge-base", timeout=5)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = " ".join([p.get_text(strip=True) for p in soup.find_all(["p", "h1", "h2", "h3"])])
        if text and query.lower() in text.lower():
            result.append(f"Vanuit Exact Online Knowledge Base: {text[:200]}... (verkort)")
    except Exception as e:
        logging.info(f"Kon Exact Online Knowledge Base niet ophalen: {e}")
    return "\n".join(result) if result else None

# ──────────────────────────────────────────────────────────────────────────────
# Ranking helpers (deterministisch, uitlegbaar)
# ──────────────────────────────────────────────────────────────────────────────
def _token_score(q: str, text: str) -> int:
    qs = [w for w in re.findall(r"\w+", q.lower()) if len(w) > 2]
    ts = set(re.findall(r"\w+", str(text).lower()))
    return sum(1 for w in qs if w in ts)

# ──────────────────────────────────────────────────────────────────────────────
# Kern-zoekfunctie: strikt binnen gekozen systeem (+ optioneel subthema)
# ──────────────────────────────────────────────────────────────────────────────
def vind_best_passend_antwoord(vraag: str, systeem: str, subthema: str | None) -> str | None:
    try:
        # 1) scope: altijd eerst filteren op gekozen systeem
        try:
            df_sys = faq_df.xs(systeem, level="Systeem", drop_level=False)
        except KeyError:
            logging.warning(f"Geen data voor systeem: {systeem}")
            return None

        # 2) optioneel subthema
        if subthema and subthema != "alles":
            try:
                df_mod = df_sys.xs(subthema, level="Subthema", drop_level=False)
            except KeyError:
                logging.info(f"Geen data voor subthema '{subthema}' binnen {systeem}; val terug op alle {systeem}-items")
                df_mod = df_sys
        else:
            df_mod = df_sys

        if df_mod.empty:
            return None

        # 3) exacte match op Omschrijving melding binnen scope
        vraag_norm = vraag.strip().lower()
        df_reset = df_mod.reset_index()
        exact = df_reset[df_reset["Omschrijving melding"].astype(str).str.strip().str.lower() == vraag_norm]
        if not exact.empty:
            return exact.iloc[0]["Antwoord of oplossing"]

        # 4) token-overlap ranking binnen scope
        cand = df_mod.reset_index().copy()
        cand["_score"] = cand["combined"].apply(lambda t: _token_score(vraag, t))
        cand = cand.sort_values("_score", ascending=False)
        top = cand.iloc[0]
        return top["Antwoord of oplossing"] if top["_score"] > 0 else None
    except Exception as e:
        logging.error(f"Error in vind_best_passend_antwoord: {e}")
        return None

# ──────────────────────────────────────────────────────────────────────────────
# UI helper
# ──────────────────────────────────────────────────────────────────────────────
AI_INFO = (
    "AI-Antwoord Info:\n"
    "1. Dit antwoord is afkomstig uit de IPAL chatbox. Controleer bij twijfel altijd de officiële documentatie.\n"
    "2. Hulp nodig met DocBase of Exact? Maak een ticket aan (bekijk eerst de FAQ)."
)

ASSISTANT_AVATAR = "aichatbox.png" if os.path.exists("aichatbox.png") else None
USER_AVATAR = "parochie.png" if os.path.exists("parochie.png") else None

TIMEZONE = pytz.timezone("Europe/Amsterdam")
MAX_HISTORY = 10


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
        if (
            m["role"] == "assistant"
            and i == len(st.session_state.history) - 1
            and st.session_state.last_question
        ):
            pdf_data = make_pdf(st.session_state.last_question, m["content"])
            st.download_button(
                "📄 Download PDF", data=pdf_data, file_name="antwoord.pdf", mime="application/pdf"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Sessie init
# ──────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.selected_product = None
    st.session_state.selected_module = None
    st.session_state.last_question = ""
    st.session_state.debug = False
    st.session_state.allow_web = False    # standaard UIT
    st.session_state.allow_ai = False     # standaard UIT


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

def main():
    with st.sidebar:
        if st.button("🔄 Nieuw gesprek", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        st.session_state.debug = st.toggle("Debug info", value=st.session_state.get("debug", False))
        st.session_state.allow_web = st.toggle("Sta web-fallback toe", value=st.session_state.get("allow_web", False))
        st.session_state.allow_ai = st.toggle("Sta AI-fallback toe", value=st.session_state.get("allow_ai", False))
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

    # Startscherm (productkeuze)
    if not st.session_state.get("selected_product"):
        video_path = "helpdesk.mp4"
        if os.path.exists(video_path):
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            try:
                st.video(video_bytes, format="video/mp4", start_time=0, autoplay=True)
            except Exception as e:
                logging.error(f"Video kon niet worden afgespeeld: {e}")
        elif os.path.exists("logo.png"):
            st.image("logo.png", width=244)

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
            st.session_state.selected_module = "alles"
            add_msg("assistant", "Gekozen: Algemeen")
            st.rerun()
        render_chat()
        return

    # Modulekeuze binnen gekozen product
    if st.session_state.selected_product in PRODUCTEN and not st.session_state.selected_module:
        opts = subthema_dict.get(st.session_state.selected_product, [])
        sel = st.selectbox("Kies onderwerp:", ["(Kies)"] + list(opts))
        if sel != "(Kies)":
            st.session_state.selected_module = sel
            add_msg(
                "assistant",
                f"Gekozen: {st.session_state.selected_product} (Module: {sel})",
            )
            st.rerun()
        render_chat()
        return

    # Chat UI
    render_chat()

    vraag = st.chat_input("Stel uw vraag:")
    if not vraag:
        return

    # UNIEK codewoord → direct antwoord uit CSV
    if vraag.strip().upper() == "UNIEKECODE123":
        df_reset = faq_df.reset_index()
        mask = df_reset["Antwoord of oplossing"].astype(str).str.contains(r"\[UNIEKECODE123\]", case=False, na=False)
        if mask.any():
            antwoord = df_reset.loc[mask].iloc[0]["Antwoord of oplossing"]
            add_msg("user", vraag)
            add_msg("assistant", antwoord + f"\n\n{AI_INFO}")
            st.rerun()
        return

    # Bewaar voor PDF
    st.session_state.last_question = vraag
    add_msg("user", vraag)

    # Veiligheidsfilter
    ok, warn = filter_topics(vraag)
    if not ok:
        add_msg("assistant", warn)
        st.rerun()
        return

    systeem = st.session_state.selected_product
    subthema = st.session_state.selected_module

    antwoord = None

    # STRIKT: binnen gekozen systeem
    if systeem in PRODUCTEN:
        antwoord = vind_best_passend_antwoord(vraag, systeem, subthema)
    else:
        # Algemeen → kies beste uit hele CSV
        df_all = faq_df.reset_index().copy()
        vraag_norm = vraag.strip().lower()
        exact = df_all[df_all["Omschrijving melding"].astype(str).str.strip().str.lower() == vraag_norm]
        if not exact.empty:
            antwoord = exact.iloc[0]["Antwoord of oplossing"]
        if not antwoord and not df_all.empty:
            df_all["_score"] = df_all["combined"].apply(lambda t: _token_score(vraag, t))
            df_all = df_all.sort_values("_score", ascending=False)
            top = df_all.iloc[0]
            antwoord = top["Antwoord of oplossing"] if top["_score"] > 0 else None

    # Optionele fallbacks (bewust opt-in)
    if not antwoord and st.session_state.allow_web:
        webbits = fetch_web_info_cached(vraag)
        if webbits:
            antwoord = webbits

    if not antwoord and st.session_state.allow_ai:
        prompt = [
            {"role": "system", "content": "Je bent een helpdeskassistent voor parochies. Geef korte, stap-voor-stap hulp in duidelijke, eenvoudige taal. Wees eerlijk wanneer je het niet zeker weet."},
            {"role": "user", "content": f"Vraag: {vraag}\nContext: Systeem={systeem}, Subthema={subthema}. Als je het niet zeker weet, zeg dat en geef 1-3 praktische suggesties."},
        ]
        try:
            antwoord = chatgpt_cached(prompt)
            antwoord = _strip_md(antwoord)
        except Exception as e:
            logging.error(f"OpenAI fout: {e}")

    add_msg("assistant", (antwoord or "Ik vond geen passend antwoord in de CSV. Probeer uw vraag specifieker te formuleren of kies een ander subthema.") + "\n\n" + AI_INFO)
    st.rerun()


if __name__ == "__main__":
    main()
