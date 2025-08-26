# IPAL Chatbox — main.py
# - Chat-wizard met 4 knoppen: Exact | DocBase | Zoeken | Internet
# - 1→7-cascade staat ALLEEN in het expander-paneel (niet erbuiten)
# - Geen sidebar
# - Intro-video bovenaan
# - CSV robuust inlezen + slimme quotes-fix
# - PDF-download + "Kopieer antwoord" voor het laatste resultaat

import os
import re
import io
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional, List

import streamlit as st
import pandas as pd
import pytz
from dotenv import load_dotenv
from openai import OpenAI
import streamlit.components.v1 as components

# Web-fallback
import requests
from bs4 import BeautifulSoup

try:
    from openai.error import RateLimitError
except ImportError:
    RateLimitError = Exception

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# PDF
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ── UI-config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="IPAL Chatbox", layout="centered", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-size:20px; }
      button[kind="primary"] { font-size:22px !important; padding:.75em 1.5em; }
      video { width: 600px !important; height: auto !important; max-width: 100%; }

      /* Sidebar en hamburger verbergen */
      [data-testid="stSidebar"],
      [data-testid="stSidebarNav"],
      [data-testid="collapsedControl"] { display: none !important; }

      /* Optioneel: standaard menu/footers weg */
      #MainMenu { visibility: hidden; }
      footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ── OpenAI (optioneel) ───────────────────────────────────────────────────────
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(1, 8), retry=retry_if_exception_type(RateLimitError))
@st.cache_data(show_spinner=False)
def chatgpt_cached(messages, temperature=0.2, max_tokens=700) -> str:
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()


# ── Helpers: tekst opschonen ────────────────────────────────────────────────
def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u00A0", " ")
    repl = {
        "\u0091": "'", "\u0092": "'", "\u0093": '"', "\u0094": '"',
        "\u0096": "-", "\u0097": "-", "\u0085": "...",
        "\u2018": "'", "\u2019": "'", "\u201A": ",", "\u201B": "'",
        "\u201C": '"', "\u201D": '"', "\u201E": '"',
        "\u00AB": '"', "\u00BB": '"', "\u2039": "'", "\u203A": "'",
        "\u2013": "-", "\u2014": "-", "\u2212": "-",
        "\u00AD": "", "\u2026": "...",
        "\u00B4": "'", "\u02BC": "'", "\u02BB": "'",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ── PDF/AI-INFO ─────────────────────────────────────────────────────────────
FAQ_LINKS = [
    ("Veelgestelde vragen DocBase nieuw 2024", "https://parochie-automatisering.nl/docbase/Templates/docbase?action=SelOpenDocument&DetailsMode=2&Docname=00328526&Type=INSTR_DOCS&LoginMode=1&LinkToVersion=1&OpenFileMode=2&password=%3Auzt7hs%23qL%2A%28&username=Externehyperlink&ID=0.07961651005089099&EC=1"),
    ("Veelgestelde vragen Exact Online", "https://parochie-automatisering.nl/docbase/Templates/docbase?action=SelOpenDocument&DetailsMode=2&Docname=00328522&Type=INSTR_DOCS&LoginMode=1&LinkToVersion=1&OpenFileMode=2&password=%3Auzt7hs%23qL%2A%28&username=Externehyperlink&ID=0.8756321684738348&EC=1"),
]

AI_INFO = """
AI-Antwoord Info:  
1. Dit is het AI-antwoord vanuit de IPAL chatbox van het Interdiocesaan Platform Automatisering & Ledenadministratie. Het is altijd een goed idee om de meest recente informatie te controleren via officiële bronnen.  
2. Heeft u hulp nodig met DocBase of Exact? Dan kunt u eenvoudig een ticket aanmaken in DocBase. Controleer vóór het invullen eerst onze FAQ (veelgestelde vragen en antwoorden). Klik hieronder om de FAQ te openen:

- [Veelgestelde vragen DocBase nieuw 2024](https://parochie-automatisering.nl/docbase/Templates/docbase?action=SelOpenDocument&DetailsMode=2&Docname=00328526&Type=INSTR_DOCS&LoginMode=1&LinkToVersion=1&OpenFileMode=2&password=%3Auzt7hs%23qL%2A%28&username=Externehyperlink&ID=0.07961651005089099&EC=1)
- [Veelgestelde vragen Exact Online](https://parochie-automatisering.nl/docbase/Templates/docbase?action=SelOpenDocument&DetailsMode=2&Docname=00328522&Type=INSTR_DOCS&LoginMode=1&LinkToVersion=1&OpenFileMode=2&password=%3Auzt7hs%23qL%2A%28&username=Externehyperlink&ID=0.8756321684738348&EC=1)
"""

if os.path.exists("Calibri.ttf"):
    pdfmetrics.registerFont(TTFont("Calibri", "Calibri.ttf"))

def _strip_md(s: str) -> str:
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"#+\s*([^\n]+)", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    return s

def _parse_ai_info(ai_info: str) -> tuple[list[str], bool]:
    numbered: list[str] = []
    show_click = False
    for raw in (ai_info or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("- ["):
            break
        if line.lower().startswith("ai-antwoord info"):
            continue
        m = re.match(r"^(\d+)\.\s*(.*)$", line)
        if m:
            txt = m.group(2)
            key = "Klik hieronder om de FAQ te openen"
            if key in txt:
                txt = txt.split(key, 1)[0].strip()
                show_click = True
            txt = clean_text(_strip_md(txt))
            if txt:
                numbered.append(txt)
        else:
            if numbered:
                extra = clean_text(_strip_md(line))
                if extra:
                    numbered[-1] += " + " + extra
    if not show_click and "Klik hieronder om de FAQ te openen" in (ai_info or ""):
        show_click = True
    return numbered, show_click

def make_pdf(question: str, answer: str) -> bytes:
    question = clean_text(question or "")
    answer = clean_text(_strip_md(answer or ""))

    numbered_items, show_click = _parse_ai_info(AI_INFO)

    buffer = io.BytesIO()
    left = right = top = bottom = 2 * cm
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=left, rightMargin=right, topMargin=top, bottomMargin=bottom
    )
    content_width = A4[0] - left - right

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontName="Helvetica",
        fontSize=11, leading=16, spaceAfter=12, alignment=TA_LEFT
    )
    heading_style = ParagraphStyle(
        "Heading", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=14, leading=18, textColor=colors.HexColor("#333"),
        spaceBefore=12, spaceAfter=6
    )

    story = []
    if os.path.exists("logopdf.png"):
        try:
            banner = Image("logopdf.png")
            banner._restrictSize(content_width, 10000)
            banner.hAlign = "LEFT"
            story.append(banner)
            story.append(Spacer(1, 8))
        except Exception as e:
            logging.error(f"Kon banner niet laden: {e}")

    story.append(Paragraph(f"Vraag: {question}", heading_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Antwoord:", heading_style))
    for line in (answer.split("\n") if answer else []):
        line = line.strip()
        if line:
            story.append(Paragraph(line, body_style))

    if numbered_items:
        story.append(Spacer(1, 12))
        story.append(Paragraph("AI-Antwoord Info:", heading_style))
        list_items = [ListItem(Paragraph(item, body_style), leftIndent=12) for item in numbered_items]
        story.append(ListFlowable(list_items, bulletType="1"))

    story.append(Spacer(1, 12))
    if show_click:
        story.append(Paragraph("Klik hieronder om de FAQ te openen:", heading_style))

    link_items = []
    for label, url in FAQ_LINKS:
        p = Paragraph(f'<link href="{url}" color="blue">{clean_text(label)}</link>', body_style)
        link_items.append(ListItem(p, leftIndent=12))
    story.append(ListFlowable(link_items, bulletType="bullet"))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ── CSV inlezen ──────────────────────────────────────────────────────────────
WANTED_COLS = [
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

@st.cache_data(show_spinner=False)
def load_faq(preferred_path: Optional[str] = None) -> pd.DataFrame:
    candidate_paths = []
    if preferred_path:
        candidate_paths.append(preferred_path)
    # omgevingsvariabele of secrets
    env_path = os.getenv("FAQ_FILE") or st.secrets.get("FAQ_FILE", None)
    if env_path:
        candidate_paths.append(env_path)
    # veelgebruikte bestandsnamen
    candidate_paths += ["faq.csv", "fac.csv", "FAQ.csv", "faq (3).csv"]

    path = next((p for p in candidate_paths if p and os.path.exists(p)), None)
    if not path:
        st.error("Geen FAQ-bestand gevonden (gezocht naar: faq.csv / fac.csv / 'FAQ_FILE').")
        return pd.DataFrame(columns=WANTED_COLS + ["combined"])

    attempts = [
        ("utf-8-sig", None),
        ("utf-8", None),
        ("cp1252", ";"),
        ("cp1252", ","),
        ("latin1", ";"),
    ]
    df = None
    for enc, sep in attempts:
        try:
            df = pd.read_csv(path, encoding=enc, sep=sep, engine="python")
            break
        except Exception:
            df = None
    if df is None:
        st.error(f"Kon CSV niet lezen: {path}. Sla bij voorkeur op als UTF-8 (met ;).")
        return pd.DataFrame(columns=WANTED_COLS + ["combined"])

    # headers + ontbrekende kolommen
    df.columns = [str(c).strip() for c in df.columns]
    for c in WANTED_COLS:
        if c not in df.columns:
            df[c] = ""

    with pd.option_context("mode.chained_assignment", None):
        df["ID"] = pd.to_numeric(df.get("ID", ""), errors="coerce").astype("Int64")
        for c in WANTED_COLS:
            df[c] = (
                df[c]
                .fillna("")
                .astype(str)
                .str.replace("\ufeff", "", regex=False)
                .str.replace("\u00A0", " ", regex=False)
            )

    keep = ["Systeem", "Subthema", "Categorie", "Omschrijving melding", "Toelichting melding"]
    df["combined"] = df[keep].fillna("").astype(str).agg(" ".join, axis=1)

    return df[WANTED_COLS + ["combined"]]

faq_df = load_faq()


# ── Chat helpers + state ─────────────────────────────────────────────────────
TIMEZONE = pytz.timezone("Europe/Amsterdam")
ASSISTANT_AVATAR = "aichatbox.png" if os.path.exists("aichatbox.png") else None
USER_AVATAR = "parochie.png" if os.path.exists("parochie.png") else None

DEFAULT_STATE = {
    "history": [],
    "pdf_ready": False,
    "last_question": "",
    "last_item_label": "",
    "selected_answer_text": None,
    "selected_image": None,
    "chat_step": "greet",
    "chat_scope": None,
    "chat_results": [],
    "chat_greeted": False,
    "processing": False,
    "min_hits": 2,
    "min_cov": 0.25,
    "auto_simple": True,
    "allow_ai": True,
    "allow_web": False,
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

def get_avatar(role: str):
    return ASSISTANT_AVATAR if role == "assistant" and ASSISTANT_AVATAR else USER_AVATAR

def add_msg(role: str, content: str):
    ts = datetime.now(TIMEZONE).strftime("%d-%m-%Y %H:%M")
    if st.session_state.history and st.session_state.history[-1].get("content") == content and st.session_state.history[-1].get("role") == role:
        return
    st.session_state.history = (st.session_state.history + [{"role": role, "content": content, "time": ts}])[-12:]

def _copy_button(text: str, key_suffix: str):
    payload = text or ""
    js_text = json.dumps(payload)
    html_code = """
<div style="margin-top:8px;">
  <button id="COPY_BTN_ID" style="padding:6px 10px;font-size:16px;">
    Kopieer antwoord
  </button>
  <span id="COPY_STATE_ID" style="margin-left:8px;font-size:14px;"></span>
  <script>
    (function(){
      const btn = document.getElementById('COPY_BTN_ID');
      const state = document.getElementById('COPY_STATE_ID');
      if (btn) {
        btn.addEventListener('click', async () => {
          try {
            await navigator.clipboard.writeText(JS_TEXT);
            state.textContent = 'Gekopieerd!';
            setTimeout(() => { state.textContent = ''; }, 1500);
          } catch (e) {
            state.textContent = 'Niet gelukt — gebruik de tekst hieronder.';
            setTimeout(() => { state.textContent = ''; }, 3000);
          }
        });
      }
    })();
  </script>
</div>
""".replace("COPY_BTN_ID", f"copybtn-{key_suffix}") \
   .replace("COPY_STATE_ID", f"copystate-{key_suffix}") \
   .replace("JS_TEXT", js_text)
    components.html(html_code, height=70)
    with st.expander("Kopiëren lukt niet? Toon tekst om handmatig te kopiëren."):
        st.text_area("Tekst", payload, height=150, key=f"copy_fallback_{key_suffix}")

def render_chat():
    for i, m in enumerate(st.session_state.history):
        st.chat_message(m["role"], avatar=get_avatar(m["role"]))\
            .markdown(f"{m['content']}\n\n_{m['time']}_")
        if m["role"] == "assistant" and i == len(st.session_state.history) - 1 and st.session_state.get("pdf_ready", False):
            q = (st.session_state.get("last_question") or st.session_state.get("last_item_label") or "Vraag")
            pdf = make_pdf(q, m["content"])
            btn_key = f"pdf_{i}_{m['time'].replace(':','-')}"
            st.download_button("📄 Download PDF", data=pdf, file_name="antwoord.pdf", mime="application/pdf", key=btn_key)
            hash_key = hashlib.md5((m["time"] + m["content"]).encode("utf-8")).hexdigest()[:8]
            _copy_button(m["content"], hash_key)
            img = st.session_state.get("selected_image")
            if img and isinstance(img, str) and img.strip():
                try:
                    st.image(img, caption="Afbeelding bij dit antwoord", use_column_width=True)
                except Exception:
                    pass


# ── Zoek & simple explain ───────────────────────────────────────────────────
STOPWORDS_NL = {
    "de","het","een","en","of","maar","want","dus","als","dan","dat","die","dit","deze",
    "ik","jij","hij","zij","wij","jullie","u","ze","je","mijn","jouw","zijn","haar","ons","hun",
    "van","voor","naar","met","bij","op","in","aan","om","tot","uit","over","onder","boven","zonder",
    "ook","nog","al","wel","niet","nooit","altijd","hier","daar","ergens","niets","iets","alles",
    "is","was","wordt","zijn","heeft","heb","hebben","doe","doet","doen","kan","kunnen","moet","moeten"
}
def filter_topics(text: str) -> tuple[bool, str]:
    t = (text or "").lower()
    verboden = ["wachtwoord","password","pin","pincode","bsn","creditcard","cvv","iban","token","api key","apikey","geheim"]
    if any(w in t for w in verboden):
        return (False, "Voor uw veiligheid: deel geen wachtwoorden, codes of privégegevens. Formuleer uw vraag zonder deze gegevens.")
    return (True, "")

def _tokenize_clean(text: str) -> list[str]:
    return [w for w in re.findall(r"[0-9A-Za-zÀ-ÿ_]+", (text or "").lower())
            if len(w) > 2 and w not in STOPWORDS_NL]

def _token_score(q: str, text: str) -> int:
    qs = set(_tokenize_clean(q)); ts = set(_tokenize_clean(text))
    return len(qs & ts)

def zoek_hele_csv(vraag: str, min_hits: int = 2, min_cov: float = 0.25, fallback_rows: int = 50) -> pd.DataFrame:
    if faq_df.empty:
        return pd.DataFrame()
    df = faq_df.reset_index(drop=True).copy()
    q_tokens = set(_tokenize_clean(vraag))
    eff_min_hits = max(1, min(min_hits, len(q_tokens)))
    df["_score"] = df["combined"].apply(lambda t: _token_score(vraag, t))
    df = df.sort_values("_score", ascending=False)
    if df.empty:
        return df
    def _ok(row):
        qs = set(_tokenize_clean(vraag)); ts = set(_tokenize_clean(str(row["combined"])))
        hits = len(qs & ts); cov = hits / max(1, len(qs))
        return hits >= eff_min_hits and cov >= min_cov
    filtered = df[df.apply(_ok, axis=1)]
    if filtered.empty:
        nonzero = df[df["_score"] > 0].head(fallback_rows)
        return nonzero if not nonzero.empty else df.head(fallback_rows)
    return filtered

def simplify_text(txt: str, max_bullets: int = 5) -> str:
    text = clean_text(txt or "")
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    keys = ["klik", "open", "ga naar", "instellingen", "fout", "oplossing", "stap", "menu", "rapport", "boek", "opslaan", "zoeken"]
    scored = []
    for s in sentences:
        score = sum(1 for k in keys if k in s.lower())
        score += min(2, len(re.findall(r"\d+", s)))
        if len(s) <= 200:
            score += 1
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    bullets = [s for _, s in scored[:max_bullets]] or sentences[:min(max_bullets, 3)]
    out = "### In het kort\n" + "\n".join(f"- {b}" for b in bullets)
    steps = [s for s in sentences if re.search(r"^(Klik|Open|Ga|Kies|Vul|Controleer|Selecteer)\b", s.strip(), re.I)]
    if steps:
        out += "\n\n### Stappenplan\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps[:max_bullets]))
    return out.strip()

def simple_from_source(text: str) -> str:
    txt = (text or "").strip()
    if not txt:
        return ""
    if client is not None:
        try:
            return chatgpt_cached(
                [
                    {"role": "system", "content":
                     "Leg in eenvoudige Nederlandse woorden uit voor een vrijwilliger zonder technische kennis. "
                     "Gebruik maximaal 5 bullets en, indien nuttig, een kort stappenplan. "
                     "Baseer ALLES uitsluitend op de gegeven bron; geen aannames."},
                    {"role": "user", "content": f"Bron:\n{txt}\n\nMaak het simpel en concreet; voeg een stappenplan toe als dat helpt."}
                ],
                temperature=0.2, max_tokens=500
            )
        except Exception:
            pass
    return simplify_text(txt)

def enrich_with_simple(answer: str) -> str:
    simple = simple_from_source(answer)
    return (answer + "\n\n---\n\n" + simple) if simple else answer


# ── Chat-wizard ──────────────────────────────────────────────────────────────
def _detect_scope(msg: str) -> Optional[str]:
    t = (msg or "").lower()
    if any(w in t for w in ["exact", "eol", "e-online", "exact online"]):
        return "Exact"
    if any(w in t for w in ["docbase", "doc base"]):
        return "DocBase"
    if any(w in t for w in ["csv", "zoeken intern", "zoek in csv", "zoeken in csv", "zoeken"]):
        return "Zoeken"
    if any(w in t for w in ["internet", "web", "algemeen", "overig", "anders", "ik weet het niet"]):
        return "Algemeen"
    return None

def _mk_label(i: int, row: pd.Series) -> str:
    oms = clean_text(str(row.get('Omschrijving melding', '')).strip())
    toel = clean_text(str(row.get('Toelichting melding', '')).strip())
    preview = oms or toel or clean_text(str(row.get('Antwoord of oplossing', '')).strip())
    preview = re.sub(r"\s+", " ", preview)[:140]
    return f"{i+1:02d}. {preview}"

def chat_wizard():
    if st.session_state.get("processing"):
        return
    st.session_state["processing"] = True
    try:
        # knoppen
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("Exact", use_container_width=True):
            st.session_state["chat_scope"] = "Exact"; st.session_state["chat_step"] = "ask_topic"
            add_msg("assistant", "Prima. Kunt u in één zin beschrijven waar uw vraag over Exact Online over gaat?")
        if c2.button("DocBase", use_container_width=True):
            st.session_state["chat_scope"] = "DocBase"; st.session_state["chat_step"] = "ask_topic"
            add_msg("assistant", "Dank u. Kunt u in één zin beschrijven waar uw vraag over DocBase over gaat?")
        if c3.button("Zoeken", use_container_width=True):
            st.session_state["chat_scope"] = "Zoeken"; st.session_state["chat_step"] = "ask_topic"
            add_msg("assistant", "Waar wilt u in de CSV op zoeken? Typ een korte zoekterm.")
        if c4.button("Internet", use_container_width=True):
            st.session_state["chat_scope"] = "Algemeen"; st.session_state["chat_step"] = "ask_topic"
            add_msg("assistant", "Waarover gaat uw vraag? Beschrijf dit kort in één zin.")

        # begroeting
        if not st.session_state.get("chat_greeted", False):
            add_msg("assistant", "👋 Waarmee kan ik u van dienst zijn? U kunt hieronder typen of een snelkeuze gebruiken (Exact, DocBase, Zoeken of Internet).")
            st.session_state["chat_greeted"] = True

        render_chat()

        # input
        step = st.session_state.get("chat_step", "greet")
        scope = st.session_state.get("chat_scope")
        placeholders = {
            "greet": "Typ uw bericht…",
            "ask_scope": "Gaat uw vraag over Exact, DocBase of iets anders?",
            "ask_topic": f"Uw vraag over {scope or '…'} in één zin…",
            "pick_item": "Kies een van de opties of stel uw vraag anders.",
            "followup": "Heeft u een vervolgvraag over dit antwoord?",
        }
        user_text = st.chat_input(placeholders.get(step, "Stel uw vraag…"))
        if not user_text:
            return
        add_msg("user", user_text)

        # filters
        ok, warn = filter_topics(user_text)
        if not ok:
            st.session_state["pdf_ready"] = False
            add_msg("assistant", warn)
            return

        if step in ("greet", "ask_scope"):
            scope_guess = _detect_scope(user_text)
            if scope_guess is None:
                hits = zoek_hele_csv(user_text, min_hits=st.session_state["min_hits"], min_cov=st.session_state["min_cov"])
                if not hits.empty:
                    st.session_state["chat_scope"] = "Zoeken"
                    st.session_state["chat_results"] = hits.to_dict("records")
                    st.session_state["chat_step"] = "pick_item"
                    add_msg("assistant", "Ik heb een aantal mogelijke matches gevonden in onze CSV. Kies er één hieronder.")
                else:
                    st.session_state["chat_scope"] = "Algemeen"
                    st.session_state["chat_step"] = "ask_topic"
                    add_msg("assistant", "Kunt u in één zin beschrijven waar uw internetvraag over gaat?")
            else:
                st.session_state["chat_scope"] = scope_guess
                st.session_state["chat_step"] = "ask_topic"
                add_msg("assistant", f"Prima. Kunt u in één zin beschrijven waar uw vraag over **{scope_guess}** over gaat?")
            return

        if step == "ask_topic":
            st.session_state["last_question"] = user_text
            scope = st.session_state.get("chat_scope")
            if scope == "Algemeen":
                antwoord = None
                if client is not None:
                    try:
                        antwoord = chatgpt_cached(
                            [
                                {"role":"system","content":"Je helpt vrijwilligers van parochies. Antwoord kort en concreet. Noem geen niet-bestaande bronnen."},
                                {"role":"user","content": f"Vraag: {user_text}"}
                            ],
                            temperature=0.2, max_tokens=500
                        )
                    except Exception:
                        antwoord = None
                if not antwoord and st.session_state.get("allow_web"):
                    webbits = fetch_web_info_cached(user_text)
                    if webbits:
                        antwoord = webbits
                st.session_state["pdf_ready"] = True
                add_msg("assistant", (antwoord or "Kunt u uw vraag iets concreter maken?") + "\n\n" + AI_INFO)
                st.session_state["chat_step"] = "followup"
                return
            else:
                hits = zoek_hele_csv(user_text, min_hits=st.session_state["min_hits"], min_cov=st.session_state["min_cov"])
                if hits.empty:
                    add_msg("assistant", "Ik vond geen goede match in de CSV. Formuleer het anders of kies **Internet**.")
                    return
                st.session_state["chat_results"] = hits.to_dict("records")
                st.session_state["chat_step"] = "pick_item"
                add_msg("assistant", "Ik heb een aantal mogelijke matches gevonden. Kies hieronder het beste passende item.")
                return

        if step == "pick_item":
            # probeer nummer te herkennen
            m = re.search(r"\b(\d{1,2})\b", user_text)
            if not (m and st.session_state.get("chat_results")):
                add_msg("assistant", "Typ het nummer van het item (bijv. 2) of stel je vraag anders.")
                return
            idx = int(m.group(1)) - 1
            opts = st.session_state["chat_results"]
            if idx < 0 or idx >= len(opts):
                add_msg("assistant", "Ongeldig nummer. Probeer opnieuw.")
                return
            row = pd.Series(opts[idx])
            ans = clean_text(str(row.get('Antwoord of oplossing', '') or '').strip())
            if not ans:
                oms = clean_text(str(row.get('Omschrijving melding', '') or '').strip())
                ans = f"(Geen uitgewerkt antwoord in CSV voor: {oms})"
            label = _mk_label(idx, row)
            img = clean_text(str(row.get('Afbeelding', '') or '').strip())
            st.session_state["selected_image"] = img if img else None
            st.session_state["last_item_label"] = label
            st.session_state["last_question"] = f"Gekozen item: {label}"
            final_ans = enrich_with_simple(ans) if st.session_state.get("auto_simple", True) else ans
            st.session_state["selected_answer_text"] = ans
            st.session_state["pdf_ready"] = True
            add_msg("assistant", final_ans + "\n\n" + AI_INFO)
            st.session_state["chat_step"] = "followup"
            return

        if step == "followup":
            vraag2 = user_text
            st.session_state["last_question"] = vraag2
            bron = str(st.session_state.get("selected_answer_text") or "").strip()
            reactie = None
            if st.session_state.get("allow_ai") and client is not None and bron:
                try:
                    reactie = chatgpt_cached(
                        [
                            {"role": "system", "content": "Beantwoord uitsluitend op basis van de bron. Kort en duidelijk in het Nederlands."},
                            {"role": "user", "content": f"Bron:\n{bron}\n\nVervolgvraag: {vraag2}"}
                        ],
                        temperature=0.1, max_tokens=600,
                    )
                except Exception:
                    reactie = None
            if not reactie:
                reactie = simplify_text(bron) if bron else "Ik heb geen detail voor dit item in de CSV. Formuleer je vraag anders of kies **Internet**."
            st.session_state["pdf_ready"] = True
            add_msg("assistant", reactie + "\n\n" + AI_INFO)
            return
    finally:
        st.session_state["processing"] = False


# ── Web-fallback (optioneel) ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_web_info_cached(query: str) -> Optional[str]:
    result = []
    try:
        r = requests.get("https://docbase.nl", timeout=5); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        txt = " ".join(p.get_text(strip=True) for p in soup.find_all(['p','h1','h2','h3']))
        if txt and (query or "").lower() in txt.lower():
            result.append(f"Vanuit docbase.nl: {txt[:200]}... (verkort)")
    except Exception:
        pass
    try:
        r = requests.get("https://support.exactonline.com/community/s/knowledge-base", timeout=5); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        txt = " ".join(p.get_text(strip=True) for p in soup.find_all(['p','h1','h2','h3']))
        if txt and (query or "").lower() in txt.lower():
            result.append(f"Vanuit Exact Online Knowledge Base: {txt[:200]}... (verkort)")
    except Exception:
        pass
    return "\n".join(result) if result else None


# ── APP ──────────────────────────────────────────────────────────────────────
def main():
    # Intro (video of logo)
    video_path = "helpdesk.mp4"
    if os.path.exists(video_path):
        try:
            with open(video_path, "rb") as f:
                st.video(f.read(), format="video/mp4", start_time=0)
        except Exception as e:
            logging.error(f"Introvideo kon niet worden afgespeeld: {e}")
    elif os.path.exists("logo.png"):
        st.image("logo.png", width=244)
    else:
        st.info("Welkom bij IPAL Chatbox")

    st.header("Welkom bij IPAL Chatbox")

    # ——— Klassieke cascade (ALLEEN hier!) ————————————————
    with st.expander("Liever de klassieke cascade openen?"):
        if faq_df.empty:
            st.info("Geen FAQ-gegevens gevonden.")
        else:
            # Lokale helpers
            def _norm(v: str) -> str:
                return re.sub(r"\s+", " ", str(v).replace("\ufeff", "").replace("\u00A0", " ")).strip().lower()
            def _sortkey(x): 
                return _norm("" if x is None else x)
            def _disp(v: str) -> str:
                v = ("" if v is None else str(v)).strip()
                return v if v else "(Leeg)"

            dfv = faq_df.reset_index(drop=True).copy()
            st.caption("Volgorde: 1) Systeem → 2) Subthema → 3) Categorie → 4) Omschrijving → 5) Toelichting → 6) Soort → 7) Antwoord")

            # 1) Systeem
            sys_opts = sorted(dfv["Systeem"].dropna().unique(), key=_sortkey)
            sel_sys = st.selectbox("1) Systeem", sys_opts, key="classic_1_sys")
            step1 = dfv[dfv["Systeem"].apply(_norm) == _norm(sel_sys)]

            # 2) Subthema
            sub_opts = sorted(step1["Subthema"].dropna().unique(), key=_sortkey)
            sel_sub = st.selectbox("2) Subthema", sub_opts, key="classic_2_sub")
            step2 = step1[step1["Subthema"].apply(_norm) == _norm(sel_sub)]

            # 3) Categorie
            cat_opts = sorted(step2["Categorie"].dropna().unique(), key=_sortkey)
            sel_cat = st.selectbox("3) Categorie", cat_opts, key="classic_3_cat")
            step3 = step2[step2["Categorie"].apply(_norm) == _norm(sel_cat)]

            # 4) Omschrijving melding
            oms_opts = sorted(step3["Omschrijving melding"].fillna("").unique(), key=_sortkey)
            sel_oms = st.selectbox("4) Omschrijving melding", oms_opts, key="classic_4_oms")
            step4 = step3[step3["Omschrijving melding"].apply(_norm) == _norm(sel_oms)]

            # 5) Toelichting melding
            toe_vals = step4["Toelichting melding"].fillna("").map(_disp).unique()
            toe_opts = sorted(toe_vals, key=_sortkey)
            sel_toe_disp = st.selectbox("5) Toelichting melding", toe_opts, key="classic_5_toe")
            sel_toe_raw = "" if sel_toe_disp == "(Leeg)" else sel_toe_disp
            step5 = step4[step4["Toelichting melding"].fillna("").apply(_norm) == _norm(sel_toe_raw)]

            # 6) Soort melding
            soort_vals = step5["Soort melding"].fillna("").map(_disp).unique()
            soort_opts = sorted(soort_vals, key=_sortkey)
            sel_soort_disp = st.selectbox("6) Soort melding", soort_opts, key="classic_6_soort")
            sel_soort_raw = "" if sel_soort_disp == "(Leeg)" else sel_soort_disp
            step6 = step5[step5["Soort melding"].fillna("").apply(_norm) == _norm(sel_soort_raw)]

            # 7) Antwoord
            if step6.empty:
                st.warning("Geen overeenkomstige rij gevonden voor deze keuzes.")
            else:
                row = step6.iloc[0]
                st.markdown("**7) Antwoord of oplossing**")
                st.write((row.get("Antwoord of oplossing", "") or "").strip())

                # optioneel: afbeelding
                try:
                    img = (row.get("Afbeelding", "") or "").strip()
                    if img:
                        st.image(img, use_column_width=True)
                except Exception:
                    pass

    # ——— Chat-wizard (blijft zoals je gewend bent) —————————
    chat_wizard()


if __name__ == "__main__":
    main()
