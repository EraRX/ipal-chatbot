# IPAL Chatbox — main.py (simpel-weergave + nette PDF-opmaak)
# - Chat-wizard: Exact | DocBase | Zoeken | Internet
# - Klassieke cascade (Systeem → Subthema → Categorie → Omschrijving → Toelichting → Soort → Antwoord)
# - PDF met banner/logo + Copy
# - CSV robustness + smart quotes fix
# - Auto-simple uitleg (vervangt het ruwe bronantwoord i.p.v. eraan toe te voegen)
# - PDF: doorlopende 1..N, sub-bullets, checkboxes; AI-INFO + klikbare FAQ-links

import os
import re
import io
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional

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
    SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem, KeepTogether
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

      /* Sidebar/hamburger verbergen */
      [data-testid="stSidebar"],
      [data-testid="stSidebarNav"],
      [data-testid="collapsedControl"] { display: none !important; }

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

# ── Helpers: text cleanup ────────────────────────────────────────────────────
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
2. Heeft u hulp nodig met DocBase of Exact? Dan kunt u eenvoudig een melding maken door een ticket aan te maken in DocBase. Maar voordat u een ticket invult, controleer eerst onze FAQ (veelgestelde vragen en antwoorden). Klik hieronder om de FAQ te openen:

- [Veelgestelde vragen DocBase nieuw 2024](https://parochie-automatisering.nl/docbase/Templates/docbase?action=SelOpenDocument&DetailsMode=2&Docname=00328526&Type=INSTR_DOCS&LoginMode=1&LinkToVersion=1&OpenFileMode=2&password=%3Auzt7hs%23qL%2A%28&username=Externehyperlink&ID=0.07961651005089099&EC=1)
- [Veelgestelde vragen Exact Online](https://parochie-automatisering.nl/docbase/Templates/docbase?action=SelOpenDocument&DetailsMode=2&Docname=00328522&Type=INSTR_DOCS&LoginMode=1&LinkToVersion=1&OpenFileMode=2&password=%3Auzt7hs%23qL%2A%28&username=Externehyperlink&ID=0.8756321684738348&EC=1)
"""

if os.path.exists("Calibri.ttf"):
    pdfmetrics.registerFont(TTFont("Calibri", "Calibri.ttf"))

# ── Helpers voor PDF/chat-inhoud ────────────────────────────────────────────
def remove_ai_info(text: str) -> str:
    """Knipt het hele AI-INFO-blok uit de tekst (zodat we het netjes opnieuw kunnen zetten in de PDF)."""
    if not text:
        return ""
    pattern = r"(?is)\bAI[\-\u2011\u2013\u2014\u00A0 ]?Antwoord\s*Info\s*:\s*.*$"
    return re.sub(pattern, "", text).rstrip()

def _md_inline_to_paragraph_text(s: str) -> str:
    """Zet simpele Markdown inline om naar ReportLab-paragraph markup (bold/italic) en escape HTML-tekens."""
    from xml.sax.saxutils import escape
    if not s:
        return ""
    t = escape(s, entities={'"': "&quot;"})
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)                      # **vet**
    t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", t)            # *cursief*
    return t

def _parse_simple_markdown_to_flowables(text: str, styles) -> list:
    """
    PDF-opmaak van eenvoudige Markdown:
    - ### / #### koppen
    - Genummerde hoofd-stappen (1., 1) of '1 ' → DOORTELLING via ListFlowable(bulletType='1')
    - Sub-bullets (•, -, *), en checkboxbullets - [ ] / - [x] → ☐ / ☑
    - Losse bullets buiten OL blijven bullets
    """
    if not text:
        return []

    lines = [ln.rstrip() for ln in text.splitlines()]
    flow = []
    body = styles["Body"]; h3 = styles["H3"]; h4 = styles["H4"]

    ol_items = []                  # (title:str, subs:list[str])
    current_ol = None              # {"title": str, "subs": []}
    ul_buf = []                    # losse bullets buiten OL

    def flush_ul():
        nonlocal ul_buf
        if not ul_buf:
            return
        items = [ListItem(Paragraph(_md_inline_to_paragraph_text(it), body), leftIndent=12) for it in ul_buf]
        flow.append(ListFlowable(items, bulletType="bullet"))
        ul_buf = []

    def flush_ol():
        nonlocal ol_items, current_ol
        if current_ol:
            ol_items.append((current_ol["title"], current_ol["subs"]))
            current_ol = None
        if not ol_items:
            return
        list_items = []
        for title, subs in ol_items:
            title_para = Paragraph(_md_inline_to_paragraph_text(title), body)
            if subs:
                sub_flows = [ListItem(Paragraph(_md_inline_to_paragraph_text(s), body), leftIndent=12) for s in subs]
                sub_list = ListFlowable(sub_flows, bulletType="bullet")
                item_flow = KeepTogether([title_para, sub_list])
            else:
                item_flow = KeepTogether([title_para])
            list_items.append(ListItem(item_flow))
        flow.append(ListFlowable(list_items, bulletType="1"))
        ol_items = []

    for raw in lines:
        line = raw.strip()

        if not line:
            flush_ul()
            flow.append(Spacer(1, 6))
            continue

        # #### / ### headings sluiten buffers
        m4 = re.match(r"^####\s+(.*)$", line)
        if m4:
            flush_ul(); flush_ol()
            flow.append(Paragraph(_md_inline_to_paragraph_text(m4.group(1)), h4))
            continue

        m3 = re.match(r"^###\s+(.*)$", line)
        if m3:
            flush_ul(); flush_ol()
            flow.append(Paragraph(_md_inline_to_paragraph_text(m3.group(1)), h3))
            continue

        # Ordered list hoofd-stap: 1. ... of 1) ... of "1 ..." (met spatie)
        m_ol = re.match(r"^\d+(?:[.)]|\s)\s*(.*)$", line)
        if m_ol:
            if current_ol:
                ol_items.append((current_ol["title"], current_ol["subs"]))
            current_ol = {"title": m_ol.group(1), "subs": []}
            continue

        # Checkbox bullets: - [ ] ... of - [x] ...
        m_cb = re.match(r"^-\s+\[([ xX])\]\s+(.*)$", line)
        if m_cb:
            mark = "☑" if m_cb.group(1).lower() == "x" else "☐"
            txt  = f"{mark} {m_cb.group(2)}"
            if current_ol:
                current_ol["subs"].append(txt)
            else:
                ul_buf.append(txt)
            continue

        # Bullets: • ... of - ... of * ...
        m_ul = re.match(r"^[\u2022\-\*]\s+(.*)$", line)
        if m_ul:
            if current_ol:
                current_ol["subs"].append(m_ul.group(1))
            else:
                ul_buf.append(m_ul.group(1))
            continue

        # Anders: gewone alinea
        flush_ul(); flush_ol()
        flow.append(Paragraph(_md_inline_to_paragraph_text(line), body))

    flush_ul(); flush_ol()
    return flow

def _extract_ai_info_points(ai_info: str) -> list[str]:
    """Pak de twee genummerde regels (1., 2.) uit AI_INFO en voeg doorlopende regel(s) samen."""
    points = []
    cur = None
    for raw in (ai_info or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("ai-antwoord info"):
            continue
        if line.lower().startswith("klik hieronder"):
            break
        if re.match(r"^\-\s*\[", line):
            break
        m = re.match(r"^\d+[.)]\s+(.*)$", line)
        if m:
            if cur:
                points.append(cur.strip())
            cur = m.group(1)
        else:
            if cur:
                cur += " " + line
    if cur:
        points.append(cur.strip())
    return points[:2]

def _extract_ai_info_links(ai_info: str) -> list[tuple[str, str]]:
    """Pak '- [Label](URL)' regels uit AI_INFO als (label, url)."""
    links = []
    for raw in (ai_info or "").splitlines():
        m = re.match(r"^\-\s*\[([^\]]+)\]\(([^)]+)\)", raw.strip())
        if m:
            label = clean_text(m.group(1))
            url   = m.group(2).strip()
            links.append((label, url))
    return links

def make_pdf(question: str, answer: str) -> bytes:
    """
    Bouwt de PDF:
    - Antwoord met nette layout (koppen, doorlopende 1..N, sub-bullets, checkboxes)
    - Onderaan: AI-Antwoord Info (2 genummerde regels)
    - Daaronder: 'Klik hieronder...' met twee klikbare FAQ-links (zonder kale URL-tekst)
    """
    from xml.sax.saxutils import escape

    question = clean_text(question or "")
    # Knip AI-INFO uit de hoofdtekst; we voegen die onderaan gestructureerd toe.
    answer_no_info = remove_ai_info(answer or "")

    buffer = io.BytesIO()
    left = right = top = bottom = 2 * cm
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=left, rightMargin=right, topMargin=top, bottomMargin=bottom
    )
    content_width = A4[0] - left - right

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Body",    parent=styles["Normal"],   fontName="Helvetica",       fontSize=11, leading=16, spaceAfter=10, alignment=TA_LEFT))
    styles.add(ParagraphStyle("Heading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#333"), spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle("H3",      parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=12, leading=16, spaceBefore=8,  spaceAfter=4))
    styles.add(ParagraphStyle("H4",      parent=styles["Heading4"], fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=6,  spaceAfter=2))

    story = []

    # Banner/logo (optioneel)
    if os.path.exists("logopdf.png"):
        try:
            banner = Image("logopdf.png")
            banner._restrictSize(content_width, 10000)
            banner.hAlign = "LEFT"
            story.append(banner)
            story.append(Spacer(1, 8))
        except Exception as e:
            logging.error(f"Kon banner niet laden: {e}")

    # Vraag
    story.append(Paragraph(f"Vraag: {question}", styles["Heading"]))
    story.append(Spacer(1, 10))

    # Antwoord
    story.append(Paragraph("Antwoord:", styles["Heading"]))
    story.extend(_parse_simple_markdown_to_flowables(answer_no_info, styles))

    # AI-Antwoord Info (twee genummerde regels)
    ai_points = _extract_ai_info_points(AI_INFO)
    if ai_points:
        story.append(Spacer(1, 12))
        story.append(Paragraph("AI-Antwoord Info:", styles["Heading"]))
        items = [ListItem(Paragraph(_md_inline_to_paragraph_text(p), styles["Body"]), leftIndent=12) for p in ai_points]
        story.append(ListFlowable(items, bulletType="1"))

    # “Klik hieronder…” + klikbare links
    ai_links = _extract_ai_info_links(AI_INFO)
    if ai_links:
        story.append(Spacer(1, 6))
        story.append(Paragraph("Klik hieronder om de FAQ te openen:", styles["H4"]))
        link_items = []
        for label, url in ai_links:
            p = Paragraph(f'<link href="{escape(url)}" color="blue">{_md_inline_to_paragraph_text(label)}</link>', styles["Body"])
            link_items.append(ListItem(p, leftIndent=12))
        story.append(ListFlowable(link_items, bulletType="bullet"))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ── CSV loading + normalization ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_faq(path: str = "faq.csv") -> pd.DataFrame:
    cols = [
        "ID", "Systeem", "Subthema", "Categorie",
        "Omschrijving melding", "Toelichting melding", "Soort melding",
        "Antwoord of oplossing", "Afbeelding"
    ]
    if not os.path.exists(path):
        st.error(f"FAQ-bestand '{path}' niet gevonden.")
        return pd.DataFrame(columns=cols).set_index(["Systeem", "Subthema", "Categorie"])

    try:
        df = pd.read_csv(path, encoding="utf-8", sep=";")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", sep=";")


    for c in cols:
        if c not in df.columns:
            df[c] = None

    norm_cols = [
        "Systeem", "Subthema", "Categorie",
        "Omschrijving melding", "Toelichting melding",
        "Soort melding", "Antwoord of oplossing", "Afbeelding"
    ]
    for c in norm_cols:
        df[c] = (df[c]
                 .fillna("")
                 .astype(str)
                 .str.replace("\u00A0", " ", regex=False)
                 .str.strip()
                 .str.replace(r"\s+", " ", regex=True))
        df[c] = df[c].apply(clean_text)

    # Normalize Systeem
    sys_raw = df["Systeem"].astype(str).str.lower().str.strip()
    direct_map = {
        "exact": "Exact",
        "exact online": "Exact",
        "eol": "Exact",
        "e-online": "Exact",
        "e online": "Exact",
        "docbase": "DocBase",
        "doc base": "DocBase",
        "sila": "DocBase",
        "algemeen": "Algemeen",
    }
    df["Systeem"] = sys_raw.replace(direct_map)

    # Combined search field
    keep = ["Systeem", "Subthema", "Categorie", "Omschrijving melding", "Toelichting melding"]
    df["combined"] = df[keep].fillna("").agg(" ".join, axis=1)

    # Index voor cascade
    try:
        df = df.set_index(["Systeem", "Subthema", "Categorie"], drop=True)
    except Exception:
        st.warning("Kon index niet goed zetten — controleer CSV kolommen Systeem/Subthema/Categorie")
        df = df.reset_index(drop=True)

    return df

faq_df = load_faq()

STOPWORDS_NL = {
    "de","het","een","en","of","maar","want","dus","als","dan","dat","die","dit","deze",
    "ik","jij","hij","zij","wij","jullie","u","ze","je","mijn","jouw","zijn","haar","ons","hun",
    "van","voor","naar","met","bij","op","in","aan","om","tot","uit","over","onder","boven","zonder",
    "ook","nog","al","wel","niet","nooit","altijd","hier","daar","ergens","niets","iets","alles",
    "is","was","wordt","zijn","heeft","heb","hebben","doe","doet","doen","kan","kunnen","moet","moeten"
}

def filter_topics(text: str) -> tuple[bool, str]:
    t = (text or "").lower()
    verboden = ["wachtwoord", "password", "pin", "pincode", "bsn", "creditcard", "cvv", "iban", "token", "api key", "apikey", "geheim"]
    if any(w in t for w in verboden):
        return (False, "Voor uw veiligheid: deel geen wachtwoorden, codes of privégegevens. Formuleer uw vraag zonder deze gegevens.")
    return (True, "")

def _tokenize_clean(text: str) -> list[str]:
    return [w for w in re.findall(r"[0-9A-Za-zÀ-ÿ_]+", (text or "").lower()) if len(w) > 2 and w not in STOPWORDS_NL]

def _relevance(q: str, t: str) -> tuple[int, float]:
    qs = set(_tokenize_clean(q)); ts = set(_tokenize_clean(t))
    hits = len(qs & ts); coverage = hits / max(1, len(qs))
    return hits, coverage

def _token_score(q: str, text: str) -> int:
    qs = set(_tokenize_clean(q)); ts = set(_tokenize_clean(text))
    return len(qs & ts)

# ── Zoekfuncties ────────────────────────────────────────────────────────────
def find_answer_by_codeword(df: pd.DataFrame, codeword: str = "[UNIEKECODE123]") -> Optional[str]:
    try:
        mask = df["Antwoord of oplossing"].astype(str).str.contains(codeword, case=False, na=False)
        if mask.any():
            return str(df.loc[mask].iloc[0]["Antwoord of oplossing"]).strip()
    except Exception:
        pass
    return None

def zoek_hele_csv(vraag: str, min_hits: int = 2, min_cov: float = 0.25, fallback_rows: int = 50) -> pd.DataFrame:
    if faq_df.empty:
        return pd.DataFrame()
    df = faq_df.reset_index().copy()
    q_tokens = set(_tokenize_clean(vraag))
    eff_min_hits = max(1, min(min_hits, len(q_tokens)))
    df["_score"] = df["combined"].apply(lambda t: _token_score(vraag, t))
    df = df.sort_values("_score", ascending=False)
    if df.empty:
        return df
    def _ok(row):
        hits, cov = _relevance(vraag, str(row["combined"]))
        return hits >= eff_min_hits and cov >= min_cov
    filtered = df[df.apply(_ok, axis=1)]
    if filtered.empty:
        q_lower = (vraag or "").strip().lower()
        sys_map = {"exact": "Exact", "docbase": "DocBase", "algemeen": "Algemeen"}
        if q_lower in sys_map:
            try:
                subset = faq_df.xs(sys_map[q_lower], level="Systeem", drop_level=False).reset_index()
                subset["_score"] = subset["combined"].apply(lambda t: _token_score(vraag, t))
                return subset.sort_values("_score", ascending=False).head(fallback_rows)
            except KeyError:
                pass
        nonzero = df[df["_score"] > 0].head(fallback_rows)
        return nonzero if not nonzero.empty else df.head(fallback_rows)
    return filtered

def zoek_in_scope(scope: Optional[str], vraag: str, topn: int = 8) -> pd.DataFrame:
    base = faq_df.reset_index()
    if scope in ("Exact", "DocBase"):
        base = base[base["Systeem"].astype(str).str.lower() == scope.lower()]
    if base.empty:
        return base
    base = base.copy()
    base["_score"] = base["combined"].apply(lambda t: _token_score(vraag, t))
    base = base.sort_values(["_score"], ascending=False)
    good = base[base["_score"] > 0]
    return (good if not good.empty else base).head(topn)

def vind_best_algemeen_AI(vraag: str) -> str:
    if client is None:
        return "Kunt u uw vraag iets concreter maken (bijv. ‘DocBase wachtwoord resetten’ of ‘Exact bankkoppeling’)?"
    sys = (
        "Je helpt vrijwilligers van parochies. "
        "Beantwoord kort en concreet. Stel maximaal één verhelderende vraag als dat echt nodig is. "
        "Noem geen niet-bestaande bronnen."
    )
    user = (
        f"Vraag: {vraag}\n\n"
        "Als de vraag niet direct over DocBase/Exact/IPAL-onderwerpen gaat, geef dan 1- of 2 praktische vervolgsuggesties "
        "of verwijs vriendelijk naar het juiste kanaal. Stel hoogstens één verhelderende vraag."
    )
    try:
        return chatgpt_cached(
            [{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=0.2, max_tokens=500
        )
    except Exception as e:
        logging.error(f"AI (Internet) fout: {e}")
        return "Kunt u uw vraag iets concreter maken?"

# ── Auto-simple explanation ──────────────────────────────────────────────────
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

    bullets = [s for _, s in scored[:max_bullets]]
    if not bullets:
        bullets = sentences[:min(max_bullets, 3)]

    out = "### Uitleg voor Vrijwilliger\n\n"
    out += "#### In het kort\n"
    for b in bullets:
        out += f"- {b}\n"

    steps = [s for s in sentences if re.search(r"^(Klik|Open|Ga|Kies|Vul|Controleer|Selecteer)\b", s.strip(), re.I)]
    if steps:
        out += "\n#### Stappenplan\n"
        for i, s in enumerate(steps[:max_bullets], 1):
            out += f"{i}. {s}\n"

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
                     "Schrijf in eenvoudig Nederlands voor een vrijwilliger zonder technische kennis. "
                     "Gebruik maximaal 5 bullets en (indien nuttig) een kort stappenplan. "
                     "Baseer ALLES uitsluitend op de gegeven bron."},
                    {"role": "user", "content": f"Bron:\n{txt}\n\nMaak het kort en concreet; een klein stappenplan is prima."}
                ],
                temperature=0.2, max_tokens=500
            )
        except Exception:
            pass
    return simplify_text(txt)

def enrich_with_simple(answer: str) -> str:
    """Vervangt het ruwe bronantwoord door de eenvoudige uitleg."""
    simple = simple_from_source(answer)
    return simple or (answer or "")

# ── Web-fallback (optioneel) ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_web_info_cached(query: str) -> Optional[str]:
    result = []
    try:
        r = requests.get("https://docbase.nl", timeout=5); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        txt = " " .join(p.get_text(strip=True) for p in soup.find_all(['p','h1','h2','h3']))
        if txt and (query or "").lower() in txt.lower():
            result.append(f"Vanuit docbase.nl: {txt[:200]}... (verkort)")
    except Exception:
        pass
    try:
        r = requests.get("https://support.exactonline.com/community/s/knowledge-base", timeout=5); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        txt = " " .join(p.get_text(strip=True) for p in soup.find_all(['p','h1','h2','h3']))
        if txt and (query or "").lower() in txt.lower():
            result.append(f"Vanuit Exact Online Knowledge Base: {txt[:200]}... (verkort)")
    except Exception:
        pass
    return "\n".join(result) if result else None

# ── UI helpers & state ───────────────────────────────────────────────────────
TIMEZONE = pytz.timezone("Europe/Amsterdam")
ASSISTANT_AVATAR = "aichatbox.png" if os.path.exists("aichatbox.png") else None
USER_AVATAR = "parochie.png" if os.path.exists("parochie.png") else None

DEFAULT_STATE = {
    "history": [],
    "selected_answer_text": None,
    "selected_image": None,
    "last_question": "",
    "last_item_label": "",
    "allow_ai": True,
    "allow_web": False,
    "auto_simple": True,
    "chat_mode": True,
    "chat_step": "greet",
    "chat_scope": None,
    "chat_results": [],
    "chat_greeted": False,
    "actionbar": None,  # dict: {question, content, image, time}
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

MAX_HISTORY = 12

def get_avatar(role: str):
    return ASSISTANT_AVATAR if role == "assistant" and ASSISTANT_AVATAR else USER_AVATAR

def add_msg(role: str, content: str):
    ts = datetime.now(TIMEZONE).strftime("%d-%m-%Y %H:%M")
    if st.session_state.history and st.session_state.history[-1].get("content") == content and st.session_state.history[-1].get("role") == role:
        return
    st.session_state.history = (st.session_state.history + [{"role": role, "content": content, "time": ts}])[-MAX_HISTORY:]

def with_info(text: str) -> str:
    txt = (text or "").strip()
    # GEEN clean_text hier → newlines blijven staan en Markdown rendert netjes
    return f"{txt}\n\n{AI_INFO}".strip()

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
    try:
        components.html(html_code, height=70)
    except Exception:
        st.info("Automatisch kopiëren niet beschikbaar. Gebruik de tekst hieronder.")
    show_fallback = st.checkbox("Kopiëren lukt niet? Toon tekst om handmatig te kopiëren.", key=f"copy_show_{key_suffix}")
    if show_fallback:
        st.text_area("Tekst", payload, height=150, key=f"copy_fallback_{key_suffix}")

def _render_actionbar():
    ab = st.session_state.get("actionbar")
    if not ab:
        return
    st.divider()
    st.caption("Acties voor het laatste antwoord:")
    pdf = make_pdf(ab["question"], ab["content"])  # content bevat AI_INFO; wordt in PDF opnieuw netjes opgebouwd
    st.download_button(
        "📄 Download PDF", data=pdf, file_name="antwoord.pdf",
        mime="application/pdf", key=f"pdf_{hash(ab['question']+ab['content'])}"
    )
    hash_key = hashlib.md5((ab["question"] + ab["content"]).encode("utf-8")).hexdigest()[:8]
    _copy_button(ab["content"], hash_key)
    img = ab.get("image")
    if img and isinstance(img, str) and img.strip():
        try:
            st.image(img, caption="Afbeelding bij dit antwoord", use_column_width=True)
        except Exception:
            pass

def render_chat():
    seen = set()
    for m in st.session_state.history:
        key = (m["role"], m["content"], m["time"])
        if key in seen:
            continue
        seen.add(key)
        st.chat_message(m["role"], avatar=get_avatar(m["role"]))\
          .markdown(f"{m['content']}\n\n_{m['time']}_")
    _render_actionbar()

# ── Conversatie-wizard ───────────────────────────────────────────────────────
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
    render_chat()

    # Knoppen
    with st.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        if c1.button("Exact", key="wizard_exact", use_container_width=True):
            st.session_state.update({"chat_scope": "Exact", "chat_step": "ask_topic"})
            add_msg("assistant", "Prima. Kunt u in één zin beschrijven waar uw vraag over Exact Online over gaat?")
            st.rerun()
        if c2.button("DocBase", key="wizard_docbase", use_container_width=True):
            st.session_state.update({"chat_scope": "DocBase", "chat_step": "ask_topic"})
            add_msg("assistant", "Dank u. Kunt u in één zin beschrijven waar uw vraag over DocBase over gaat?")
            st.rerun()
        if c3.button("Zoeken", key="wizard_search", use_container_width=True):
            st.session_state.update({"chat_scope": "Zoeken", "chat_step": "ask_topic"})
            add_msg("assistant", "Waar wilt u in de kennisbank op zoeken? Typ een korte zoekterm.")
            st.rerun()
        if c4.button("Internet", use_container_width=True, key="wizard_internet"):
            st.session_state.update({"chat_scope": "Algemeen", "chat_step": "ask_topic"})
            add_msg("assistant", "Waarover gaat uw vraag? Beschrijf dit kort in één zin.")
            st.rerun()
        if c5.button("🔄 Reset", key="wizard_reset", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            for k, v in DEFAULT_STATE.items():
                st.session_state[k] = v
            st.cache_data.clear()
            st.rerun()

    if not st.session_state.get("chat_greeted", False):
        add_msg("assistant", "👋 Waarmee kan ik u van dienst zijn? U kunt hieronder typen of een snelkeuze gebruiken (Exact, DocBase, Zoeken of Internet).")
        st.session_state["chat_greeted"] = True
        render_chat()

    step = st.session_state.get("chat_step", "greet")
    scope = st.session_state.get("chat_scope")
    placeholders = {
        "greet": "Typ uw bericht…",
        "ask_scope": "Gaat uw vraag over Exact, DocBase of iets anders?",
        "ask_topic": f"Uw vraag over {scope or '…'} in één zin…",
        "pick_item": "Kies een van de opties of stel uw vraag anders.",
        "followup": "Heeft u een vervolgvraag over dit antwoord?",
    }

    user_text = st.chat_input(placeholders.get(step, "Stel uw vraag…"), key=f"chat_input_{step}_{scope or 'none'}")
    if not user_text:
        if step == "pick_item" and st.session_state.get("chat_results"):
            opts = st.session_state["chat_results"]
            labels = [_mk_label(i, pd.Series(r)) for i, r in enumerate(opts)]
            chosen = st.radio("Kies het beste passende item:", labels, key=f"radio_{hash(str(opts))}")
            if st.button("Toon antwoord", key=f"toon_{hash(str(opts))}"):
                idx = labels.index(chosen)
                _show_item_answer(idx)
                st.rerun()
        return

    add_msg("user", user_text)

    # Veiligheid
    ok, warn = filter_topics(user_text)
    if not ok:
        add_msg("assistant", warn)
        st.rerun(); return

    # UNIEKECODE123
    if (user_text or "").strip().upper() == "UNIEKECODE123":
        cw = find_answer_by_codeword(faq_df.reset_index())
        if cw:
            st.session_state["last_question"] = user_text
            content = with_info(cw)
            add_msg("assistant", content)
            st.session_state["actionbar"] = {
                "question": st.session_state.get("last_question") or "Vraag",
                "content": content,
                "image": st.session_state.get("selected_image"),
                "time": datetime.now(TIMEZONE).isoformat()
            }
            st.rerun(); return

    if step in ("greet", "ask_scope"):
        scope_guess = _detect_scope(user_text)
        if scope_guess is None and user_text.strip().lower() in ("ik heb een vraag", "ik heb een vraag.", "vraag", "hallo", "goedemiddag"):
            st.session_state["chat_step"] = "ask_scope"
            add_msg("assistant", "Gaat uw vraag over **Exact**, **DocBase** of **iets anders**?")
            st.rerun(); return
        if scope_guess is None:
            hits = zoek_in_scope(None, user_text, topn=6)
            if not hits.empty:
                st.session_state["chat_scope"] = "Zoeken"
                st.session_state["chat_results"] = hits.to_dict("records")
                st.session_state["chat_step"] = "pick_item"
                add_msg("assistant", "Ik heb een aantal mogelijke matches gevonden in onze kennisbank. Kies er één hieronder.")
                st.rerun(); return
            else:
                st.session_state["chat_scope"] = "Algemeen"
                st.session_state["chat_step"] = "ask_topic"
                add_msg("assistant", "Kunt u in één zin beschrijven waar uw internetvraag over gaat?")
                st.rerun(); return
        else:
            st.session_state["chat_scope"] = scope_guess
            st.session_state["chat_step"] = "ask_topic"
            add_msg("assistant", f"Prima. Kunt u in één zin beschrijven waar uw vraag over **{scope_guess}** over gaat?")
            st.rerun(); return

    if step == "ask_topic":
        scope = st.session_state.get("chat_scope")
        if scope == "Algemeen":
            st.session_state["last_question"] = user_text
            antwoord = vind_best_algemeen_AI(user_text)
            if not antwoord and st.session_state.get("allow_web"):
                webbits = fetch_web_info_cached(user_text)
                if webbits:
                    antwoord = webbits
            content = with_info(antwoord or "Kunt u uw vraag iets concreter maken?")
            add_msg("assistant", content)
            st.session_state["actionbar"] = {
                "question": st.session_state.get("last_question") or "Vraag",
                "content": content,
                "image": st.session_state.get("selected_image"),
                "time": datetime.now(TIMEZONE).isoformat()
            }
            st.session_state["chat_step"] = "followup"
            st.rerun(); return
        else:
            st.session_state["last_question"] = user_text
            hits = zoek_in_scope(None if scope == "Zoeken" else scope, user_text, topn=8)
            if hits.empty:
                add_msg("assistant", "Ik vond geen goede match in de kennisbank. Formuleer het iets anders of kies **Internet**.")
                st.rerun(); return
            st.session_state["chat_results"] = hits.to_dict("records")
            st.session_state["chat_step"] = "pick_item"
            add_msg("assistant", "Ik heb een aantal mogelijke matches gevonden. Kies hieronder het beste passende item.")
            st.rerun(); return

    if step == "pick_item":
        m = re.search(r"\b(\d{1,2})\b", user_text)
        if m and st.session_state.get("chat_results"):
            idx = int(m.group(1)) - 1
            _show_item_answer(idx)
            st.rerun(); return
        add_msg("assistant", "Gebruik de selectie hierboven om een item te kiezen, of typ het nummer (bijv. 2).")
        st.rerun(); return

    if step == "followup":
        vraag2 = user_text
        st.session_state["last_question"] = vraag2
        low = vraag2.strip().lower()
        if any(k in low for k in ["ander", "andere", "volgende", "nog een", "iets anders"]):
            scope = st.session_state.get("chat_scope")
            hits = zoek_in_scope(None if scope == "Zoeken" else scope, vraag2, topn=8)
            if hits.empty:
                add_msg("assistant", "Ik vond geen andere goede match. Formuleer het anders of kies **Internet**.")
            else:
                st.session_state["chat_results"] = hits.to_dict("records")
                st.session_state["chat_step"] = "pick_item"
                add_msg("assistant", "Ik heb nieuwe opties gevonden. Kies hieronder een ander item.")
            st.rerun(); return

        ok, warn = filter_topics(vraag2)
        if not ok:
            add_msg("assistant", warn)
            st.rerun(); return

        bron = str(st.session_state.get("selected_answer_text") or "").strip()
        reactie = None
        if st.session_state.get("allow_ai") and client is not None and bron:
            try:
                reactie = chatgpt_cached(
                    [
                        {"role": "system", "content": "Beantwoord uitsluitend op basis van de meegegeven bron. Geen aannames buiten de bron. Schrijf kort en duidelijk in het Nederlands."},
                        {"role": "user", "content": f"Bron:\n{bron}\n\nVervolgvraag: {vraag2}"}
                    ],
                    temperature=0.1, max_tokens=600,
                )
            except Exception as e:
                logging.error(f"AI-QA fout: {e}")
                reactie = None

        if not reactie:
            reactie = simplify_text(bron) if bron else "Ik heb geen detail voor dit item in de kennisbank. Kies een ander item of stel de vraag via **Internet**."

        content = with_info(reactie)
        add_msg("assistant", content)
        st.session_state["actionbar"] = {
            "question": st.session_state.get("last_question") or st.session_state.get("last_item_label") or "Vraag",
            "content": content,
            "image": st.session_state.get("selected_image"),
            "time": datetime.now(TIMEZONE).isoformat()
        }
        st.rerun(); return

def _show_item_answer(idx: int):
    """Toon een geselecteerd CSV-item als antwoord (wizard)."""
    opts = st.session_state["chat_results"]
    if not (0 <= idx < len(opts)):
        add_msg("assistant", "Ongeldige keuze.")
        return
    row = pd.Series(opts[idx])
    ans = clean_text(str(row.get('Antwoord of oplossing', '') or '').strip())
    if not ans:
        oms = clean_text(str(row.get('Omschrijving melding', '')).strip())
        ans = f"(Geen uitgewerkt antwoord in onze kennisbank voor: {oms})"
    label = _mk_label(idx, row)
    img = clean_text(str(row.get('Afbeelding', '') or '').strip())
    st.session_state["selected_image"] = img if img else None
    st.session_state["last_item_label"] = label
    st.session_state["last_question"] = f"Gekozen item: {label}"
    final_ans = enrich_with_simple(ans) if st.session_state.get("auto_simple", True) else ans
    st.session_state["selected_answer_text"] = ans
    content = with_info(final_ans)
    add_msg("assistant", content)
    st.session_state["actionbar"] = {
        "question": st.session_state.get("last_question") or "Vraag",
        "content": content,
        "image": st.session_state.get("selected_image"),
        "time": datetime.now(TIMEZONE).isoformat()
    }
    st.session_state["chat_step"] = "followup"

# ── App ──────────────────────────────────────────────────────────────────────
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

    # ── Klassieke cascade (expander) ─────────────────────────────────────────
    with st.expander("Maak hier uw keuze:"):
        if faq_df is None or faq_df.empty:
            st.info("Geen FAQ-gegevens gevonden.")
        else:
            dfv = faq_df.reset_index(drop=False).copy()

            def _norm(v: str) -> str:
                return re.sub(r"\s+", " ", str(v).replace("\ufeff","").replace("\u00A0"," ")).strip().lower()

            def _disp(v: str) -> str:
                v = ("" if v is None else str(v)).strip()
                return v if v else "(Leeg)"

            def _opts(series: pd.Series) -> list[str]:
                return sorted({_disp(x) for x in series.dropna().astype(str).tolist()}, key=lambda x: _norm(x))

            st.caption("Volgorde: 1) Systeem → 2) Subthema → 3) Categorie → 4) Omschrijving → 5) Toelichting → 6) Soort → 7) Antwoord")

            # 1) Systeem
            sys_opts = _opts(dfv["Systeem"])
            sel_sys = st.selectbox("1) Systeem", ["(Kies)"] + sys_opts, key="c1_sys")
            step1 = dfv[dfv["Systeem"].apply(_norm) == _norm(sel_sys)] if sel_sys != "(Kies)" else pd.DataFrame(columns=dfv.columns)

            # 2) Subthema
            sub_opts = _opts(step1["Subthema"]) if not step1.empty else []
            sel_sub = st.selectbox("2) Subthema", ["(Kies)"] + sub_opts, key="c2_sub")
            step2 = step1[step1["Subthema"].apply(_norm) == _norm(sel_sub)] if sel_sub != "(Kies)" else pd.DataFrame(columns=dfv.columns)

            # 3) Categorie
            cat_opts = _opts(step2["Categorie"]) if not step2.empty else []
            sel_cat = st.selectbox("3) Categorie", ["(Kies)"] + cat_opts, key="c3_cat")
            step3 = step2[step2["Categorie"].apply(_norm) == _norm(sel_cat)] if sel_cat != "(Kies)" else pd.DataFrame(columns=dfv.columns)

            # 4) Omschrijving melding
            oms_opts = _opts(step3["Omschrijving melding"]) if not step3.empty else []
            sel_oms = st.selectbox("4) Omschrijving melding", ["(Kies)"] + oms_opts, key="c4_oms")
            step4 = step3[step3["Omschrijving melding"].apply(_norm) == _norm(sel_oms)] if sel_oms != "(Kies)" else pd.DataFrame(columns=dfv.columns)

            # 5) Toelichting melding
            toe_opts = _opts(step4["Toelichting melding"]) if not step4.empty else []
            sel_toe = st.selectbox("5) Toelichting melding", ["(Kies)"] + toe_opts, key="c5_toe")
            sel_toe_raw = "" if sel_toe in ("(Kies)", "(Leeg)") else sel_toe
            step5 = step4[step4["Toelichting melding"].fillna("").apply(_norm) == _norm(sel_toe_raw)] if not step4.empty else pd.DataFrame(columns=dfv.columns)

            # 6) Soort melding
            soort_opts = _opts(step5["Soort melding"]) if not step5.empty else []
            sel_soort = st.selectbox("6) Soort melding", ["(Kies)"] + soort_opts, key="c6_soort")
            sel_soort_raw = "" if sel_soort in ("(Kies)", "(Leeg)") else sel_soort
            step6 = step5[step5["Soort melding"].fillna("").apply(_norm) == _norm(sel_soort_raw)] if not step5.empty else pd.DataFrame(columns=dfv.columns)

            # 7) Antwoord of oplossing
            if sel_soort != "(Kies)":
                st.markdown("**7) Antwoord of oplossing**")

                if step6.empty:
                    st.warning("Geen overeenkomstige rij gevonden voor deze keuzes.")
                else:
                    row = step6.iloc[0]
                    antwoord   = (row.get("Antwoord of oplossing", "") or "").strip()
                    afbeelding = (row.get("Afbeelding", "") or "").strip()

                    # Eenvoudige uitleg i.p.v. ruwe bron (fallback indien leeg)
                    final_ans = enrich_with_simple(antwoord) if st.session_state.get("auto_simple", True) else antwoord
                    if not final_ans:
                        final_ans = "_Geen uitgewerkt antwoord in de kennisbank voor deze combinatie._"

                    # Toon antwoord + AI-INFO binnen de cascade
                    display_ans = f"{final_ans}\n\n{AI_INFO}"
                    st.markdown(display_ans)

                    # Eventuele afbeelding
                    if afbeelding:
                        try:
                            st.image(afbeelding, use_column_width=True)
                            st.session_state["selected_image"] = afbeelding
                        except Exception:
                            pass

                    # Labels / context voor PDF & actie-balk
                    label = " › ".join([x for x in [sel_sys, sel_sub, sel_cat, sel_oms] if x and x != "(Kies)"])
                    st.session_state["last_item_label"] = label
                    st.session_state["last_question"]   = label

                    # Kopieer en PDF (PDF bouwt AI-INFO en links zelf in nette vorm op)
                    _copy_button(display_ans, hashlib.md5(display_ans.encode("utf-8")).hexdigest()[:8])
                    pdf = make_pdf(label, display_ans)
                    st.download_button(
                        "📄 Download PDF",
                        data=pdf,
                        file_name="antwoord.pdf",
                        mime="application/pdf",
                        key=f"cascade_pdf_{hash(label+display_ans)}"
                    )

                    # Actie-balk onderaan met dezelfde inhoud als zichtbaar
                    st.session_state["actionbar"] = {
                        "question": label,
                        "content": display_ans,
                        "image": st.session_state.get("selected_image"),
                        "time": datetime.now(TIMEZONE).isoformat()
                    }

    # ── Wizard ───────────────────────────────────────────────────────────────
    if st.session_state.get("chat_mode", True):
        chat_wizard()
        return

if __name__ == "__main__":
    main()





